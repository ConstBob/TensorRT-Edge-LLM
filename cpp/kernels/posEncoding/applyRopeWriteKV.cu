/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "applyRopeWriteKV.h"

#include <cstdint>
#include <cuda_fp16.h>

namespace drivellm
{
namespace kernel
{

// Define template type to load/store vectorized data.
template <typename T>
struct DVec
{
    static constexpr uint32_t vec_size = 0;
    inline T& operator[](uint32_t idx);
    inline T const& operator[](uint32_t idx) const;
    inline void load(T const* ptr);
    inline void store(T* ptr) const;
};

// Store float[8] to align with load/store of activation data.
// Use this to load cos/sin cache.
template <>
struct DVec<float>
{
    float4 data[2];
    static constexpr uint32_t vec_size = 8;
    __device__ __forceinline__ float& operator[](uint32_t idx)
    {
        return ((float*) (data))[idx];
    }
    __device__ __forceinline__ float const& operator[](uint32_t idx) const
    {
        return ((float const*) (data))[idx];
    }
    __device__ __forceinline__ void load(float const* ptr)
    {
        data[0] = *(reinterpret_cast<float4 const*>(ptr));
        data[1] = *(reinterpret_cast<float4 const*>(ptr + 4));
    }
    __device__ __forceinline__ void store(float* ptr) const
    {
        *(reinterpret_cast<float4*>(ptr)) = data[0];
        *(reinterpret_cast<float4*>(ptr + 4)) = data[1];
    }
};

// half[8] into uint4 and enforce granularity of 16 bytes load/store from global memory.
template <>
struct DVec<half>
{
    uint4 data;
    static constexpr uint32_t vec_size = 8;
    __device__ __forceinline__ half& operator[](uint32_t idx)
    {
        return reinterpret_cast<half*>(&data)[idx];
    }
    __device__ __forceinline__ half const& operator[](uint32_t idx) const
    {
        return reinterpret_cast<half const*>(&data)[idx];
    }
    __device__ __forceinline__ void load(half const* ptr)
    {
        data = *(reinterpret_cast<uint4 const*>(ptr));
    }
    __device__ __forceinline__ void store(half* ptr) const
    {
        *(reinterpret_cast<uint4*>(ptr)) = data;
    }
};

template <typename T>
__device__ __forceinline__ T applyRope(T const& x, T const& y, float const& cos, float const& sin, bool const isLeft);

template <>
__device__ __forceinline__ half applyRope<half>(
    half const& x, half const& y, float const& cos, float const& sin, bool const isLeft)
{
    float val
        = isLeft ? (__half2float(x) * cos - __half2float(y) * sin) : (__half2float(x) * cos + __half2float(y) * sin);
    return __float2half(val);
}

template <typename T>
__device__ __forceinline__ DVec<T> vecApplyRopeNonInterleave(
    T* dataPtr, DVec<float> const& cosVec, DVec<float> const& sinVec, uint32_t const rotaryDim)
{
    DVec<T> result;
    DVec<T> input;
    DVec<T> permuteInput;

    uint32_t const vecOffset = threadIdx.x * DVec<T>::vec_size;
    input.load(dataPtr + vecOffset);

    if (vecOffset < rotaryDim)
    {
        uint32_t const permuteOffset
            = (vecOffset < rotaryDim / 2) ? vecOffset + rotaryDim / 2 : vecOffset - rotaryDim / 2;
        permuteInput.load(dataPtr + permuteOffset);

#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            result[i] = applyRope(input[i], permuteInput[i], cosVec[i], sinVec[i], (vecOffset < rotaryDim / 2));
        }
        return result;
    }
    else
    {
        return input;
    }
}

template <typename T>
__global__ void applyRopeWriteKV(T* qkv, T* kvCache, T* qOut, float const* cosSinCache, int32_t const* kvCacheEndLens,
    int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim)
{
    // Each CTA will process multiple tokens of a single head which each thread handles 16 / sizeof(T) elements.
    // blockDim.x: number of threads to process each token, blockDim.y: number of tokens processed by each CTA.
    // In this kernel we assume:
    //     1. The input tokens are batched with [B, qSeqLen], we use batchIdx info to write KVCache.
    //     2. Always write KVCache with layout of [B, Hk + Hv, S, headDim] where S = kvCacheCapacityLen.
    //     3. The QKV tensor has layout of [B, S, Hq+Hk+Hv, headDim] where S = qSeqLen.
    //     4. The cosSinCache has layout of [S_max, rotaryDim] where S_max = maxPositionEmbeddings.
    //     5. Write to qOut ([B, SHq, headDim] layout) if qOut is provided, otherwise overwrite QKV.
    //     6. kvCacheEndLens: Length of KVCache after insertion the entries by this kernel.

    // TODO (fans): Unify and improve the logic of computing token positions/ kvcache insertion positions.

    uint32_t const bIdx = blockIdx.x;
    uint32_t const bIdy = blockIdx.y;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const tokenIdx = bIdx * bDimY + tIdy;
    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    // We assume all the batches have the same qSeqLen (non-ragged)
    int32_t const batchIdx = tokenIdx / qSeqLen;

    // Determine the position of CosSin Cache to read from.
    // Need to handle three scenarios: Context, vanllia decode, and tree attention.
    // Workaround: For vanllia decode use kvCacheEndLens to compute token positions.
    int32_t sinCosCachePos{};
    if (tokenPosIds != nullptr)
    {
        sinCosCachePos = tokenPosIds[tokenIdx];
    }
    else
    {
        int32_t const posStartId = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        sinCosCachePos = posStartId + tokenIdx % qSeqLen;
    }

    // Vectorized load sin/cos cache from global memory.
    // If pos ids are not provided, use token idx in the sequence as cos/sinc cache posId.
    // non-interleaved rope:
    //      - cosVec = cosSinCache[sinCosCachePos][(tx * vec_size) % (rotaryDim / 2)]
    //      - sinVec = cosSinCache[sinCosCachePos][(tx * vec_size) % (rotaryDim / 2) + rotaryDim / 2]
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t cosOffset;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    cosVec.load(cosSinCache + sinCosCachePos * rotaryDim + cosOffset);
    sinVec.load(cosSinCache + sinCosCachePos * rotaryDim + (cosOffset + sinOffset));

    // tokenIdx is the index of the token in the "flattened" BxS sequence
    int32_t const eleOffsetToken = tokenIdx * (numQHead + numKVHead * 2) * headDim;

    if (bIdy < numQHead)
    {
        int32_t const qHeadIdx = bIdy;
        T* qPTr = qkv + eleOffsetToken + qHeadIdx * headDim;
        DVec<T> qRoped;
        qRoped = vecApplyRopeNonInterleave(qPTr, cosVec, sinVec, rotaryDim);

        if (qOut != nullptr)
        {
            int32_t const qOutOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim + DVec<T>::vec_size * tIdx;
            qRoped.store(qOut + qOutOffset);
        }
        else
        {
            int32_t const qOutOffset = eleOffsetToken + qHeadIdx * headDim + DVec<T>::vec_size * tIdx;
            qRoped.store(qkv + qOutOffset);
        }
    }
    else
    {
        int32_t const kvHeadIdx = bIdy - numQHead;
        int32_t const kvCacheStartIdx = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        int32_t const tokenIdxInCache = kvCacheStartIdx + tokenIdx % qSeqLen;
        int32_t const cacheOffsetSequence = batchIdx * 2 * numKVHead * kvCacheCapacity * headDim;

        int32_t const srcVOffset = eleOffsetToken + (numQHead + numKVHead) * headDim + kvHeadIdx * headDim;
        DVec<T> vSrc;
        vSrc.load(qkv + srcVOffset + DVec<T>::vec_size * tIdx);

        int32_t const srcKOffset = eleOffsetToken + numQHead * headDim + kvHeadIdx * headDim;
        DVec<T> kRoped;
        kRoped = vecApplyRopeNonInterleave(qkv + srcKOffset, cosVec, sinVec, rotaryDim);

        // This is an unique semantics that we only write kRoped back to QKV when qOut is not provided.
        // When qOut is supplied, the decoding attention kernel will directly read KVCache.
        if (qOut == nullptr)
        {
            kRoped.store(qkv + srcKOffset + DVec<T>::vec_size * tIdx);
        }

        // Save to KVCache which assume to have layout of [B, Hk + Hv, S, D]
        int32_t cacheOffsetK = cacheOffsetSequence + kvHeadIdx * kvCacheCapacity * headDim + tokenIdxInCache * headDim
            + DVec<T>::vec_size * tIdx;
        int32_t cacheOffsetV = cacheOffsetSequence + (numKVHead + kvHeadIdx) * kvCacheCapacity * headDim
            + tokenIdxInCache * headDim + DVec<T>::vec_size * tIdx;
        kRoped.store(kvCache + cacheOffsetK);
        vSrc.store(kvCache + cacheOffsetV);
    }
}

void launchApplyRopeWriteKV(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    cudaStream_t stream)
{
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t threadsPerBlock = 128;

    // How many tokens can be processed by each CTA.
    uint32_t const tokenPerBlock = threadsPerBlock * vecSize / headDim;

    uint32_t const bDimX = headDim / vecSize;
    uint32_t const bDimY = tokenPerBlock;
    uint32_t const gDimX = (totalNumTokens + tokenPerBlock - 1) / tokenPerBlock;
    uint32_t const gDimY = numQHead + numKVHead;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    applyRopeWriteKV<half><<<grid, block, 0, stream>>>(qkv, kvCache, qOut, cosSinCache, kvCacheEndLens, tokenPosIds,
        qSeqLen, totalNumTokens, kvCacheCapacity, numQHead, numKVHead, headDim, rotaryDim);
}

void launchApplyRopeWriteKVContext(half* qkv, half* kvCache, float const* cosSinCache, int32_t qSeqLen,
    int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, cudaStream_t stream)
{
    // For current context phase design, we always write to KVCache from start and inplace update QKV.
    half* qOut = nullptr;
    int32_t* kvCacheEndLens = nullptr;
    int32_t* tokenPosIds = nullptr;
    launchApplyRopeWriteKV(qkv, kvCache, qOut, cosSinCache, kvCacheEndLens, tokenPosIds, qSeqLen, totalNumTokens,
        kvCacheCapacity, numQHead, numKVHead, headDim, rotaryDim, stream);
}

void launchApplyRopeWriteKVDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, cudaStream_t stream)
{
    int32_t* tokenPosIds = nullptr;
    launchApplyRopeWriteKV(qkv, kvCache, qOut, cosSinCache, kvCacheEndLens, tokenPosIds, qSeqLen, totalNumTokens,
        kvCacheCapacity, numQHead, numKVHead, headDim, rotaryDim, stream);
}

void launchApplyRopeWriteKVTreeDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    cudaStream_t stream)
{
    launchApplyRopeWriteKV(qkv, kvCache, qOut, cosSinCache, kvCacheEndLens, tokenPosIds, qSeqLen, totalNumTokens,
        kvCacheCapacity, numQHead, numKVHead, headDim, rotaryDim, stream);
}

} // namespace kernel
} // namespace drivellm