/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvfp4A16BlackwellMoeDecodeKernels.h"
#include "nvfp4A16BlackwellMoePdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{
namespace
{

// The CUDA-core dequant-GEMV core below mirrors kernelSrcs/nvfp4A16BlackwellGemv:
// 8 warps per CTA, one CTA per 128-row N tile, 16 rows per warp, two lanes per
// output row (each lane owns 32 of the 64 K values of a tile), 128-bit
// streaming weight loads, cvt.rn.f16x2.e2m1x2 / .e4m3x2 dequant, fp32 FMA.
constexpr int32_t kWarpSize{32};
constexpr int32_t kWarpsPerBlock{8};
constexpr int32_t kThreadsPerBlock{kWarpSize * kWarpsPerBlock};
constexpr int32_t kNTile{128};
constexpr int32_t kKTile{64};
constexpr int32_t kRowsPerWarp{16};
constexpr int32_t kPackedBytesPerRowTile{kKTile / 2};
constexpr int32_t kScalesPerRowTile{kKTile / 16};
constexpr int32_t kPackedBytesPerThread{kPackedBytesPerRowTile / 2};
constexpr int32_t kSmemHalfStride{kKTile / 2 + 4};
constexpr int32_t kSmemRowStride{kKTile + 4};
constexpr int32_t kMaxSplitK{8};
//! Weight row tiles each lane keeps in flight before consuming them.  Thor's
//! LPDDR5X latency is high; with one 16-byte load per lane per tile the kernel
//! is latency bound (measured 84 us for 17 MB), with four it streams.
constexpr int32_t kPrefetchTiles{4};
//! CTAs per SM the FC1/FC2 kernels are register-bounded to (launch bounds).
//! With 8 warps and kPrefetchTiles 16-byte loads per lane each CTA keeps 16 KB
//! of weight traffic in flight; Thor's LPDDR5X needs >= 32 KB per SM to stream
//! near its ~235 GB/s practical ceiling with only 20 SMs.  FC1 compiles to ~72
//! registers once the routing lives outside the kernel.  FC2 (top-k slot loop)
//! wants ~87 unbounded; the (256, 3) bound caps it at 80 (65536 / 768 rounded
//! down to the 8-register allocation unit) and CUDA 13.2 lands at 77 with no
//! spills, so the bound is binding by a few registers (-warn-spills guards it).
constexpr int32_t kDecodeFc1CtasPerSm{2};
constexpr int32_t kDecodeFc2CtasPerSm{3};
//! FC2 may copy the weight tiles (this CTA's K range, codes + scales) of its
//! first params.fc2PrefetchSlots slots into shared memory with cp.async BEFORE
//! griddepcontrol.wait: their expert ids are routing results, complete and
//! visible when FC2 starts because FC1 (and its reduce) wait before they
//! trigger, so the copies overlap the predecessor's tail instead of trailing
//! the wait.  Nemotron: 6.4 KB activations + 18 KB per staged slot.
constexpr int32_t kCodeBytesPerTile{kNTile * kPackedBytesPerRowTile};
constexpr int32_t kScaleBytesPerTile{kNTile * kScalesPerRowTile};
constexpr int32_t kSmemBytesPerStagedTile{kSmemRowStride * static_cast<int32_t>(sizeof(float))};

struct __align__(16) Uint4
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

__device__ __forceinline__ Uint4 loadGlobal128NoAllocate(void const* const pointer)
{
    Uint4 value{};
    asm("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ Uint4 loadGlobal128Retain(void const* const pointer)
{
    Uint4 value{};
    asm("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ float2 fp4PairToFloat2(unsigned int const packedByte)
{
    __half2_raw const raw = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packedByte), __NV_E2M1);
    return __half22float2(static_cast<half2>(raw));
}

__device__ __forceinline__ float2 fp8PairToFloat2(unsigned short const packedBytes)
{
    __half2_raw const raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packedBytes), __NV_E4M3);
    return __half22float2(static_cast<half2>(raw));
}

template <typename T>
struct ActTraits;

template <>
struct ActTraits<half>
{
    static __device__ __forceinline__ float2 pairToFloat2(half const* const p)
    {
        return __half22float2(*reinterpret_cast<half2 const*>(p));
    }
    static __device__ __forceinline__ half fromFloat(float const v)
    {
        return __float2half_rn(v);
    }
    static __device__ __forceinline__ void storePair(half* const p, float2 const v)
    {
        *reinterpret_cast<half2*>(p) = __floats2half2_rn(v.x, v.y);
    }
};

template <>
struct ActTraits<__nv_bfloat16>
{
    static __device__ __forceinline__ float2 pairToFloat2(__nv_bfloat16 const* const p)
    {
        return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(p));
    }
    static __device__ __forceinline__ __nv_bfloat16 fromFloat(float const v)
    {
        return __float2bfloat16_rn(v);
    }
    static __device__ __forceinline__ void storePair(__nv_bfloat16* const p, float2 const v)
    {
        *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v.x, v.y);
    }
};

//! Stage K tiles [kBlockBegin, kBlockEnd) of one activation row into shared memory
//! as fp32, one kSmemRowStride-float slot per tile with the 4-float skew between the
//! two K halves (bank-conflict free reads).  All threads of the CTA cooperate; the
//! caller synchronizes.
template <typename T>
__device__ __forceinline__ void stageActivationRange(
    T const* const row, int32_t const kBlockBegin, int32_t const kBlockEnd, float* const sharedActivation)
{
    constexpr int32_t kElementsPerVector{static_cast<int32_t>(sizeof(Uint4) / sizeof(T))};
    constexpr int32_t kVectorsPerTile{kKTile / kElementsPerVector};
    int32_t const numVectors = (kBlockEnd - kBlockBegin) * kVectorsPerTile;
    for (int32_t vector = static_cast<int32_t>(threadIdx.x); vector < numVectors;
         vector += static_cast<int32_t>(blockDim.x))
    {
        int32_t const tile = vector / kVectorsPerTile;
        int32_t const logicalK = (vector - tile * kVectorsPerTile) * kElementsPerVector;
        Uint4 const packed = loadGlobal128Retain(row + (kBlockBegin + tile) * kKTile + logicalK);
        T const* const elements = reinterpret_cast<T const*>(&packed);
        int32_t const stagedK = tile * kSmemRowStride + logicalK + (logicalK >= kKTile / 2 ? 4 : 0);
        float2* const staged = reinterpret_cast<float2*>(sharedActivation + stagedK);
#pragma unroll
        for (int32_t pair = 0; pair < kElementsPerVector / 2; ++pair)
        {
            staged[pair] = ActTraits<T>::pairToFloat2(elements + pair * 2);
        }
    }
}

//! One lane's half of a weight row tile: 16 packed bytes (32 E2M1 codes) and the
//! two E4M3 block scales covering them.
struct RowTileHalf
{
    Uint4 codes;
    unsigned short scales;
};

//! ``rowTileIndex`` addresses BLACKWELL_MOE_N128_K64_V1:
//! ((n/128)*kBlocks + kBlock)*128 + n%128 inside the expert plane.  The 32 code
//! bytes of a row carry the TMA SWIZZLE_32B image: rows with bit 2 of n%128 set
//! store their two 16-byte halves swapped (bit 2 of rowTileIndex is that bit).
__device__ __forceinline__ RowTileHalf loadRowTileHalf(unsigned char const* __restrict__ const qweights,
    unsigned char const* __restrict__ const blockScales, long long const rowTileIndex, int32_t const kHalf)
{
    RowTileHalf tile{};
    int32_t const storedHalf = kHalf ^ static_cast<int32_t>((rowTileIndex >> 2) & 1);
    tile.codes = loadGlobal128NoAllocate(
        qweights + rowTileIndex * kPackedBytesPerRowTile + storedHalf * kPackedBytesPerThread);
    tile.scales = *reinterpret_cast<unsigned short const*>(blockScales + rowTileIndex * kScalesPerRowTile + kHalf * 2);
    return tile;
}

//! Same row tile half read from the shared-memory staging of one slot's K range
//! (kCodeBytesPerTile / kScaleBytesPerTile per tile, tiles consecutive).  The
//! swizzle bit of the row tile index is bit 2 of the row inside the tile.
__device__ __forceinline__ RowTileHalf loadRowTileHalfSmem(unsigned char const* const sCodes,
    unsigned char const* const sScales, int32_t const tile, int32_t const rowInTile, int32_t const kHalf)
{
    RowTileHalf value{};
    int32_t const storedHalf = kHalf ^ ((rowInTile >> 2) & 1);
    value.codes = *reinterpret_cast<Uint4 const*>(
        sCodes + tile * kCodeBytesPerTile + rowInTile * kPackedBytesPerRowTile + storedHalf * kPackedBytesPerThread);
    value.scales = *reinterpret_cast<unsigned short const*>(
        sScales + tile * kScaleBytesPerTile + rowInTile * kScalesPerRowTile + kHalf * 2);
    return value;
}

//! Dot product of a lane's 32 dequantized weights with the matching staged
//! activation half (``stagedHalf`` already offset by tile and K half).
__device__ __forceinline__ float dotRowTileHalf(RowTileHalf const& tile, float const* const stagedHalf)
{
    float2 const scales = fp8PairToFloat2(tile.scales);
    unsigned int const words[4]{tile.codes.x, tile.codes.y, tile.codes.z, tile.codes.w};
    float acc = 0.0f;
#pragma unroll
    for (int32_t word = 0; word < 4; ++word)
    {
#pragma unroll
        for (int32_t byte = 0; byte < 4; ++byte)
        {
            int32_t const pair = word * 4 + byte;
            unsigned int const packedByte = (words[word] >> (byte * 8)) & 0xFFU;
            float2 const weights = fp4PairToFloat2(packedByte);
            float const scale = pair < 8 ? scales.x : scales.y;
            float2 const values = *reinterpret_cast<float2 const*>(stagedHalf + pair * 2);
            float const dot = fmaf(values.x, weights.x, values.y * weights.y);
            acc = fmaf(dot, scale, acc);
        }
    }
    return acc;
}

//! Dot of one weight row (this lane's half) over the numTiles tiles staged in
//! shared memory by the pre-wait cp.async prefetch.
__device__ __forceinline__ float dotStagedRowRange(unsigned char const* const sCodes,
    unsigned char const* const sScales, int32_t const numTiles, int32_t const rowInTile, int32_t const kHalf,
    float const* const stagedRange)
{
    float acc = 0.0f;
    float const* const stagedHalf = stagedRange + kHalf * kSmemHalfStride;
    for (int32_t tile = 0; tile < numTiles; ++tile)
    {
        acc += dotRowTileHalf(
            loadRowTileHalfSmem(sCodes, sScales, tile, rowInTile, kHalf), stagedHalf + tile * kSmemRowStride);
    }
    return acc;
}

//! Streams K tiles [kBlockBegin, kBlockEnd) of one weight row (this lane's half)
//! against the activation staged for the same range, kPrefetchTiles loads in
//! flight.  One predicated loop (no separate tail copy) keeps the live tile
//! array at kPrefetchTiles entries, which is what bounds the register count.
__device__ __forceinline__ float streamRowRange(unsigned char const* __restrict__ const qweights,
    unsigned char const* __restrict__ const blockScales, long long const rowTileBase, int32_t const kBlockBegin,
    int32_t const kBlockEnd, int32_t const kHalf, float const* const stagedRange)
{
    float acc = 0.0f;
    float const* const stagedHalf = stagedRange + kHalf * kSmemHalfStride;
    for (int32_t kBlock = kBlockBegin; kBlock < kBlockEnd; kBlock += kPrefetchTiles)
    {
        RowTileHalf tiles[kPrefetchTiles];
#pragma unroll
        for (int32_t j = 0; j < kPrefetchTiles; ++j)
        {
            if (kBlock + j < kBlockEnd)
            {
                tiles[j] = loadRowTileHalf(
                    qweights, blockScales, rowTileBase + static_cast<long long>(kBlock + j) * kNTile, kHalf);
            }
        }
#pragma unroll
        for (int32_t j = 0; j < kPrefetchTiles; ++j)
        {
            if (kBlock + j < kBlockEnd)
            {
                acc += dotRowTileHalf(tiles[j], stagedHalf + (kBlock + j - kBlockBegin) * kSmemRowStride);
            }
        }
    }
    return acc;
}

//! K tiles one CTA of a split-K launch stages (the largest split range).
__host__ __device__ __forceinline__ int32_t maxTilesPerSplit(int32_t const kBlocks, int32_t const splitK)
{
    return (kBlocks + splitK - 1) / splitK;
}

__host__ __device__ __forceinline__ int32_t fc1SharedBytes(int32_t const kBlocks, int32_t const splitK)
{
    return maxTilesPerSplit(kBlocks, splitK) * kSmemBytesPerStagedTile;
}

__host__ __device__ __forceinline__ int32_t fc2SharedBytes(
    int32_t const topK, int32_t const kBlocks, int32_t const splitK, int32_t const prefetchSlots)
{
    int32_t const tiles = maxTilesPerSplit(kBlocks, splitK);
    return topK * tiles * kSmemBytesPerStagedTile + prefetchSlots * tiles * (kCodeBytesPerTile + kScaleBytesPerTile);
}

//! FC1: grid (N1_pad/128, numTokens*topK, fc1SplitK), 256 threads.  The routing
//! kernel (launchSigmoidTopkRoute / moeSigmoidGroupTopk) has already resolved
//! slot -> expert into topkIndices; each CTA stages its token's activation K
//! range and computes one 128-row tile of relu(alpha * x[token] . W1[expert])^2,
//! or an fp32 partial when split-K > 1.  Keeping the routing out of the kernel
//! holds it at GEMV register pressure so kDecodeFc1CtasPerSm CTAs co-reside per SM
//! (the fused version needed 131 registers, one CTA per SM, and streamed FC1 at
//! ~150 GB/s inside the engine).
template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock, kDecodeFc1CtasPerSm) void decodeFc1Kernel(DecodeMoeParams const params)
{
    extern __shared__ __align__(16) unsigned char dynamicSmem[];
    float* const sharedActivation = reinterpret_cast<float*>(dynamicSmem);

    int32_t const slot = static_cast<int32_t>(blockIdx.y);
    int32_t const token = slot / params.topK;
    int32_t const nFeatures = params.interSizePadded;
    int32_t const kFeatures = params.hiddenSize;
    int32_t const kBlocks = kFeatures / kKTile;
    int32_t const split = static_cast<int32_t>(blockIdx.z);
    int32_t const kBlockBegin = kBlocks * split / params.fc1SplitK;
    int32_t const kBlockEnd = kBlocks * (split + 1) / params.fc1SplitK;
    T const* const activationRow
        = static_cast<T const*>(params.hiddenStates) + static_cast<long long>(token) * kFeatures;
    // The activation row belongs to the previous layer, which had completed
    // before the routing kernel passed its own wait (routing triggers only
    // after that wait), so it is staged before this kernel's wait; only the
    // expert id (routing output) has to wait (trigger placement: see Pdl.cuh).
    stageActivationRange<T>(activationRow, kBlockBegin, kBlockEnd, sharedActivation);
    pdlWait();
    pdlTrigger();
    int32_t const expert = params.topkIndices[slot];
    __syncthreads();

    int32_t const lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const rowInTile = warp * kRowsPerWarp + lane / 2;
    int32_t const kHalf = lane & 1;
    int32_t const nBlock = static_cast<int32_t>(blockIdx.x);

    long long const expertPlaneCodes = static_cast<long long>(nFeatures) * kFeatures / 2;
    unsigned char const* const qweights
        = static_cast<unsigned char const*>(params.fc1QWeights) + static_cast<long long>(expert) * expertPlaneCodes;
    unsigned char const* const blockScales = static_cast<unsigned char const*>(params.fc1BlockScales)
        + static_cast<long long>(expert) * (expertPlaneCodes / 8);

    long long const rowTileBase = static_cast<long long>(nBlock) * kBlocks * kNTile + rowInTile;
    float acc = streamRowRange(qweights, blockScales, rowTileBase, kBlockBegin, kBlockEnd, kHalf, sharedActivation);
    acc += __shfl_xor_sync(0xFFFFFFFFU, acc, 1, kWarpSize);
    if (kHalf == 0)
    {
        int32_t const n = nBlock * kNTile + rowInTile;
        if (params.fc1SplitK == 1)
        {
            float const scaled = fmaxf(acc * params.fc1GlobalScales[expert], 0.0f);
            static_cast<T*>(params.fc1Output)[static_cast<long long>(slot) * nFeatures + n]
                = ActTraits<T>::fromFloat(scaled * scaled);
        }
        else
        {
            params
                .fc1Partials[(static_cast<long long>(split) * (params.numTokens * params.topK) + slot) * nFeatures + n]
                = acc;
        }
    }
}

//! FC1 split-K finalize: sum partials, apply alpha[expert], relu^2, narrow.
template <typename T>
__global__ void decodeFc1ReduceKernel(DecodeMoeParams const params)
{
    long long const pairIndex = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t const nFeatures = params.interSizePadded;
    long long const numSlots = static_cast<long long>(params.numTokens) * params.topK;
    long long const numPairs = numSlots * nFeatures / 2;
    bool const active = pairIndex < numPairs;
    long long const elementIndex = pairIndex * 2;
    // Routing results are complete and visible (FC1 waited before it triggered);
    // only the partials need this kernel's wait.
    float alpha = 0.0f;
    if (active)
    {
        int32_t const slot = static_cast<int32_t>(elementIndex / nFeatures);
        alpha = params.fc1GlobalScales[params.topkIndices[slot]];
    }
    pdlWait();
    pdlTrigger();
    if (active)
    {
        float2 sum{0.0f, 0.0f};
        for (int32_t split = 0; split < params.fc1SplitK; ++split)
        {
            float2 const v = *reinterpret_cast<float2 const*>(
                params.fc1Partials + static_cast<long long>(split) * numSlots * nFeatures + elementIndex);
            sum.x += v.x;
            sum.y += v.y;
        }
        float const a = fmaxf(sum.x * alpha, 0.0f);
        float const b = fmaxf(sum.y * alpha, 0.0f);
        ActTraits<T>::storePair(static_cast<T*>(params.fc1Output) + elementIndex, make_float2(a * a, b * b));
    }
}

//! FC2: grid (H/128, numTokens, fc2SplitK), 256 threads.  Each CTA loops over the
//! token's top-k slots and accumulates alpha[e] * w[token, slot] * a_slot . W2[e]
//! for its 128 hidden features in fp32, then writes the token output (or an fp32
//! partial when split-K > 1).  No atomics: the result is deterministic.
template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock, kDecodeFc2CtasPerSm) void decodeFc2Kernel(DecodeMoeParams const params)
{
    extern __shared__ __align__(16) unsigned char dynamicSmem[];
    float* const sharedActivation = reinterpret_cast<float*>(dynamicSmem);

    int32_t const token = static_cast<int32_t>(blockIdx.y);
    int32_t const nBlock = static_cast<int32_t>(blockIdx.x);
    int32_t const split = static_cast<int32_t>(blockIdx.z);
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const rowInTile = warp * kRowsPerWarp + lane / 2;
    int32_t const kHalf = lane & 1;
    int32_t const nFeatures = params.hiddenSize;
    int32_t const kFeatures = params.interSize;
    int32_t const kBlocks = kFeatures / kKTile;
    int32_t const kBlockBegin = kBlocks * split / params.fc2SplitK;
    int32_t const kBlockEnd = kBlocks * (split + 1) / params.fc2SplitK;
    long long const expertPlaneCodes = static_cast<long long>(nFeatures) * kFeatures / 2;
    int32_t const stagedTilesPerSlot = maxTilesPerSplit(kBlocks, params.fc2SplitK);
    int32_t const numTiles = kBlockEnd - kBlockBegin;
    int32_t const numPrefetchSlots = params.fc2PrefetchSlots;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    unsigned char* const sCodes = dynamicSmem + params.topK * stagedTilesPerSlot * kSmemBytesPerStagedTile;
    unsigned char* const sScales = sCodes + numPrefetchSlots * stagedTilesPerSlot * kCodeBytesPerTile;
    unsigned char const* const codesBase = static_cast<unsigned char const*>(params.fc2QWeights);
    unsigned char const* const scalesBase = static_cast<unsigned char const*>(params.fc2BlockScales);
    // This CTA's K range of one expert is contiguous in the layout: row tiles
    // (nBlock * kBlocks + kBlock) * 128 .. for kBlock in [begin, end).
    long long const rangeTileBase = (static_cast<long long>(nBlock) * kBlocks + kBlockBegin) * kNTile;

    // Before the wait: topkIndices / topkWeights / global scales are routing
    // results, complete and visible because FC1 (and its reduce) waited before
    // they triggered.  Copy the first slots' weight tiles into shared memory so
    // the traffic overlaps the predecessor's tail; fc1Output must wait.
    for (int32_t slotInToken = 0; slotInToken < numPrefetchSlots; ++slotInToken)
    {
        int32_t const expert = params.topkIndices[static_cast<long long>(token) * params.topK + slotInToken];
        unsigned char const* const codes
            = codesBase + static_cast<long long>(expert) * expertPlaneCodes + rangeTileBase * kPackedBytesPerRowTile;
        unsigned char const* const scales
            = scalesBase + static_cast<long long>(expert) * (expertPlaneCodes / 8) + rangeTileBase * kScalesPerRowTile;
        unsigned char* const dstCodes = sCodes + slotInToken * stagedTilesPerSlot * kCodeBytesPerTile;
        unsigned char* const dstScales = sScales + slotInToken * stagedTilesPerSlot * kScaleBytesPerTile;
        for (int32_t v = tid; v < numTiles * (kCodeBytesPerTile / 16); v += kThreadsPerBlock)
        {
            cpAsync16(dstCodes + v * 16, codes + v * 16);
        }
        for (int32_t v = tid; v < numTiles * (kScaleBytesPerTile / 16); v += kThreadsPerBlock)
        {
            cpAsync16(dstScales + v * 16, scales + v * 16);
        }
    }
    cpAsyncCommit();
    pdlWait();
    pdlTrigger();
    for (int32_t slotInToken = 0; slotInToken < params.topK; ++slotInToken)
    {
        long long const slot = static_cast<long long>(token) * params.topK + slotInToken;
        T const* const activationRow = static_cast<T const*>(params.fc1Output) + slot * params.interSizePadded;
        stageActivationRange<T>(activationRow, kBlockBegin, kBlockEnd,
            sharedActivation + slotInToken * stagedTilesPerSlot * kSmemRowStride);
    }
    cpAsyncWaitAll();
    __syncthreads();

    long long const rowTileBase = static_cast<long long>(nBlock) * kBlocks * kNTile + rowInTile;
    float total = 0.0f;
    for (int32_t slotInToken = 0; slotInToken < params.topK; ++slotInToken)
    {
        long long const slot = static_cast<long long>(token) * params.topK + slotInToken;
        int32_t const expert = params.topkIndices[slot];
        float const weight = params.topkWeights[slot] * params.fc2GlobalScales[expert];
        float const* const stagedRange = sharedActivation + slotInToken * stagedTilesPerSlot * kSmemRowStride;
        float acc = 0.0f;
        if (slotInToken < numPrefetchSlots)
        {
            acc = dotStagedRowRange(sCodes + slotInToken * stagedTilesPerSlot * kCodeBytesPerTile,
                sScales + slotInToken * stagedTilesPerSlot * kScaleBytesPerTile, numTiles, rowInTile, kHalf,
                stagedRange);
        }
        else
        {
            unsigned char const* const qweights = codesBase + static_cast<long long>(expert) * expertPlaneCodes;
            unsigned char const* const blockScales
                = scalesBase + static_cast<long long>(expert) * (expertPlaneCodes / 8);
            acc = streamRowRange(qweights, blockScales, rowTileBase, kBlockBegin, kBlockEnd, kHalf, stagedRange);
        }
        total = fmaf(weight, acc, total);
    }
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 1, kWarpSize);
    if (kHalf == 0)
    {
        int32_t const n = nBlock * kNTile + rowInTile;
        if (params.fc2SplitK == 1)
        {
            static_cast<T*>(params.output)[static_cast<long long>(token) * nFeatures + n]
                = ActTraits<T>::fromFloat(total);
        }
        else
        {
            params.fc2Partials[(static_cast<long long>(split) * params.numTokens + token) * nFeatures + n] = total;
        }
    }
}

//! FC2 split-K finalize: sum partials, narrow.
template <typename T>
__global__ void decodeFc2ReduceKernel(DecodeMoeParams const params)
{
    long long const pairIndex = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long const numElements = static_cast<long long>(params.numTokens) * params.hiddenSize;
    pdlWait(); // partials from FC2 (nothing of this kernel's inputs predates them)
    pdlTrigger();
    if (pairIndex * 2 < numElements)
    {
        long long const elementIndex = pairIndex * 2;
        float2 sum{0.0f, 0.0f};
        for (int32_t split = 0; split < params.fc2SplitK; ++split)
        {
            float2 const v = *reinterpret_cast<float2 const*>(
                params.fc2Partials + static_cast<long long>(split) * numElements + elementIndex);
            sum.x += v.x;
            sum.y += v.y;
        }
        ActTraits<T>::storePair(static_cast<T*>(params.output) + elementIndex, sum);
    }
}

constexpr int32_t kDefaultMaxDynamicSmemBytes{48 * 1024};

template <typename Kernel>
cudaError_t optInDynamicSmem(Kernel const kernel, int32_t const smemBytes)
{
    if (smemBytes <= kDefaultMaxDynamicSmemBytes)
    {
        return cudaSuccess;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
}

template <typename T>
cudaError_t launchFc1Typed(DecodeMoeParams const& params, cudaStream_t const stream)
{
    dim3 const grid(params.interSizePadded / kNTile, params.numTokens * params.topK, params.fc1SplitK);
    int32_t const smemBytes = fc1SharedBytes(params.hiddenSize / kKTile, params.fc1SplitK);
    cudaError_t const attr = optInDynamicSmem(decodeFc1Kernel<T>, smemBytes);
    if (attr != cudaSuccess)
    {
        return attr;
    }
    cudaError_t const launched = launchKernelPdl(decodeFc1Kernel<T>, grid, dim3(kThreadsPerBlock),
        static_cast<size_t>(smemBytes), stream, params.enablePdl, params);
    if (launched != cudaSuccess || params.fc1SplitK == 1)
    {
        return launched;
    }
    long long const numPairs = static_cast<long long>(params.numTokens) * params.topK * params.interSizePadded / 2;
    dim3 const reduceGrid(static_cast<unsigned int>((numPairs + kThreadsPerBlock - 1) / kThreadsPerBlock));
    return launchKernelPdl(
        decodeFc1ReduceKernel<T>, reduceGrid, dim3(kThreadsPerBlock), 0, stream, params.enablePdl, params);
}

template <typename T>
cudaError_t launchFc2Typed(DecodeMoeParams const& params, cudaStream_t const stream)
{
    dim3 const grid(params.hiddenSize / kNTile, params.numTokens, params.fc2SplitK);
    int32_t const smemBytes
        = fc2SharedBytes(params.topK, params.interSize / kKTile, params.fc2SplitK, params.fc2PrefetchSlots);
    cudaError_t const attr = optInDynamicSmem(decodeFc2Kernel<T>, smemBytes);
    if (attr != cudaSuccess)
    {
        return attr;
    }
    cudaError_t const launched = launchKernelPdl(decodeFc2Kernel<T>, grid, dim3(kThreadsPerBlock),
        static_cast<size_t>(smemBytes), stream, params.enablePdl, params);
    if (launched != cudaSuccess || params.fc2SplitK == 1)
    {
        return launched;
    }
    long long const numPairs = static_cast<long long>(params.numTokens) * params.hiddenSize / 2;
    dim3 const reduceGrid(static_cast<unsigned int>((numPairs + kThreadsPerBlock - 1) / kThreadsPerBlock));
    return launchKernelPdl(
        decodeFc2ReduceKernel<T>, reduceGrid, dim3(kThreadsPerBlock), 0, stream, params.enablePdl, params);
}

} // namespace

size_t getDecodeFc1PartialBytes(DecodeMoeParams const& params)
{
    if (params.fc1SplitK <= 1)
    {
        return 0;
    }
    return static_cast<size_t>(params.fc1SplitK) * params.numTokens * params.topK * params.interSizePadded
        * sizeof(float);
}

size_t getDecodeFc2PartialBytes(DecodeMoeParams const& params)
{
    if (params.fc2SplitK <= 1)
    {
        return 0;
    }
    return static_cast<size_t>(params.fc2SplitK) * params.numTokens * params.hiddenSize * sizeof(float);
}

char const* validateDecodeParams(DecodeMoeParams const& p, DecodeDtype const dtype)
{
    if (dtype != DecodeDtype::kFP16 && dtype != DecodeDtype::kBF16)
    {
        return "unsupported activation dtype";
    }
    if (p.numTokens <= 0 || p.numExperts <= 0 || p.numExperts > 512)
    {
        return "numTokens must be positive and numExperts in [1, 512]";
    }
    if (p.topK <= 0 || p.topK > 32 || p.topK > p.numExperts)
    {
        return "topK must be in [1, 32] and <= numExperts";
    }
    if (p.hiddenSize <= 0 || p.hiddenSize % kNTile != 0)
    {
        return "hiddenSize must be a positive multiple of 128";
    }
    if (p.fc2PrefetchSlots < 0 || p.fc2PrefetchSlots > p.topK)
    {
        return "fc2PrefetchSlots must be in [0, topK]";
    }
    if (p.interSize <= 0 || p.interSize % kKTile != 0)
    {
        return "interSize must be a positive multiple of 64";
    }
    if (p.interSizePadded < p.interSize || p.interSizePadded % kNTile != 0)
    {
        return "interSizePadded must be interSize rounded up to a multiple of 128";
    }
    if (p.fc1SplitK < 1 || p.fc1SplitK > kMaxSplitK || p.fc1SplitK > p.hiddenSize / kKTile)
    {
        return "fc1SplitK out of range";
    }
    if (p.fc2SplitK < 1 || p.fc2SplitK > kMaxSplitK || p.fc2SplitK > p.interSize / kKTile)
    {
        return "fc2SplitK out of range";
    }
    if (p.topkIndices == nullptr || p.topkWeights == nullptr || p.hiddenStates == nullptr || p.fc1Output == nullptr
        || p.output == nullptr || p.fc1QWeights == nullptr || p.fc1BlockScales == nullptr
        || p.fc1GlobalScales == nullptr || p.fc2QWeights == nullptr || p.fc2BlockScales == nullptr
        || p.fc2GlobalScales == nullptr)
    {
        return "required pointer is null";
    }
    if ((p.fc1SplitK > 1 && p.fc1Partials == nullptr) || (p.fc2SplitK > 1 && p.fc2Partials == nullptr))
    {
        return "split-K requires a partials buffer";
    }
    auto const aligned16 = [](void const* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 15U) == 0; };
    if (!aligned16(p.hiddenStates) || !aligned16(p.fc1Output) || !aligned16(p.fc1QWeights) || !aligned16(p.fc2QWeights)
        || !aligned16(p.fc2BlockScales))
    {
        return "activations, weights and FC2 block scales must be 16-byte aligned";
    }
    return nullptr;
}

int32_t decodeFc2SharedBytes(
    int32_t const topK, int32_t const kBlocks, int32_t const splitK, int32_t const prefetchSlots)
{
    return fc2SharedBytes(topK, kBlocks, splitK, prefetchSlots);
}

cudaError_t launchDecodeFc1(DecodeMoeParams const& params, DecodeDtype const dtype, cudaStream_t const stream)
{
    if (char const* reason = validateDecodeParams(params, dtype); reason != nullptr)
    {
        return cudaErrorInvalidValue;
    }
    return dtype == DecodeDtype::kFP16 ? launchFc1Typed<half>(params, stream)
                                       : launchFc1Typed<__nv_bfloat16>(params, stream);
}

cudaError_t launchDecodeFc2(DecodeMoeParams const& params, DecodeDtype const dtype, cudaStream_t const stream)
{
    if (char const* reason = validateDecodeParams(params, dtype); reason != nullptr)
    {
        return cudaErrorInvalidValue;
    }
    return dtype == DecodeDtype::kFP16 ? launchFc2Typed<half>(params, stream)
                                       : launchFc2Typed<__nv_bfloat16>(params, stream);
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
