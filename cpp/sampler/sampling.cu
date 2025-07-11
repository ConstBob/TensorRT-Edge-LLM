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

// clang-format off
#include "sampling.h"
#include "common/common.h"
// clang-format on
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>

namespace drivellm
{

// Define missing constants for half precision
#ifndef HALF_FLT_MAX
#define HALF_FLT_MAX 65504.0f
#endif

// ========================================================================
// WORKSPACE MANAGEMENT STRUCTURE
// ========================================================================

// Internal workspace structure for memory management
template <typename T>
struct SamplingWorkspace
{
    void* ptr;
    size_t size;

    // Buffer pointers and sizes for different sampling methods
    T* topkTempLogits;
    int32_t* topkIndices;
    T* topkValues;

    void* toppTempStorage; // Keep as void* for CUB temp storage
    T* toppProbs;
    T* toppSortedProbs;
    int32_t* toppSortedIdVals;
    int32_t* toppIdVals;
    int32_t* toppOffsetBuf;
    int32_t* toppBeginOffsetBuf;
    int32_t* toppTopKIndices;
    T* toppTopKValues;
    int32_t* toppEarlyExitFlags;

    SamplingWorkspace()
        : ptr(nullptr)
        , size(0)
        , topkTempLogits(nullptr)
        , topkIndices(nullptr)
        , topkValues(nullptr)
        , toppTempStorage(nullptr)
        , toppProbs(nullptr)
        , toppSortedProbs(nullptr)
        , toppSortedIdVals(nullptr)
        , toppIdVals(nullptr)
        , toppOffsetBuf(nullptr)
        , toppBeginOffsetBuf(nullptr)
        , toppTopKIndices(nullptr)
        , toppTopKValues(nullptr)
        , toppEarlyExitFlags(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, SamplingParams const& params)
    {
        ptr = workspace;
        size = workspaceSize;

        // Add alignment function (same as in size calculation)
        auto alignSize = [](size_t size) -> size_t {
            const size_t alignment = 256; // 256-byte alignment for optimal GPU memory access
            return (size + alignment - 1) & ~(alignment - 1);
        };

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        if (params.useTopK)
        {
            // Top-K workspace layout
            size_t tempLogitsSize = alignSize(params.batchSize * params.vocabSize * sizeof(T));
            size_t indicesSize = alignSize(params.batchSize * 8 * params.topK * sizeof(int32_t)); // BLOCKS_PER_BEAM = 8
            size_t valuesSize = alignSize(params.batchSize * 8 * params.topK * sizeof(T));

            topkTempLogits = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
            offset += tempLogitsSize;

            topkIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += indicesSize;

            topkValues = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
            offset += valuesSize;
        }
        else if (params.useTopP)
        {
            // Top-P workspace layout
            // Calculate CUB temp storage size
            size_t cubTempStorageSize;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, static_cast<T*>(nullptr),
                static_cast<T*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
                static_cast<int32_t>(params.vocabSize * params.batchSize), params.batchSize,
                static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr));

            toppTempStorage = static_cast<char*>(ptr) + offset;
            offset += alignSize(cubTempStorageSize);

            toppProbs = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(T));

            toppSortedProbs = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(T));

            toppSortedIdVals = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppIdVals = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppOffsetBuf = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize((params.batchSize + 1) * sizeof(int32_t));

            toppBeginOffsetBuf = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize((params.batchSize + 1) * sizeof(int32_t));

            toppTopKIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppTopKValues = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(T));

            toppEarlyExitFlags = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * sizeof(int32_t));
        }

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }

    // Setup workspace for selectAllTopK
    void setupWorkspaceForTopK(
        void* workspace, size_t workspaceSize, int32_t batchSize, int32_t vocabSize, int32_t topK)
    {
        ptr = workspace;
        size = workspaceSize;

        // Add alignment function (same as in size calculation)
        auto alignSize = [](size_t size) -> size_t {
            const size_t alignment = 256; // 256-byte alignment for optimal GPU memory access
            return (size + alignment - 1) & ~(alignment - 1);
        };

        size_t offset = 0;

        // Same layout as top-K sampling
        size_t tempLogitsSize = alignSize(batchSize * vocabSize * sizeof(T));
        size_t indicesSize = alignSize(batchSize * 8 * topK * sizeof(int32_t)); // BLOCKS_PER_BEAM = 8
        size_t valuesSize = alignSize(batchSize * 8 * topK * sizeof(T));

        topkTempLogits = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
        offset += tempLogitsSize;

        topkIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += indicesSize;

        topkValues = reinterpret_cast<T*>(static_cast<char*>(ptr) + offset);
        offset += valuesSize;

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Workspace size too small for topK. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }
};

// ========================================================================
// WORKSPACE SIZE CALCULATION
// ========================================================================

// Enhanced workspace size calculation that handles alignment
template <typename T>
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params)
{
    size_t workspaceSize = 0;

    // Add alignment padding between buffers
    auto alignSize = [](size_t size) -> size_t {
        const size_t alignment = 256; // 256-byte alignment for optimal GPU memory access
        return (size + alignment - 1) & ~(alignment - 1);
    };

    if (params.useTopK)
    {
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(T));             // temp logits
        workspaceSize += alignSize(batchSize * 8 * params.topK * sizeof(int32_t)); // top-k indices
        workspaceSize += alignSize(batchSize * 8 * params.topK * sizeof(T));       // top-k values
    }
    else if (params.useTopP)
    {
        // Calculate CUB temp storage size
        size_t cubTempStorageSize;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, static_cast<T*>(nullptr),
            static_cast<T*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<int32_t>(vocabSize * batchSize), batchSize, static_cast<int32_t*>(nullptr),
            static_cast<int32_t*>(nullptr));

        workspaceSize += alignSize(cubTempStorageSize);                      // CUB temp storage
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(T));       // probs
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(T));       // sorted probs
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // sorted id vals
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // id vals
        workspaceSize += alignSize((batchSize + 1) * sizeof(int32_t));       // offset buf
        workspaceSize += alignSize((batchSize + 1) * sizeof(int32_t));       // begin offset buf
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // top-k indices
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(T));       // top-k values
        workspaceSize += alignSize(batchSize * sizeof(int32_t));             // early exit flags
    }
    else
    {
        // No filtering - error out
        throw std::runtime_error("Either topK or topP must be set");
    }

    return workspaceSize;
}

// Calculate workspace size for selectAllTopK
template <typename T>
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK)
{
    auto alignSize = [](size_t size) -> size_t {
        const size_t alignment = 256;
        return (size + alignment - 1) & ~(alignment - 1);
    };

    size_t workspaceSize = 0;
    workspaceSize += alignSize(batchSize * vocabSize * sizeof(T));      // temp logits
    workspaceSize += alignSize(batchSize * 8 * topK * sizeof(int32_t)); // top-k indices
    workspaceSize += alignSize(batchSize * 8 * topK * sizeof(T));       // top-k values

    return workspaceSize;
}

// Device functions for math operations
template <typename T>
__device__ T exp_device(T x);

template <>
__device__ float exp_device<float>(float x)
{
    return expf(x);
}

template <>
__device__ half exp_device<half>(half x)
{
    return __float2half(__expf(__half2float(x)));
}

template <>
__device__ __nv_bfloat16 exp_device<__nv_bfloat16>(__nv_bfloat16 x)
{
    return __float2bfloat16(__expf(__bfloat162float(x)));
}

template <typename T>
__device__ T max_device(T a, T b);

template <>
__device__ float max_device<float>(float a, float b)
{
    return fmaxf(a, b);
}

template <>
__device__ half max_device<half>(half a, half b)
{
    return __hmax(a, b);
}

template <>
__device__ __nv_bfloat16 max_device<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __hmax(a, b);
}

// Helper function to convert T to float
template <typename T>
__device__ float toFloat(T x)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return x;
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return __half2float(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __bfloat162float(x);
    }
}

// Helper function to convert float to T
template <typename T>
__device__ T fromFloat(float x)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return x;
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return __float2half(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __float2bfloat16(x);
    }
}

// Helper structures for top-k reduction operations (similar to TRT-LLM)
template <typename T>
struct TopK_2
{
    T value;
    int32_t index;

    __device__ __forceinline__ void init()
    {
        value = -FLT_MAX;
        index = -1;
    }

    __device__ __forceinline__ void insert(T elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

// Reduction operator for top-k
template <typename T>
struct reduce_topk_op_2
{
    __device__ __forceinline__ TopK_2<T> operator()(TopK_2<T> const& a, TopK_2<T> const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Stage 2 kernel for returnAllTopK that matches the old sampler's approach
template <typename T, int BLOCK_SIZE>
__global__ void topKStage2ReturnAllTopK(int32_t const* __restrict topKTmpIdBuf, T* topKTmpValBuf,
    int64_t* outputIndices, float* outputValues, T* outputTValues, int32_t batchSize, int32_t vocabSize, int32_t topK,
    int32_t blocksPerBeam, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs)
{
    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x);

    if (batchIdx >= batchSize)
        return;

    auto const size = topK * blocksPerBeam;
    auto const stride = topK * blocksPerBeam;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    __shared__ float sSum;
    T* sVal = topKTmpValBuf + batchIdx * stride;
    auto* sId = reinterpret_cast<int32_t*>(array);
    if (tid == 0)
    {
        sSum = 0.0f;
    }
    TopK_2<float> partial;

    auto sVal2 = reinterpret_cast<float*>(sId + topK);
    float maxLogit;
    for (int32_t ite = 0; ite < topK; ite++)
    {
        partial.init();
#pragma unroll
        for (int32_t i = tid; i < size; i += BLOCK_SIZE)
        {
            partial.insert((float) sVal[i], i);
        }

        TopK_2<float> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<float>());

        if (tid == 0)
        {
            if (ite == 0)
            {
                maxLogit = total.value;
            }
            sId[ite] = total.index;
            sVal[total.index] = -FLT_MAX;

            // when cumLogProbs are computed, topKTmpValBuf (logits_buf_) are
            // already pre-processed by softmax_kernel
            if (!inputHasProbs)
            {
                total.value = __expf(total.value - maxLogit);
            }
            sVal2[ite] = total.value;
            sSum += total.value;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        for (int32_t ki = 0; ki < topK; ki++)
        {
            auto expLogit = sVal2[ki];
            auto idx = sId[ki];

            if (outputIndices != nullptr)
            {
                auto outputId = idx != -1 ? topKTmpIdBuf[batchIdx * stride + idx] % vocabSize : vocabSize - 1;
                outputId = outputId == -1 ? vocabSize - 1 : outputId;
                outputIndices[batchIdx * topK + ki] = outputId;
            }

            if (outputValues != nullptr)
            {
                auto logProb = logf(expLogit);
                auto const normalizedProb = normalizeLogProbs ? logProb - logf(sSum) : logProb;
                outputValues[batchIdx * topK + ki] = normalizedProb;
            }

            if (outputTValues != nullptr)
            {
                outputTValues[batchIdx * topK + ki] = fromFloat<T>(expLogit);
            }
        }
    }
}

// Block prefix callback for cumulative sum computation
struct BlockPrefixCallbackOp
{
    float runningTotal;

    __device__ BlockPrefixCallbackOp(float runningTotal)
        : runningTotal(runningTotal)
    {
    }

    __device__ float operator()(float blockAggregate)
    {
        float oldPrefix = runningTotal;
        runningTotal += blockAggregate;
        return oldPrefix;
    }
};

// =======================================================================================
// TOP-K SAMPLING KERNELS (Based on TensorRT-LLM two-stage approach)
// =======================================================================================

// Stage 1: Find top-K elements using iterative block-level reduction
template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKStage1(
    T const* __restrict__ logits, T* tmpLogits, int32_t* topKTmpIdBuf, T* topKTmpValBuf, SamplingParams const params)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    auto const batchId = bid / BLOCKS_PER_BEAM_;
    auto const blockLane = bid % BLOCKS_PER_BEAM_;

    if (batchId >= params.batchSize)
        return;

    auto const vocabSize = params.vocabSize;
    auto const k = params.topK;
    auto const temperature = params.temperature;
    auto const invTemp = (temperature == 0.0f) ? 0.0f : 1.0f / temperature;

    auto const tmpLogBufIndex = batchId * vocabSize;
    auto const tmpTopKBufIndex = batchId * BLOCKS_PER_BEAM_ * k + blockLane * k;

    TopK_2<T> partial;
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? fromFloat<T>(HALF_FLT_MAX) : fromFloat<T>(FLT_MAX);

    // Copy logits to temporary buffer and apply temperature
    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        auto localIndex = elemId + tmpLogBufIndex;
        T logit = logits[localIndex];
        float floatLogit = toFloat(logit) * invTemp;
        tmpLogits[localIndex] = fromFloat<T>(floatLogit);
    }

    // Find top-K elements iteratively
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            auto index = elemId + tmpLogBufIndex;
            partial.insert(tmpLogits[index], index);
        }

        TopK_2<T> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<T>());

        if (tid == 0)
        {
            auto const index = tmpTopKBufIndex + ite;
            topKTmpIdBuf[index] = total.index;
            topKTmpValBuf[index] = total.value;

            if (total.index >= 0)
            {
                tmpLogits[total.index] = -MAX_T_VAL;
            }
        }
        __syncthreads();
    }
}

// Stage 2: Sample from top-K elements using softmax
template <typename T, int BLOCK_SIZE_>
__global__ void topKStage2Sampling(int32_t const* __restrict__ topKTmpIdBuf, T* topKTmpValBuf,
    int64_t* __restrict__ selectedIndices, SamplingParams const params, uint64_t philoxSeed, uint64_t philoxOffset)
{
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? fromFloat<T>(HALF_FLT_MAX) : fromFloat<T>(FLT_MAX);

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x);

    if (batchIdx >= params.batchSize)
    {
        return;
    }

    auto const k = params.topK;
    auto const topP = params.topP;
    auto const size = k * 8; // BLOCKS_PER_BEAM = 8
    auto const stride = k * 8;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    __shared__ float sSum;
    T* sVal = topKTmpValBuf + batchIdx * stride;
    auto* sId = reinterpret_cast<int32_t*>(array);

    if (tid == 0)
    {
        sSum = 0.0f;
    }
    TopK_2<float> partial;

    auto sVal2 = reinterpret_cast<float*>(sId + k);
    float maxLogit;

    // Sort top-K elements and compute softmax
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int32_t i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert(static_cast<float>(sVal[i]), i);
        }

        TopK_2<float> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<float>());

        if (tid == 0)
        {
            if (ite == 0)
            {
                maxLogit = total.value;
            }

            if (total.index >= 0 && total.index < size)
            {
                sId[ite] = total.index;
                sVal[total.index] = -MAX_T_VAL;

                total.value = __expf(total.value - maxLogit);
                sVal2[ite] = total.value;
                sSum += total.value;
            }
            else
            {
                sId[ite] = -1;
                sVal2[ite] = 0.0f;
            }
        }
        __syncthreads();
    }

    // Sample from the distribution
    if (tid == 0)
    {
        // Initialize curand state for this batch
        curandState_t localState;
        curand_init(philoxSeed, batchIdx, philoxOffset, &localState);

        // When generating random number, topP filtering is applied to ensure the sum of the probabilities is less than
        // topP
        auto randNum = static_cast<float>(curand_uniform(&localState) * topP * sSum);

        for (int32_t ki = 0; ki < k; ki++)
        {
            auto expLogit = sVal2[ki];
            randNum = randNum - expLogit;

            if (randNum <= 0.0f || ki == k - 1)
            {
                auto idx = sId[ki];

                if (idx >= 0 && idx < stride)
                {
                    auto globalIdx = topKTmpIdBuf[batchIdx * stride + idx];
                    auto outputId = (globalIdx != -1) ? (globalIdx % params.vocabSize) : (params.vocabSize - 1);
                    outputId = (outputId == -1) ? (params.vocabSize - 1) : outputId;

                    if (outputId >= 0 && outputId < params.vocabSize)
                    {
                        selectedIndices[batchIdx] = outputId;
                    }
                    else
                    {
                        selectedIndices[batchIdx] = params.vocabSize - 1; // Safe fallback
                    }
                }
                else
                {
                    selectedIndices[batchIdx] = params.vocabSize - 1; // Safe fallback
                }
                break;
            }
        }
    }
}

// =======================================================================================
// SOFTMAX KERNEL (Required for Top-P sampling)
// =======================================================================================

template <typename T, int BLOCK_SIZE>
__global__ void softmaxKernel(T const* logits, T* probs, int32_t batchSize, int32_t vocabSize, float temperature)
{
    auto const batchId = static_cast<int32_t>(blockIdx.x);
    auto const tid = static_cast<int32_t>(threadIdx.x);

    if (batchId >= batchSize)
        return;

    auto const offset = batchId * vocabSize;
    auto const invTemp = (temperature == 0.0f) ? 0.0f : 1.0f / temperature;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float maxLogit;
    __shared__ float sumExp;

    // Find max logit for numerical stability
    float threadMax = -FLT_MAX;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto logit = toFloat(logits[offset + i]) * invTemp;
        threadMax = fmaxf(threadMax, logit);
    }

    float blockMax = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
    if (tid == 0)
    {
        maxLogit = blockMax;
        sumExp = 0.0f;
    }
    __syncthreads();

    // Compute exp(logit - maxLogit) and sum
    float threadSum = 0.0f;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto logit = toFloat(logits[offset + i]) * invTemp;
        auto expLogit = expf(logit - maxLogit);
        probs[offset + i] = fromFloat<T>(expLogit);
        threadSum += expLogit;
    }

    float blockSum = BlockReduce(tempStorage).Reduce(threadSum, cub::Sum());
    if (tid == 0)
    {
        sumExp = blockSum;
    }
    __syncthreads();

    // Normalize to get probabilities
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto prob = toFloat(probs[offset + i]) / sumExp;
        probs[offset + i] = fromFloat<T>(prob);
    }
}

// =======================================================================================
// TOP-P SAMPLING KERNELS (based on TensorRT-LLM implementation)
// =======================================================================================

// Initialize ID values and offsets for top-p sampling
__global__ void topPInitialize(
    int32_t* topPIdValBuf, int32_t* topPOffsetBuf, int32_t* beginTopPOffsetBuf, int32_t batchSize, int32_t vocabSize)
{
    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    if (bid == 0)
    {
        for (auto i = tid; i < batchSize + 1; i += static_cast<int32_t>(blockDim.x))
        {
            int32_t expectedOffset = i * vocabSize;
            topPOffsetBuf[i] = expectedOffset;
            beginTopPOffsetBuf[i] = expectedOffset;
        }
    }

    auto index = tid + bid * static_cast<int32_t>(blockDim.x);

    while (index < batchSize * vocabSize)
    {
        topPIdValBuf[index] = index % vocabSize;
        index += static_cast<int32_t>(blockDim.x * gridDim.x);
    }
}

// Early exit optimization: check if highest probability token exceeds threshold
template <typename T, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(T const* probs, int32_t* topKTmpIdBuf,
    T* topKTmpValBuf, int32_t* earlyExitFlags, int32_t vocabSize, float topP, int32_t batchSize)
{
    auto const threadId = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);

    if (batchId >= batchSize)
        return;

    float pThreshold = topP;

    typedef cub::BlockReduce<TopK_2<T>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_2<T> partial;

    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? fromFloat<T>(HALF_FLT_MAX) : fromFloat<T>(FLT_MAX);

    partial.value = -MAX_T_VAL;
    partial.index = -1;

#pragma unroll
    for (int32_t elemId = static_cast<int32_t>(threadId); elemId < vocabSize; elemId += THREADBLOCK_SIZE)
    {
        auto index = elemId + batchId * vocabSize;
        partial.insert(probs[index], elemId);
    }

    TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>());

    if (threadId == 0)
    {
        T sumProb = total.value;

        if (static_cast<float>(sumProb) >= pThreshold)
        {
            // Early exit: set flag and store the selected token
            earlyExitFlags[batchId] = 1;
            auto index = batchId * vocabSize;
            topKTmpIdBuf[index] = total.index; // Store the actual token ID
            topKTmpValBuf[index] = total.value;
        }
        else
        {
            // No early exit
            earlyExitFlags[batchId] = 0;
        }
    }
}

// Final sampling stage using block-level prefix sum
template <typename T, int blockSize>
__global__ void topPSampling(T const* sortedProbs, int32_t const* sortedIdVals, int64_t* selectedIndices,
    int32_t const* topKTmpIdBuf, int32_t const* earlyExitFlags, int32_t vocabSize, uint64_t philoxSeed,
    uint64_t philoxOffset, float topP, int32_t batchSize)
{
    __shared__ float randNumS;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);

    if (batchId >= batchSize)
        return;

    auto const probThreshold = topP;

    if (threadIdx.x == 0)
    {
        // Initialize curand state for this batch
        curandState_t localState;
        curand_init(philoxSeed, batchId, philoxOffset, &localState);

        auto const randomNumber = curand_uniform(&localState);
        randNumS = randomNumber * probThreshold;
    }

    typedef cub::BlockScan<float, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;
    BlockPrefixCallbackOp prefixOp(0);

    __syncthreads();

    auto offset = batchId * vocabSize;
    auto end = ((vocabSize + blockSize - 1) / blockSize) * blockSize;
    int32_t selectedTokenId = 0;
    float threadOffset = 0;
    int32_t count = 0;

    for (int vi = tid; vi < end; vi += blockSize)
    {
        auto threadProb = (vi < vocabSize) ? static_cast<float>(sortedProbs[offset + vi]) : 0.f;
        BlockScan(tempStorage).InclusiveSum(threadProb, threadOffset, prefixOp);
        count = __syncthreads_count(randNumS <= threadOffset);
        selectedTokenId = vi;

        if (count != 0)
        {
            break;
        }
    }

    selectedTokenId = min(selectedTokenId, vocabSize - 1);

    if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
    {
        int32_t finalToken = sortedIdVals[offset + selectedTokenId];
        selectedIndices[batchId] = finalToken;
    }
}

// =======================================================================================
// HOST WRAPPER FUNCTIONS
// =======================================================================================

// Updated sampling function with automatic workspace allocation fallback
template <typename T>
void topKtopPSamplingFromLogits(T const* logits, int64_t* selectedIndices, SamplingParams const& params,
    void* workspace, size_t workspaceSize, cudaStream_t stream, uint64_t philoxSeed, uint64_t philoxOffset)
{
    assert(logits != nullptr && selectedIndices != nullptr);
    assert(params.batchSize > 0 && params.vocabSize > 0);

    int const BLOCK_SIZE = 256;
    int const BLOCKS_PER_BEAM = 8;

    // Setup workspace partitioning
    SamplingWorkspace<T> ws;
    ws.setupWorkspace(workspace, workspaceSize, params);

    // Validate workspace buffers
    if (params.useTopK)
    {
        assert(ws.topkTempLogits != nullptr && ws.topkIndices != nullptr && ws.topkValues != nullptr);
    }

    if (params.useTopK)
    {

        // Stage 1: Find top-K elements
        dim3 grid1(params.batchSize * BLOCKS_PER_BEAM);
        dim3 block1(BLOCK_SIZE);

        topKStage1<T, BLOCK_SIZE, BLOCKS_PER_BEAM>
            <<<grid1, block1, 0, stream>>>(logits, ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);

        // Stage 2: Sample from top-K elements
        dim3 grid2(params.batchSize);
        dim3 block2(BLOCK_SIZE);
        size_t sharedMemSize = params.topK * sizeof(int32_t) + params.topK * sizeof(float);

        topKStage2Sampling<T, BLOCK_SIZE><<<grid2, block2, sharedMemSize, stream>>>(
            ws.topkIndices, ws.topkValues, selectedIndices, params, philoxSeed, philoxOffset);
    }
    else if (params.useTopP)
    {
        // Top-P only sampling using workspace

        // Stage 0: Convert logits to probabilities using softmax
        int const SOFTMAX_BLOCK_SIZE = 256;

        softmaxKernel<T, SOFTMAX_BLOCK_SIZE><<<params.batchSize, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
            logits, ws.toppProbs, params.batchSize, params.vocabSize, params.temperature);

        // Stage 1: Initialize
        topPInitialize<<<32, 512, 0, stream>>>(
            ws.toppIdVals, ws.toppOffsetBuf, ws.toppBeginOffsetBuf, params.batchSize, params.vocabSize);

        // Stage 2: Early exit optimization
        int const BLOCK_SIZE_TOPK = 256;

        topPBeamTopKKernel<T, BLOCK_SIZE_TOPK><<<params.batchSize, BLOCK_SIZE_TOPK, 0, stream>>>(ws.toppProbs,
            ws.toppTopKIndices, ws.toppTopKValues, ws.toppEarlyExitFlags, params.vocabSize, params.topP,
            params.batchSize);

        // Stage 3: Sort probabilities in descending order
        size_t cubTempStorageSize;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, static_cast<T*>(nullptr),
            static_cast<T*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<int32_t>(params.vocabSize * params.batchSize), params.batchSize, static_cast<int32_t*>(nullptr),
            static_cast<int32_t*>(nullptr));

        cub::DeviceSegmentedRadixSort::SortPairsDescending(ws.toppTempStorage, cubTempStorageSize, ws.toppProbs,
            ws.toppSortedProbs, ws.toppIdVals, ws.toppSortedIdVals, params.vocabSize * params.batchSize,
            params.batchSize, ws.toppBeginOffsetBuf, ws.toppOffsetBuf + 1, 0, sizeof(T) * 8, stream);

        // Stage 4: Sample using block-level prefix sum
        int const SAMPLING_BLOCK_SIZE = 256;

        topPSampling<T, SAMPLING_BLOCK_SIZE><<<params.batchSize, SAMPLING_BLOCK_SIZE, 0, stream>>>(ws.toppSortedProbs,
            ws.toppSortedIdVals, selectedIndices, ws.toppTopKIndices, ws.toppEarlyExitFlags, params.vocabSize,
            philoxSeed, philoxOffset, params.topP, params.batchSize);
    }
}

// selectAllTopK function with automatic workspace allocation fallback
template <typename T>
void selectAllTopKFromLogits(T const* input, float* topKValues, int64_t* topKIndices, int32_t batchSize,
    int32_t vocabSize, int32_t topK, void* workspace, size_t workspaceSize, cudaStream_t stream, bool returnLogProbs,
    bool normalizeLogProbs, bool inputHasProbs)
{
    if (topK <= 0 || topK > vocabSize)
    {
        return;
    }

    // Validate that topKValues is not nullptr when returnLogProbs is true, or null when returnLogProbs is false
    if ((returnLogProbs && topKValues == nullptr) || (!returnLogProbs && topKValues != nullptr))
    {
        throw std::invalid_argument(
            "topKValues must be non-null when returnLogProbs is true, or null when returnLogProbs is false");
    }

    constexpr int32_t BLOCK_SIZE = 256;
    constexpr int32_t BLOCKS_PER_BEAM = 8;

    // Setup workspace partitioning
    SamplingWorkspace<T> ws;
    ws.setupWorkspaceForTopK(workspace, workspaceSize, batchSize, vocabSize, topK);

    // Create sampling parameters with temperature = 1.0 (no modification of input values)
    SamplingParams params(batchSize, vocabSize, 1.0f, topK);

    // Stage 1: Find top-K elements using existing kernel
    dim3 grid1(batchSize * BLOCKS_PER_BEAM);
    dim3 block1(BLOCK_SIZE);

    topKStage1<T, BLOCK_SIZE, BLOCKS_PER_BEAM>
        <<<grid1, block1, 0, stream>>>(input, ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);

    // Stage 2: Second top-K selection from 8*K results (matches old sampler for returnAllTopK)
    dim3 grid2(batchSize);
    dim3 block2(BLOCK_SIZE);
    size_t sharedMemSize = topK * sizeof(int32_t) + topK * sizeof(float);

    topKStage2ReturnAllTopK<T, BLOCK_SIZE><<<grid2, block2, sharedMemSize, stream>>>(ws.topkIndices, ws.topkValues,
        topKIndices, topKValues, returnLogProbs ? ws.topkTempLogits : nullptr, batchSize, vocabSize, topK,
        BLOCKS_PER_BEAM, returnLogProbs, normalizeLogProbs, inputHasProbs);
}

// Explicit template instantiations
template void topKtopPSamplingFromLogits<float>(float const* logits, int64_t* selectedIndices,
    SamplingParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream, uint64_t philoxSeed,
    uint64_t philoxOffset);

template void topKtopPSamplingFromLogits<half>(half const* logits, int64_t* selectedIndices,
    SamplingParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream, uint64_t philoxSeed,
    uint64_t philoxOffset);

template void topKtopPSamplingFromLogits<__nv_bfloat16>(__nv_bfloat16 const* logits, int64_t* selectedIndices,
    SamplingParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream, uint64_t philoxSeed,
    uint64_t philoxOffset);

template void selectAllTopKFromLogits<float>(float const* input, float* topKValues, int64_t* topKIndices,
    int32_t batchSize, int32_t vocabSize, int32_t topK, void* workspace, size_t workspaceSize, cudaStream_t stream,
    bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs);

template void selectAllTopKFromLogits<half>(half const* input, float* topKValues, int64_t* topKIndices,
    int32_t batchSize, int32_t vocabSize, int32_t topK, void* workspace, size_t workspaceSize, cudaStream_t stream,
    bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs);

template void selectAllTopKFromLogits<__nv_bfloat16>(__nv_bfloat16 const* input, float* topKValues,
    int64_t* topKIndices, int32_t batchSize, int32_t vocabSize, int32_t topK, void* workspace, size_t workspaceSize,
    cudaStream_t stream, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs);

template size_t getTopKtopPSamplingWorkspaceSize<float>(
    int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

template size_t getTopKtopPSamplingWorkspaceSize<half>(
    int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

template size_t getTopKtopPSamplingWorkspaceSize<__nv_bfloat16>(
    int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

template size_t getSelectAllTopKWorkspaceSize<float>(int32_t batchSize, int32_t vocabSize, int32_t topK);

template size_t getSelectAllTopKWorkspaceSize<half>(int32_t batchSize, int32_t vocabSize, int32_t topK);

template size_t getSelectAllTopKWorkspaceSize<__nv_bfloat16>(int32_t batchSize, int32_t vocabSize, int32_t topK);

} // namespace drivellm