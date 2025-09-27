/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "eagleAcceptKernels.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace drivellm
{
namespace kernel
{

// Shared alignment function for workspace buffers
inline size_t alignWorkspaceSize(size_t size)
{
    size_t const alignment = 256; // 256-byte alignment for optimal GPU memory access
    return (size + alignment - 1) & ~(alignment - 1);
}

// Internal workspace structure for memory management (similar to SamplingWorkspace)
struct EagleAcceptWorkspace
{
    // Device pointer, owned by external caller - must remain valid for kernel lifetime
    void* ptr;
    // Size in bytes of the workspace buffer
    size_t size;

    // Device pointer to top-1 tokens buffer, owned by external caller
    // Array size: batchSize * numTokens elements (int32_t each)
    int32_t* top1Tokens;

    EagleAcceptWorkspace()
        : ptr(nullptr)
        , size(0)
        , top1Tokens(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, int32_t batchSize, int32_t numTokens)
    {
        // Check that workspace is aligned to 256 bytes for optimal GPU memory access
        if (reinterpret_cast<uintptr_t>(workspace) % 256 != 0)
        {
            throw std::runtime_error("Workspace must be aligned to 256 bytes");
        }

        ptr = workspace;
        size = workspaceSize;

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        // Top-1 tokens buffer
        size_t top1TokensSize = alignWorkspaceSize(batchSize * numTokens * sizeof(int32_t));
        top1Tokens = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TokensSize;

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Eagle workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }
};

// Calculate workspace size for Eagle accept algorithm
size_t getEagleAcceptWorkspaceSize(int32_t batchSize, int32_t numTokens)
{
    // Top-1 tokens buffer
    return alignWorkspaceSize(batchSize * numTokens * sizeof(int32_t));
}

namespace
{

// Helper structure for top-1 selection
struct Top1Helper
{
    float value;
    int32_t index;

    __device__ __forceinline__ Top1Helper()
        : value(-FLT_MAX)
        , index(-1)
    {
    }

    __device__ __forceinline__ void update(float elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

// Reduction operator for top-1 selection
struct top1MaxOpFunctor
{
    __device__ __forceinline__ Top1Helper operator()(Top1Helper const& a, Top1Helper const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Inline function for shared memory alignment
__forceinline__ size_t alignSharedMem(size_t size)
{
    return ((size + 15) / 16) * 16; // Align to 16 bytes
}

// Stage 1: Compute top-1 tokens for all positions using sampling strategy
template <int32_t BLOCK_SIZE>
__global__ void eagleComputeTop1Kernel(
    float const* logits, int32_t* top1Tokens, int32_t batchSize, int32_t numTokens, int32_t vocabSize)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x / numTokens);
    auto const tokenIdx = static_cast<int32_t>(blockIdx.x % numTokens);

    if (batchIdx >= batchSize || tokenIdx >= numTokens)
        return;

    // Calculate logits offset for this batch and token position
    int32_t const logitsOffset = batchIdx * numTokens * vocabSize + tokenIdx * vocabSize;

    Top1Helper partial;

    // Find top-1 token using parallel reduction across vocab
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE)
    {
        float logitValue = logits[logitsOffset + v];
        partial.update(logitValue, v);
    }

    // Block-level reduction to find global max
    Top1Helper blockMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    // Store result
    if (tid == 0)
    {
        int32_t outputIdx = batchIdx * numTokens + tokenIdx;
        top1Tokens[outputIdx] = (blockMax.index != -1) ? blockMax.index : 0;
    }
}
// Helper function to compute tree depth - count total connections (sum of 1s)
__device__ int32_t computeTokenDepth(int32_t tokenIdx, int8_t const* attentionMask, int32_t numTokens)
{
    int32_t depth = 0;
    for (int32_t i = 0; i < numTokens; ++i)
    {
        if (attentionMask[tokenIdx * numTokens + i] == 1)
        {
            depth++;
        }
    }
    return depth;
}

// CUDA kernel for eagle accept algorithm - optimized for concurrent batch processing
__global__ void eagleAcceptKernel(int32_t const* top1Tokens, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t numTokens,
    int32_t maxDepth)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    if (batchIdx >= batchSize)
        return;

    // Minimal shared memory usage - only for token depths and communication
    extern __shared__ char sharedMem[];
    int32_t* tokenDepths = reinterpret_cast<int32_t*>(sharedMem);

    // Initialize output arrays for this batch
    if (tid == 0)
    {
        acceptLength[batchIdx] = 1; // First token is always accepted
        for (int32_t i = 0; i < maxDepth; ++i)
        {
            acceptedTokenIds[batchIdx * maxDepth + i] = -1;
            acceptedIndices[batchIdx * maxDepth + i] = -1;
        }
        // First token is always accepted
        acceptedTokenIds[batchIdx * maxDepth + 0] = tokenIds[batchIdx * numTokens + 0];
        acceptedIndices[batchIdx * maxDepth + 0] = 0;
    }

    // Precompute token depths for this batch using batch-specific attention mask
    int8_t const* batchAttentionMask = attentionMask + batchIdx * numTokens * numTokens;
    for (int32_t i = tid; i < numTokens; i += blockSize)
    {
        tokenDepths[i] = computeTokenDepth(i, batchAttentionMask, numTokens);
    }
    __syncthreads();

    // Process this batch - each batch processes independently
    int32_t currentDepth = 1;
    int32_t currentTokenIdx = 0;
    int32_t expectedNextDepth = tokenDepths[0] + 1; // Next depth should be current token's depth + 1

    for (int32_t step = 0; step < maxDepth - 1; ++step)
    {
        // All threads check the same condition since variables are the same for all
        if (currentDepth >= maxDepth || currentTokenIdx >= numTokens - 1)
            break;

        // Step 1: Get precomputed top-1 token for current position
        __shared__ int32_t selectedTokenId;
        __shared__ int32_t nextTokenIdx;

        if (tid == 0)
        {
            // Use precomputed top-1 token instead of computing argmax
            int32_t top1Idx = batchIdx * numTokens + currentTokenIdx;
            selectedTokenId = top1Tokens[top1Idx];
            nextTokenIdx = -1;
        }
        __syncthreads();

        // Step 2: Find which token in the tree matches the selected token and is at the correct depth
        // Use parallel search across all threads
        int32_t const* batchTokenIds = tokenIds + batchIdx * numTokens;
        for (int32_t checkIdx = tid + 1; checkIdx < numTokens; checkIdx += blockSize)
        {
            if (batchTokenIds[checkIdx] == selectedTokenId && tokenDepths[checkIdx] == expectedNextDepth)
            {
                // Check attention mask: does checkIdx attend to currentTokenIdx?
                int32_t maskOffset = batchIdx * numTokens * numTokens + checkIdx * numTokens + currentTokenIdx;
                if (attentionMask[maskOffset] == 1)
                {
                    // Found a valid next token - use atomic to ensure only first match is taken
                    atomicCAS(&nextTokenIdx, -1, checkIdx);
                }
            }
        }
        __syncthreads();

        // Step 3: Update results if valid token found
        // Use shared memory to communicate updated values to all threads
        __shared__ int32_t newCurrentDepth;
        __shared__ int32_t newCurrentTokenIdx;
        __shared__ int32_t newExpectedNextDepth;

        if (tid == 0)
        {
            if (nextTokenIdx != -1)
            {
                acceptedTokenIds[batchIdx * maxDepth + currentDepth] = selectedTokenId;
                acceptedIndices[batchIdx * maxDepth + currentDepth] = nextTokenIdx;
                acceptLength[batchIdx] = currentDepth + 1;
                newCurrentTokenIdx = nextTokenIdx;
                newCurrentDepth = currentDepth + 1;
                newExpectedNextDepth = expectedNextDepth + 1;
            }
            else
            {
                // Signal no update
                newCurrentDepth = currentDepth;
                newCurrentTokenIdx = currentTokenIdx;
                newExpectedNextDepth = expectedNextDepth;
            }
        }
        __syncthreads();

        // All threads update their local variables
        currentDepth = newCurrentDepth;
        currentTokenIdx = newCurrentTokenIdx;
        expectedNextDepth = newExpectedNextDepth;

        // Break out of loop if no valid token found
        if (nextTokenIdx == -1)
            break;
    }
}

// Optimized kernel launcher function using workspace and two-stage approach
void launchEagleAcceptKernel(float const* logits, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t numTokens,
    int32_t vocabSize, int32_t maxDepth, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;

    // Setup workspace partitioning
    EagleAcceptWorkspace ws;
    ws.setupWorkspace(workspace, workspaceSize, batchSize, numTokens);

    // Validate workspace buffer
    assert(ws.top1Tokens != nullptr);

    // Stage 1: Compute top-1 tokens for all positions
    dim3 const gridSizeStage1(batchSize * numTokens);
    dim3 const blockSizeStage1(blockSize);

    // Calculate shared memory for stage 1 (only CUB temp storage)
    size_t sharedMemSizeStage1 = sizeof(typename cub::BlockReduce<Top1Helper, blockSize>::TempStorage);
    sharedMemSizeStage1 = alignSharedMem(sharedMemSizeStage1);

    eagleComputeTop1Kernel<blockSize><<<gridSizeStage1, blockSizeStage1, sharedMemSizeStage1, stream>>>(
        logits, ws.top1Tokens, batchSize, numTokens, vocabSize);

    // Stage 2: Run optimized eagle accept algorithm
    dim3 const gridSizeStage2(batchSize);
    dim3 const blockSizeStage2(blockSize);

    // Calculate shared memory for stage 2 (only token depths - much smaller!)
    size_t sharedMemSizeStage2 = numTokens * sizeof(int32_t);
    sharedMemSizeStage2 = alignSharedMem(sharedMemSizeStage2);

    eagleAcceptKernel<<<gridSizeStage2, blockSizeStage2, sharedMemSizeStage2, stream>>>(ws.top1Tokens, tokenIds,
        attentionMask, acceptedTokenIds, acceptedIndices, acceptLength, batchSize, numTokens, maxDepth);
}

} // namespace

void eagleAccept(rt::Tensor const& logits, rt::Tensor const& tokenIds, rt::Tensor const& attentionMask,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedIndices, rt::Tensor& acceptLength, int32_t maxDepth,
    void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // Validate input shapes
    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();
    auto const maskShape = attentionMask.getShape();
    auto const acceptedTokenIdsShape = acceptedTokenIds.getShape();
    auto const acceptedIndicesShape = acceptedIndices.getShape();
    auto const acceptLengthShape = acceptLength.getShape();

    check::check(logitsShape.getNumDims() == 3, "logits must be 3D tensor [batch_size, num_tokens, vocab_size]");
    check::check(tokenIdsShape.getNumDims() == 2, "tokenIds must be 2D tensor [batch_size, num_tokens]");
    check::check(maskShape.getNumDims() == 3, "attentionMask must be 3D tensor [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape.getNumDims() == 2, "acceptedTokenIds must be 2D tensor [batch_size, max_depth]");
    check::check(acceptedIndicesShape.getNumDims() == 2, "acceptedIndices must be 2D tensor [batch_size, max_depth]");
    check::check(acceptLengthShape.getNumDims() == 1, "acceptLength must be 1D tensor [batch_size]");

    int32_t const batchSize = logitsShape[0];
    int32_t const numTokens = logitsShape[1];
    int32_t const vocabSize = logitsShape[2];

    check::check(
        tokenIdsShape[0] == batchSize && tokenIdsShape[1] == numTokens, "tokenIds must be [batch_size, num_tokens]");
    check::check(maskShape[0] == batchSize && maskShape[1] == numTokens && maskShape[2] == numTokens,
        "attentionMask must be [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape[0] == batchSize && acceptedTokenIdsShape[1] == maxDepth,
        "acceptedTokenIds must be [batch_size, max_depth]");
    check::check(acceptedIndicesShape[0] == batchSize && acceptedIndicesShape[1] == maxDepth,
        "acceptedIndices must be [batch_size, max_depth]");
    check::check(acceptLengthShape[0] == batchSize, "acceptLength length must match batch_size");

    // Validate data types
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "logits must be FP32");
    check::check(tokenIds.getDataType() == nvinfer1::DataType::kINT32, "tokenIds must be INT32");
    check::check(attentionMask.getDataType() == nvinfer1::DataType::kINT8, "attentionMask must be INT8");
    check::check(acceptedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "acceptedTokenIds must be INT32");
    check::check(acceptedIndices.getDataType() == nvinfer1::DataType::kINT32, "acceptedIndices must be INT32");
    check::check(acceptLength.getDataType() == nvinfer1::DataType::kINT32, "acceptLength must be INT32");

    // Validate device types - all tensors must be on GPU
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU, "logits must be on GPU device");
    check::check(tokenIds.getDeviceType() == rt::DeviceType::kGPU, "tokenIds must be on GPU device");
    check::check(attentionMask.getDeviceType() == rt::DeviceType::kGPU, "attentionMask must be on GPU device");
    check::check(acceptedTokenIds.getDeviceType() == rt::DeviceType::kGPU, "acceptedTokenIds must be on GPU device");
    check::check(acceptedIndices.getDeviceType() == rt::DeviceType::kGPU, "acceptedIndices must be on GPU device");
    check::check(acceptLength.getDeviceType() == rt::DeviceType::kGPU, "acceptLength must be on GPU device");
    check::check(maxDepth > 0 && maxDepth <= numTokens, "maxDepth must be positive and <= numTokens");

    // Get device pointers
    float const* logitsPtr = logits.dataPointer<float>();
    int32_t const* tokenIdsPtr = tokenIds.dataPointer<int32_t>();
    int8_t const* attentionMaskPtr = attentionMask.dataPointer<int8_t>();
    int32_t* acceptedTokenIdsPtr = acceptedTokenIds.dataPointer<int32_t>();
    int32_t* acceptedIndicesPtr = acceptedIndices.dataPointer<int32_t>();
    int32_t* acceptLengthPtr = acceptLength.dataPointer<int32_t>();

    // Validate workspace size
    size_t requiredWorkspaceSize = getEagleAcceptWorkspaceSize(batchSize, numTokens);
    if (workspaceSize < requiredWorkspaceSize)
    {
        throw std::runtime_error("Eagle workspace size too small. Required: " + std::to_string(requiredWorkspaceSize)
            + ", provided: " + std::to_string(workspaceSize));
    }

    // Launch kernel
    launchEagleAcceptKernel(logitsPtr, tokenIdsPtr, attentionMaskPtr, acceptedTokenIdsPtr, acceptedIndicesPtr,
        acceptLengthPtr, batchSize, numTokens, vocabSize, maxDepth, workspace, workspaceSize, stream);
}

} // namespace kernel
} // namespace drivellm