/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/cudaUtils.h"
#include "moeMarlinIndicesKernels.h"

#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

// Build Marlin indices from slot lists
__global__ void buildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize)
{
    int32_t expertId = blockIdx.x;
    if (expertId >= numExperts)
        return;

    int32_t count = slotsPerExpertWorkspace[expertId];
    int32_t paddedCount = paddedCounts[expertId];
    if (paddedCount <= 0)
        return;

    int32_t totalSlots = numTokens * topK;
    int32_t outStart = paddedOffsets[expertId];

    for (int32_t i = threadIdx.x; i < paddedCount; i += blockDim.x)
    {
        if (i < count)
        {
            int32_t slot = slotsByExpertWorkspace[expertId * totalSlots + i];
            sortedTokenIds[outStart + i] = slot;
            topkWeightsFlat[outStart + i] = topkWeights[slot];
        }
        else
        {
            sortedTokenIds[outStart + i] = totalSlots;
            topkWeightsFlat[outStart + i] = 0.0f;
        }
    }

    int32_t numBlocks = paddedCount / moeBlockSize;
    int32_t blockOffset = outStart / moeBlockSize;
    for (int32_t b = threadIdx.x; b < numBlocks; b += blockDim.x)
    {
        expertIds[blockOffset + b] = expertId;
    }
}

// Aggregate slot outputs back to tokens: sum over topK in slot order
__global__ void aggregateSlotOutputsKernel(
    half const* slotOutputs, half* aggregatedOutput, int32_t numTokens, int32_t topK, int32_t outDim)
{
    int32_t tokenId = blockIdx.x;
    int32_t dimIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (tokenId >= numTokens || dimIdx >= outDim)
        return;

    float accum = 0.0f;
    int32_t base = tokenId * topK;
    for (int32_t k = 0; k < topK; ++k)
    {
        int32_t slot = base + k;
        accum += __half2float(slotOutputs[slot * outDim + dimIdx]);
    }

    aggregatedOutput[tokenId * outDim + dimIdx] = __float2half(accum);
}

void launchBuildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize, cudaStream_t stream)
{
    buildMarlinIndicesKernel<<<numExperts, 256, 0, stream>>>(slotsByExpertWorkspace, slotsPerExpertWorkspace,
        paddedCounts, paddedOffsets, topkWeights, sortedTokenIds, topkWeightsFlat, expertIds, numTokens, topK,
        numExperts, moeBlockSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchAggregateSlotOutputsKernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens, int32_t topK,
    int32_t outDim, cudaStream_t stream)
{
    dim3 grid(numTokens, static_cast<uint32_t>(trt_edgellm::divUp(outDim, 256)));
    dim3 block(256);
    aggregateSlotOutputsKernel<<<grid, block, 0, stream>>>(
        static_cast<half const*>(slotOutputs), static_cast<half*>(aggregatedOutput), numTokens, topK, outDim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
