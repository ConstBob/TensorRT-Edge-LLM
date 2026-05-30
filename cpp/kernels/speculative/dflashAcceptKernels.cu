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

#include "common/checkMacros.h"
#include "kernels/speculative/dflashAcceptKernels.h"

#include <cfloat>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

// ---------------------------------------------------------------------------
// Optimized DFlash sequential accept with parallel argmax reduction
// ---------------------------------------------------------------------------
//
// The old kernel used 128 threads doing sequential O(V) scans per position,
// taking ~18ms on Thor for V=248K. This version:
//   Phase 1: one CTA per (batch, position), 256 threads cooperate on argmax
//            via warp shuffle + shared-memory reduction. ~0.1ms per position.
//   Phase 2: tiny kernel, one thread per batch, sequential accept walk.
//

static constexpr int32_t kArgmaxBlockSize = 256;

/// Phase 1: Parallel argmax — one CTA per (batch, position), all threads
/// cooperate to find argmax over vocabSize.
__global__ void dflashParallelArgmaxKernel(float const* __restrict__ baseLogits, // [B * VF * V]
    int32_t* __restrict__ argmaxResults,                                         // [B * VF]
    int32_t totalPositions, int32_t vocabSize)
{
    int32_t const posIdx = blockIdx.x;
    if (posIdx >= totalPositions)
    {
        return;
    }

    float const* posLogits = baseLogits + static_cast<int64_t>(posIdx) * vocabSize;

    // Each thread finds local max over its stride of the vocab
    float localMax = -FLT_MAX;
    int32_t localIdx = 0;

    for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
    {
        float val = posLogits[v];
        // Tie-break by lower vocab ID for determinism
        if (val > localMax || (val == localMax && v < localIdx))
        {
            localMax = val;
            localIdx = v;
        }
    }

    // Warp-level reduction via shuffle (tie-break: lower index wins)
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        float otherMax = __shfl_down_sync(0xFFFFFFFF, localMax, offset);
        int32_t otherIdx = __shfl_down_sync(0xFFFFFFFF, localIdx, offset);
        if (otherMax > localMax || (otherMax == localMax && otherIdx < localIdx))
        {
            localMax = otherMax;
            localIdx = otherIdx;
        }
    }

    // Inter-warp reduction via shared memory
    __shared__ float sMaxVal[32];
    __shared__ int32_t sMaxIdx[32];

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const numWarps = (blockDim.x + 31) / 32;

    if (laneId == 0)
    {
        sMaxVal[warpId] = localMax;
        sMaxIdx[warpId] = localIdx;
    }
    __syncthreads();

    // Final reduction in first warp only
    if (warpId == 0)
    {
        float wMax = (laneId < numWarps) ? sMaxVal[laneId] : -FLT_MAX;
        int32_t wIdx = (laneId < numWarps) ? sMaxIdx[laneId] : 0;

        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            float otherMax = __shfl_down_sync(0xFFFFFFFF, wMax, offset);
            int32_t otherIdx = __shfl_down_sync(0xFFFFFFFF, wIdx, offset);
            if (otherMax > wMax || (otherMax == wMax && otherIdx < wIdx))
            {
                wMax = otherMax;
                wIdx = otherIdx;
            }
        }

        if (laneId == 0)
        {
            argmaxResults[posIdx] = wIdx;
        }
    }
}

/// Phase 2: Sequential accept walk — one thread per batch
__global__ void dflashSequentialAcceptWalkKernel(int32_t const* __restrict__ argmaxResults, // [B, VF]
    int32_t const* __restrict__ draftTokenIds,                                              // [B, VF]
    int32_t* __restrict__ acceptedTokenIds,                                                 // [B, VF]
    int32_t* __restrict__ acceptLength,                                                     // [B]
    int32_t verifyLen)
{
    int32_t const batchIdx = blockIdx.x;

    int32_t const* batchArgmax = argmaxResults + batchIdx * verifyLen;
    int32_t const* batchDraft = draftTokenIds + batchIdx * verifyLen;
    int32_t* batchAccepted = acceptedTokenIds + batchIdx * verifyLen;

    // Always accept base prediction after y0 (position 0)
    batchAccepted[0] = batchArgmax[0];
    int32_t accepted = 1;

    for (int32_t i = 1; i < verifyLen; ++i)
    {
        // Does base prediction at position i-1 match draft at position i?
        if (batchArgmax[i - 1] == batchDraft[i])
        {
            batchAccepted[accepted] = batchArgmax[i];
            accepted++;
        }
        else
        {
            break;
        }
    }

    acceptLength[batchIdx] = accepted;
}

/// Build verify token IDs on GPU.
__global__ void dflashBuildVerifyTokensKernel(int32_t const* __restrict__ lastAcceptedTokens, // [B]
    int32_t const* __restrict__ draftTokenIds,                                                // [B, BS]
    int32_t* __restrict__ verifyTokenIds,                                                     // [B, BS]
    int32_t blockSize, int32_t totalElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }
    int32_t const batchIdx = idx / blockSize;
    int32_t const posIdx = idx % blockSize;

    if (posIdx == 0)
    {
        verifyTokenIds[batchIdx * blockSize] = lastAcceptedTokens[batchIdx];
    }
    else
    {
        verifyTokenIds[batchIdx * blockSize + posIdx] = draftTokenIds[batchIdx * blockSize + posIdx];
    }
}

} // anonymous namespace

void dflashSequentialAccept(rt::Tensor const& baseLogits, rt::Tensor const& draftTokenIds, rt::Tensor& acceptedTokenIds,
    rt::Tensor& acceptLength, rt::Tensor& argmaxScratch, int32_t batchSize, int32_t verifyLen, int32_t vocabSize,
    cudaStream_t stream)
{
    int32_t const totalPositions = batchSize * verifyLen;
    int32_t* argmaxResults = static_cast<int32_t*>(argmaxScratch.rawPointer());

    // Phase 1: Parallel argmax — one CTA per position, 256 threads cooperate
    dflashParallelArgmaxKernel<<<totalPositions, kArgmaxBlockSize, 0, stream>>>(
        static_cast<float const*>(baseLogits.rawPointer()), argmaxResults, totalPositions, vocabSize);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Sequential accept walk — one thread per batch (trivially fast)
    dflashSequentialAcceptWalkKernel<<<batchSize, 1, 0, stream>>>(argmaxResults,
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<int32_t*>(acceptedTokenIds.rawPointer()),
        static_cast<int32_t*>(acceptLength.rawPointer()), verifyLen);
    CUDA_CHECK(cudaGetLastError());
}

void dflashBuildVerifyTokens(rt::Tensor const& lastAcceptedTokens, rt::Tensor const& draftTokenIds,
    rt::Tensor& verifyTokenIds, int32_t batchSize, int32_t blockSize, cudaStream_t stream)
{
    int32_t const totalThreads = batchSize * blockSize;
    int32_t const numThreads = 256;
    int32_t const numBlocks = (totalThreads + numThreads - 1) / numThreads;

    dflashBuildVerifyTokensKernel<<<numBlocks, numThreads, 0, stream>>>(
        static_cast<int32_t const*>(lastAcceptedTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<int32_t*>(verifyTokenIds.rawPointer()),
        blockSize, totalThreads);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
