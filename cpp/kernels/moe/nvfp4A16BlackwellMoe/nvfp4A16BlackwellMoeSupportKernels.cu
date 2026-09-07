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

#include "nvfp4A16BlackwellMoeSupportKernels.h"

#include "nvfp4A16BlackwellMoePdl.cuh"

#include "moeSigmoidGroupTopkDevice.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{
namespace
{

constexpr int32_t kThreads{256};
//! Experts per lane of the warp routing (numExperts <= 512).
constexpr int32_t kRouteMaxExpertsPerLane{16};

template <typename T>
__global__ __launch_bounds__(kThreads) void gatherPermutedRowsKernel(T const* __restrict__ hidden,
    int32_t const* __restrict__ permutedIdx, int32_t const* __restrict__ numValidTiles, int32_t const tokenTile,
    int32_t const topK, int32_t const hiddenSize, int32_t const numTokens, int64_t const maxRowsPadded,
    T* __restrict__ permuted, T* __restrict__ output)
{
    constexpr int32_t kElementsPerVector{static_cast<int32_t>(16 / sizeof(T))};
    int32_t const vectorsPerRow = hiddenSize / kElementsPerVector;
    int64_t const block = static_cast<int64_t>(blockIdx.x);
    uint4 const zero{0U, 0U, 0U, 0U};
    // permutedIdx / numValidTiles come from the layout kernel and hidden from the
    // previous layer: nothing below may be read before the grid dependency resolves.
    pdlWait();
    if (block < maxRowsPadded)
    {
        int64_t const validRows = static_cast<int64_t>(numValidTiles[0]) * tokenTile;
        // Rows past the valid tiles and padding rows need no fill: FC1 produces a
        // discarded row from whatever is there and the FC2 epilogue skips it
        // through permuted_idx.
        int32_t const expanded = block < validRows ? permutedIdx[block] : -1;
        if (expanded >= 0)
        {
            uint4* const dst = reinterpret_cast<uint4*>(permuted + block * hiddenSize);
            int32_t const token = expanded / topK;
            uint4 const* const src = reinterpret_cast<uint4 const*>(hidden + static_cast<int64_t>(token) * hiddenSize);
            for (int32_t v = threadIdx.x; v < vectorsPerRow; v += kThreads)
            {
                dst[v] = src[v];
            }
        }
    }
    else
    {
        int64_t const token = block - maxRowsPadded;
        if (token < numTokens)
        {
            uint4* const dst = reinterpret_cast<uint4*>(output + token * hiddenSize);
            for (int32_t v = threadIdx.x; v < vectorsPerRow; v += kThreads)
            {
                dst[v] = zero;
            }
        }
    }
    pdlTrigger();
}

//! Warp-per-token sigmoid top-k routing for the ungrouped contract (nGroup == 1).
//! grid = ceil(numTokens / 8), 256 threads; dynamic smem = 8 * (numExperts + 2 * topK) words.
template <int32_t kMaxExpertsPerLane>
__global__ __launch_bounds__(kThreads) void sigmoidTopkRouteKernel(float const* __restrict__ logits,
    float const* __restrict__ correctionBias, int32_t const numTokens, int32_t const numExperts, int32_t const topK,
    bool const normTopkProb, float const routedScalingFactor, int32_t* __restrict__ topkIndices,
    float* __restrict__ topkWeights)
{
    extern __shared__ __align__(16) float routeSmem[];
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / 32;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const token = static_cast<int32_t>(blockIdx.x) * (kThreads / 32) + warp;
    // The router logits are written by the preceding layer.
    pdlWait();
    if (token < numTokens)
    {
        int32_t const wordsPerWarp = numExperts + 2 * topK;
        float* const sSigmoid = routeSmem + warp * wordsPerWarp;
        float* const sWeight = sSigmoid + numExperts;
        int32_t* const sIdx = reinterpret_cast<int32_t*>(sWeight + topK);
        sigmoidTopkWarp<kMaxExpertsPerLane>(logits + static_cast<int64_t>(token) * numExperts, correctionBias,
            numExperts, topK, normTopkProb, routedScalingFactor, sSigmoid, sIdx, sWeight);
        if (lane < topK)
        {
            topkIndices[static_cast<int64_t>(token) * topK + lane] = sIdx[lane];
            topkWeights[static_cast<int64_t>(token) * topK + lane] = sWeight[lane];
        }
    }
    pdlTrigger();
}

//! Single-CTA expert-contiguous, tile-padded layout (same contract as buildLayoutGpu
//! minus tile_mn_limit, which the grouped GEMM does not read): permuted_idx[r] =
//! expanded slot (token * topK + k) or -1 for a pad row, tile_group_idx[t] = expert of
//! token tile t, num_valid_tiles[0] = tile count.  Rows of one expert are contiguous
//! and start at a tile boundary; experts with no rows get no tiles.  1024 threads,
//! dynamic smem = 4 * numExperts words.
constexpr int32_t kLayoutThreads{1024};

__global__ __launch_bounds__(kLayoutThreads) void buildTileLayoutKernel(int32_t const* __restrict__ topkIndices,
    int32_t const numSlots, int32_t const numExperts, int32_t const tokenTile, int32_t* __restrict__ permutedIdx,
    int32_t* __restrict__ tileGroupIdx, int32_t* __restrict__ numValidTiles)
{
    extern __shared__ int32_t layoutSmem[];
    int32_t* const count = layoutSmem;
    int32_t* const rowOffset = layoutSmem + numExperts;
    int32_t* const tileOffset = layoutSmem + 2 * numExperts;
    int32_t* const cursor = layoutSmem + 3 * numExperts;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    for (int32_t e = tid; e < numExperts; e += kLayoutThreads)
    {
        count[e] = 0;
        cursor[e] = 0;
    }
    __syncthreads();
    // topkIndices is written by the routing kernel.
    pdlWait();
    for (int32_t i = tid; i < numSlots; i += kLayoutThreads)
    {
        int32_t const expert = topkIndices[i];
        if (expert >= 0 && expert < numExperts)
        {
            atomicAdd_block(&count[expert], 1);
        }
    }
    __syncthreads();
    // Warp 0: exclusive scan of padded rows and tiles over experts (<= 32 experts per lane).
    if (tid < 32)
    {
        int32_t const perLane = (numExperts + 31) / 32;
        int32_t rows = 0;
        int32_t tiles = 0;
        for (int32_t j = 0; j < perLane; ++j)
        {
            int32_t const e = tid * perLane + j;
            if (e < numExperts)
            {
                int32_t const t = (count[e] + tokenTile - 1) / tokenTile;
                rows += t * tokenTile;
                tiles += t;
            }
        }
        int32_t rowsIncl = rows;
        int32_t tilesIncl = tiles;
#pragma unroll
        for (int32_t offset = 1; offset < 32; offset <<= 1)
        {
            int32_t const r = __shfl_up_sync(0xFFFFFFFFU, rowsIncl, offset);
            int32_t const t = __shfl_up_sync(0xFFFFFFFFU, tilesIncl, offset);
            if (tid >= offset)
            {
                rowsIncl += r;
                tilesIncl += t;
            }
        }
        int32_t rowBase = rowsIncl - rows;
        int32_t tileBase = tilesIncl - tiles;
        for (int32_t j = 0; j < perLane; ++j)
        {
            int32_t const e = tid * perLane + j;
            if (e < numExperts)
            {
                rowOffset[e] = rowBase;
                tileOffset[e] = tileBase;
                int32_t const t = (count[e] + tokenTile - 1) / tokenTile;
                rowBase += t * tokenTile;
                tileBase += t;
            }
        }
        if (tid == 31)
        {
            numValidTiles[0] = tilesIncl;
        }
    }
    __syncthreads();
    // Tile owners and pad rows, one expert per thread.
    for (int32_t e = tid; e < numExperts; e += kLayoutThreads)
    {
        int32_t const c = count[e];
        int32_t const t = (c + tokenTile - 1) / tokenTile;
        for (int32_t i = 0; i < t; ++i)
        {
            tileGroupIdx[tileOffset[e] + i] = e;
        }
        for (int32_t r = c; r < t * tokenTile; ++r)
        {
            permutedIdx[rowOffset[e] + r] = -1;
        }
    }
    // Scatter slots into their expert's row range.
    for (int32_t i = tid; i < numSlots; i += kLayoutThreads)
    {
        int32_t const expert = topkIndices[i];
        if (expert >= 0 && expert < numExperts)
        {
            int32_t const pos = atomicAdd_block(&cursor[expert], 1);
            permutedIdx[rowOffset[expert] + pos] = i;
        }
    }
    pdlTrigger();
}

} // namespace

cudaError_t launchGatherPermutedRows(DecodeDtype const dtype, void const* const hiddenStates,
    int32_t const* const permutedIdx, int32_t const* const numValidTiles, int32_t const tokenTile, int32_t const topK,
    int32_t const hiddenSize, int32_t const numTokens, int64_t const maxRowsPadded, void* const permutedActivations,
    void* const output, bool const enablePdl, cudaStream_t const stream)
{
    if (hiddenSize % 8 != 0 || topK <= 0 || tokenTile <= 0 || numTokens <= 0 || maxRowsPadded <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const blocks = maxRowsPadded + numTokens;
    if (blocks > 0x7FFFFFFFLL)
    {
        return cudaErrorInvalidValue;
    }
    dim3 const grid(static_cast<unsigned int>(blocks));
    if (dtype == DecodeDtype::kFP16)
    {
        return launchKernelPdl(gatherPermutedRowsKernel<half>, grid, dim3(kThreads), 0, stream, enablePdl,
            static_cast<half const*>(hiddenStates), permutedIdx, numValidTiles, tokenTile, topK, hiddenSize, numTokens,
            maxRowsPadded, static_cast<half*>(permutedActivations), static_cast<half*>(output));
    }
    return launchKernelPdl(gatherPermutedRowsKernel<__nv_bfloat16>, grid, dim3(kThreads), 0, stream, enablePdl,
        static_cast<__nv_bfloat16 const*>(hiddenStates), permutedIdx, numValidTiles, tokenTile, topK, hiddenSize,
        numTokens, maxRowsPadded, static_cast<__nv_bfloat16*>(permutedActivations),
        static_cast<__nv_bfloat16*>(output));
}

cudaError_t launchSigmoidTopkRoute(float const* logits, float const* correctionBias, int32_t numTokens,
    int32_t numExperts, int32_t topK, bool normTopkProb, float routedScalingFactor, int32_t* topkIndices,
    float* topkWeights, bool enablePdl, cudaStream_t stream)
{
    if (numExperts <= 0 || numExperts > 32 * kRouteMaxExpertsPerLane || topK <= 0 || topK > 32 || numTokens <= 0)
    {
        return cudaErrorInvalidValue;
    }
    constexpr int32_t kWarps{kThreads / 32};
    dim3 const grid(static_cast<unsigned int>((numTokens + kWarps - 1) / kWarps));
    size_t const smem = static_cast<size_t>(kWarps) * (numExperts + 2 * topK) * sizeof(float);
    return launchKernelPdl(sigmoidTopkRouteKernel<kRouteMaxExpertsPerLane>, grid, dim3(kThreads), smem, stream,
        enablePdl, logits, correctionBias, numTokens, numExperts, topK, normTopkProb, routedScalingFactor, topkIndices,
        topkWeights);
}

cudaError_t launchBuildTileLayout(int32_t const* topkIndices, int32_t numSlots, int32_t numExperts, int32_t tokenTile,
    int32_t* permutedIdx, int32_t* tileGroupIdx, int32_t* numValidTiles, bool enablePdl, cudaStream_t stream)
{
    if (numExperts <= 0 || numExperts > 32 * 32 || tokenTile <= 0 || numSlots < 0)
    {
        return cudaErrorInvalidValue;
    }
    size_t const smem = static_cast<size_t>(4) * numExperts * sizeof(int32_t);
    return launchKernelPdl(buildTileLayoutKernel, dim3(1), dim3(kLayoutThreads), smem, stream, enablePdl, topkIndices,
        numSlots, numExperts, tokenTile, permutedIdx, tileGroupIdx, numValidTiles);
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
