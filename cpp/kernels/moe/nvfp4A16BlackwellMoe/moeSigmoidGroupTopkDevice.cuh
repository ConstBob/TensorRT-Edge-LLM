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

#pragma once

#include <cfloat>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Block-cooperative sigmoid + grouped top-k routing for ONE token, matching
//! moeSigmoidGroupTopkKernels.cu (NemotronH routing) bit for bit in its
//! selection order: sigmoid -> +bias -> top-2-sum per group -> topkGroup
//! groups -> mask -> iterative arg-max (ties resolve to the lower expert id)
//! -> weights gathered from the UNbiased sigmoid -> optional renorm -> scale.
//!
//! Shared memory provided by the caller (floats unless noted):
//!   sSigmoid[numExperts], sBiased[numExperts], sGroupScores[nGroup],
//!   sGroupSelected[nGroup] (int32), sReduceVal[kThreads/32],
//!   sReduceIdx[kThreads/32] (int32), sTopkIdx[topK] (int32), sTopkWeight[topK].
//! All threads of the block must call it; results are visible to every thread
//! after the final __syncthreads() inside.
template <int32_t kThreads>
__device__ __forceinline__ void sigmoidGroupTopkBlock(float const* __restrict__ logits,
    float const* __restrict__ correctionBias, int32_t const numExperts, int32_t const topK, int32_t const nGroup,
    int32_t const topkGroup, bool const normTopkProb, float const routedScalingFactor, float* sSigmoid, float* sBiased,
    float* sGroupScores, int32_t* sGroupSelected, float* sReduceVal, int32_t* sReduceIdx, int32_t* sTopkIdx,
    float* sTopkWeight)
{
    static_assert(kThreads % 32 == 0, "kThreads must be a whole number of warps");
    constexpr int32_t kWarps = kThreads / 32;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const lane = tid & 31;
    int32_t const warp = tid >> 5;
    int32_t const expertsPerGroup = numExperts / nGroup;

    // Step 1: sigmoid + bias.
    for (int32_t e = tid; e < numExperts; e += kThreads)
    {
        float const sig = 1.0f / (1.0f + expf(-logits[e]));
        sSigmoid[e] = sig;
        sBiased[e] = correctionBias != nullptr ? sig + correctionBias[e] : sig;
    }
    __syncthreads();

    // Step 2: top-2 sum per group.
    for (int32_t g = tid; g < nGroup; g += kThreads)
    {
        float top1 = -FLT_MAX;
        float top2 = -FLT_MAX;
        int32_t const start = g * expertsPerGroup;
        for (int32_t i = 0; i < expertsPerGroup; ++i)
        {
            float const v = sBiased[start + i];
            if (v > top1)
            {
                top2 = top1;
                top1 = v;
            }
            else if (v > top2)
            {
                top2 = v;
            }
        }
        sGroupScores[g] = top1 + top2;
        sGroupSelected[g] = 0;
    }
    __syncthreads();

    // Step 3: thread 0 picks topkGroup groups (nGroup is small).
    if (tid == 0)
    {
        for (int32_t i = 0; i < topkGroup; ++i)
        {
            int32_t best = -1;
            float bestScore = -FLT_MAX;
            for (int32_t g = 0; g < nGroup; ++g)
            {
                if (sGroupSelected[g] == 0 && sGroupScores[g] > bestScore)
                {
                    bestScore = sGroupScores[g];
                    best = g;
                }
            }
            if (best >= 0)
            {
                sGroupSelected[best] = 1;
            }
        }
    }
    __syncthreads();

    // Step 4: mask unselected groups.
    for (int32_t e = tid; e < numExperts; e += kThreads)
    {
        if (sGroupSelected[e / expertsPerGroup] == 0)
        {
            sBiased[e] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Step 5: iterative arg-max over the masked biased scores.
    for (int32_t k = 0; k < topK; ++k)
    {
        float bestVal = -FLT_MAX;
        int32_t bestIdx = 0x7FFFFFFF;
        for (int32_t e = tid; e < numExperts; e += kThreads)
        {
            float const v = sBiased[e];
            if (v > bestVal || (v == bestVal && e < bestIdx))
            {
                bestVal = v;
                bestIdx = e;
            }
        }
#pragma unroll
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            float const otherVal = __shfl_xor_sync(0xFFFFFFFFU, bestVal, offset);
            int32_t const otherIdx = __shfl_xor_sync(0xFFFFFFFFU, bestIdx, offset);
            if (otherVal > bestVal || (otherVal == bestVal && otherIdx < bestIdx))
            {
                bestVal = otherVal;
                bestIdx = otherIdx;
            }
        }
        if (lane == 0)
        {
            sReduceVal[warp] = bestVal;
            sReduceIdx[warp] = bestIdx;
        }
        __syncthreads();
        if (tid == 0)
        {
            float winVal = sReduceVal[0];
            int32_t winIdx = sReduceIdx[0];
            for (int32_t w = 1; w < kWarps; ++w)
            {
                float const v = sReduceVal[w];
                int32_t const i = sReduceIdx[w];
                if (v > winVal || (v == winVal && i < winIdx))
                {
                    winVal = v;
                    winIdx = i;
                }
            }
            sTopkIdx[k] = winIdx;
            sTopkWeight[k] = sSigmoid[winIdx];
            sBiased[winIdx] = -FLT_MAX;
        }
        __syncthreads();
    }

    // Step 6: renormalize + scale (thread 0, then publish).
    if (tid == 0)
    {
        float sum = 0.0f;
        for (int32_t k = 0; k < topK; ++k)
        {
            sum += sTopkWeight[k];
        }
        float const inv = (normTopkProb && sum > 0.0f) ? 1.0f / sum : 1.0f;
        for (int32_t k = 0; k < topK; ++k)
        {
            sTopkWeight[k] = sTopkWeight[k] * inv * routedScalingFactor;
        }
    }
    __syncthreads();
}

//! Single-warp variant of sigmoidGroupTopkBlock for the ungrouped case
//! (nGroup == 1, so every expert is a candidate): identical selection order and
//! weights, but the whole routing runs in warp 0's registers with shuffle
//! arg-max rounds, no block barriers.  numExperts <= 32 * kMaxExpertsPerLane.
//! Shared memory: sSigmoid[numExperts], sTopkIdx[topK] (int32), sTopkWeight[topK].
//! Only the calling warp writes them; the caller synchronizes the block.
template <int32_t kMaxExpertsPerLane = 16>
__device__ __forceinline__ void sigmoidTopkWarp(float const* __restrict__ logits,
    float const* __restrict__ correctionBias, int32_t const numExperts, int32_t const topK, bool const normTopkProb,
    float const routedScalingFactor, float* sSigmoid, int32_t* sTopkIdx, float* sTopkWeight)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    float biased[kMaxExpertsPerLane];
#pragma unroll
    for (int32_t i = 0; i < kMaxExpertsPerLane; ++i)
    {
        int32_t const e = lane + 32 * i;
        biased[i] = -FLT_MAX;
        if (e < numExperts)
        {
            float const sig = 1.0f / (1.0f + expf(-logits[e]));
            sSigmoid[e] = sig;
            biased[i] = correctionBias != nullptr ? sig + correctionBias[e] : sig;
        }
    }
    __syncwarp();
    float sum = 0.0f;
    for (int32_t k = 0; k < topK; ++k)
    {
        float bestVal = -FLT_MAX;
        int32_t bestIdx = 0x7FFFFFFF;
#pragma unroll
        for (int32_t i = 0; i < kMaxExpertsPerLane; ++i)
        {
            // Lane-local candidates are in increasing expert order, so the first
            // strictly greater value wins and ties keep the lower expert id.
            if (biased[i] > bestVal)
            {
                bestVal = biased[i];
                bestIdx = lane + 32 * i;
            }
        }
#pragma unroll
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            float const otherVal = __shfl_xor_sync(0xFFFFFFFFU, bestVal, offset);
            int32_t const otherIdx = __shfl_xor_sync(0xFFFFFFFFU, bestIdx, offset);
            if (otherVal > bestVal || (otherVal == bestVal && otherIdx < bestIdx))
            {
                bestVal = otherVal;
                bestIdx = otherIdx;
            }
        }
        // bestIdx is warp-uniform here; the owning lane retires it.
        if ((bestIdx & 31) == lane)
        {
#pragma unroll
            for (int32_t i = 0; i < kMaxExpertsPerLane; ++i)
            {
                if (i == (bestIdx >> 5))
                {
                    biased[i] = -FLT_MAX;
                }
            }
        }
        float const weight = sSigmoid[bestIdx];
        sum += weight;
        if (lane == 0)
        {
            sTopkIdx[k] = bestIdx;
            sTopkWeight[k] = weight;
        }
    }
    __syncwarp();
    if (lane < topK)
    {
        float const inv = (normTopkProb && sum > 0.0f) ? 1.0f / sum : 1.0f;
        sTopkWeight[lane] = sTopkWeight[lane] * inv * routedScalingFactor;
    }
    __syncwarp();
}

//! Bytes of dynamic shared memory sigmoidGroupTopkBlock needs (16-byte rounded).
__host__ __device__ __forceinline__ int32_t sigmoidGroupTopkSharedBytes(
    int32_t const numExperts, int32_t const nGroup, int32_t const topK, int32_t const kThreads)
{
    int32_t const floats = 2 * numExperts + nGroup + topK + (kThreads / 32);
    int32_t const ints = nGroup + topK + (kThreads / 32);
    int32_t const bytes = (floats + ints) * static_cast<int32_t>(sizeof(float));
    return (bytes + 15) / 16 * 16;
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
