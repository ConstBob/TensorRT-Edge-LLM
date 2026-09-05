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

#include "nvfp4A16BlackwellMoeDecodeKernels.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Sigmoid top-k routing for the ungrouped contract (nGroup == 1): one warp per
//! token, same selection order and weights as moeSigmoidGroupTopk (ties to the
//! lower expert id, weights from the unbiased sigmoid, optional renorm, scale).
//! Writes topkIndices[T, topK] and topkWeights[T, topK]. numExperts <= 512.
cudaError_t launchSigmoidTopkRoute(float const* logits, float const* correctionBias, int32_t numTokens,
    int32_t numExperts, int32_t topK, bool normTopkProb, float routedScalingFactor, int32_t* topkIndices,
    float* topkWeights, cudaStream_t stream);

//! Expert-contiguous, tile-padded layout of the numSlots = T * topK routed rows
//! (single CTA): permutedIdx[r] = token * topK + k or -1 for a pad row (rows of
//! one expert are contiguous from a tile boundary; experts without rows get no
//! tiles), tileGroupIdx[t] = expert owning token tile t, numValidTiles[0] = tile
//! count. Row order inside an expert follows the atomic scatter and is not
//! deterministic; every consumer is order independent. Buffers must hold
//! numSlots + numExperts * (tokenTile - 1) rows and the matching tile count.
cudaError_t launchBuildTileLayout(int32_t const* topkIndices, int32_t numSlots, int32_t numExperts, int32_t tokenTile,
    int32_t* permutedIdx, int32_t* tileGroupIdx, int32_t* numValidTiles, cudaStream_t stream);

//! Build the permuted, expert-contiguous, tile-padded activation buffer the
//! grouped GEMM reads, and zero the token output the FC2 scatter-add
//! accumulates into.
//!
//! grid = maxRowsPadded + numTokens blocks. Block r < maxRowsPadded copies
//! hidden[permutedIdx[r] / topK] into permuted[r] for routed rows, but only for
//! r < numValidTiles[0] * tokenTile (device tile count), so the launch is
//! CUDA-graph stable while touching only the rows the GEMM will read. Pad rows
//! (permutedIdx[r] < 0) are left untouched: FC1 turns them into rows the FC2
//! epilogue discards through the same permutedIdx. The remaining numTokens
//! blocks zero output rows.
cudaError_t launchGatherPermutedRows(DecodeDtype dtype, void const* hiddenStates, int32_t const* permutedIdx,
    int32_t const* numValidTiles, int32_t tokenTile, int32_t topK, int32_t hiddenSize, int32_t numTokens,
    int64_t maxRowsPadded, void* permutedActivations, void* output, cudaStream_t stream);

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
