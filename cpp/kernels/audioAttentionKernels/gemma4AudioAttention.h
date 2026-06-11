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

#pragma once

#include "common/tensor.h"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Attention-window configuration for the Gemma 4 audio-encoder chunked local attention.
//!
//! Holds only the parameters that are *not* derivable from the tensor shapes (the batch B,
//! sequence S, head count H, head dim D, and relative-position length P are read from the
//! input tensors' shapes). Defaults for the Gemma 4 audio tower: chunkSize=12,
//! leftHorizon=12 (= attention_context_left - 1), contextSize=24 (= chunk + left + right),
//! logitCap=50. The kernel is specialized to this exact config (see gemma4AudioAttentionForward).
struct Gemma4AudioAttentionParams
{
    int chunkSize;   //!< C (query block size)
    int leftHorizon; //!< L (effective left context, attention_context_left - 1)
    int contextSize; //!< M (gathered K/V context size, chunk + left + right)
    float logitCap;  //!< tanh soft-cap on logits
};

//! \brief Fused forward for the Gemma 4 audio-encoder attention body.
//!
//! Computes the attention region *after* Q/K/V projection and *before* the output
//! projection: per-dim learned Q scaling (softplus(gamma)) + fixed K scaling, overlapping
//! K/V context gather, content scores, query-dependent relative-position scores with the
//! HF relative shift, tanh soft-cap, local-causal + padding mask, fp32 softmax over the
//! context, value mix, and crop back to seqLen. One CUDA block handles one (b, h, n) chunk.
//!
//! Shapes (B, S, H, D, P) are read from the tensor extents and the element type is read from
//! qRaw/kRaw/v/relKey/out (which must all share one floating type: half, bfloat16, or float).
//! All tensors are row-major and contiguous, on the GPU. Internal accumulation (scores,
//! softmax) is fp32; the float path keeps full precision throughout and matches the HF fp32
//! reference. Specialized to the Gemma 4 audio config (chunkSize=12, leftHorizon=12,
//! contextSize=24, relPosLen=13, headDim=128); the function validates the supplied shapes and
//! params against that specialization.
//!
//! \param[in] qRaw   Query projections, [B, S, H, D] (half/bf16/float).
//! \param[in] kRaw   Key projections, [B, S, H, D] (same type as qRaw).
//! \param[in] v      Value projections, [B, S, H, D] (same type as qRaw).
//! \param[in] gamma  Per-dim learned query scale (per_dim_scale), [D] (float).
//! \param[in] relKey Projected relative-position embeddings, [P, H, D] (same type as qRaw).
//! \param[in] valid  Audio validity mask (true = real token), [B, S] (bool).
//! \param[out] out   Attention output, [B, S, H, D] (same type as qRaw).
//! \param[in] params Attention-window config and soft-cap (shapes come from the tensors).
//! \param[in] stream CUDA stream for execution.
void gemma4AudioAttentionForward(rt::Tensor const& qRaw, rt::Tensor const& kRaw, rt::Tensor const& v,
    rt::Tensor const& gamma, rt::Tensor const& relKey, rt::Tensor const& valid, rt::Tensor& out,
    Gemma4AudioAttentionParams const& params, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
