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

// Adapted from SGLang: https://github.com/sgl-project/sglang
// Originally from vLLM and TensorRT-LLM MoE kernels

#pragma once

#include "common/tensor.h"
namespace trt_edgellm
{
namespace kernel
{

/**
 * @brief Calculate workspace size required for MoE TopK Softmax kernel
 *
 * The workspace is only needed when numExperts is not a power of 2 (1-256),
 * in which case a fallback path using separate softmax and topk kernels is used.
 *
 * @param numTokens Number of tokens to process
 * @param numExperts Number of experts in the MoE layer
 * @return Required workspace size in bytes (0 if optimized path is used)
 */
size_t getMoeTopkSoftmaxWorkspaceSize(int32_t numTokens, int32_t numExperts);

/**
 * @brief MoE TopK Softmax kernel for Mixture of Experts gating
 *
 * This kernel implements the gating mechanism for MoE layers:
 * 1. Takes gating logits of shape [numTokens, numExperts]
 * 2. Applies optional tanh softcapping to limit logit range
 * 3. Applies optional correction bias for expert load balancing
 * 4. Computes softmax over the expert dimension
 * 5. Selects top-k experts with highest probabilities
 * 6. Optionally renormalizes the selected weights to sum to 1
 *
 * Algorithm:
 * - For power-of-2 experts (1-256): Uses optimized fused kernel with warp-level parallelism
 * - For other expert counts: Falls back to separate softmax + topk kernels
 *
 * Optimizations:
 * - Fused softmax and top-k selection in a single kernel pass
 * - Warp-level butterfly reductions (no shared memory needed)
 * - Vectorized memory loads for better bandwidth utilization
 * - Multiple rows processed per warp for high occupancy
 *
 * @param gatingOutput Input gating logits [numTokens, numExperts] (FP32/FP16/BF16, GPU)
 * @param topkWeights Output selected expert weights [numTokens, topk] (FP32, GPU)
 * @param topkIndices Output selected expert indices [numTokens, topk] (INT32, GPU)
 * @param topk Number of experts to select per token
 * @param workspace Workspace buffer for fallback path (can be nullptr if not needed)
 * @param workspaceSize Size of workspace buffer in bytes
 * @param stream CUDA stream for execution
 * @param renormalize Whether to renormalize topk weights to sum to 1 (default: false)
 * @param moeSoftcapping Softcapping value (0.0 to disable): val = tanh(val/cap) * cap (default: 0.0)
 * @param correctionBias Optional bias tensor [numExperts] for expert load balancing (FP32, GPU)
 *
 * @note All tensor parameters must be allocated on GPU device
 * @note Workspace is only required when numExperts is not a power of 2 in range [1, 256]
 * @note Use getMoeTopkSoftmaxWorkspaceSize() to determine required workspace size
 */
void moeTopkSoftmax(rt::Tensor const& gatingOutput, rt::Tensor& topkWeights, rt::Tensor& topkIndices, int32_t topk,
    void* workspace, size_t workspaceSize, cudaStream_t stream, bool renormalize = false, float moeSoftcapping = 0.0f,
    rt::OptionalInputTensor correctionBias = std::nullopt);

} // namespace kernel
} // namespace trt_edgellm
