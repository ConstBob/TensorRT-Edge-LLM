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

#include <common/tensor.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// @brief Scatter accepted recurrent states (FP32) back to the main state pool
///        after MTP speculative decoding verification.
///
/// After the base model runs MTP verify with intermediate state caching,
/// each batch item's recurrent state must be updated to the last accepted step.
/// The MTP GDN kernel (cache ON, state update ON) writes h0_out to the state
/// after ALL T steps, and saves per-step snapshots to intermediate_states.
/// When only L < T tokens are accepted (partial accept), the main state must
/// be corrected from S_{T-1} to S_{L-1}.
///
/// For each batch item b (acceptedSteps[b] = last accepted step, 0-based):
///   - acceptedSteps[b] < 0             → skip (invalid / not participating)
///   - acceptedSteps[b] == maxSteps - 1  → no-op (all T tokens accepted, state already correct)
///   - acceptedSteps[b] < maxSteps - 1   → partial accept, scatter:
///                                         dst[b, :] = src[b, acceptedSteps[b], :]
///
/// Reference: SGLang `fused_mamba_state_scatter_with_mask` (Triton kernel).
///
/// @param src            Intermediate state buffer [batchSize, maxSteps, hv, k, v] FP32 (GPU, input).
///                       This is one layer's MTP plugin 3rd output.
/// @param acceptedSteps  Per-batch accepted step index [batchSize] INT32 (GPU, input).
///                       -1 = skip, 0..maxSteps-2 = partial accept (need scatter), maxSteps-1 = all accept (no-op).
/// @param dst            Main recurrent state pool [batchSize, hv, k, v] FP32 (GPU, in/out).
///                       This is one layer's slice from LinearKVCache.mDeviceRecurrentStates.
/// @param stream         CUDA stream for asynchronous execution.
///
/// @note dst and src trailing dims (hv*k*v) must be divisible by 8 (DVec<float> vec_size).
/// @note This function processes a single layer. Call once per recurrent layer.
/// @throws std::runtime_error if tensors are not on GPU, or dtypes are incorrect.
void mtpScatterRecurrentStates(
    rt::Tensor const& src, rt::Tensor const& acceptedSteps, rt::Tensor& dst, cudaStream_t stream);

/// @brief Scatter accepted conv states (FP16) back to the main state pool
///        after MTP speculative decoding verification.
///
/// Same semantics as mtpScatterRecurrentStates but for FP16 causal conv1d states.
///
/// @param src            Intermediate conv state buffer [batchSize, maxSteps, dim, width] FP16 (GPU, input).
/// @param acceptedSteps  Per-batch accepted step index [batchSize] INT32 (GPU, input).
/// @param dst            Main conv state pool [batchSize, dim, width] FP16 (GPU, in/out).
/// @param stream         CUDA stream for asynchronous execution.
///
/// @note dst and src trailing dims (dim*width) must be divisible by 8 (DVec<half> vec_size).
/// @note This function processes a single layer. Call once per conv layer.
/// @throws std::runtime_error if tensors are not on GPU, or dtypes are incorrect.
void mtpScatterConvStates(rt::Tensor const& src, rt::Tensor const& acceptedSteps, rt::Tensor& dst, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
