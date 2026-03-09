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

/*
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Extracted selective state update kernel interface for TensorRT Edge-LLM integration
 * - Added explicit stride parameters for non-contiguous/padded memory layouts
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 * - Added setContiguousStrides() helper function
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

// =============================================================================
// Allowed dispatch values for kernel instantiation
// =============================================================================
using AllowedDims = std::integer_sequence<int, 64, 80, 128, 256>;    // 80 for Nemotron-9B
using AllowedDstates = std::integer_sequence<int, 64, 80, 128, 256>; // 80 for padded layout testing

// =============================================================================
// Tensor bundle for SSM state update
// =============================================================================

/*!
 * \brief Input/output tensors for the selective SSM state update.
 *
 * Required tensors use non-null references (via pointer); optional tensors
 * (D, z) are nullable. The dispatcher derives all shapes and strides from
 * the tensor metadata, so callers do not need to populate a params struct.
 *
 * Decode layout  (3D x):  [batch, nheads, dim]
 * Prefill layout (4D x):  [batch, seq_len, nheads, dim]
 */
struct SsmUpdateTensors
{
    trt_edgellm::rt::Tensor const* x;          //!< [batch, (seq_len,) nheads, dim]
    trt_edgellm::rt::Tensor const* A;          //!< [nheads], always FP32
    trt_edgellm::rt::Tensor const* B;          //!< [batch, (seq_len,) ngroups, dstate]
    trt_edgellm::rt::Tensor const* C;          //!< [batch, (seq_len,) ngroups, dstate]
    trt_edgellm::rt::Tensor const* dt;         //!< [batch, (seq_len,) nheads]
    trt_edgellm::rt::Tensor const* dt_bias;    //!< [nheads]
    trt_edgellm::rt::Tensor const* D{nullptr}; //!< [nheads], optional skip connection
    trt_edgellm::rt::Tensor const* z{nullptr}; //!< optional SiLU gate, same shape as x
    trt_edgellm::rt::Tensor* state;            //!< [batch, nheads, dim, dstate], updated in-place
    trt_edgellm::rt::Tensor* output;           //!< [batch, (seq_len,) nheads, dim]
};

// =============================================================================
// Kernel launch function declarations
// =============================================================================

/*!
 * \brief Launch the decode selective state update kernel (seq_len == 1).
 *
 * Computes:
 *   new_state = state * exp(A * dt) + B * dt * x
 *   output    = sum_i(new_state_i * C_i) + D * x
 *   if z is non-null: output *= silu(z)
 *
 * \tparam input_t   Input/output dtype (half or float)
 * \tparam weight_t  Weight dtype for dt, D, dt_bias
 * \tparam matrixA_t Dtype for matrix A exponentiation (use float for precision)
 * \tparam state_t   SSM state cache dtype (use float to avoid drift)
 * \tparam stateIndex_t State index dtype (typically int32_t)
 */
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
void invokeSelectiveStateUpdate(SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

/*!
 * \brief Launch the prefill selective state update kernel (seq_len > 1).
 *
 * Processes all seq_len tokens in a single kernel launch, keeping the SSM
 * state in fp32 registers throughout and writing to global memory only once.
 * x must be 4D: [batch, seq_len, nheads, dim].
 */
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
void invokeSelectiveStateUpdatePrefill(SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

} // namespace mamba_ssm
