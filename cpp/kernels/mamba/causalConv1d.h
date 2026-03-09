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
 * This file contains code derived from causal-conv1d
 * (https://github.com/Dao-AILab/causal-conv1d)
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Licensed under the BSD 3-Clause License.
 *
 * Modifications by NVIDIA:
 * - Adapted causal depthwise conv1d kernel interface for TensorRT Edge-LLM integration
 * - Added stride, dilation, and padding parameters for generalized conv1d
 * - Added decode-mode, state capture, and shift-insert kernel interfaces
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

/*!
 * \brief Input/output tensors for the prefill causal depthwise conv1d.
 *
 * x:      [batch, seq_len, dim]
 * weight: [dim, 1, width]
 * bias:   [dim]
 * out:    [batch, out_seq_len, dim]
 */
struct CausalConv1dTensors
{
    trt_edgellm::rt::Tensor const* x;
    trt_edgellm::rt::Tensor const* weight;
    trt_edgellm::rt::Tensor const* bias;
    trt_edgellm::rt::Tensor* out;
};

template <typename T>
void invokeCausalConv1d(
    CausalConv1dTensors const& tensors, int32_t stride, int32_t padding, int32_t dilation, cudaStream_t stream);

//! Decode-mode conv1d: dot product of conv_state and weight per channel.
//! conv_state: [batch, dim, width]
//! weight:     [dim, 1, width]
//! bias:       [dim]
//! output:     [batch, 1, dim]
template <typename T>
void invokeCausalConv1dDecode(void const* convState, void const* weight, void const* bias, void* output, int32_t batch,
    int32_t dim, int32_t width, cudaStream_t stream);

//! Capture the last `width` columns of x into conv_state (transposed).
//! x:          [batch, seqLen, dim]
//! convState:  [batch, dim, width]  (output, zero-initialized before call)
template <typename T>
void invokeCaptureConvState(
    void const* x, void* convState, int32_t batch, int32_t seqLen, int32_t dim, int32_t width, cudaStream_t stream);

//! Shift conv_state left by 1 and insert new values at position width-1.
//! convState:  [batch, dim, width]  (in-place)
//! newCol:     [batch, 1, dim]  (the new single-token input)
template <typename T>
void invokeConvStateShiftInsert(
    void* convState, void const* newCol, int32_t batch, int32_t dim, int32_t width, cudaStream_t stream);

} // namespace mamba_ssm
