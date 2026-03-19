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
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/**
 * @brief Apply SwiGLU activation: silu(gate) * up
 *
 * This kernel applies the SwiGLU (Swish-Gated Linear Unit) activation function,
 * commonly used in MoE models like Qwen, Llama, and Mistral.
 *
 * Given an input of shape [N, 2*D], it:
 * 1. Splits the input into gate [N, D] and up [N, D]
 * 2. Applies SiLU (Swish) activation to gate: silu(x) = x * sigmoid(x)
 * 3. Multiplies element-wise: output = silu(gate) * up
 *
 * Optimizations:
 * - Vectorized 128-bit loads/stores (8 FP16 elements) for better memory bandwidth
 * - Fused split, activation, and multiplication in a single pass
 * - No intermediate storage required
 *
 * @param gateUpInput Input tensor [numTokens, 2*intermediateDim] (FP16, GPU)
 * @param output Output tensor [numTokens, intermediateDim] (FP16, GPU)
 * @param numTokens Number of tokens
 * @param intermediateDim Intermediate dimension (output will be this size)
 * @param stream CUDA stream
 *
 * @throws std::runtime_error If any of the following preconditions are violated:
 *   - gateUpInput is not 2D with shape [numTokens, 2*intermediateDim]
 *   - output is not 2D with shape [numTokens, intermediateDim]
 *   - Either tensor is not FP16 or not on GPU
 *   - intermediateDim is not a multiple of 8 (required for 128-bit vectorized access)
 *   - Data pointers are not 16-byte aligned
 */
void swiGluActivation(
    rt::Tensor const& gateUpInput, rt::Tensor& output, int64_t numTokens, int64_t intermediateDim, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
