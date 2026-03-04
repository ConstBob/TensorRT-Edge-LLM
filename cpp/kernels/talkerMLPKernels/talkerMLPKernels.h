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

#pragma once

#include "common/tensor.h"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Two-layer MLP with SiLU activation (Talker projection layers)
//!
//! Performs: output = FC2(SiLU(FC1(input) + bias1)) + bias2
//! Where FC1: [inputDim → hiddenDim], FC2: [hiddenDim → outputDim]
//!
//! Architecture:
//!   input [N, 2048]
//!     ↓ FC1 (Linear)
//!   [N, 2048] + bias1
//!     ↓ SiLU
//!   [N, 2048]
//!     ↓ FC2 (Linear)
//!   [N, 1024] + bias2
//!     ↓
//!   output [N, 1024]
//!
//! \param[in] cublasHandle cuBLAS handle for GEMM operations
//! \param[in] input Input tensor with shape [numTokens, 2048] (FP16)
//! \param[in] fc1Weight FC1 weight matrix with shape [2048, 2048] (FP16, column-major)
//! \param[in] fc1Bias FC1 bias vector with shape [2048] (FP16)
//! \param[in] fc2Weight FC2 weight matrix with shape [2048, 1024] (FP16, column-major)
//! \param[in] fc2Bias FC2 bias vector with shape [1024] (FP16)
//! \param[out] output Output tensor with shape [numTokens, 1024] (FP16)
//! \param[in,out] workspace Workspace buffer for intermediate FC1 output [numTokens, 2048] (FP16)
//! \param[in] stream CUDA stream for execution
//!
//! \note Weight matrices are stored in column-major format (cuBLAS convention)
//! \note Workspace must be pre-allocated with size [numTokens, 2048] * sizeof(half)
void invokeTalkerMLP(void* cublasHandle, rt::Tensor const& input, rt::Tensor const& fc1Weight,
    rt::Tensor const& fc1Bias, rt::Tensor const& fc2Weight, rt::Tensor const& fc2Bias, rt::Tensor& output,
    rt::Tensor& workspace, cudaStream_t stream);

//! \brief Gather operation: select rows from source tensor by indices
//!
//! Performs: output[i] = source[indices[i]]
//! where each row has hiddenDim elements.
//!
//! \param[in] source Source tensor with shape [srcNumTokens, hiddenDim] (FP16)
//! \param[in] indices Indices tensor with shape [numIndices] (INT32)
//! \param[out] output Output tensor with shape [numIndices, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeGather(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream);

//! \brief Scatter operation: place rows from source to output by indices
//!
//! Performs: output[indices[i]] = source[i]
//! where each row has hiddenDim elements.
//!
//! \param[in] source Source tensor with shape [numIndices, hiddenDim] (FP16)
//! \param[in] indices Indices tensor with shape [numIndices] (INT32)
//! \param[out] output Output tensor with shape [dstNumTokens, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeScatter(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream);

//! \brief Sum reduction over sequence dimension for residual connection
//!
//! Performs: output[0, 0, :] = sum(input[0, :, :], dim=1)
//! where input has shape [1, seqLen, hiddenDim] and output has shape [1, 1, hiddenDim]
//!
//! \param[in] input Input tensor with shape [1, seqLen, hiddenDim] (FP16)
//! \param[out] output Output tensor with shape [1, 1, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void sumReduceOverSequence(rt::Tensor const& input, rt::Tensor& output, cudaStream_t stream);

//! \brief Element-wise addition: output = a + b
//!
//! \param[out] output Output tensor (FP16)
//! \param[in] a First input tensor (FP16)
//! \param[in] b Second input tensor (FP16)
//! \param[in] stream CUDA stream for execution
void invokeElementwiseAdd(rt::Tensor& output, rt::Tensor const& a, rt::Tensor const& b, cudaStream_t stream);

//! \brief In-place element-wise addition: data += addend
//!
//! \param[in,out] data Input/output tensor (FP16)
//! \param[in] addend Tensor to add (FP16)
//! \param[in] numElements Number of elements to add (0 = use full tensor volume)
//! \param[in] dataOffset Element offset into data (default 0)
//! \param[in] addendOffset Element offset into addend (default 0)
//! \param[in] stream CUDA stream for execution
void invokeElementwiseAddInplace(rt::Tensor& data, rt::Tensor const& addend, int64_t numElements, int64_t dataOffset,
    int64_t addendOffset, cudaStream_t stream);

//! \brief Suppress logits for a range of token IDs by setting them to -infinity
//!
//! For all positions in [suppressStart, suppressEnd), sets logits[i] = -inf
//! UNLESS i == exceptTokenId (typically the EOS token that should remain sampable).
//! Operates on FP32 logits tensor with shape [1, vocabSize].
//!
//! \param[in,out] logits Logits tensor [1, vocabSize] (FP32, in-place)
//! \param[in] suppressStart Start of suppress range (inclusive)
//! \param[in] suppressEnd End of suppress range (exclusive)
//! \param[in] exceptTokenId Token ID to exempt from suppression (-1 to suppress all)
//! \param[in] stream CUDA stream for execution
void invokeSuppressLogits(
    rt::Tensor& logits, int32_t suppressStart, int32_t suppressEnd, int32_t exceptTokenId, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
