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

#include "common/tensor.h"
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/**
 * @brief DFlash sequential accept for speculative decoding.
 *
 * Much simpler than Eagle tree accept: for each batch element, walks forward
 * comparing base argmax tokens to draft tokens sequentially.  Accepts until
 * first mismatch + bonus token at the mismatch position.
 *
 * @param baseLogits      Base model logits [batch, verifyLen, vocabSize] (FP32, GPU)
 * @param draftTokenIds   Draft token IDs [batch, verifyLen] (INT32, GPU)
 * @param acceptedTokenIds  Output accepted token IDs [batch, maxAcceptLen] (INT32, GPU)
 * @param acceptLength    Output accept length per batch [batch] (INT32, GPU)
 * @param batchSize       Number of batch elements
 * @param verifyLen       Number of tokens to verify per batch
 * @param vocabSize       Vocabulary size
 * @param stream          CUDA stream
 */
void dflashSequentialAccept(rt::Tensor const& baseLogits, rt::Tensor const& draftTokenIds, rt::Tensor& acceptedTokenIds,
    rt::Tensor& acceptLength, rt::Tensor& argmaxScratch, int32_t batchSize, int32_t verifyLen, int32_t vocabSize,
    cudaStream_t stream);

/**
 * @brief Build verify token IDs on GPU.
 *
 * verify[b][0] = lastAcceptedTokens[b], verify[b][j] = draftTokenIds[b][j] for j >= 1.
 * Replaces the host-side roundtrip used in the debug prototype.
 *
 * @param lastAcceptedTokens  Last accepted token per batch [batch] (INT32, GPU)
 * @param draftTokenIds       Draft argmax token IDs [batch, blockSize] (INT32, GPU)
 * @param verifyTokenIds      Output verify token IDs [batch, blockSize] (INT32, GPU)
 * @param batchSize           Number of batch elements
 * @param blockSize           DFlash block size (number of draft positions)
 * @param stream              CUDA stream
 */
void dflashBuildVerifyTokens(rt::Tensor const& lastAcceptedTokens, rt::Tensor const& draftTokenIds,
    rt::Tensor& verifyTokenIds, int32_t batchSize, int32_t blockSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
