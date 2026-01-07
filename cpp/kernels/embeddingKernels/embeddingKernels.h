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

//! \brief Standard embedding lookup kernel
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Embedding table with shape [vocabSize, hiddenSize]
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
void embeddingLookup(
    rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::Tensor& output, cudaStream_t stream);

//! \brief Embedding lookup with image embedding insertion following PromptTuningEmbedding logic
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Embedding table with shape [vocabSize, hiddenSize]
//! \param[in] imageEmbeds Image embeddings with shape [imageTokenLen, hiddenSize]
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
void embeddingLookupWithImageInsertion(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::Tensor const& imageEmbeds, rt::Tensor& output, cudaStream_t stream);

//! \brief Assemble deepstack embeddings by extracting image token embeddings from deepstack features
//!
//! This function processes input token IDs and selectively extracts embeddings for image tokens from
//! the provided deepstack features. Image tokens are identified by token IDs >= vocabSize. Regular
//! text tokens (IDs < vocabSize) are assigned zero embeddings. This is typically used in multi-modal
//! models where deepstack visual features need to be combined with text embeddings.
//!
//! Token ID Mapping:
//! - Token ID < vocabSize: Zero embedding (text tokens are handled separately)
//! - Token ID >= vocabSize: Extract from deepstackFeatures at index (tokenId - vocabSize)
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] deepstackFeatures Deepstack image features with shape [numImageTokens, hiddenSize]
//! \param[in] vocabSize Vocabulary size threshold for distinguishing text vs image tokens
//! \param[out] deepstackEmbeds Output embeddings with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures, int32_t vocabSize,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm