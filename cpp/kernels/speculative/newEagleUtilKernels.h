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
#include <cstdint>

namespace drivellm
{
namespace kernel
{

// Disable clang-format to explicitly format the documentation.
// clang-format off

// The kernel will prepare required inputs to execute the eagle prefill step.
// Inputs:
//     sequenceLength: Sequence length of the prefill input.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
//     selectTokenIndices [GPU, Int32]: Denote the position to gather the hidden states and logits output.
void prepareEaglePrefillInputs(
    rt::Tensor& sequenceContextLengths, rt::Tensor& selectTokenIndices, int32_t const sequenceLength, cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle draft proposal step.
// In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
// MISC inputs to execute the draft proposal step.
// Inputs:
//     draftTreeMask [GPU, Int8]: unpacked draft tree mask denote the relationship between the draft tree nodes.
//         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//     draftTreeLength [GPU, Int32]: Real length of the draft tree.
//     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedDraftTreeMask [GPU, Int32]: Packed tree mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of draft tree nodes among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
void prepareEagleDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
    rt::Tensor const& sequenceStartIndices, rt::Tensor& packedDraftTreeMask, rt::Tensor& tensorPositionIndices,
    rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths, cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle accept decode token step.
// Since we reuse the logic of tree attention, instead of draft tree mask, we will prepare casual mask and corresponding
// position indices of each accepted token.
// Inputs:
//     sequenceStartIndices [GPU, Int32]: The start indices of the first accepted token.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedTreeMask [GPU, Int32]: Packed casual tree mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of accepted tokens among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position (always the last one) to gather the hidden states and logits.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
//     acceptedTokenNum: Number of accepted tokens.
void prepareEagleAcceptDecodeTokenInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor& packedTreeMask,
    rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths,
    int32_t const acceptedTokenNum, cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle draft proposal step.
// In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
// MISC inputs to execute the draft proposal step.
// Inputs:
//     baseTreeDecodingMask [GPU, Int8]: unpacked base tree decoding mask denotes the relationship between the base tree decoding nodes.
//         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedBaseTreeDecodingMask [GPU, Int32]: Packed base tree decoding mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of base tree decoding nodes among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
void prepareEagleBaseTreeDecodingInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
    rt::Tensor& packedBaseTreeDecodingMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream);

// The kernel will commit KVCache and assemble the hidden state inplace according to the accepted indices and accept lengths.
// Inputs:
//     acceptedIndices [GPU, Int32]: The accepted indices with shape [batch, max-depth].
//     acceptLengths [GPU, Int32]: The accept lengths with shape [batch].
//     kvCacheBuffer [GPU, Half]: The KVCache buffer with shape [num-layers, max-batch-size, 2, num-heads, max-seq-len, hidden-size-per-head].
//     kvCacheLengths [GPU, Int32]: The KVCache lengths with shape [batch].
//     hiddenState [GPU, Half]: The hidden state with shape [batch, num-tokens, base-hidden-dim].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     kvCacheBuffer [GPU, Half]: The updated KVCache buffer.
//     hiddenState [GPU, Half]: The updated hidden state.
void eagleBaseCommitKVCacheAndAssembleHiddenState(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, rt::Tensor& kvCacheBuffer, rt::Tensor& hiddenState, cudaStream_t stream);

// clang-format on

} // namespace kernel
} // namespace drivellm