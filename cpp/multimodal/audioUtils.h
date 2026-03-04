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
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

//! Chunk metadata for Qwen3-Omni audio preprocessing
struct ChunkInfo
{
    int64_t numChunks;
    std::vector<int64_t> chunkLengths;
    std::vector<int64_t> chunkOffsets;
    int64_t maxChunkLength;
};

//! Compute CNN output length (three 2x downsampling layers)
int64_t computeFeatExtractOutputLength(int64_t inputLength);

//! Compute chunk split information for audio features
ChunkInfo computeChunkInfo(int64_t featureLength, int32_t nWindow);

//! Chunk and pad audio features to uniform size
bool chunkAndPadFeatures(
    rt::Tensor const& melSpectrogram, ChunkInfo const& chunkInfo, rt::Tensor& paddedFeature, cudaStream_t stream);

//! Create validity mask for tokens after CNN downsampling
bool createPaddedMask(ChunkInfo const& chunkInfo, int32_t nWindow, rt::Tensor& paddedMask,
    std::vector<int64_t>& afterCNNLens, cudaStream_t stream);

//! Preprocess audio for Qwen3-Omni encoder: chunk, pad, and create masks
bool preprocessAudioForEncoder(rt::Tensor const& melSpectrogram, int32_t nWindow, rt::Tensor& paddedFeature,
    rt::Tensor& paddedMaskAfterCNN, std::vector<int64_t>& afterCNNLens, cudaStream_t stream);

//! Convert boolean mask to nonzero indices (equivalent to torch.nonzero)
//! This function implements the NonZero operation that was removed from the ONNX model.
//! It converts a 2D boolean mask into indices of nonzero elements.
//!
//! @param paddedMask Input boolean mask [num_chunks, max_len_after_cnn]
//! @param paddedMaskIndices Output indices [num_valid_elements, 2] where each row is [chunk_idx, position_idx]
//! @param stream CUDA stream for async operations
//! @return true on success, false on failure
//!
//! Example:
//!   Input mask: [[1, 1, 0], [1, 0, 0]]
//!   Output indices: [[0, 0], [0, 1], [1, 0]]
bool convertMaskToIndices(rt::Tensor const& paddedMask, rt::Tensor& paddedMaskIndices, cudaStream_t stream);

//! Create block-diagonal attention mask for chunk-wise attention
//! Each chunk can only attend to tokens within the same chunk, preventing
//! cross-contamination between different audio segments.
//!
//! @param afterCNNLens Vector of chunk lengths after CNN downsampling
//! @param attentionMask Output attention mask [total_len, total_len]
//!        Values: 0.0 for allowed attention, -65504.0 (FP16 min) for blocked
//! @param stream CUDA stream for async operations
//! @return true on success, false on failure
//!
//! Example:
//!   Input: afterCNNLens = [26, 12]
//!   Output: 38x38 block-diagonal matrix with two blocks (26x26 and 12x12)
bool createChunkwiseAttentionMask(
    std::vector<int64_t> const& afterCNNLens, rt::Tensor& attentionMask, cudaStream_t stream);

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
