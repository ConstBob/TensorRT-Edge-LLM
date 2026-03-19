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
int64_t computeFeatExtractOutputLength(int64_t inputLength, int32_t nWindow);

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

//! Create block-diagonal attention mask matching _prepare_attention_mask + cu_seqlens logic.
//! Merges per-chunk after-CNN lengths into larger windows using n_window_infer,
//! then builds a block-diagonal mask where each window allows bidirectional attention.
//!
//! @param afterCNNLens Per-chunk after-CNN lengths
//! @param nWindow Audio encoder n_window parameter (default 50)
//! @param nWindowInfer Audio encoder n_window_infer parameter (default 200)
//! @param attentionMask Output attention mask [total_len, total_len]
//! @param stream CUDA stream for async operations
//! @return true on success, false on failure
bool createChunkwiseAttentionMask(std::vector<int64_t> const& afterCNNLens, int32_t nWindow, int32_t nWindowInfer,
    rt::Tensor& attentionMask, cudaStream_t stream);

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
