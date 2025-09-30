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

namespace drivellm
{
namespace kernel
{

// The kernel will normalize image data and convert to half
// Inputs:
//     originalImage [GPU, UInt8]: [batch, height, width, channels]
//     mean [GPU, Float]: [channels]
//     std [GPU, Float]: [channels]
//     stream: CUDA stream for execution
// Outputs:
//     normalizedImage [GPU, Half]: [batch, height, width, channels]
void normalizeImage(rt::Tensor const& originalImage, rt::Tensor const& mean, rt::Tensor const& std,
    rt::Tensor& normalizedImage, cudaStream_t stream);

// The kernel will transpose image data to patch format for Qwen2-VL and Qwen2.5-VL VIT
// The transpose is corresponding to the following python code:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L299
// Inputs:
//     originalImage [GPU, Half]: Current image [T, height, width, channels]
//     inputOffset: Offset of the input patches, denoting the start index of the current image
//     temporalPatchSize: Temporal patch size for the vision transformer
//     patchSize: Patch size for the vision transformer
//     mergeSize: Merge size for the vision transformer
//     stream: CUDA stream for execution
// Outputs:
//     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalSeqLength, inputDim]
//         curSeqLength = gridT * gridH * gridW * mergeSize * mergeSize
//         totalSeqLength = sum(curSeqLength) over all images
//         inputDim = channels * temporalPatchSize * patchSize * patchSize
void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int32_t const inputOffset,
    int32_t const temporalPatchSize, int32_t const patchSize, int32_t const mergeSize, cudaStream_t stream);

// The kernel will initialize the attention mask for Qwen2-VL and Qwen2.5-VL VIT
// Inputs:
//     cuSeqlens [GPU, Int32]: Cumulative sequence lengths [num]
//     stream: CUDA stream for execution
// Outputs:
//     attentionMask [GPU, Half]: Attention mask tensor [1, curHW, curHW]
void initAttentionMaskQwenViT(rt::Tensor const& cuSeqlens, rt::Tensor& attentionMask, cudaStream_t stream);

// The kernel will initialize the rotary position embeddings for Qwen2-VL and Qwen2.5-VL VIT
// Inputs:
//     posIds [GPU, Int32]: [totalSeqLength*2]
//     stream: CUDA stream for execution
// Outputs:
//     rotaryPosEmb [GPU, Float]: Rotary position embeddings tensor [totalSeqLength, vitPosEmbDim]
void initRotaryPosEmbQwenViT(rt::Tensor const& posIds, rt::Tensor& rotaryPosEmb, float const rotaryBaseFrequency,
    float const scale, cudaStream_t stream);

// The kernel will transpose image data to patch format for InternVL VIT
// Inputs:
//     originalImage [GPU, Half]: Current image [1, height, width, channels]
//     inputOffset: Offset of the input patches, denoting the start index of the current image
//     stream: CUDA stream for execution
// Outputs:
//     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalNumBlocks, channels, blockSizeH, blockSizeW]
//         curNumBlocks = blockH * blockW
//         totalNumBlocks = sum(curNumBlocks) over all images
void transposeToPatchInternVL(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int32_t const inputOffset, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm
