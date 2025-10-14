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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "imageUtilKernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

namespace drivellm
{
namespace kernel
{

__global__ void normalizeImageKernel(unsigned char const* originalImage, float const* mean, float const* std,
    half* normalizedImage, int32_t const batch, int32_t const height, int32_t const width, int32_t const channels)
{
    // Each thread processes one pixel
    // originalImage format: [batch, height, width, channels]
    // normalizedImage format: [batch, height, width, channels]
    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const totalPixels = batch * height * width * channels;
    if (tid >= totalPixels)
        return;

    int32_t const channel = tid % channels;
    unsigned char val = originalImage[tid];
    float normalized = (val / 255.0f - mean[channel]) / std[channel];
    normalizedImage[tid] = __float2half(normalized);
}

void normalizeImage(rt::Tensor const& originalImage, rt::Tensor const& mean, rt::Tensor const& std,
    rt::Tensor& normalizedImage, cudaStream_t stream)
{
    check::check(originalImage.getDeviceType() == rt::DeviceType::kGPU && mean.getDeviceType() == rt::DeviceType::kGPU
            && std.getDeviceType() == rt::DeviceType::kGPU && normalizedImage.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kUINT8 && mean.getDataType() == DataType::kFLOAT
            && std.getDataType() == DataType::kFLOAT && normalizedImage.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && normalizedImage.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [batch, height, width, channels] and [batch, height, width, channels] "
        "respectively.");

    int32_t const batch = originalImage.getShape()[0];
    int32_t const height = originalImage.getShape()[1];
    int32_t const width = originalImage.getShape()[2];
    int32_t const channels = originalImage.getShape()[3];
    check::check(
        channels == mean.getShape()[0] && channels == std.getShape()[0], "Channels mismatch for mean and std.");

    // Each CTA get assigned 256 threads.
    int32_t const blockSize = 256;
    int32_t const totalPixels = batch * height * width * channels;
    int32_t const gridSize = (totalPixels + blockSize - 1) / blockSize;

    normalizeImageKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<unsigned char>(),
        mean.dataPointer<float>(), std.dataPointer<float>(), normalizedImage.dataPointer<half>(), batch, height, width,
        channels);
}

__global__ void transposeToPatchQwenKernel(half const* originalImage, half* inputPatches, int32_t const T,
    int32_t const H, int32_t const W, int32_t const C, int32_t const temporalPatchSize, int32_t const patchSize,
    int32_t const mergeSize, int32_t const inputOffset)
{
    // This is a naive implementation of 9D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [T, H, W, C]
    //      T = gridT * temporalPatchSize
    //      H = gridH * mergeSize * patchSize
    //      W = gridW * mergeSize * patchSize
    //      C = channels
    // Transposed format: [seqLength, inputDim]
    //      seqLength = gridT * gridH * gridW * mergeSize * mergeSize
    //      inputDim = C * temporalPatchSize * patchSize * patchSize

    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t const gridT = T / temporalPatchSize;
    int32_t const gridH = H / (mergeSize * patchSize);
    int32_t const gridW = W / (mergeSize * patchSize);
    int32_t const seqLength = gridT * gridH * gridW * mergeSize * mergeSize;
    int32_t const inputDim = C * temporalPatchSize * patchSize * patchSize;
    int32_t const totalElements = seqLength * inputDim;

    if (tid >= totalElements)
        return;

    // Calculate which sequence and element this thread handles
    int32_t const seqIdx = tid / inputDim;
    int32_t const elemIdx = tid % inputDim;

    // Calculate sequence coordinates
    int32_t const tIdx = seqIdx / (gridH * gridW * mergeSize * mergeSize);
    int32_t const hIdx = (seqIdx % (gridH * gridW * mergeSize * mergeSize)) / (gridW * mergeSize * mergeSize);
    int32_t const wIdx = (seqIdx % (gridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    int32_t const mergeH = (seqIdx % (mergeSize * mergeSize)) / mergeSize;
    int32_t const mergeW = seqIdx % mergeSize;

    // Calculate coordinates within the patch
    int32_t const cIdx = elemIdx / (temporalPatchSize * patchSize * patchSize);
    int32_t const tPatchIdx = (elemIdx % (temporalPatchSize * patchSize * patchSize)) / (patchSize * patchSize);
    int32_t const patchH = (elemIdx % (patchSize * patchSize)) / patchSize;
    int32_t const patchW = elemIdx % patchSize;

    // Calculate source coordinates
    int32_t const srcT = tIdx * temporalPatchSize + tPatchIdx;
    int32_t const srcH = hIdx * mergeSize * patchSize + mergeH * patchSize + patchH;
    int32_t const srcW = wIdx * mergeSize * patchSize + mergeW * patchSize + patchW;
    int32_t const srcC = cIdx;

    // Calculate indices
    int32_t const srcIdx = srcT * H * W * C + srcH * W * C + srcW * C + srcC;
    int32_t const dstIdx = inputOffset + seqIdx * inputDim + elemIdx;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int32_t const inputOffset,
    int32_t const temporalPatchSize, int32_t const patchSize, int32_t const mergeSize, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 2,
        "Input and output tensor shapes shall be [T, H, W, C] and [totalSeqLength, inputDim] respectively.");
    // Get tensor dimensions
    int32_t const T = originalImage.getShape()[0];
    int32_t const H = originalImage.getShape()[1];
    int32_t const W = originalImage.getShape()[2];
    int32_t const C = originalImage.getShape()[3];
    int32_t const inputDim = inputPatches.getShape()[1];

    // Assertions for dimension assumptions
    check::check(inputDim == C * temporalPatchSize * patchSize * patchSize,
        "inputDim must be equal to C * temporalPatchSize * patchSize * patchSize: inputDim=" + std::to_string(inputDim)
            + ", C * temporalPatchSize * patchSize * patchSize="
            + std::to_string(C * temporalPatchSize * patchSize * patchSize));
    check::check(T % temporalPatchSize == 0,
        "T must be multiple of temporalPatchSize: T=" + std::to_string(T)
            + ", temporalPatchSize=" + std::to_string(temporalPatchSize));
    check::check(H % (mergeSize * patchSize) == 0,
        "H must be multiple of mergeSize * patchSize: H=" + std::to_string(H)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));
    check::check(W % (mergeSize * patchSize) == 0,
        "W must be multiple of mergeSize * patchSize: W=" + std::to_string(W)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));

    uint32_t const blockSize = 256;
    uint32_t const totalElements = T * H * W * C;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchQwenKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), T, H, W, C, temporalPatchSize, patchSize, mergeSize, inputOffset);
}

__global__ void transposeToPatchInternVLKernel(half const* originalImage, half* inputPatches, int32_t const inputOffset,
    int32_t const height, int32_t const width, int32_t const channels, int32_t const blockImageSizeH,
    int32_t const blockImageSizeW)
{
    // This is a naive implementation of 5D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [1,H, W, C]
    //      H = gridH * blockImageSizeH
    //      W = gridW * blockImageSizeW
    //      C = channels
    // Transposed format: [gridH * gridW, channels, blockSizeH, blockSizeW]

    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const gridH = height / blockImageSizeH;
    int32_t const gridW = width / blockImageSizeW;
    int32_t const numBlocks = gridH * gridW;
    int32_t const totalElements = numBlocks * channels * blockImageSizeH * blockImageSizeW;

    if (tid >= totalElements)
        return;

    // Calculate indices
    int32_t const gridHIdx = tid / (gridW * channels * blockImageSizeH * blockImageSizeW);
    int32_t const gridWIdx = (tid % (gridW * channels * blockImageSizeH * blockImageSizeW))
        / (channels * blockImageSizeH * blockImageSizeW);
    int32_t const cIdx = (tid % (channels * blockImageSizeH * blockImageSizeW)) / (blockImageSizeH * blockImageSizeW);
    int32_t const blockHIdx = (tid % (blockImageSizeH * blockImageSizeW)) / blockImageSizeW;
    int32_t const blockWIdx = tid % blockImageSizeW;

    int32_t const srcHIdx = gridHIdx * blockImageSizeH + blockHIdx;
    int32_t const srcWIdx = gridWIdx * blockImageSizeW + blockWIdx;
    int32_t const srcCIdx = cIdx;
    int32_t const srcIdx = srcHIdx * width * channels + srcWIdx * channels + srcCIdx;
    int32_t const dstIdx = inputOffset + tid;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchInternVL(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int32_t const inputOffset, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [1, height, width, channels] and [totalNumBlocks, channels, "
        "blockSizeH, blockSizeW] respectively.");
    check::check(originalImage.getShape()[0] == 1, "Original image shape shall be [1, height, width, channels].");

    int32_t const height = originalImage.getShape()[1];
    int32_t const width = originalImage.getShape()[2];
    int32_t const channels = originalImage.getShape()[3];
    int32_t const blockSizeH = inputPatches.getShape()[2];
    int32_t const blockSizeW = inputPatches.getShape()[3];

    uint32_t const blockSize = 256;
    uint32_t const totalElements = height * width * channels;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchInternVLKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), inputOffset, height, width, channels, blockSizeH, blockSizeW);
}

__global__ void initMaskToMinKernel(half* attentionMask, int32_t const totalElements)
{
    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }

    // Mask used to disable attention between "patches". Use -20000.0F to avoid data overflow in
    // TensorRT which could produce NaN output if we supply -MAX_FLOAT_FP16 mask values.
    half const disabledMaskValue{-20000.0F};
    attentionMask[tid] = disabledMaskValue;
}

__global__ void initAttentionMaskKernel(
    int32_t const* cuSeqlens, half* attentionMask, int32_t const cuSeqlensSize, int32_t const curHW)
{
    // Each CTA process one cuSeqlen and set zero to attentionMask[start:end, start:end]
    // Each CTA get assigned (16, 16) threads.
    int32_t const bIdx = blockIdx.x;
    int32_t const start = cuSeqlens[bIdx];
    int32_t const end = cuSeqlens[bIdx + 1];
    int32_t const tIdx = threadIdx.x;
    int32_t const tidy = threadIdx.y;

    for (int32_t i = start + tIdx; i < end; i += 16)
    {
        for (int32_t j = start + tidy; j < end; j += 16)
        {
            int32_t const posIdx = i * curHW + j;
            attentionMask[posIdx] = CUDART_ZERO_FP16;
        }
    }
}

void initAttentionMaskQwenViT(rt::Tensor const& cuSeqlens, rt::Tensor& attentionMask, cudaStream_t stream)
{
    check::check(
        cuSeqlens.getDeviceType() == rt::DeviceType::kGPU && attentionMask.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(cuSeqlens.getDataType() == DataType::kINT32 && attentionMask.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(cuSeqlens.getShape().getNumDims() == 1, "Cu seqlens shape shall be [num].");
    check::check(attentionMask.getShape().getNumDims() == 3 && attentionMask.getShape()[0] == 1,
        "Attention mask shape shall be [1, curHW, curHW].");

    int32_t const cuSeqlensSize = cuSeqlens.getShape()[0];
    int32_t const curHW = attentionMask.getShape()[1];
    int32_t const totalElements = curHW * curHW;

    // Initialize attention mask to small value to indicate "disabled" attention.
    uint32_t const initBlockSize = 256;
    uint32_t const initGridSize = (totalElements + initBlockSize - 1) / initBlockSize;
    initMaskToMinKernel<<<initGridSize, initBlockSize, 0, stream>>>(attentionMask.dataPointer<half>(), totalElements);

    // Set zero to target positions
    dim3 blockSize{16, 16};
    uint32_t const gridSize = cuSeqlensSize - 1;
    initAttentionMaskKernel<<<gridSize, blockSize, 0, stream>>>(
        cuSeqlens.dataPointer<int32_t>(), attentionMask.dataPointer<half>(), cuSeqlensSize, curHW);
}

__global__ void initRotaryPosEmbQwenKernel(int32_t const* posIds, float* rotaryPosEmb, int32_t const totalSeqLength,
    int32_t const vitPosEmbDim, float const rotaryBaseFrequency, float const scale)
{
    // Each CTA get assigned 256 threads. Each thread processes one element
    // posIds: [totalSeqLength*2]
    // rotaryPosEmb: [totalSeqLength*2, vitPosEmbDim/2]
    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const totalElements = totalSeqLength * vitPosEmbDim;
    if (tid >= totalElements)
        return;

    int32_t const posIdx = tid / (vitPosEmbDim / 2);
    int32_t const dimIdx = tid % (vitPosEmbDim / 2);

    float invFreq = posIds[posIdx] * scale / pow(rotaryBaseFrequency, 2 * dimIdx / (float) vitPosEmbDim);
    rotaryPosEmb[tid] = invFreq;
}

void initRotaryPosEmbQwenViT(rt::Tensor const& posIds, rt::Tensor& rotaryPosEmb, float const rotaryBaseFrequency,
    float const scale, cudaStream_t stream)
{
    check::check(posIds.getDeviceType() == rt::DeviceType::kGPU && rotaryPosEmb.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(posIds.getDataType() == DataType::kINT32 && rotaryPosEmb.getDataType() == DataType::kFLOAT,
        "Data type check failed for the input tensors.");
    check::check(posIds.getShape().getNumDims() == 1, "Pos ids shape shall be [totalSeqLength*2].");
    check::check(rotaryPosEmb.getShape().getNumDims() == 2,
        "Rotary position embeddings shape shall be [totalSeqLength, vitPosEmbDim].");

    int32_t const totalSeqLength = rotaryPosEmb.getShape()[0];
    int32_t const vitPosEmbDim = rotaryPosEmb.getShape()[1];
    check::check(totalSeqLength == posIds.getShape()[0] / 2, "Total sequence length mismatch.");

    uint32_t const blockSize = 256;
    uint32_t const totalElements = totalSeqLength * vitPosEmbDim;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    initRotaryPosEmbQwenKernel<<<gridSize, blockSize, 0, stream>>>(posIds.dataPointer<int32_t>(),
        rotaryPosEmb.dataPointer<float>(), totalSeqLength, vitPosEmbDim, rotaryBaseFrequency, scale);
}

} // namespace kernel
} // namespace drivellm
