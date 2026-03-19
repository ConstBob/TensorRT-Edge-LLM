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

/*
 * This file contains code derived from causal-conv1d
 * (https://github.com/Dao-AILab/causal-conv1d)
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Licensed under the BSD 3-Clause License.
 *
 * Modifications by NVIDIA:
 * - Adapted causal depthwise conv1d kernel for TensorRT Edge-LLM integration
 * - Added stride, dilation, and padding parameters for generalized conv1d
 * - Added decode-mode kernel (conv_state dot weight)
 * - Added conv state capture and shift-insert kernels
 */

#include "causalConv1d.h"

#include "common/checkMacros.h"
#include "conversion.cuh"

#include <cuda_fp16.h>
#include <stdexcept>

namespace mamba_ssm
{

// Prefill causal conv1d kernel (templated).
template <typename T>
__global__ void causalConv1dKernel(T const* x, T const* weight, T const* bias, T* out, int32_t batch, int32_t seqLen,
    int32_t outSeqLen, int32_t dim, int32_t width, int32_t stride, int32_t padding, int32_t dilation,
    int64_t xStrideBatch, int64_t xStrideSeq, int64_t xStrideDim, int64_t weightStrideChannel,
    int64_t weightStrideKernel, int64_t outStrideBatch, int64_t outStrideSeq, int64_t outStrideDim)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const xBatchOffset = static_cast<int64_t>(batchIdx) * xStrideBatch;
    int64_t const outBatchOffset = static_cast<int64_t>(batchIdx) * outStrideBatch;
    int64_t const weightChannelOffset = static_cast<int64_t>(dimIdx) * weightStrideChannel;
    float const biasValue = bias == nullptr ? 0.F : conversion::toFloat(bias[dimIdx]);

    for (int32_t outPos = 0; outPos < outSeqLen; ++outPos)
    {
        float acc = biasValue;
        int32_t const inBase = outPos * stride - padding;
        for (int32_t k = 0; k < width; ++k)
        {
            int32_t const inPos = inBase + k * dilation;
            if (inPos >= 0 && inPos < seqLen)
            {
                int64_t const xIdx = xBatchOffset + static_cast<int64_t>(inPos) * xStrideSeq
                    + static_cast<int64_t>(dimIdx) * xStrideDim;
                int64_t const wIdx = weightChannelOffset + static_cast<int64_t>(k) * weightStrideKernel;
                acc += conversion::toFloat(x[xIdx]) * conversion::toFloat(weight[wIdx]);
            }
        }
        int64_t const outIdx = outBatchOffset + static_cast<int64_t>(outPos) * outStrideSeq
            + static_cast<int64_t>(dimIdx) * outStrideDim;
        conversion::convertAndStore(&out[outIdx], acc);
    }
}

void invokeCausalConv1d(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out, int32_t stride, int32_t padding,
    int32_t dilation, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);
    int32_t const outSeqLen = static_cast<int32_t>(out.getShape()[1]);

    int64_t const xStrideBatch = x.getStride(0);
    int64_t const xStrideSeq = x.getStride(1);
    int64_t const xStrideDim = x.getStride(2);
    int64_t const weightStrideChannel = weight.getStride(0);
    int64_t const weightStrideKernel = weight.getStride(2);
    int64_t const outStrideBatch = out.getStride(0);
    int64_t const outStrideSeq = out.getStride(1);
    int64_t const outStrideDim = out.getStride(2);

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));

    if (x.getDataType() != nvinfer1::DataType::kHALF || weight.getDataType() != nvinfer1::DataType::kHALF
        || out.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("invokeCausalConv1d: only FP16 (half) is supported.");
    }
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    causalConv1dKernel<half><<<grid, block, 0, stream>>>(x.dataPointer<half>(), weight.dataPointer<half>(), biasPtr,
        out.dataPointer<half>(), batch, seqLen, outSeqLen, dim, width, stride, padding, dilation, xStrideBatch,
        xStrideSeq, xStrideDim, weightStrideChannel, weightStrideKernel, outStrideBatch, outStrideSeq, outStrideDim);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Decode kernel: conv_state dot weight + bias.
template <typename T>
__global__ void causalConv1dDecodeKernel(
    T const* convState, T const* weight, T const* bias, T* output, int32_t dim, int32_t width)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    int64_t const stateOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;

    for (int32_t k = 0; k < width; ++k)
    {
        acc += conversion::toFloat(convState[stateOffset + k]) * conversion::toFloat(weight[weightOffset + k]);
    }

    int64_t const outIdx = static_cast<int64_t>(batchIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
}

void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    if (convState.getDataType() != nvinfer1::DataType::kHALF || weight.getDataType() != nvinfer1::DataType::kHALF
        || out.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("invokeCausalConv1dDecode: only FP16 (half) is supported.");
    }

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    causalConv1dDecodeKernel<half><<<grid, block, 0, stream>>>(
        convState.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(), dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Capture last `width` time-steps from x into conv_state (transposed).
template <typename T>
__global__ void captureConvStateKernel(T const* x, T* convState, int32_t seqLen, int32_t dim, int32_t width)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const tailLen = (seqLen >= width) ? width : seqLen;
    int32_t const tailStart = seqLen - tailLen;
    int32_t const dstOffset = width - tailLen;

    for (int32_t t = 0; t < tailLen; ++t)
    {
        int64_t const srcIdx = (static_cast<int64_t>(batchIdx) * seqLen + tailStart + t) * dim + dimIdx;
        int64_t const dstIdx = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width + dstOffset + t;
        convState[dstIdx] = x[srcIdx];
    }
}

void invokeCaptureConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor& convState, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    if (x.getDataType() != nvinfer1::DataType::kHALF || convState.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("invokeCaptureConvState: only FP16 (half) is supported.");
    }

    size_t const elemSize = sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(convState.rawPointer(), 0, static_cast<size_t>(batch) * dim * width * elemSize, stream));

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    captureConvStateKernel<half>
        <<<grid, block, 0, stream>>>(x.dataPointer<half>(), convState.dataPointer<half>(), seqLen, dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Shift conv_state left by 1, insert new column at position width-1.
template <typename T>
__global__ void convStateShiftInsertKernel(T* convState, T const* newCol, int32_t batch, int32_t dim, int32_t width)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const rowOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    T* row = convState + rowOffset;

    for (int32_t k = 0; k < width - 1; ++k)
    {
        row[k] = row[k + 1];
    }
    row[width - 1] = newCol[static_cast<int64_t>(batchIdx) * dim + dimIdx];
}

void invokeConvStateShiftInsert(
    trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    if (convState.getDataType() != nvinfer1::DataType::kHALF || newCol.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("invokeConvStateShiftInsert: only FP16 (half) is supported.");
    }

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    convStateShiftInsertKernel<half>
        <<<grid, block, 0, stream>>>(convState.dataPointer<half>(), newCol.dataPointer<half>(), batch, dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace mamba_ssm
