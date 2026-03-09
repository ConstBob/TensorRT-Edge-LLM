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

namespace mamba_ssm
{

// Internal params struct — not exposed in the public header.
struct CausalConv1dParams
{
    int32_t batch{};
    int32_t seqLen{};
    int32_t outSeqLen{};
    int32_t dim{};
    int32_t width{};
    int32_t stride{1};
    int32_t padding{};
    int32_t dilation{1};

    int64_t xStrideBatch{};
    int64_t xStrideSeq{};
    int64_t xStrideDim{};

    int64_t weightStrideChannel{};
    int64_t weightStrideKernel{};

    int64_t outStrideBatch{};
    int64_t outStrideSeq{};
    int64_t outStrideDim{};

    void const* x{nullptr};
    void const* weight{nullptr};
    void const* bias{nullptr};
    void* out{nullptr};
};

template <typename T>
__global__ void causalConv1dKernel(CausalConv1dParams params)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= params.dim)
    {
        return;
    }

    auto const* x = reinterpret_cast<T const*>(params.x);
    auto const* weight = reinterpret_cast<T const*>(params.weight);
    auto const* bias = reinterpret_cast<T const*>(params.bias);
    auto* out = reinterpret_cast<T*>(params.out);

    int64_t const xBatchOffset = static_cast<int64_t>(batchIdx) * params.xStrideBatch;
    int64_t const outBatchOffset = static_cast<int64_t>(batchIdx) * params.outStrideBatch;
    int64_t const weightChannelOffset = static_cast<int64_t>(dimIdx) * params.weightStrideChannel;
    float const biasValue = bias == nullptr ? 0.F : conversion::toFloat(bias[dimIdx]);

    for (int32_t outPos = 0; outPos < params.outSeqLen; ++outPos)
    {
        float acc = biasValue;
        int32_t const inBase = outPos * params.stride - params.padding;
        for (int32_t k = 0; k < params.width; ++k)
        {
            int32_t const inPos = inBase + k * params.dilation;
            if (inPos >= 0 && inPos < params.seqLen)
            {
                int64_t const xIdx = xBatchOffset + static_cast<int64_t>(inPos) * params.xStrideSeq
                    + static_cast<int64_t>(dimIdx) * params.xStrideDim;
                int64_t const wIdx = weightChannelOffset + static_cast<int64_t>(k) * params.weightStrideKernel;
                acc += conversion::toFloat(x[xIdx]) * conversion::toFloat(weight[wIdx]);
            }
        }
        int64_t const outIdx = outBatchOffset + static_cast<int64_t>(outPos) * params.outStrideSeq
            + static_cast<int64_t>(dimIdx) * params.outStrideDim;
        conversion::convertAndStore(&out[outIdx], acc);
    }
}

template <typename T>
void invokeCausalConv1d(
    CausalConv1dTensors const& tensors, int32_t stride, int32_t padding, int32_t dilation, cudaStream_t stream)
{
    CausalConv1dParams params{};
    params.batch = static_cast<int32_t>(tensors.x->getShape()[0]);
    params.seqLen = static_cast<int32_t>(tensors.x->getShape()[1]);
    params.dim = static_cast<int32_t>(tensors.x->getShape()[2]);
    params.width = static_cast<int32_t>(tensors.weight->getShape()[2]);
    params.outSeqLen = static_cast<int32_t>(tensors.out->getShape()[1]);
    params.stride = stride;
    params.padding = padding;
    params.dilation = dilation;
    params.xStrideBatch = tensors.x->getStride(0);
    params.xStrideSeq = tensors.x->getStride(1);
    params.xStrideDim = tensors.x->getStride(2);
    params.weightStrideChannel = tensors.weight->getStride(0);
    params.weightStrideKernel = tensors.weight->getStride(2);
    params.outStrideBatch = tensors.out->getStride(0);
    params.outStrideSeq = tensors.out->getStride(1);
    params.outStrideDim = tensors.out->getStride(2);
    params.x = tensors.x->rawPointer();
    params.weight = tensors.weight->rawPointer();
    params.bias = tensors.bias->rawPointer();
    params.out = tensors.out->rawPointer();

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(params.batch, static_cast<uint32_t>((params.dim + kThreads - 1) / kThreads));
    causalConv1dKernel<T><<<grid, block, 0, stream>>>(params);
    CUDA_CHECK(cudaPeekAtLastError());
}

template void invokeCausalConv1d<half>(
    CausalConv1dTensors const& tensors, int32_t stride, int32_t padding, int32_t dilation, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Decode kernel: conv_state[batch, dim, width] dot weight[dim, 1, width] + bias
// ---------------------------------------------------------------------------

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

    // output layout: [batch, 1, dim]
    int64_t const outIdx = static_cast<int64_t>(batchIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
}

template <typename T>
void invokeCausalConv1dDecode(void const* convState, void const* weight, void const* bias, void* output, int32_t batch,
    int32_t dim, int32_t width, cudaStream_t stream)
{
    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    causalConv1dDecodeKernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<T const*>(convState),
        reinterpret_cast<T const*>(weight), reinterpret_cast<T const*>(bias), reinterpret_cast<T*>(output), dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

template void invokeCausalConv1dDecode<half>(void const* convState, void const* weight, void const* bias, void* output,
    int32_t batch, int32_t dim, int32_t width, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Capture last `width` time-steps from x into conv_state (transposed).
// x layout:         [batch, seqLen, dim]  (row-major)
// convState layout: [batch, dim, width]   (row-major, must be zero-initialized)
// Each thread handles one (batch, dim) element, writing up to `width` entries.
// ---------------------------------------------------------------------------

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
        // src: x[batchIdx, tailStart + t, dimIdx]
        int64_t const srcIdx = (static_cast<int64_t>(batchIdx) * seqLen + tailStart + t) * dim + dimIdx;
        // dst: convState[batchIdx, dimIdx, dstOffset + t]
        int64_t const dstIdx = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width + dstOffset + t;
        convState[dstIdx] = x[srcIdx];
    }
}

template <typename T>
void invokeCaptureConvState(
    void const* x, void* convState, int32_t batch, int32_t seqLen, int32_t dim, int32_t width, cudaStream_t stream)
{
    // Zero the output first
    size_t const elemSize = sizeof(T);
    CUDA_CHECK(cudaMemsetAsync(convState, 0, static_cast<size_t>(batch) * dim * width * elemSize, stream));

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    captureConvStateKernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<T const*>(x), reinterpret_cast<T*>(convState), seqLen, dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

template void invokeCaptureConvState<half>(
    void const* x, void* convState, int32_t batch, int32_t seqLen, int32_t dim, int32_t width, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Shift conv_state left by 1, insert new column at position width-1.
// convState: [batch, dim, width]
// newCol:    [batch, 1, dim]  (i.e. contiguous dim elements per batch)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void convStateShiftInsertKernel(T* convState, T const* newCol, int32_t dim, int32_t width)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const rowOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    T* row = convState + rowOffset;

    // Shift left by 1
    for (int32_t k = 0; k < width - 1; ++k)
    {
        row[k] = row[k + 1];
    }

    // Insert new value at position width-1
    row[width - 1] = newCol[static_cast<int64_t>(batchIdx) * dim + dimIdx];
}

template <typename T>
void invokeConvStateShiftInsert(
    void* convState, void const* newCol, int32_t batch, int32_t dim, int32_t width, cudaStream_t stream)
{
    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    convStateShiftInsertKernel<T>
        <<<grid, block, 0, stream>>>(reinterpret_cast<T*>(convState), reinterpret_cast<T const*>(newCol), dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

template void invokeConvStateShiftInsert<half>(
    void* convState, void const* newCol, int32_t batch, int32_t dim, int32_t width, cudaStream_t stream);

} // namespace mamba_ssm
