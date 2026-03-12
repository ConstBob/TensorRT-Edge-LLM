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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "moeActivationKernels.h"
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

using format::fmtstr;

// ====================== Helper Functions ======================

__device__ __forceinline__ half silu(half x)
{
    float xf = __half2float(x);
    return __float2half(xf / (1.0f + expf(-xf)));
}

// 128-bit aligned vector type for coalesced 8×FP16 loads/stores.
constexpr int32_t kElemPerVec = 8;

struct alignas(16) HalfVec8
{
    half data[kElemPerVec];
    __device__ __host__ half& operator[](int32_t idx)
    {
        return data[idx];
    }
    __device__ __host__ half const& operator[](int32_t idx) const
    {
        return data[idx];
    }
};

// ====================== SwiGLU Kernel ======================

// Input layout: [numTokens, 2 * intermediateDim] with gate in the first half and up in the second.
// The kernel receives a single base pointer typed as HalfVec8 and computes per-token gate/up offsets.
// numVecsPerRow = intermediateDim / kElemPerVec (the number of vecs covering one gate or up segment).
__global__ void swiGluKernel(
    HalfVec8 const* __restrict__ input, HalfVec8* __restrict__ output, int64_t numVecsPerRow, int64_t numTokens)
{
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    // Per-token vec offsets: gate and up are stored contiguously with stride 2 * numVecsPerRow.
    int64_t const gateStart = tokenIdx * 2 * numVecsPerRow;
    int64_t const upStart = gateStart + numVecsPerRow;
    int64_t const outStart = tokenIdx * numVecsPerRow;

    int64_t const tid = threadIdx.x;
    int64_t const stride = blockDim.x;

    for (int64_t i = tid; i < numVecsPerRow; i += stride)
    {
        HalfVec8 gateVec = input[gateStart + i];
        HalfVec8 upVec = input[upStart + i];
        HalfVec8 outVec;

#pragma unroll
        for (int32_t j = 0; j < kElemPerVec; j++)
        {
            outVec[j] = silu(gateVec[j]) * upVec[j];
        }

        output[outStart + i] = outVec;
    }
}

// ====================== Public API ======================

void swiGluActivation(
    rt::Tensor const& gateUpInput, rt::Tensor& output, int64_t numTokens, int64_t intermediateDim, cudaStream_t stream)
{
    auto const inputShape = gateUpInput.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "gateUpInput must be 2D tensor [numTokens, 2*intermediateDim]");
    check::check(outputShape.getNumDims() == 2, "output must be 2D tensor [numTokens, intermediateDim]");
    check::check(inputShape[0] == numTokens, "gateUpInput first dimension must match numTokens");
    check::check(inputShape[1] == 2 * intermediateDim,
        fmtstr("gateUpInput second dimension must be 2*intermediateDim = %ld", 2 * intermediateDim));
    check::check(outputShape[0] == numTokens, "output first dimension must match numTokens");
    check::check(outputShape[1] == intermediateDim,
        fmtstr("output second dimension must be intermediateDim = %ld", intermediateDim));

    check::check(gateUpInput.getDataType() == nvinfer1::DataType::kHALF, "gateUpInput must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    check::check(gateUpInput.getDeviceType() == rt::DeviceType::kGPU, "gateUpInput must be on GPU");
    check::check(output.getDeviceType() == rt::DeviceType::kGPU, "output must be on GPU");

    check::check(intermediateDim % kElemPerVec == 0,
        fmtstr("intermediateDim (%ld) must be a multiple of %d for vectorized access", intermediateDim, kElemPerVec));

    auto const* inputRawPtr = gateUpInput.dataPointer<half>();
    auto* outputRawPtr = output.dataPointer<half>();

    check::check(reinterpret_cast<uintptr_t>(inputRawPtr) % alignof(HalfVec8) == 0,
        "gateUpInput pointer must be 16-byte aligned for vectorized access");
    check::check(reinterpret_cast<uintptr_t>(outputRawPtr) % alignof(HalfVec8) == 0,
        "output pointer must be 16-byte aligned for vectorized access");

    int64_t const numVecsPerRow = intermediateDim / kElemPerVec;

    auto const* inputVec = reinterpret_cast<HalfVec8 const*>(inputRawPtr);
    auto* outputVec = reinterpret_cast<HalfVec8*>(outputRawPtr);

    int64_t const blocks = numTokens;
    int32_t const threads = 1024;

    swiGluKernel<<<blocks, threads, 0, stream>>>(inputVec, outputVec, numVecsPerRow, numTokens);
}

} // namespace kernel
} // namespace trt_edgellm
