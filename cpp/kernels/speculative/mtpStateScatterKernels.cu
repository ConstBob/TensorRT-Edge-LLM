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

// MTP State Scatter Kernels
//
// After MTP speculative decoding verification, the base model's recurrent
// (GDN) and conv1d states must be updated to the last accepted step.
// During verify, the MTP kernel (with cache ON, state update ON) writes:
//   - h0_out = state after ALL T steps (wrong when only L < T tokens accepted)
//   - intermediate_states[t] = state snapshot after step t
//
// This kernel copies the correct snapshot back to the main state pool:
//   dst[batch, :] = src[batch, accepted_step, :]
//
// Design follows SGLang's fused_mamba_state_scatter_with_mask (Triton),
// adapted to CUDA using edge-llm's DVec<T> vectorized load/store.
//
// Grid: (batchSize, 1, ceil(vecCount / blockDim.x))
//   - Per-layer invocation (caller loops over layers)
//   - Early exit for skip (accepted < 0) and all-accept (accepted == maxSteps-1)
//   - DVec<float> = 8 floats (256-bit), DVec<half> = 8 halves (128-bit)

#include "mtpStateScatterKernels.h"

#include "kernels/common/vectorizedTypes.cuh"

#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

// ---------------------------------------------------------------------------
// Generic scatter kernel using DVec<T> for vectorized 256/128-bit access.
//
// Each thread copies one DVec (8 elements).  Grid z tiles the vectorized
// element count for states larger than blockDim.x vectors.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void mtpStateScatterKernel(T* __restrict__ dst, // [batchSize, stateElements]
    T const* __restrict__ src,                             // [batchSize, maxSteps, stateElements]
    int32_t const* __restrict__ acceptedSteps,             // [batchSize]
    int32_t maxSteps,
    int32_t vecCount) // stateElements / DVec<T>::vec_size
{
    int32_t const b = blockIdx.x; // batch index
    int32_t const step = acceptedSteps[b];

    // Skip invalid batch items (step < 0) or all-T-tokens-accepted (step == maxSteps-1,
    // meaning h0 already holds the correct final state after all T steps).
    if (step < 0 || step >= maxSteps - 1)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;

    // Flat index into the vectorized element array for this thread.
    int32_t const vecIdx = blockIdx.z * blockDim.x + threadIdx.x;
    if (vecIdx >= vecCount)
    {
        return;
    }

    // dst layout: [batchSize, stateElements]
    int64_t const stateElems = static_cast<int64_t>(vecCount) * kVecSize;
    int64_t const dstScalar = static_cast<int64_t>(b) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    // src layout: [batchSize, maxSteps, stateElements]
    int64_t const srcScalar
        = (static_cast<int64_t>(b) * maxSteps + step) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    DVec<T> v;
    v.load(src + srcScalar);
    v.store(dst + dstScalar);
}

// ---------------------------------------------------------------------------
// Validation + launch helper
// ---------------------------------------------------------------------------
template <typename T>
void launchScatter(rt::Tensor const& src, rt::Tensor const& acceptedSteps, rt::Tensor& dst, cudaStream_t stream)
{
    // Validate device placement.
    if (dst.getDeviceType() != rt::DeviceType::kGPU || src.getDeviceType() != rt::DeviceType::kGPU
        || acceptedSteps.getDeviceType() != rt::DeviceType::kGPU)
    {
        throw std::runtime_error("mtpStateScatter: all tensors must be on GPU");
    }

    // Validate dtypes.
    auto const expectedType = std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
    if (dst.getDataType() != expectedType || src.getDataType() != expectedType)
    {
        throw std::runtime_error("mtpStateScatter: dst/src dtype mismatch");
    }
    if (acceptedSteps.getDataType() != nvinfer1::DataType::kINT32)
    {
        throw std::runtime_error("mtpStateScatter: acceptedSteps must be INT32");
    }

    // Extract shapes.
    // dst: [batchSize, ...trailing...]   → flatten trailing to stateElements
    // src: [batchSize, maxSteps, ...trailing...]
    auto const dstShape = dst.getShape();
    auto const srcShape = src.getShape();
    auto const accShape = acceptedSteps.getShape();

    if (dstShape.getNumDims() < 1 || srcShape.getNumDims() < 2)
    {
        throw std::runtime_error("mtpStateScatter: dst must be ≥1D, src must be ≥2D");
    }

    int32_t const batchSize = dstShape[0];
    int32_t const maxSteps = srcShape[1];

    if (srcShape[0] != batchSize)
    {
        throw std::runtime_error("mtpStateScatter: batch size mismatch between dst and src");
    }
    if (accShape[0] != batchSize)
    {
        throw std::runtime_error("mtpStateScatter: acceptedSteps batch size mismatch");
    }

    // Compute stateElements = product of trailing dims of dst.
    int64_t stateElements = 1;
    for (int32_t d = 1; d < dstShape.getNumDims(); ++d)
    {
        stateElements *= dstShape[d];
    }

    if (batchSize == 0 || stateElements == 0)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;
    if (stateElements % kVecSize != 0)
    {
        throw std::runtime_error("mtpStateScatter: stateElements must be divisible by DVec vec_size (8)");
    }

    int32_t const vecCount = static_cast<int32_t>(stateElements / kVecSize);

    // 256 threads per block — good default for memory-bound vectorized copy.
    constexpr int32_t kThreads = 256;

    // Grid: (batch, 1, ceil(vecCount / kThreads))
    int32_t const zBlocks = (vecCount + kThreads - 1) / kThreads;
    dim3 const grid(batchSize, 1, zBlocks);
    dim3 const block(kThreads);

    mtpStateScatterKernel<T><<<grid, block, 0, stream>>>(
        dst.dataPointer<T>(), src.dataPointer<T>(), acceptedSteps.dataPointer<int32_t>(), maxSteps, vecCount);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void mtpScatterRecurrentStates(
    rt::Tensor const& src, rt::Tensor const& acceptedSteps, rt::Tensor& dst, cudaStream_t stream)
{
    // FP32 recurrent states — DVec<float> loads 8 floats (256-bit) per thread.
    launchScatter<float>(src, acceptedSteps, dst, stream);
}

void mtpScatterConvStates(rt::Tensor const& src, rt::Tensor const& acceptedSteps, rt::Tensor& dst, cudaStream_t stream)
{
    // FP16 conv states — DVec<half> loads 8 halves (128-bit) per thread.
    launchScatter<half>(src, acceptedSteps, dst, stream);
}

} // namespace kernel
} // namespace trt_edgellm
