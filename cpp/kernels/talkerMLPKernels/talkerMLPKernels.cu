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

#include "talkerMLPKernels.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <dlfcn.h>

// ============================================================================
// cuBLAS dynamic loading (avoids compile-time cublas dependency for edgellmCore)
// ============================================================================

// cuBLAS function pointer types (use int to avoid cublas header dependency)
using CublasSetStreamFn = int (*)(void*, cudaStream_t);
using CublasGemmExFn = int (*)(void*, int, int, int, int, int, void const*, void const*, cudaDataType_t, int,
    void const*, cudaDataType_t, int, void const*, void*, cudaDataType_t, int, int, int);

class CublasLoader
{
public:
    CublasLoader(CublasLoader const&) = delete;
    CublasLoader& operator=(CublasLoader const&) = delete;
    CublasLoader(CublasLoader&&) = delete;
    CublasLoader& operator=(CublasLoader&&) = delete;

    static CublasLoader& getInstance()
    {
        static CublasLoader instance;
        return instance;
    }

    void* libHandle{nullptr};
    CublasSetStreamFn setStream{nullptr};
    CublasGemmExFn gemmEx{nullptr};

private:
    CublasLoader()
    {
        libHandle = dlopen("libcublas.so", RTLD_LAZY);
        if (!libHandle)
        {
            return;
        }
        setStream = reinterpret_cast<CublasSetStreamFn>(dlsym(libHandle, "cublasSetStream_v2"));
        gemmEx = reinterpret_cast<CublasGemmExFn>(dlsym(libHandle, "cublasGemmEx"));
        if (!setStream || !gemmEx)
        {
            dlclose(libHandle);
            libHandle = nullptr;
        }
    }

    ~CublasLoader()
    {
        if (libHandle)
        {
            dlclose(libHandle);
        }
    }
};

// cuBLAS constants (match cublas_api.h values)
constexpr int kCUBLAS_OP_N = 0;
constexpr int kCUBLAS_OP_T = 1;
constexpr int kCUBLAS_STATUS_SUCCESS = 0;
// WAR: Use CUBLAS_COMPUTE_32F (FP32 accumulation) instead of CUBLAS_COMPUTE_16F.
// FP16 accumulation overflows when Thinker layer-14 hidden states have large values
// (e.g. maxAbs ~8912). The 2560-dim dot product partial sums exceed FP16 max (65504)
// even though the final result fits in FP16.
constexpr int kCUBLAS_COMPUTE_FP32 = 4;            // CUBLAS_COMPUTE_32F
constexpr int kCUBLAS_GEMM_DEFAULT_TENSOR_OP = 99; // CUBLAS_GEMM_DEFAULT_TENSOR_OP

namespace trt_edgellm
{
namespace kernel
{

namespace
{

//! \brief SiLU activation for FP16
//! \param x Input value
//! \return silu(x) = x / (1 + exp(-x))
__device__ __forceinline__ half silu(half x)
{
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + __expf(-fx)));
}

//! \brief Fused bias addition and SiLU activation kernel (vectorized)
//!
//! Each block processes one token, threads within block handle different dimensions.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//! Requires hiddenDim to be a multiple of 8 and pointers to be 16-byte aligned.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size (must be multiple of 8)
__global__ void biasAndSiLUKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int64_t numTokens, int64_t hiddenDim)
{
    constexpr int32_t kVEC_SIZE = 8; // sizeof(uint4) / sizeof(half)

    // Each block handles one token
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    half* tokenData = data + tokenIdx * hiddenDim;

    using vec_t = uint4;
    int64_t const numVecs = hiddenDim / kVEC_SIZE;

    for (int64_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[i];
        vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[i];

        half* dataPtr = reinterpret_cast<half*>(&dataVec);
        half const* biasPtr = reinterpret_cast<half const*>(&biasVec);

#pragma unroll
        for (int32_t j = 0; j < kVEC_SIZE; ++j)
        {
            dataPtr[j] = silu(__hadd(dataPtr[j], biasPtr[j]));
        }

        reinterpret_cast<vec_t*>(tokenData)[i] = dataVec;
    }
}

//! \brief Vectorized bias addition kernel using half2
//!
//! Uses 2D grid to avoid expensive modulo operation.
//! blockIdx.x handles dimension chunks, blockIdx.y handles tokens.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void addBiasKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int32_t numTokens, int32_t hiddenDim)
{
    using vec_t = uint4;

    int32_t const tokenIdx = blockIdx.y;
    int32_t const vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    if (tokenIdx >= numTokens || vecIdx >= numVecs)
    {
        return;
    }

    half* tokenData = data + static_cast<int64_t>(tokenIdx) * hiddenDim;

    vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[vecIdx];
    vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[vecIdx];

    half2* dataPtr = reinterpret_cast<half2*>(&dataVec);
    half2 const* biasPtr = reinterpret_cast<half2 const*>(&biasVec);

#pragma unroll
    for (int32_t i = 0; i < VEC_SIZE / 2; ++i)
    {
        dataPtr[i] = __hadd2(dataPtr[i], biasPtr[i]);
    }

    reinterpret_cast<vec_t*>(tokenData)[vecIdx] = dataVec;
}

//! \brief Vectorized gather kernel
//!
//! Each block handles one output row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [srcNumTokens, hiddenDim] (FP16)
//! \param[in] indices Indices to gather [numIndices] (INT32)
//! \param[out] output Output tensor [numIndices, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to gather
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void gatherKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one output row
    int32_t const outIdx = blockIdx.x;
    if (outIdx >= numIndices)
    {
        return;
    }

    // Get source row index
    int32_t const srcIdx = indices[outIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(outIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

//! \brief Vectorized scatter kernel
//!
//! Each block handles one source row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [numIndices, hiddenDim] (FP16)
//! \param[in] indices Indices to scatter to [numIndices] (INT32)
//! \param[out] output Output tensor [dstNumTokens, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to scatter
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void scatterKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one source row
    int32_t const srcIdx = blockIdx.x;
    if (srcIdx >= numIndices)
    {
        return;
    }

    // Get destination row index
    int32_t const dstIdx = indices[srcIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(dstIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

//! \brief GPU Sum Reduction Kernel using CUB WarpReduce
//!
//! Each warp handles one dimension position, threads within warp load seqLen values.
//! Uses CUB's WarpReduce for efficient parallel reduction.
//!
//! \param[in] input Input tensor [seqLen, hiddenDim] (FP16)
//! \param[out] output Output tensor [hiddenDim] (FP16)
//! \param[in] seqLen Sequence length to sum over
//! \param[in] hiddenDim Hidden dimension size
__global__ void sumReduceKernelCUB(
    half const* __restrict__ input, half* __restrict__ output, int32_t seqLen, int32_t hiddenDim)
{
    constexpr int32_t kWarpSize = 32;
    constexpr int32_t kWarpsPerBlock = 256 / kWarpSize;

    // Each warp handles one dimension position
    int32_t const warpId = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    int32_t const laneId = threadIdx.x % kWarpSize;

    if (warpId >= hiddenDim)
    {
        return;
    }

    // Each lane loads one element from the sequence (if within range)
    float val = 0.0f;
    if (laneId < seqLen)
    {
        val = __half2float(input[laneId * hiddenDim + warpId]);
    }

    // Use CUB WarpReduce for efficient parallel sum
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpsPerBlock];

    int32_t const warpIdInBlock = threadIdx.x / kWarpSize;
    float sum = WarpReduce(tempStorage[warpIdInBlock]).Sum(val);

    // Lane 0 writes the result
    if (laneId == 0)
    {
        output[warpId] = __float2half(sum);
    }
}

// Internal host function wrappers for kernel launches (not exposed in header)

void invokeBiasAndSiLU(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "hiddenDim must be a multiple of 8 for vectorized access");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int64_t const numTokens = data.getShape()[0];
    int64_t const hiddenDim = data.getShape()[1];

    dim3 const grid(numTokens);
    dim3 const block(256);

    biasAndSiLUKernelVectorized<<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeAddBias(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "Hidden dimension must be divisible by 8 for vectorization");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int32_t const numTokens = static_cast<int32_t>(data.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(data.getShape()[1]);

    constexpr int32_t VEC_SIZE = 8;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    dim3 const block(256);
    dim3 const grid((numVecs + block.x - 1) / block.x, numTokens);

    addBiasKernelVectorized<VEC_SIZE><<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace

// Host function implementations
void invokeTalkerMLP(void* cublasHandle, rt::Tensor const& input, rt::Tensor const& fc1Weight,
    rt::Tensor const& fc1Bias, rt::Tensor const& fc2Weight, rt::Tensor const& fc2Bias, rt::Tensor& output,
    rt::Tensor& workspace, cudaStream_t stream)
{
    auto inputShape = input.getShape();
    auto outputShape = output.getShape();
    auto workspaceShape = workspace.getShape();
    auto fc1WeightShape = fc1Weight.getShape();
    auto fc2WeightShape = fc2Weight.getShape();

    if (input.getDataType() != nvinfer1::DataType::kHALF || fc1Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc1Bias.getDataType() != nvinfer1::DataType::kHALF || fc2Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc2Bias.getDataType() != nvinfer1::DataType::kHALF || output.getDataType() != nvinfer1::DataType::kHALF
        || workspace.getDataType() != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("All tensors must be FP16");
        return;
    }

    if (inputShape.getNumDims() != 2 || outputShape.getNumDims() != 2 || workspaceShape.getNumDims() != 2)
    {
        LOG_ERROR("Tensors must be 2D");
        return;
    }

    int64_t const numTokens = inputShape[0];
    int64_t const inputDim = inputShape[1];
    int64_t const hiddenDim = fc1WeightShape[0];
    int64_t const outputDim = fc2WeightShape[0];

    if (fc1WeightShape[0] != hiddenDim || fc1WeightShape[1] != inputDim)
    {
        LOG_ERROR("FC1 weight shape mismatch: expected [%ld, %ld], got [%ld, %ld]", hiddenDim, inputDim,
            fc1WeightShape[0], fc1WeightShape[1]);
        return;
    }

    if (fc2WeightShape[0] != outputDim || fc2WeightShape[1] != hiddenDim)
    {
        LOG_ERROR("FC2 weight shape mismatch: expected [%ld, %ld], got [%ld, %ld]", outputDim, hiddenDim,
            fc2WeightShape[0], fc2WeightShape[1]);
        return;
    }

    if (outputShape[1] != outputDim)
    {
        LOG_ERROR("Output dimension mismatch: expected %ld, got %ld", outputDim, outputShape[1]);
        return;
    }

    if (outputShape[0] != numTokens || workspaceShape[0] != numTokens)
    {
        LOG_ERROR("Batch size mismatch: output[0]=%ld, workspace[0]=%ld, expected=%ld", outputShape[0],
            workspaceShape[0], numTokens);
        return;
    }

    if (workspaceShape[1] != hiddenDim)
    {
        LOG_ERROR("Workspace dimension mismatch: expected [%ld, %ld], got [%ld, %ld]", numTokens, hiddenDim,
            workspaceShape[0], workspaceShape[1]);
        return;
    }

    auto& cublas = CublasLoader::getInstance();
    if (!cublas.libHandle)
    {
        LOG_ERROR("cuBLAS not available (dlopen failed)");
        return;
    }

    cublas.setStream(cublasHandle, stream);

    // PyTorch Linear: output = input @ weight.T + bias
    // cuBLAS column-major: treat as output^T = weight @ input^T
    // Use CUBLAS_OP_T on weight (stored row-major) to get weight @ input^T
    float const alphaF32 = 1.0f;
    float const betaF32 = 0.0f;

    // FC1 GEMM: workspace = input @ fc1Weight.T
    int status = cublas.gemmEx(cublasHandle, kCUBLAS_OP_T, kCUBLAS_OP_N, hiddenDim, numTokens, inputDim, &alphaF32,
        fc1Weight.rawPointer(), CUDA_R_16F, inputDim, input.rawPointer(), CUDA_R_16F, inputDim, &betaF32,
        workspace.rawPointer(), CUDA_R_16F, hiddenDim, kCUBLAS_COMPUTE_FP32, kCUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != kCUBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("FC1 GEMM failed with status %d", status);
        return;
    }

    // Bias + SiLU activation
    invokeBiasAndSiLU(workspace, fc1Bias, stream);

    // FC2 GEMM: output = workspace @ fc2Weight.T
    status = cublas.gemmEx(cublasHandle, kCUBLAS_OP_T, kCUBLAS_OP_N, outputDim, numTokens, hiddenDim, &alphaF32,
        fc2Weight.rawPointer(), CUDA_R_16F, hiddenDim, workspace.rawPointer(), CUDA_R_16F, hiddenDim, &betaF32,
        output.rawPointer(), CUDA_R_16F, outputDim, kCUBLAS_COMPUTE_FP32, kCUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != kCUBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("FC2 GEMM failed with status %d", status);
        return;
    }

    // Add FC2 bias
    invokeAddBias(output, fc2Bias, stream);
}

void invokeGather(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    gatherKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeScatter(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    scatterKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void sumReduceOverSequence(rt::Tensor const& input, rt::Tensor& output, cudaStream_t stream)
{
    // input: [1, seqLen, hiddenDim] → output: [1, 1, hiddenDim]
    // Sum over sequence dimension (dim=1)

    check::check(input.getDataType() == nvinfer1::DataType::kHALF, "Input tensor must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto inputShape = input.getShape();
    auto outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 3, "Input must be 3D [1, seqLen, hiddenDim]");
    check::check(outputShape.getNumDims() == 3, "Output must be 3D [1, 1, hiddenDim]");
    check::check(inputShape[0] == 1 && outputShape[0] == 1, "Batch size must be 1");
    check::check(outputShape[1] == 1, "Output seqLen must be 1");
    check::check(inputShape[2] == outputShape[2], "Hidden dimension must match");
    check::check(inputShape[1] <= 32, "seqLen must be <= 32 for warp-based reduction");

    int64_t const seqLen = inputShape[1];
    int64_t const hiddenDim = inputShape[2];

    half const* inputPtr = static_cast<half const*>(input.rawPointer());
    half* outputPtr = static_cast<half*>(output.rawPointer());

    // Launch CUB-based kernel: each warp handles one dimension position
    // 256 threads = 8 warps per block, need hiddenDim/8 blocks
    constexpr int32_t BLOCK_SIZE = 256;
    constexpr int32_t WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    dim3 const block(BLOCK_SIZE);
    dim3 const grid((hiddenDim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    sumReduceKernelCUB<<<grid, block, 0, stream>>>(
        inputPtr, outputPtr, static_cast<int32_t>(seqLen), static_cast<int32_t>(hiddenDim));

    CUDA_CHECK(cudaPeekAtLastError());
}

// ========== Vectorized Elementwise Add Kernels ==========

//! \brief Vectorized elementwise add using half2 (2x throughput)
template <int32_t VEC_SIZE = 8>
__global__ void elementwiseAddKernelVectorized(
    half* __restrict__ output, half const* __restrict__ a, half const* __restrict__ b, int32_t size)
{
    using vec_t = uint4; // 8 half = 16 bytes

    int32_t const vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const numVecs = size / VEC_SIZE;

    if (vecIdx < numVecs)
    {
        vec_t aVec = reinterpret_cast<vec_t const*>(a)[vecIdx];
        vec_t bVec = reinterpret_cast<vec_t const*>(b)[vecIdx];

        half2* aPtr = reinterpret_cast<half2*>(&aVec);
        half2 const* bPtr = reinterpret_cast<half2 const*>(&bVec);

        // Use __hadd2 for vectorized FP16 addition
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE / 2; ++i)
        {
            aPtr[i] = __hadd2(aPtr[i], bPtr[i]);
        }

        reinterpret_cast<vec_t*>(output)[vecIdx] = aVec;
    }

    // Handle remainder (single thread handles tail elements)
    if (vecIdx == 0)
    {
        int32_t const remainderStart = numVecs * VEC_SIZE;
        for (int32_t i = remainderStart; i < size; ++i)
        {
            output[i] = __hadd(a[i], b[i]);
        }
    }
}

//! \brief Vectorized inplace elementwise add using half2
template <int32_t VEC_SIZE = 8>
__global__ void elementwiseAddInplaceKernelVectorized(
    half* __restrict__ data, half const* __restrict__ addend, int32_t size)
{
    using vec_t = uint4;

    int32_t const vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const numVecs = size / VEC_SIZE;

    if (vecIdx < numVecs)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(data)[vecIdx];
        vec_t addVec = reinterpret_cast<vec_t const*>(addend)[vecIdx];

        half2* dataPtr = reinterpret_cast<half2*>(&dataVec);
        half2 const* addPtr = reinterpret_cast<half2 const*>(&addVec);

#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE / 2; ++i)
        {
            dataPtr[i] = __hadd2(dataPtr[i], addPtr[i]);
        }

        reinterpret_cast<vec_t*>(data)[vecIdx] = dataVec;
    }

    // Handle remainder
    if (vecIdx == 0)
    {
        int32_t const remainderStart = numVecs * VEC_SIZE;
        for (int32_t i = remainderStart; i < size; ++i)
        {
            data[i] = __hadd(data[i], addend[i]);
        }
    }
}

void invokeElementwiseAdd(rt::Tensor& output, rt::Tensor const& a, rt::Tensor const& b, cudaStream_t stream)
{
    int64_t const size = a.getShape().volume();
    constexpr int32_t kVEC_SIZE = 8;
    int32_t const numVecs = static_cast<int32_t>((size + kVEC_SIZE - 1) / kVEC_SIZE);
    dim3 const block(256);
    dim3 const grid((numVecs + block.x - 1) / block.x);

    elementwiseAddKernelVectorized<kVEC_SIZE><<<grid, block, 0, stream>>>(static_cast<half*>(output.rawPointer()),
        a.dataPointer<half>(), b.dataPointer<half>(), static_cast<int32_t>(size));
    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeElementwiseAddInplace(rt::Tensor& data, rt::Tensor const& addend, int64_t numElements, int64_t dataOffset,
    int64_t addendOffset, cudaStream_t stream)
{
    int64_t const size = (numElements > 0) ? numElements : data.getShape().volume();
    half* dataPtr = static_cast<half*>(data.rawPointer()) + dataOffset;
    half const* addendPtr = addend.dataPointer<half>() + addendOffset;

    constexpr int32_t kVEC_SIZE = 8;
    int32_t const numVecs = static_cast<int32_t>((size + kVEC_SIZE - 1) / kVEC_SIZE);
    dim3 const block(256);
    dim3 const grid((numVecs + block.x - 1) / block.x);

    elementwiseAddInplaceKernelVectorized<kVEC_SIZE>
        <<<grid, block, 0, stream>>>(dataPtr, addendPtr, static_cast<int32_t>(size));
    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeSuppressLogits(
    rt::Tensor& logits, int32_t suppressStart, int32_t suppressEnd, int32_t exceptTokenId, cudaStream_t stream)
{
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "Logits tensor must be FP32");

    int32_t const count = suppressEnd - suppressStart;
    if (count <= 0)
    {
        return;
    }

    // Pre-built host buffer of -inf values (allocated once, never freed).
    static constexpr int32_t kMaxSuppressRange = 2048;
    static float const* sNegInfBuffer = []() {
        static float buf[kMaxSuppressRange];
        for (int32_t i = 0; i < kMaxSuppressRange; ++i)
        {
            buf[i] = -INFINITY;
        }
        return buf;
    }();

    check::check(count <= kMaxSuppressRange, "Suppress range exceeds pre-allocated buffer");

    float* logitsPtr = static_cast<float*>(logits.rawPointer());

    bool const exceptInRange = (exceptTokenId >= suppressStart && exceptTokenId < suppressEnd);

    if (!exceptInRange)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            logitsPtr + suppressStart, sNegInfBuffer, count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    else
    {
        // Write -inf in two segments, skipping the excepted token to preserve its logit value.
        int32_t const seg1Count = exceptTokenId - suppressStart;
        int32_t const seg2Count = suppressEnd - exceptTokenId - 1;

        if (seg1Count > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                logitsPtr + suppressStart, sNegInfBuffer, seg1Count * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        if (seg2Count > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(logitsPtr + exceptTokenId + 1, sNegInfBuffer, seg2Count * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }
    }
}

} // namespace kernel
} // namespace trt_edgellm
