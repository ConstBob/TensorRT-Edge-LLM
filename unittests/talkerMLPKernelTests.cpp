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
#include "common/tensor.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "testUtils.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{

float siluF32(float x)
{
    return x / (1.0f + std::exp(-x));
}

// CPU reference: output = FC2(SiLU(FC1(input) + bias1)) + bias2
// Weight layout: row-major [outDim, inDim] (same as PyTorch Linear.weight)
void referenceTalkerMLP(std::vector<half> const& input, std::vector<half> const& fc1Weight,
    std::vector<half> const& fc1Bias, std::vector<half> const& fc2Weight, std::vector<half> const& fc2Bias,
    std::vector<half>& output, int64_t numTokens, int64_t inputDim, int64_t hiddenDim, int64_t outputDim)
{
    std::vector<float> workspace(numTokens * hiddenDim, 0.0f);
    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t h = 0; h < hiddenDim; ++h)
        {
            float acc = 0.0f;
            for (int64_t k = 0; k < inputDim; ++k)
            {
                acc += __half2float(input[n * inputDim + k]) * __half2float(fc1Weight[h * inputDim + k]);
            }
            acc += __half2float(fc1Bias[h]);
            workspace[n * hiddenDim + h] = siluF32(acc);
        }
    }

    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t o = 0; o < outputDim; ++o)
        {
            float acc = 0.0f;
            for (int64_t h = 0; h < hiddenDim; ++h)
            {
                acc += workspace[n * hiddenDim + h] * __half2float(fc2Weight[o * hiddenDim + h]);
            }
            acc += __half2float(fc2Bias[o]);
            output[n * outputDim + o] = __float2half(acc);
        }
    }
}

} // namespace

// ============================================================================
// Fixture for tests that require cuBLAS (invokeTalkerMLP)
// ============================================================================
class TalkerMLPTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};
    void* cublasLib{nullptr};
    void* cublasHandle{nullptr};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));

        cublasLib = dlopen("libcublas.so", RTLD_LAZY);
        if (!cublasLib)
        {
            GTEST_SKIP() << "cuBLAS not available";
        }
        auto createFn = reinterpret_cast<int (*)(void**)>(dlsym(cublasLib, "cublasCreate_v2"));
        if (!createFn || createFn(&cublasHandle) != 0)
        {
            dlclose(cublasLib);
            cublasLib = nullptr;
            GTEST_SKIP() << "Failed to create cuBLAS handle";
        }
    }

    void TearDown() override
    {
        if (cublasHandle && cublasLib)
        {
            auto destroyFn = reinterpret_cast<int (*)(void*)>(dlsym(cublasLib, "cublasDestroy_v2"));
            if (destroyFn)
            {
                destroyFn(cublasHandle);
            }
        }
        if (cublasLib)
        {
            dlclose(cublasLib);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ============================================================================
// Fixture for tests that only need CUDA (no cuBLAS dependency)
// ============================================================================
class TalkerKernelTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ===== invokeTalkerMLP =====

TEST_F(TalkerMLPTest, MLPAccuracy)
{
    // (numTokens, inputDim, hiddenDim, outputDim)
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> testCases = {
        {1, 64, 64, 32},
        {2, 64, 64, 32},
        {4, 2048, 2048, 1024},
    };

    for (auto const& [numTokens, inputDim, hiddenDim, outputDim] : testCases)
    {
        SCOPED_TRACE("numTokens=" + std::to_string(numTokens) + ", inputDim=" + std::to_string(inputDim)
            + ", hiddenDim=" + std::to_string(hiddenDim) + ", outputDim=" + std::to_string(outputDim));

        // Scale init range down for large dimensions to avoid FP16 overflow in accumulation
        float const initScale = (inputDim > 256) ? 0.1f : 0.5f;

        std::vector<half> hostInput(numTokens * inputDim);
        std::vector<half> hostFc1W(hiddenDim * inputDim);
        std::vector<half> hostFc1B(hiddenDim);
        std::vector<half> hostFc2W(outputDim * hiddenDim);
        std::vector<half> hostFc2B(outputDim);

        uniformFloatInitialization(hostInput, -initScale * 2, initScale * 2);
        uniformFloatInitialization(hostFc1W, -initScale, initScale);
        uniformFloatInitialization(hostFc1B, -0.1f, 0.1f);
        uniformFloatInitialization(hostFc2W, -initScale, initScale);
        uniformFloatInitialization(hostFc2B, -0.1f, 0.1f);

        std::vector<half> refOutput(numTokens * outputDim);
        referenceTalkerMLP(
            hostInput, hostFc1W, hostFc1B, hostFc2W, hostFc2B, refOutput, numTokens, inputDim, hiddenDim, outputDim);

        rt::Coords inputShape{numTokens, inputDim};
        rt::Coords fc1WShape{hiddenDim, inputDim};
        rt::Coords fc1BShape{hiddenDim};
        rt::Coords fc2WShape{outputDim, hiddenDim};
        rt::Coords fc2BShape{outputDim};
        rt::Coords outputShape{numTokens, outputDim};
        rt::Coords workspaceShape{numTokens, hiddenDim};

        rt::Tensor gpuInput(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1W(fc1WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1B(fc1BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2W(fc2WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2B(fc2BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuOutput(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuWorkspace(workspaceShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        CUDA_CHECK(cudaMemcpy(
            gpuInput.rawPointer(), hostInput.data(), hostInput.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(gpuFc1W.rawPointer(), hostFc1W.data(), hostFc1W.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(gpuFc1B.rawPointer(), hostFc1B.data(), hostFc1B.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(gpuFc2W.rawPointer(), hostFc2W.data(), hostFc2W.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(gpuFc2B.rawPointer(), hostFc2B.data(), hostFc2B.size() * sizeof(half), cudaMemcpyHostToDevice));

        kernel::invokeTalkerMLP(
            cublasHandle, gpuInput, gpuFc1W, gpuFc1B, gpuFc2W, gpuFc2B, gpuOutput, gpuWorkspace, stream);

        std::vector<half> gpuResult(numTokens * outputDim);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), gpuOutput.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // For large-dim GEMM, allow a small fraction of outliers due to FP16 accumulation differences
        auto [rtol, atol] = getTolerance<half>();
        int32_t mismatches = 0;
        for (size_t i = 0; i < gpuResult.size(); ++i)
        {
            if (!isclose(gpuResult[i], refOutput[i], rtol, atol))
            {
                ++mismatches;
            }
        }
        EXPECT_LT(mismatches, std::max(1, static_cast<int32_t>(gpuResult.size() / 100)))
            << "Too many mismatches: " << mismatches << " / " << gpuResult.size();
    }
}

// ===== Gather / Scatter =====

TEST_F(TalkerKernelTest, GatherScatterRoundTrip)
{
    int64_t const srcTokens = 8;
    int64_t const hiddenDim = 64;
    int64_t const numIndices = 4;

    std::vector<half> hostSource(srcTokens * hiddenDim);
    uniformFloatInitialization(hostSource, -1.0f, 1.0f);
    std::vector<int32_t> hostIndices = {2, 5, 0, 7};

    rt::Coords srcShape{srcTokens, hiddenDim};
    rt::Coords idxShape{numIndices};
    rt::Coords gatherShape{numIndices, hiddenDim};
    rt::Coords scatterShape{srcTokens, hiddenDim};

    rt::Tensor gpuSource(srcShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuIndices(idxShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor gpuGatherOut(gatherShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuScatterOut(scatterShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(
        gpuSource.rawPointer(), hostSource.data(), hostSource.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpuIndices.rawPointer(), hostIndices.data(), hostIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpuScatterOut.rawPointer(), 0, srcTokens * hiddenDim * sizeof(half)));

    kernel::invokeGather(gpuSource, gpuIndices, gpuGatherOut, stream);
    kernel::invokeScatter(gpuGatherOut, gpuIndices, gpuScatterOut, stream);

    std::vector<half> scatterResult(srcTokens * hiddenDim);
    CUDA_CHECK(cudaMemcpy(
        scatterResult.data(), gpuScatterOut.rawPointer(), scatterResult.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t idx = 0; idx < numIndices; ++idx)
    {
        int32_t const srcRow = hostIndices[idx];
        for (int64_t d = 0; d < hiddenDim; ++d)
        {
            EXPECT_TRUE(isclose(scatterResult[srcRow * hiddenDim + d], hostSource[srcRow * hiddenDim + d], 0.f, 0.f))
                << "Gather-scatter round trip mismatch at row " << srcRow << " dim " << d;
        }
    }
}

// ===== SumReduceOverSequence =====

TEST_F(TalkerKernelTest, SumReduceOverSequence)
{
    int64_t const seqLen = 8;
    int64_t const hiddenDim = 64;

    std::vector<half> hostInput(seqLen * hiddenDim);
    uniformFloatInitialization(hostInput, -1.0f, 1.0f);

    std::vector<half> refOutput(hiddenDim);
    for (int64_t d = 0; d < hiddenDim; ++d)
    {
        float acc = 0.0f;
        for (int64_t s = 0; s < seqLen; ++s)
        {
            acc += __half2float(hostInput[s * hiddenDim + d]);
        }
        refOutput[d] = __float2half(acc);
    }

    rt::Coords inputShape{1, seqLen, hiddenDim};
    rt::Coords outputShape{1, 1, hiddenDim};

    rt::Tensor gpuInput(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuOutput(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(
        cudaMemcpy(gpuInput.rawPointer(), hostInput.data(), hostInput.size() * sizeof(half), cudaMemcpyHostToDevice));

    kernel::sumReduceOverSequence(gpuInput, gpuOutput, stream);

    std::vector<half> gpuResult(hiddenDim);
    CUDA_CHECK(
        cudaMemcpy(gpuResult.data(), gpuOutput.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();
    for (int64_t d = 0; d < hiddenDim; ++d)
    {
        EXPECT_TRUE(isclose(gpuResult[d], refOutput[d], rtol, atol))
            << "Sum reduce mismatch at dim " << d << ": gpu=" << __half2float(gpuResult[d])
            << ", ref=" << __half2float(refOutput[d]);
    }
}

// ===== ElementwiseAdd =====

TEST_F(TalkerKernelTest, ElementwiseAdd)
{
    int64_t const size = 256;
    std::vector<half> hostA(size), hostB(size);
    uniformFloatInitialization(hostA, -1.0f, 1.0f);
    uniformFloatInitialization(hostB, -1.0f, 1.0f);

    rt::Coords shape{size};
    rt::Tensor gpuA(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuB(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuOut(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(gpuA.rawPointer(), hostA.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuB.rawPointer(), hostB.data(), size * sizeof(half), cudaMemcpyHostToDevice));

    kernel::invokeElementwiseAdd(gpuOut, gpuA, gpuB, stream);

    std::vector<half> gpuResult(size);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), gpuOut.rawPointer(), size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i = 0; i < size; ++i)
    {
        half expected = __float2half(__half2float(hostA[i]) + __half2float(hostB[i]));
        EXPECT_TRUE(isclose(gpuResult[i], expected, 1e-3f, 1e-3f)) << "ElementwiseAdd mismatch at " << i;
    }
}

TEST_F(TalkerKernelTest, ElementwiseAddInplace)
{
    int64_t const size = 256;
    std::vector<half> hostData(size), hostAddend(size);
    uniformFloatInitialization(hostData, -1.0f, 1.0f);
    uniformFloatInitialization(hostAddend, -1.0f, 1.0f);

    rt::Coords shape{size};
    rt::Tensor gpuData(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuAddend(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(gpuData.rawPointer(), hostData.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuAddend.rawPointer(), hostAddend.data(), size * sizeof(half), cudaMemcpyHostToDevice));

    kernel::invokeElementwiseAddInplace(gpuData, gpuAddend, 0, 0, 0, stream);

    std::vector<half> gpuResult(size);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), gpuData.rawPointer(), size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i = 0; i < size; ++i)
    {
        half expected = __float2half(__half2float(hostData[i]) + __half2float(hostAddend[i]));
        EXPECT_TRUE(isclose(gpuResult[i], expected, 1e-3f, 1e-3f)) << "ElementwiseAddInplace mismatch at " << i;
    }
}

TEST_F(TalkerKernelTest, ElementwiseAddInplaceWithOffset)
{
    int64_t const totalSize = 512;
    int64_t const numElements = 128;
    int64_t const dataOffset = 64;
    int64_t const addendOffset = 32;

    std::vector<half> hostData(totalSize), hostAddend(totalSize);
    uniformFloatInitialization(hostData, -1.0f, 1.0f);
    uniformFloatInitialization(hostAddend, -1.0f, 1.0f);

    rt::Coords shape{totalSize};
    rt::Tensor gpuData(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuAddend(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(gpuData.rawPointer(), hostData.data(), totalSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuAddend.rawPointer(), hostAddend.data(), totalSize * sizeof(half), cudaMemcpyHostToDevice));

    kernel::invokeElementwiseAddInplace(gpuData, gpuAddend, numElements, dataOffset, addendOffset, stream);

    std::vector<half> gpuResult(totalSize);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), gpuData.rawPointer(), totalSize * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i = 0; i < numElements; ++i)
    {
        half expected
            = __float2half(__half2float(hostData[dataOffset + i]) + __half2float(hostAddend[addendOffset + i]));
        EXPECT_TRUE(isclose(gpuResult[dataOffset + i], expected, 1e-3f, 1e-3f)) << "Offset add mismatch at " << i;
    }

    for (int64_t i = 0; i < dataOffset; ++i)
    {
        EXPECT_TRUE(isclose(gpuResult[i], hostData[i], 0.f, 0.f)) << "Data before offset was modified at " << i;
    }
}

// ===== SuppressLogits =====

TEST_F(TalkerKernelTest, SuppressLogitsWithException)
{
    int32_t const vocabSize = 256;
    int32_t const suppressStart = 50;
    int32_t const suppressEnd = 150;
    int32_t const exceptTokenId = 100;

    std::vector<float> hostLogits(vocabSize, 1.0f);

    rt::Coords shape{1, static_cast<int64_t>(vocabSize)};
    rt::Tensor gpuLogits(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(
        cudaMemcpy(gpuLogits.rawPointer(), hostLogits.data(), vocabSize * sizeof(float), cudaMemcpyHostToDevice));

    kernel::invokeSuppressLogits(gpuLogits, suppressStart, suppressEnd, exceptTokenId, stream);

    std::vector<float> gpuResult(vocabSize);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), gpuLogits.rawPointer(), vocabSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < vocabSize; ++i)
    {
        if (i >= suppressStart && i < suppressEnd && i != exceptTokenId)
        {
            EXPECT_TRUE(std::isinf(gpuResult[i]) && gpuResult[i] < 0)
                << "Token " << i << " should be -inf, got " << gpuResult[i];
        }
        else
        {
            EXPECT_FLOAT_EQ(gpuResult[i], 1.0f) << "Token " << i << " should be unchanged";
        }
    }
}

TEST_F(TalkerKernelTest, SuppressLogitsNoException)
{
    int32_t const vocabSize = 128;
    int32_t const suppressStart = 0;
    int32_t const suppressEnd = 64;

    std::vector<float> hostLogits(vocabSize, 2.0f);

    rt::Coords shape{1, static_cast<int64_t>(vocabSize)};
    rt::Tensor gpuLogits(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(
        cudaMemcpy(gpuLogits.rawPointer(), hostLogits.data(), vocabSize * sizeof(float), cudaMemcpyHostToDevice));

    kernel::invokeSuppressLogits(gpuLogits, suppressStart, suppressEnd, -1, stream);

    std::vector<float> gpuResult(vocabSize);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), gpuLogits.rawPointer(), vocabSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < suppressEnd; ++i)
    {
        EXPECT_TRUE(std::isinf(gpuResult[i]) && gpuResult[i] < 0) << "Token " << i << " should be -inf";
    }
    for (int32_t i = suppressEnd; i < vocabSize; ++i)
    {
        EXPECT_FLOAT_EQ(gpuResult[i], 2.0f) << "Token " << i << " should be unchanged";
    }
}
