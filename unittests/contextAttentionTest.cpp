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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

void TestContextAttentionAccuracy(
    int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool causal = true)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    // Check if context FMHA is supported for this configuration
    if (!ContextFMHARunner::canImplement(headSize, smVersion, DataType::kHALF))
    {
        GTEST_SKIP() << "Context FMHA not supported for headSize=" << headSize << ", SM=" << smVersion;
    }

    // Calculate total elements
    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headSize;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLen * numKVHeads * headSize;
    size_t const outSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headSize;

    // Initialize input data in BSHD layout: [B, S, H, D]
    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    // Create Tensor objects (they allocate device memory internally)
    rt::Tensor qTensor({batchSize, seqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, seqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, seqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorRef({batchSize, seqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorKernel({batchSize, seqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));

    // Compute reference output using the BSHD reference kernel
    cudaStream_t stream = nullptr;
    rt::launchFmhaReferenceBshd(qTensor, kTensor, vTensor, oTensorRef, causal, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy reference output to host
    std::vector<half> outReference(outSize);
    CUDA_CHECK(
        cudaMemcpy(outReference.data(), oTensorRef.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    // Simple case for batches of fixed length sequences.
    // TODO: update to take ragged layout with variable sequence lengths.
    std::vector<int32_t> cuSeqLens(batchSize + 1);
    for (int32_t i = 0; i <= batchSize; i++)
    {
        cuSeqLens[i] = i * seqLen;
    }

    // Create Tensor for cu_seqlens and copy data
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpy(
        cuSeqLensTensor.rawPointer(), cuSeqLens.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Load context FMHA kernels
    EXPECT_TRUE(ContextFMHARunner::loadContextFMHAKernels(smVersion, DataType::kHALF));

    // Create context FMHA runner with SEPARATE_Q_K_V layout
    ContextFMHARunner runner(DataType::kHALF, batchSize, seqLen, numQHeads, numKVHeads, headSize, smVersion,
        AttentionInputLayout::SEPARATE_Q_K_V);

    // Setup parameters
    FusedMultiheadAttentionParamsV2 params;
    runner.setupParams(params);

    // Set device pointers
    params.s_kv = seqLen;
    params.q_ptr = qTensor.rawPointer();
    params.k_ptr = kTensor.rawPointer();
    params.v_ptr = vTensor.rawPointer();
    params.o_ptr = oTensorKernel.rawPointer();
    params.cu_q_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
    params.cu_kv_seqlens = cuSeqLensTensor.dataPointer<int32_t>();

    // Dispatch kernel
    runner.dispatchFMHAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy output to host
    std::vector<half> outHost(outSize);
    CUDA_CHECK(cudaMemcpy(outHost.data(), oTensorKernel.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    // Check accuracy
    bool NanValueDetected = false;
    int32_t numCloseWithin1E_3 = 0;
    int64_t totalElements = static_cast<int64_t>(outSize);

    for (int64_t i = 0; i < totalElements; ++i)
    {
        ASSERT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2))
            << "Mismatch at index=" << i << " expected=" << __half2float(outReference[i])
            << " actual=" << __half2float(outHost[i]);

        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numCloseWithin1E_3++;
        }
        if (__hisnan(outHost[i]))
        {
            NanValueDetected = true;
        }
    }

    float passRate1E_3 = static_cast<float>(numCloseWithin1E_3) / totalElements;

    std::cout << "Context Attention test. " << (causal ? "[Causal] " : "[Non-causal] ") << "batch_size: " << batchSize
              << " seq_len: " << seqLen << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads
              << " head_size: " << headSize << " pass_rate_1e-3: " << passRate1E_3 << std::endl;

    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

// Test cases with different head ratios (similar to XQA tests)

TEST(ContextAttentionTest, accuracyKVRatio1_Causal)
{
    // MHA: num_Q_heads == num_KV_heads
    TestContextAttentionAccuracy(1, 512, 8, 8, 128, true);
    TestContextAttentionAccuracy(2, 256, 16, 16, 64, true);
    TestContextAttentionAccuracy(4, 512, 4, 4, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio3_Causal)
{
    // GQA with ratio 3
    TestContextAttentionAccuracy(1, 512, 24, 8, 64, true);
    TestContextAttentionAccuracy(4, 512, 12, 4, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio4_Causal)
{
    // GQA with ratio 4
    TestContextAttentionAccuracy(1, 132, 32, 8, 64, true);
    TestContextAttentionAccuracy(2, 260, 32, 8, 128, true);
    TestContextAttentionAccuracy(4, 520, 16, 4, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio7_Causal)
{
    // GQA with ratio 7
    TestContextAttentionAccuracy(1, 784, 28, 4, 64, true);
    TestContextAttentionAccuracy(2, 512, 14, 2, 128, true);
    TestContextAttentionAccuracy(4, 256, 14, 2, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio8_Causal)
{
    // GQA with ratio 8
    TestContextAttentionAccuracy(1, 128, 32, 4, 64, true);
    TestContextAttentionAccuracy(2, 256, 16, 2, 128, true);
    TestContextAttentionAccuracy(4, 512, 16, 2, 128, true);
}

// Long sequence tests
TEST(ContextAttentionTest, longSequence_Causal)
{
    TestContextAttentionAccuracy(1, 1024, 12, 4, 128, true);
    TestContextAttentionAccuracy(1, 1024, 12, 2, 128, true);
    TestContextAttentionAccuracy(1, 2048, 24, 3, 64, true);
}
