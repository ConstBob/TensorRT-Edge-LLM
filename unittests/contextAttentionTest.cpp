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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

void TestContextAttentionAccuracy(std::vector<int32_t> const& cuSeqlens, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t maxSeqLen, bool isPackedViT = false, bool causal = true,
    std::optional<float> attentionScale = std::nullopt)
{
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    bool const canImplement = isPackedViT ? CuteDslFMHAV2Runner::canImplementViT(headSize, smVersion, DataType::kHALF)
                                          : causal
            && CuteDslFMHAV2Runner::canImplement(
                numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL);
    ASSERT_TRUE(canImplement) << "FMHA-v2 CuTe DSL unexpectedly unsupported for headSize=" << headSize
                              << ", SM=" << smVersion;

    // Calculate total elements
    int32_t const batchSize = static_cast<int32_t>(cuSeqlens.size()) - 1;
    int32_t const totalTokens = isPackedViT ? cuSeqlens.back() : batchSize * maxSeqLen;

    size_t const qSize = static_cast<size_t>(totalTokens) * numQHeads * headSize;
    size_t const kvSize = static_cast<size_t>(totalTokens) * numKVHeads * headSize;
    size_t const outSize = static_cast<size_t>(totalTokens) * numQHeads * headSize;

    // Initialize input data
    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    // Create Tensor objects based on layout (they allocate device memory internally)
    rt::Tensor qTensor, kTensor, vTensor, oTensorRef, oTensorKernel;
    if (isPackedViT)
    {
        qTensor = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        kTensor = rt::Tensor({totalTokens, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        vTensor = rt::Tensor({totalTokens, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorRef = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorKernel = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    }
    else
    {
        qTensor = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        kTensor = rt::Tensor({batchSize, maxSeqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        vTensor = rt::Tensor({batchSize, maxSeqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorRef = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorKernel = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    }

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpy(
        cuSeqLensTensor.rawPointer(), cuSeqlens.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    rt::Tensor paddedCuKVSeqLensTensor;
    if (!isPackedViT)
    {
        std::vector<int32_t> inputSeqLens(batchSize);
        for (int32_t i = 0; i < batchSize; ++i)
        {
            inputSeqLens[i] = cuSeqlens[i + 1] - cuSeqlens[i];
        }

        rt::Tensor inputSeqLensTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(
            inputSeqLensTensor.rawPointer(), inputSeqLens.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor kvCacheEndIdxsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        paddedCuKVSeqLensTensor = rt::Tensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        kernel::calCuQCuKVSeqLensAndKVEndIdxs(inputSeqLensTensor, rt::Tensor{}, cuQSeqLensTensor, cuKVSeqLensTensor,
            kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, maxSeqLen, nullptr);
    }

    // Compute reference output
    cudaStream_t stream = nullptr;
    if (isPackedViT)
    {
        rt::launchFmhaReferenceCompact(
            qTensor, kTensor, vTensor, oTensorRef, cuSeqLensTensor, maxSeqLen, false, resolvedAttentionScale, stream);
    }
    else
    {
        rt::launchFmhaReferenceBshd(qTensor, kTensor, vTensor, oTensorRef, causal, resolvedAttentionScale, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy reference output to host
    std::vector<half> outReference(outSize);
    CUDA_CHECK(
        cudaMemcpy(outReference.data(), oTensorRef.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headSize, batchSize, maxSeqLen, maxSeqLen,
        /*useSmallD64=*/maxSeqLen <= 512);
    bool ranKernel = false;
    if (isPackedViT)
    {
        ASSERT_TRUE(CuteDslFMHAV2Runner::loadViTKernelModule());
        ranKernel = runner.run(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
            oTensorKernel.rawPointer(), cuSeqLensTensor.dataPointer<int32_t>(), totalTokens, maxSeqLen, batchSize,
            stream, resolvedAttentionScale);
    }
    else
    {
        ASSERT_TRUE(CuteDslFMHAV2Runner::loadLLMKernelModule());
        ranKernel = runner.run(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
            oTensorKernel.rawPointer(), paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, resolvedAttentionScale);
    }
    ASSERT_TRUE(ranKernel);
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
        if (std::isnan(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }

    float passRate1E_3 = static_cast<float>(numCloseWithin1E_3) / totalElements;

    std::string layoutStr = isPackedViT ? "[Compact]" : "[Padded]";
    std::string maskStr = causal ? "[Causal] " : "[Non-causal] ";
    std::cout << "Context Attention test. " << layoutStr << maskStr << "batch_size: " << batchSize;
    if (isPackedViT)
    {
        std::cout << " total_tokens: " << totalTokens << " max_seq_len: " << maxSeqLen;
    }
    else
    {
        std::cout << " seq_len: " << maxSeqLen;
    }
    std::cout << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;

    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

// Convenience wrapper for padded layout (fixed sequence length)
void TestContextAttentionAccuracy(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, bool causal = true, std::optional<float> attentionScale = std::nullopt)
{
    // Generate cu_seqlens for fixed-length sequences
    std::vector<int32_t> cuSeqlens(batchSize + 1);
    for (int32_t i = 0; i <= batchSize; i++)
    {
        cuSeqlens[i] = i * seqLen;
    }
    TestContextAttentionAccuracy(cuSeqlens, numQHeads, numKVHeads, headSize, seqLen, false, causal, attentionScale);
}

// Test cases with different head ratios (similar to XQA tests)

TEST(ContextAttentionTest, accuracyKVRatio1_Causal)
{
    // MHA: num_Q_heads == num_KV_heads
    TestContextAttentionAccuracy(1, 512, 8, 8, 128, true);
    TestContextAttentionAccuracy(2, 256, 16, 16, 64, true);
    TestContextAttentionAccuracy(4, 512, 4, 4, 128, true);
    TestContextAttentionAccuracy(1, 512, 4, 4, 256, true);
    TestContextAttentionAccuracy(1, 64, 4, 4, 256, true);
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
    TestContextAttentionAccuracy(1, 512, 16, 4, 256, true);
    TestContextAttentionAccuracy(2, 48, 16, 4, 256, true);
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
    TestContextAttentionAccuracy(2, 256, 16, 2, 256, true);
}

TEST(ContextAttentionTest, paddedLayout_VariableSequenceLengths)
{
    TestContextAttentionAccuracy({0, 64, 160}, 8, 2, 128, 128);
}

// Long sequence tests
TEST(ContextAttentionTest, longSequence_Causal)
{
    TestContextAttentionAccuracy(1, 1024, 12, 4, 128, true);
    TestContextAttentionAccuracy(1, 1024, 12, 2, 128, true);
    TestContextAttentionAccuracy(1, 2048, 24, 3, 64, true);
    TestContextAttentionAccuracy(1, 1024, 8, 2, 256, true);
}

// Convenience wrapper for compact layout (variable sequence lengths, non-causal)
void TestContextAttentionCompactAccuracy(std::vector<int32_t> const& cuSeqlens, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t maxSeqLen, std::optional<float> attentionScale = std::nullopt)
{
    TestContextAttentionAccuracy(cuSeqlens, numQHeads, numKVHeads, headSize, maxSeqLen, true, false, attentionScale);
}

TEST(ContextAttentionTest, compactLayout_NonCausal)
{
    // VIT attention with compact layout and variable sequence lengths (non-causal)
    TestContextAttentionCompactAccuracy({0, 32, 60, 88, 128}, 16, 16, 64, 128);
    TestContextAttentionCompactAccuracy({0, 16, 64}, 16, 16, 72, 128);
    TestContextAttentionCompactAccuracy({0, 100, 200, 300}, 8, 8, 80, 512);
}

TEST(ContextAttentionTest, configurableScale)
{
    float constexpr kIDENTITY_SCALE = 1.0F;
    float constexpr kCUSTOM_SCALE = 0.37F;

    TestContextAttentionAccuracy(1, 64, 8, 8, 64, true, kIDENTITY_SCALE);
    TestContextAttentionAccuracy(1, 64, 8, 2, 128, true, kCUSTOM_SCALE);
    TestContextAttentionAccuracy(1, 256, 8, 1, 256, true, kIDENTITY_SCALE);
    TestContextAttentionAccuracy(1, 256, 8, 1, 256, true, kCUSTOM_SCALE);
    TestContextAttentionCompactAccuracy({0, 16, 48}, 8, 8, 64, 48, kIDENTITY_SCALE);
    TestContextAttentionCompactAccuracy({0, 24, 64}, 8, 8, 80, 64, kCUSTOM_SCALE);
}

namespace
{

void fillNonCausalVarlenReference(std::vector<half> const& qInput, std::vector<half> const& kInput,
    std::vector<half> const& vInput, std::vector<half>& output, std::vector<int32_t> const& qLens,
    std::vector<int32_t> const& kvLens, int32_t seqLenQ, int32_t seqLenK, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize)
{
    int32_t const batchSize = static_cast<int32_t>(qLens.size());
    float const softmaxScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    std::fill(output.begin(), output.end(), __float2half(0.0F));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t qPos = 0; qPos < qLens[static_cast<size_t>(b)]; ++qPos)
        {
            for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
            {
                int32_t const kvHead = qHead * numKVHeads / numQHeads;
                float rowMax = -std::numeric_limits<float>::infinity();
                for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                {
                    float score = 0.0F;
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        size_t const qIdx
                            = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                        size_t const kIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        score += __half2float(qInput[qIdx]) * __half2float(kInput[kIdx]);
                    }
                    rowMax = std::max(rowMax, score * softmaxScale);
                }

                float rowSum = 0.0F;
                std::vector<float> weights(static_cast<size_t>(kvLens[static_cast<size_t>(b)]));
                for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                {
                    float score = 0.0F;
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        size_t const qIdx
                            = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                        size_t const kIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        score += __half2float(qInput[qIdx]) * __half2float(kInput[kIdx]);
                    }
                    float const weight = std::exp((score * softmaxScale) - rowMax);
                    weights[static_cast<size_t>(kPos)] = weight;
                    rowSum += weight;
                }

                for (int32_t d = 0; d < headSize; ++d)
                {
                    float acc = 0.0F;
                    for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                    {
                        size_t const vIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        acc += (weights[static_cast<size_t>(kPos)] / rowSum) * __half2float(vInput[vIdx]);
                    }
                    size_t const outIdx
                        = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                    output[outIdx] = __float2half(acc);
                }
            }
        }
    }
}

void TestContextAttentionDenoisePaddingVarlen(std::vector<int32_t> const& qLens, std::vector<int32_t> const& kvLens)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    int32_t constexpr headSize = 256;
    int32_t constexpr numQHeads = 16;
    int32_t constexpr numKVHeads = 8;
    int32_t const batchSize = static_cast<int32_t>(qLens.size());
    int32_t const seqLenQ = *std::max_element(qLens.begin(), qLens.end());
    int32_t const seqLenK = *std::max_element(kvLens.begin(), kvLens.end());

    if (!CuteDslFMHAV2Runner::canImplement(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING))
    {
        GTEST_SKIP() << "FMHA-v2 CuTe DSL PADDING is not supported for headSize=" << headSize << ", SM=" << smVersion;
    }

    size_t const qSize = static_cast<size_t>(batchSize) * seqLenQ * numQHeads * headSize;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLenK * numKVHeads * headSize;
    size_t const outSize = qSize;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);
    uniformFloatInitialization(qInput, -1.0F, 1.0F);
    uniformFloatInitialization(kInput, -1.0F, 1.0F);
    uniformFloatInitialization(vInput, -1.0F, 1.0F);

    std::vector<int32_t> cuQ(static_cast<size_t>(batchSize + 1), 0);
    std::vector<int32_t> cuKV(static_cast<size_t>(batchSize + 1), 0);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        cuQ[static_cast<size_t>(b + 1)] = cuQ[static_cast<size_t>(b)] + qLens[static_cast<size_t>(b)];
        cuKV[static_cast<size_t>(b + 1)] = cuKV[static_cast<size_t>(b)] + kvLens[static_cast<size_t>(b)];
    }

    rt::Tensor qTensor({batchSize, seqLenQ, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, seqLenK, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, seqLenK, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensor({batchSize, seqLenQ, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuQTensor.rawPointer(), cuQ.data(), static_cast<size_t>(batchSize + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuKVTensor.rawPointer(), cuKV.data(), static_cast<size_t>(batchSize + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    ASSERT_TRUE(CuteDslFMHAV2Runner::loadLLMKernelModule());
    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headSize, batchSize, seqLenQ, seqLenK);
    ASSERT_TRUE(runner.runPadding(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
        oTensor.rawPointer(), cuQTensor.dataPointer<int32_t>(), cuKVTensor.dataPointer<int32_t>(), nullptr,
        1.0F / std::sqrt(static_cast<float>(headSize))));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> expected(outSize);
    fillNonCausalVarlenReference(
        qInput, kInput, vInput, expected, qLens, kvLens, seqLenQ, seqLenK, numQHeads, numKVHeads, headSize);
    std::vector<half> actual(outSize);
    CUDA_CHECK(cudaMemcpy(actual.data(), oTensor.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t qPos = 0; qPos < qLens[static_cast<size_t>(b)]; ++qPos)
        {
            for (int32_t h = 0; h < numQHeads; ++h)
            {
                for (int32_t d = 0; d < headSize; ++d)
                {
                    size_t const idx = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + h) * headSize) + d;
                    ASSERT_TRUE(isclose(actual[idx], expected[idx], 1e-2F, 1e-2F))
                        << "DiffusionGemma denoise PADDING mismatch at b=" << b << " q=" << qPos << " h=" << h
                        << " d=" << d << " expected=" << __half2float(expected[idx])
                        << " actual=" << __half2float(actual[idx]);
                }
            }
        }
    }
}

} // namespace

TEST(ContextAttentionTest, diffusionGemmaDenoisePaddingVarlen)
{
    TestContextAttentionDenoisePaddingVarlen({4}, {24});
    TestContextAttentionDenoisePaddingVarlen({4, 3}, {24, 11});
}
