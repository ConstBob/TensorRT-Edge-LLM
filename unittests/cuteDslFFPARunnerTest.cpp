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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include "kernels/contextAttentionKernels/cuteDslFFPARunner.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

bool isFfpaSupportedSm(int32_t smVersion)
{
    return CuteDslFFPARunner::canImplement(512, smVersion);
}

void initializeFp16(rt::Tensor& tensor, int32_t seed)
{
    std::vector<__half> host(static_cast<size_t>(tensor.getShape().volume()));
    for (size_t index = 0; index < host.size(); ++index)
    {
        int32_t const value = (static_cast<int32_t>((index * 17 + seed * 13) % 67) - 33);
        host[index] = __float2half(static_cast<float>(value) * 0.0078125F);
    }
    copyHostToDevice(tensor, host);
}

void initializeNormalFp16(rt::Tensor& tensor, uint32_t seed)
{
    std::vector<__half> host(static_cast<size_t>(tensor.getShape().volume()));
    std::mt19937 rng{seed};
    // N(0, 1/sqrt(D)) keeps QK^T row variance unit-ish after the 1/sqrt(D) scale,
    // exercising the realistic softmax dynamic range rather than near-zero scores.
    std::normal_distribution<float> dist{0.0F, 1.0F / std::sqrt(512.0F)};
    for (size_t index = 0; index < host.size(); ++index)
    {
        host[index] = __float2half(dist(rng));
    }
    copyHostToDevice(tensor, host);
}

void expectFp16Close(rt::Tensor const& actual, rt::Tensor const& expected, std::string const& label)
{
    ASSERT_EQ(actual.getShape().volume(), expected.getShape().volume()) << label;
    auto const actualHost = copyDeviceToHost<__half>(actual);
    auto const expectedHost = copyDeviceToHost<__half>(expected);
    auto const& shape = actual.getShape();

    bool nanDetected = false;
    int64_t closeWithin1e3 = 0;
    int64_t const totalElements = static_cast<int64_t>(actualHost.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actualHost[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expectedHost[static_cast<size_t>(idx)]);

        ASSERT_FALSE(std::isnan(actualValue)) << label << " produced NaN at " << formatTensorIndex(shape, idx);
        ASSERT_FALSE(std::isinf(actualValue)) << label << " produced Inf at " << formatTensorIndex(shape, idx);

        ASSERT_TRUE(isclose(actualHost[static_cast<size_t>(idx)], expectedHost[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        if (isclose(actualHost[static_cast<size_t>(idx)], expectedHost[static_cast<size_t>(idx)], 1e-3F, 1e-3F))
        {
            ++closeWithin1e3;
        }
        nanDetected = nanDetected || std::isnan(actualValue);
    }

    float const passRate1e3 = static_cast<float>(closeWithin1e3) / static_cast<float>(totalElements);
    EXPECT_GT(passRate1e3, 0.9F) << label << " pass-rate at 1e-3 = " << passRate1e3;
    EXPECT_FALSE(nanDetected) << label;
}

struct ShapeParam
{
    int32_t batchSize;
    int32_t seqLen;
    int32_t numQHeads;
    int32_t numKVHeads; // == numQHeads for MHA; < numQHeads for GQA/MQA (must divide numQHeads)
    bool useNormalInit;
    char const* name;
};

void runAccuracyCase(ShapeParam const& p, cudaStream_t stream)
{
    int32_t constexpr kHeadDim = 512;
    // Q / O carry the full Q-head count; K / V carry the (possibly smaller) KV-head
    // count.  The FP32 BSHD reference maps q_head -> kv_head as q_head * Hkv / Hq,
    // which matches the kernel's q_head // (Hq / Hkv) when Hq % Hkv == 0.
    rt::Coords const qShape{p.batchSize, p.seqLen, p.numQHeads, kHeadDim};
    rt::Coords const kvShape{p.batchSize, p.seqLen, p.numKVHeads, kHeadDim};
    rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference(qShape, rt::DeviceType::kGPU, DataType::kHALF);

    if (p.useNormalInit)
    {
        initializeNormalFp16(q, 11);
        initializeNormalFp16(k, 29);
        initializeNormalFp16(v, 47);
    }
    else
    {
        initializeFp16(q, 11);
        initializeFp16(k, 29);
        initializeFp16(v, 47);
    }

    rt::launchFmhaReferenceBshd(q, k, v, outputReference, true, stream);

    CuteDslFFPAParams params;
    params.q = q.rawPointer();
    params.k = k.rawPointer();
    params.v = v.rawPointer();
    params.o = output.rawPointer();
    params.batchSize = p.batchSize;
    params.seqlenQ = p.seqLen;
    params.seqlenK = p.seqLen;
    params.numQHeads = p.numQHeads;
    params.numKVHeads = p.numKVHeads;
    params.headDim = kHeadDim;
    params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    ASSERT_EQ(CuteDslFFPARunner::run(params, stream), 0) << p.name;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectFp16Close(output, outputReference, p.name);
}

class CuteDslFFPABase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mSmVersion = getSMVersion();
        if (!isFfpaSupportedSm(mSmVersion))
        {
            GTEST_SKIP() << "FFPA d512 causal CuTe DSL kernel is not supported on SM" << mSmVersion;
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&mStream));
        if (!CuteDslFFPARunner::loadKernelModule())
        {
            CUDA_CHECK(cudaStreamDestroy(mStream));
            mStream = nullptr;
            GTEST_SKIP() << "Failed to load FFPA d512 causal CuTe DSL kernel module";
        }
    }

    void TearDown() override
    {
        CuteDslFFPARunner::unloadKernelModule();
        if (mStream != nullptr)
        {
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
    }

    cudaStream_t mStream{nullptr};
    int32_t mSmVersion{0};
};

class CuteDslFFPAAccuracySweep : public CuteDslFFPABase, public ::testing::WithParamInterface<ShapeParam>
{
};

TEST_P(CuteDslFFPAAccuracySweep, Causal)
{
    runAccuracyCase(GetParam(), mStream);
}

INSTANTIATE_TEST_SUITE_P(FP16Causal, CuteDslFFPAAccuracySweep,
    ::testing::Values(
        // --- MHA (numKVHeads == numQHeads) ---
        ShapeParam{1, 16, 1, 1, /*useNormalInit=*/false, "B1_S16_H1"},
        ShapeParam{2, 24, 2, 2, /*useNormalInit=*/false, "B2_S24_H2"},
        ShapeParam{1, 8, 1, 1, /*useNormalInit=*/true, "sub_Bc"},
        ShapeParam{1, 16, 4, 4, /*useNormalInit=*/true, "eq_Bc"},
        ShapeParam{1, 64, 8, 8, /*useNormalInit=*/true, "eq_Br"},
        ShapeParam{1, 128, 8, 8, /*useNormalInit=*/true, "multi_block_M"},
        ShapeParam{1, 130, 4, 4, /*useNormalInit=*/true, "multi_block_M_unaligned"},
        ShapeParam{1, 1024, 8, 8, /*useNormalInit=*/true, "llm_1k"},
        ShapeParam{1, 2048, 8, 8, /*useNormalInit=*/true, "llm_2k"},
        ShapeParam{4, 256, 4, 4, /*useNormalInit=*/true, "batch"},
        // --- GQA group size 4 (the primary target: H_q / H_kv == 4) ---
        ShapeParam{1, 128, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2"},
        ShapeParam{1, 1024, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2_1k"},
        ShapeParam{2, 256, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2_batch"},
        // --- Other GQA / MQA group sizes for coverage ---
        ShapeParam{1, 128, 8, 4, /*useNormalInit=*/true, "gqa_g2_H8_KV4"},
        ShapeParam{1, 256, 4, 1, /*useNormalInit=*/true, "mqa_g4_H4_KV1"},
        ShapeParam{1, 1024, 8, 1, /*useNormalInit=*/true, "mqa_H8_KV1_1k"}),
    [](::testing::TestParamInfo<ShapeParam> const& info) { return std::string{info.param.name}; });

class CuteDslFFPACausalProperty : public CuteDslFFPABase
{
protected:
    // Run the causal kernel for a B=1, H=numHeads prefix [0..seqLen) drawn from
    // the leading rows of a single full-length input buffer. The runner re-derives
    // strides from seqlenK, so reusing the full-S input pointer with a shorter
    // seqlenQ/seqlenK correctly addresses the contiguous leading-row prefix.
    rt::Tensor runPrefix(
        rt::Tensor const& fullQ, rt::Tensor const& fullK, rt::Tensor const& fullV, int32_t seqLen, int32_t numHeads)
    {
        int32_t constexpr kHeadDim = 512;
        rt::Coords const outShape{1, seqLen, numHeads, kHeadDim};
        rt::Tensor output(outShape, rt::DeviceType::kGPU, DataType::kHALF);

        CuteDslFFPAParams params;
        params.q = fullQ.rawPointer();
        params.k = fullK.rawPointer();
        params.v = fullV.rawPointer();
        params.o = output.rawPointer();
        params.batchSize = 1;
        params.seqlenQ = seqLen;
        params.seqlenK = seqLen;
        params.numQHeads = numHeads;
        params.numKVHeads = numHeads;
        params.headDim = kHeadDim;
        params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

        EXPECT_EQ(CuteDslFFPARunner::run(params, mStream), 0) << "prefix S=" << seqLen;
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        CUDA_CHECK(cudaGetLastError());
        return output;
    }
};

// Prefix invariance: causal attention's output at rows [0..k] depends only on
// Q[:k+1], K[:k+1], V[:k+1].  Running the kernel at S=16, 64, 128 on the same
// leading-row inputs must yield identical outputs for the rows they share.
// Picking 16/64/128 crosses the Br=64 tile boundary so we exercise both
// single-block and multi-block Q traversal.
TEST_F(CuteDslFFPACausalProperty, PrefixEquivalence)
{
    int32_t constexpr kHeadDim = 512;
    int32_t constexpr kNumHeads = 2;
    int32_t constexpr kFullSeq = 128;

    rt::Coords const fullShape{1, kFullSeq, kNumHeads, kHeadDim};
    rt::Tensor fullQ(fullShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor fullK(fullShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor fullV(fullShape, rt::DeviceType::kGPU, DataType::kHALF);

    initializeNormalFp16(fullQ, 101);
    initializeNormalFp16(fullK, 211);
    initializeNormalFp16(fullV, 307);

    rt::Tensor out128 = runPrefix(fullQ, fullK, fullV, 128, kNumHeads);
    rt::Tensor out64 = runPrefix(fullQ, fullK, fullV, 64, kNumHeads);
    rt::Tensor out16 = runPrefix(fullQ, fullK, fullV, 16, kNumHeads);

    auto const host128 = copyDeviceToHost<__half>(out128);
    auto const host64 = copyDeviceToHost<__half>(out64);
    auto const host16 = copyDeviceToHost<__half>(out16);

    int64_t const rowStride = static_cast<int64_t>(kNumHeads) * kHeadDim;

    // Rows [0..15] must match across all three (sub-Bc prefix; single Q block in all).
    for (int64_t row = 0; row < 16; ++row)
    {
        for (int64_t col = 0; col < rowStride; ++col)
        {
            int64_t const idx = row * rowStride + col;
            float const v128 = __half2float(host128[static_cast<size_t>(idx)]);
            float const v64 = __half2float(host64[static_cast<size_t>(idx)]);
            float const v16 = __half2float(host16[static_cast<size_t>(idx)]);
            ASSERT_TRUE(isclose(host128[static_cast<size_t>(idx)], host64[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..15] S=128 vs S=64 diverge at flat_index=" << idx << " row=" << row << " S=128=" << v128
                << " S=64=" << v64;
            ASSERT_TRUE(isclose(host64[static_cast<size_t>(idx)], host16[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..15] S=64 vs S=16 diverge at flat_index=" << idx << " row=" << row << " S=64=" << v64
                << " S=16=" << v16;
        }
    }

    // Rows [0..63] must match between S=64 and S=128 (crosses the Br=64 boundary).
    for (int64_t row = 0; row < 64; ++row)
    {
        for (int64_t col = 0; col < rowStride; ++col)
        {
            int64_t const idx = row * rowStride + col;
            float const v128 = __half2float(host128[static_cast<size_t>(idx)]);
            float const v64 = __half2float(host64[static_cast<size_t>(idx)]);
            ASSERT_TRUE(isclose(host128[static_cast<size_t>(idx)], host64[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..63] S=128 vs S=64 diverge at flat_index=" << idx << " row=" << row << " S=128=" << v128
                << " S=64=" << v64;
        }
    }
}

class CuteDslFFPANegativePath : public CuteDslFFPABase
{
};

// Runtime guard: params.headDim != 512 must be rejected with a non-zero return code,
// even though canImplement() filters this at the API entry.
TEST_F(CuteDslFFPANegativePath, RejectsUnsupportedHeadDim)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 128; // not supported — the kernel is D=512 only
    params.softmaxScale = 1.0F / std::sqrt(128.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: GQA is supported (num_kv_heads is a runtime kernel argument), but
// the Q heads must partition evenly across the K/V heads.  numQHeads % numKVHeads != 0
// (here 8 % 3) has no valid group size and must be rejected before any kernel launch.
// Note: numQHeads != numKVHeads on its own is *not* an error anymore — see the GQA
// cases in CuteDslFFPAAccuracySweep, which exercise the divisible path end-to-end.
TEST_F(CuteDslFFPANegativePath, RejectsIndivisibleKvHeads)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 8;
    params.numKVHeads = 3; // 8 % 3 != 0 — no integer GQA group size
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: a null Q pointer must be rejected before any kernel launch.
TEST_F(CuteDslFFPANegativePath, RejectsNullPointers)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = nullptr; // intentionally null
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

TEST(CuteDslFFPARunnerStaticTest, CanImplementSupportedHeadDimAndSMs)
{
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 80));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 86));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 87));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 89));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 100));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 101));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 110));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 120));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 121));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(128, 100));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, 75));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, 90));
}

} // namespace

#endif // defined(CUTE_DSL_FFPA_ENABLED)
