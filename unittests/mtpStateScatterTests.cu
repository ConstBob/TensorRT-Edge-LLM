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

#include "kernels/speculative/mtpStateScatterKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm::kernel;
using namespace trt_edgellm::rt;

// ============================================================================
// Helper: create a non-owning GPU Tensor from a device pointer
// ============================================================================

namespace
{

/// Fill host buffer with deterministic values: value = batch*10000 + step*100 + elemIdx%100
inline void fillRecurrentRef(std::vector<float>& buf, int32_t batch, int32_t maxSteps, int32_t stateElems)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t t = 0; t < maxSteps; ++t)
        {
            for (int32_t e = 0; e < stateElems; ++e)
            {
                int64_t idx = (static_cast<int64_t>(b) * maxSteps + t) * stateElems + e;
                buf[idx] = static_cast<float>(b * 10000 + t * 100 + (e % 100));
            }
        }
    }
}

constexpr float kSentinel = -999.0f;

} // anonymous namespace

// ============================================================================
// FP32 Recurrent State Scatter Tests
// ============================================================================

class MTPStateScatterRecurrentTest : public ::testing::Test
{
protected:
    // stateShape = trailing dims of dst, e.g. {hv, k, v} or {stateElems} for 1D
    void runTest(int32_t batchSize, int32_t maxSteps, std::vector<int64_t> const& stateShape,
        std::vector<int32_t> const& acceptedStepsHost)
    {
        ASSERT_EQ(static_cast<int32_t>(acceptedStepsHost.size()), batchSize);

        int64_t stateElems = 1;
        for (auto d : stateShape)
            stateElems *= d;
        ASSERT_EQ(stateElems % 8, 0) << "stateElems must be divisible by 8 for DVec<float>";

        int64_t const srcTotal = static_cast<int64_t>(batchSize) * maxSteps * stateElems;
        int64_t const dstTotal = static_cast<int64_t>(batchSize) * stateElems;

        // Prepare host data.
        std::vector<float> hSrc(srcTotal);
        fillRecurrentRef(hSrc, batchSize, maxSteps, static_cast<int32_t>(stateElems));
        std::vector<float> hDst(dstTotal, kSentinel);

        // Allocate device memory.
        float *dSrc, *dDst;
        int32_t* dAccepted;
        cudaMalloc(&dSrc, srcTotal * sizeof(float));
        cudaMalloc(&dDst, dstTotal * sizeof(float));
        cudaMalloc(&dAccepted, batchSize * sizeof(int32_t));

        cudaMemcpy(dSrc, hSrc.data(), srcTotal * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dDst, hDst.data(), dstTotal * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dAccepted, acceptedStepsHost.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice);

        // Build rt::Tensor wrappers (non-owning).
        // dst shape: [batchSize, ...stateShape]
        std::vector<int64_t> dstDims = {batchSize};
        dstDims.insert(dstDims.end(), stateShape.begin(), stateShape.end());
        nvinfer1::Dims dstTrtDims{};
        dstTrtDims.nbDims = static_cast<int32_t>(dstDims.size());
        for (int i = 0; i < dstTrtDims.nbDims; ++i)
            dstTrtDims.d[i] = dstDims[i];
        Tensor tensorDst(dDst, Coords(dstTrtDims), DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        // src shape: [batchSize, maxSteps, ...stateShape]
        std::vector<int64_t> srcDims = {batchSize, maxSteps};
        srcDims.insert(srcDims.end(), stateShape.begin(), stateShape.end());
        nvinfer1::Dims srcTrtDims{};
        srcTrtDims.nbDims = static_cast<int32_t>(srcDims.size());
        for (int i = 0; i < srcTrtDims.nbDims; ++i)
            srcTrtDims.d[i] = srcDims[i];
        Tensor tensorSrc(dSrc, Coords(srcTrtDims), DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        // acceptedSteps shape: [batchSize]
        nvinfer1::Dims accDims{};
        accDims.nbDims = 1;
        accDims.d[0] = batchSize;
        Tensor tensorAcc(dAccepted, Coords(accDims), DeviceType::kGPU, nvinfer1::DataType::kINT32);

        // Run kernel.
        mtpScatterRecurrentStates(tensorSrc, tensorAcc, tensorDst, nullptr);
        cudaDeviceSynchronize();

        // Read back.
        std::vector<float> hResult(dstTotal);
        cudaMemcpy(hResult.data(), dDst, dstTotal * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify.
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const step = acceptedStepsHost[b];
            int64_t const dstBase = static_cast<int64_t>(b) * stateElems;

            if (step < 0 || step >= maxSteps - 1)
            {
                // Should be untouched (sentinel).
                for (int64_t e = 0; e < stateElems; ++e)
                {
                    EXPECT_EQ(hResult[dstBase + e], kSentinel)
                        << "batch=" << b << " elem=" << e << " (step=" << step << ", should be untouched)";
                }
            }
            else
            {
                // Should match src[b, step, :].
                int64_t const srcBase = (static_cast<int64_t>(b) * maxSteps + step) * stateElems;
                for (int64_t e = 0; e < stateElems; ++e)
                {
                    EXPECT_EQ(hResult[dstBase + e], hSrc[srcBase + e])
                        << "batch=" << b << " step=" << step << " elem=" << e;
                }
            }
        }

        cudaFree(dSrc);
        cudaFree(dDst);
        cudaFree(dAccepted);
    }
};

// Basic: partial reject (accepted step 0 out of 4).
TEST_F(MTPStateScatterRecurrentTest, SingleBatch_PartialReject)
{
    runTest(/*batchSize=*/1, /*maxSteps=*/4, /*stateShape=*/{128}, /*accepted=*/{0});
}

// All accept: step == maxSteps-1, dst should remain untouched.
TEST_F(MTPStateScatterRecurrentTest, SingleBatch_AllAccept)
{
    runTest(1, 4, {128}, {3});
}

// Skip: step == -1, dst should remain untouched.
TEST_F(MTPStateScatterRecurrentTest, SingleBatch_Skip)
{
    runTest(1, 4, {128}, {-1});
}

// Mixed batch: some accept, some reject, some skip.
TEST_F(MTPStateScatterRecurrentTest, MixedBatch)
{
    //                                reject  all-accept  skip   partial
    runTest(/*batchSize=*/4, /*maxSteps=*/4, {256}, {1, 3, -1, 2});
}

// Multi-dim state shape: [hv=32, k=16, v=16] = 8192 elements (proxy for real 128*128).
TEST_F(MTPStateScatterRecurrentTest, MultiDimState)
{
    runTest(2, 2, {32, 16, 16}, {0, 1});
}

// Large state: tests vectorized path with stateElems >> blockDim (grid z > 1).
TEST_F(MTPStateScatterRecurrentTest, LargeState_GridZ)
{
    // stateElems=4096 → vecCount=512 → with 256 threads, zBlocks=2
    runTest(2, 4, {4096}, {1, 2});
}

// ============================================================================
// FP16 Conv State Scatter Tests
// ============================================================================

class MTPStateScatterConvTest : public ::testing::Test
{
protected:
    void runTest(int32_t batchSize, int32_t maxSteps, std::vector<int64_t> const& stateShape,
        std::vector<int32_t> const& acceptedStepsHost)
    {
        ASSERT_EQ(static_cast<int32_t>(acceptedStepsHost.size()), batchSize);

        int64_t stateElems = 1;
        for (auto d : stateShape)
            stateElems *= d;
        ASSERT_EQ(stateElems % 8, 0) << "stateElems must be divisible by 8 for DVec<half>";

        int64_t const srcTotal = static_cast<int64_t>(batchSize) * maxSteps * stateElems;
        int64_t const dstTotal = static_cast<int64_t>(batchSize) * stateElems;

        // Prepare host data.
        std::vector<__half> hSrc(srcTotal);
        for (int64_t i = 0; i < srcTotal; ++i)
            hSrc[i] = __float2half(static_cast<float>(i % 1000));

        __half const hSentinel = __float2half(-999.0f);
        std::vector<__half> hDst(dstTotal, hSentinel);

        // Allocate device memory.
        __half *dSrc, *dDst;
        int32_t* dAccepted;
        cudaMalloc(&dSrc, srcTotal * sizeof(__half));
        cudaMalloc(&dDst, dstTotal * sizeof(__half));
        cudaMalloc(&dAccepted, batchSize * sizeof(int32_t));

        cudaMemcpy(dSrc, hSrc.data(), srcTotal * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dDst, hDst.data(), dstTotal * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dAccepted, acceptedStepsHost.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice);

        // Build rt::Tensor wrappers.
        std::vector<int64_t> dstDims = {batchSize};
        dstDims.insert(dstDims.end(), stateShape.begin(), stateShape.end());
        nvinfer1::Dims dstTrtDims{};
        dstTrtDims.nbDims = static_cast<int32_t>(dstDims.size());
        for (int i = 0; i < dstTrtDims.nbDims; ++i)
            dstTrtDims.d[i] = dstDims[i];
        Tensor tensorDst(dDst, Coords(dstTrtDims), DeviceType::kGPU, nvinfer1::DataType::kHALF);

        std::vector<int64_t> srcDims = {batchSize, maxSteps};
        srcDims.insert(srcDims.end(), stateShape.begin(), stateShape.end());
        nvinfer1::Dims srcTrtDims{};
        srcTrtDims.nbDims = static_cast<int32_t>(srcDims.size());
        for (int i = 0; i < srcTrtDims.nbDims; ++i)
            srcTrtDims.d[i] = srcDims[i];
        Tensor tensorSrc(dSrc, Coords(srcTrtDims), DeviceType::kGPU, nvinfer1::DataType::kHALF);

        nvinfer1::Dims accDims{};
        accDims.nbDims = 1;
        accDims.d[0] = batchSize;
        Tensor tensorAcc(dAccepted, Coords(accDims), DeviceType::kGPU, nvinfer1::DataType::kINT32);

        // Run kernel.
        mtpScatterConvStates(tensorSrc, tensorAcc, tensorDst, nullptr);
        cudaDeviceSynchronize();

        // Read back.
        std::vector<__half> hResult(dstTotal);
        cudaMemcpy(hResult.data(), dDst, dstTotal * sizeof(__half), cudaMemcpyDeviceToHost);

        // Verify.
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const step = acceptedStepsHost[b];
            int64_t const dstBase = static_cast<int64_t>(b) * stateElems;

            if (step < 0 || step >= maxSteps - 1)
            {
                for (int64_t e = 0; e < stateElems; ++e)
                {
                    EXPECT_EQ(__half2float(hResult[dstBase + e]), __half2float(hSentinel))
                        << "batch=" << b << " elem=" << e;
                }
            }
            else
            {
                int64_t const srcBase = (static_cast<int64_t>(b) * maxSteps + step) * stateElems;
                for (int64_t e = 0; e < stateElems; ++e)
                {
                    EXPECT_EQ(__half2float(hResult[dstBase + e]), __half2float(hSrc[srcBase + e]))
                        << "batch=" << b << " step=" << step << " elem=" << e;
                }
            }
        }

        cudaFree(dSrc);
        cudaFree(dDst);
        cudaFree(dAccepted);
    }
};

// Conv1d: small config (dim=64, width=4 → stateElems=256).
TEST_F(MTPStateScatterConvTest, SmallConv_PartialReject)
{
    runTest(/*batchSize=*/2, /*maxSteps=*/4, /*stateShape=*/{64, 4}, /*accepted=*/{1, 0});
}

// Conv1d: all accept → no scatter.
TEST_F(MTPStateScatterConvTest, AllAccept)
{
    runTest(2, 4, {64, 4}, {3, 3});
}

// Conv1d: realistic dim=4096, width=4 → stateElems=16384.
TEST_F(MTPStateScatterConvTest, LargeDim)
{
    runTest(4, 2, {4096, 4}, {0, 1, 0, -1});
}
