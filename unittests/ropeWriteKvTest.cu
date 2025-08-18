/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/cudaUtils.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

using namespace drivellm;
using namespace drivellm::kernel;

struct AttnParams
{
    uint32_t numQHeads;
    uint32_t numKVHeads;
    uint32_t headDim;
    uint32_t rotaryDim;
};

void TestRopeWriteKvPrefill(uint32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qSeqLen, float ropeTheta = 10000.0f, int32_t cosSinCacheBatchSize = 1, int32_t cosSinCacheSeqLen = 0)
{
    cudaStream_t stream{nullptr};

    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const kvCacheVolume = batchSize * (numKVHeads + numKVHeads) * kvCacheCapacity * headDim;

    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    if (cosSinCacheSeqLen == 0)
    {
        cosSinCacheSeqLen = kvCacheCapacity;
    }
    int32_t const cosSinCacheVolume = cosSinCacheBatchSize * cosSinCacheSeqLen * rotaryDim;

    std::vector<half> qkvInput;
    std::vector<half> qkvReference;

    bool const permuteRope = true;
    float const ropeScale = 1.0f;
    thrust::device_vector<float> cosSinCacheDevice(cosSinCacheVolume);
    std::vector<float> cosSinCache(cosSinCacheVolume);
    bool const useRegularRope = cosSinCacheBatchSize == 1 && rotaryDim % 64 == 0;
    if (useRegularRope)
    {
        // Initialize normal CosSinCache to real values.
        initializeNormalRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()), ropeTheta, ropeScale, rotaryDim,
            kvCacheCapacity, stream);
    }
    else
    {
        // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatinitialization(cosSinCache, -1, 1);
        thrust::copy(cosSinCache.begin(), cosSinCache.end(), cosSinCacheDevice.begin());
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        for (int32_t j = 0; j < qSeqLen; j++)
        {
            std::vector<half> qij(numQHeads * headDim);
            std::vector<half> kij(numKVHeads * headDim);
            std::vector<half> vij(numKVHeads * headDim);

            uniformFloatinitialization(qij);
            uniformFloatinitialization(kij);
            uniformFloatinitialization(vij);
            // QKV input has layout of [B, S, H, D]

            qkvInput.insert(qkvInput.end(), qij.begin(), qij.end());
            qkvInput.insert(qkvInput.end(), kij.begin(), kij.end());
            qkvInput.insert(qkvInput.end(), vij.begin(), vij.end());

            std::vector<half> qRoped;
            std::vector<half> kRoped;
            if (useRegularRope)
            {
                qRoped = ropeRef(qij, numQHeads, headDim, rotaryDim, j, ropeScale, ropeTheta, permuteRope);
                kRoped = ropeRef(kij, numKVHeads, headDim, rotaryDim, j, ropeScale, ropeTheta, permuteRope);
            }
            else
            {
                // Calculate the correct batch index for cosSinCache
                int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : i;
                int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + j * rotaryDim;
                auto const cosVec = std::vector<float>(
                    cosSinCache.begin() + cosSinCacheOffset, cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2);
                auto const sinVec = std::vector<float>(cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2,
                    cosSinCache.begin() + cosSinCacheOffset + rotaryDim);

                qRoped = ropeRefCosSin(qij, numQHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                kRoped = ropeRefCosSin(kij, numKVHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
            }

            qkvReference.insert(qkvReference.end(), qRoped.begin(), qRoped.end());
            qkvReference.insert(qkvReference.end(), kRoped.begin(), kRoped.end());
            qkvReference.insert(qkvReference.end(), vij.begin(), vij.end());
        }
    }

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<half> kvCacheDevice(kvCacheVolume);

    int32_t const tokenToProcess = batchSize * qSeqLen;

    // Set qOut, kvCacheStartIds, tokenPosIds to nullptr since they are not used in prefill case.
    launchApplyRopeWriteKVContext(thrust::raw_pointer_cast(qkvDevice.data()),
        thrust::raw_pointer_cast(kvCacheDevice.data()), thrust::raw_pointer_cast(cosSinCacheDevice.data()), qSeqLen,
        tokenToProcess, kvCacheCapacity, numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize,
        cosSinCacheSeqLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::host_vector<half> qkvOut(qkvInput.size());
    thrust::host_vector<half> kvCacheOut(kvCacheVolume);
    thrust::copy(qkvDevice.begin(), qkvDevice.end(), qkvOut.begin());
    thrust::copy(kvCacheDevice.begin(), kvCacheDevice.end(), kvCacheOut.begin());

    KvCacheIndexer kvIndexer(batchSize, numKVHeads, kvCacheCapacity, headDim);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        int32_t const batchOffset = i * qSeqLen * (numQHeads + 2 * numKVHeads) * headDim;
        for (int32_t j = 0; j < qSeqLen; ++j)
        {
            int32_t const tokenOffset = j * (numQHeads + 2 * numKVHeads) * headDim;
            for (int32_t hq = 0; hq < numQHeads; ++hq)
            {
                int32_t const qOffset = batchOffset + tokenOffset + hq * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const qVal = qkvOut[qOffset + d];
                    half const qRefVal = qkvReference[qOffset + d];
                    ASSERT_TRUE(isclose(qVal, qRefVal, 1e-3, 1e-3));
                }
            }
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                int32_t const kOffset = batchOffset + tokenOffset + numQHeads * headDim + hkv * headDim;
                int32_t const vOffset
                    = batchOffset + tokenOffset + numQHeads * headDim + numKVHeads * headDim + hkv * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const kVal = qkvOut[kOffset + d];
                    half const kCacheVal = kvCacheOut[kvIndexer.indexK(i, hkv, j, d)];
                    half const kRefVal = qkvReference[kOffset + d];
                    half const vVal = qkvOut[vOffset + d];
                    half const vCacheVal = kvCacheOut[kvIndexer.indexV(i, hkv, j, d)];
                    half const vRefVal = qkvReference[vOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(kCacheVal, kVal, 1e-5, 1e-5));
                    ASSERT_TRUE(isclose(vCacheVal, vVal, 1e-5, 1e-5));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvPrefill "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize
              << " cosSinCacheSeqLen: " << cosSinCacheSeqLen << std::endl;
}

void TestRopeWriteKvDecode(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qLen, float ropeTheta = 10000.0f, bool const isTreeAttention = false,
    int32_t cosSinCacheBatchSize = 1, int32_t cosSinCacheSeqLen = 0)
{
    // Not tested for MROPE which supply positional encoding coefficients as input tensor.
    EXPECT_TRUE(qLen == 1 || isTreeAttention);
    cudaStream_t stream{nullptr};

    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;

    // QKV tensor has layout [B, S, Hq+Hk+Hv, D]. KV cache has layout [B, 2, S, Hkv, D].
    std::vector<half> qkvInput;
    std::vector<half> kvCache(batchSize * 2 * numKVHeads * kvCacheCapacity * headDim, 0);
    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    if (cosSinCacheSeqLen == 0)
    {
        cosSinCacheSeqLen = kvCacheCapacity;
    }
    int32_t const cosSinCacheVolume = cosSinCacheBatchSize * cosSinCacheSeqLen * rotaryDim;

    // Reference output of Q, K, V all have layout [B, S, H, D].
    std::vector<half> qreference;
    std::vector<half> kreference;
    std::vector<half> vreference;

    // Random initialized the total length which is committed kv-cache length + new tokens length.
    std::vector<int32_t> fullSeqLens(batchSize);
    uniformIntInitialization(fullSeqLens, kvCacheCapacity / 4, kvCacheCapacity);
    std::vector<int32_t> customSeqLens;

    bool const permuteRope = true;
    float const ropeScale = 1.0f;
    thrust::device_vector<float> cosSinCacheDevice(cosSinCacheVolume);
    std::vector<float> cosSinCache(cosSinCacheVolume);
    bool const useRegularRope = cosSinCacheBatchSize == 1 && rotaryDim % 64 == 0;
    if (useRegularRope)
    {
        // Initialize normal CosSinCache to real values.
        initializeNormalRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()), ropeTheta, ropeScale, rotaryDim,
            kvCacheCapacity, stream);
    }
    else
    {
        // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatinitialization(cosSinCache, -1, 1);
        thrust::copy(cosSinCache.begin(), cosSinCache.end(), cosSinCacheDevice.begin());
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        int32_t const qStartIdx = fullSeqLens[i] - qLen;
        // With speculative decoding, the sequence index is not identical to kvcache index.
        std::vector<int32_t> customSeqLen(qLen);
        uniformIntInitialization(customSeqLen, qStartIdx, qStartIdx + qLen - 1);
        customSeqLens.insert(customSeqLens.end(), customSeqLen.begin(), customSeqLen.end());

        for (int32_t j = 0; j < qLen; j++)
        {
            std::vector<half> qi(numQHeads * headDim);
            std::vector<half> ki(numKVHeads * headDim);
            std::vector<half> vi(numKVHeads * headDim);

            uniformFloatinitialization(qi);
            uniformFloatinitialization(ki);
            uniformFloatinitialization(vi);

            qkvInput.insert(qkvInput.end(), qi.begin(), qi.end());
            qkvInput.insert(qkvInput.end(), ki.begin(), ki.end());
            qkvInput.insert(qkvInput.end(), vi.begin(), vi.end());

            int32_t seqIdx = qStartIdx + j;
            if (isTreeAttention)
            {
                // Pick custom sequence index if tree attention is enabled.
                seqIdx = customSeqLen[j];
            }

            std::vector<half> qRefij;
            std::vector<half> kRefij;
            if (useRegularRope)
            {
                qRefij = ropeRef(qi, numQHeads, headDim, rotaryDim, seqIdx, ropeScale, ropeTheta, permuteRope);
                kRefij = ropeRef(ki, numKVHeads, headDim, rotaryDim, seqIdx, ropeScale, ropeTheta, permuteRope);
            }
            else
            {
                // Calculate the correct batch index for cosSinCache
                int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : i;
                int32_t const cosSinCacheOffset
                    = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + seqIdx * rotaryDim;

                auto const cosVec = std::vector<float>(
                    cosSinCache.begin() + cosSinCacheOffset, cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2);
                auto const sinVec = std::vector<float>(cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2,
                    cosSinCache.begin() + cosSinCacheOffset + rotaryDim);

                qRefij = ropeRefCosSin(qi, numQHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                kRefij = ropeRefCosSin(ki, numKVHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
            }

            qreference.insert(qreference.end(), qRefij.begin(), qRefij.end());
            kreference.insert(kreference.end(), kRefij.begin(), kRefij.end());
            vreference.insert(vreference.end(), vi.begin(), vi.end());
        }
    }

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<half> qOutDevice(batchSize * qLen * numQHeads * headDim);
    thrust::device_vector<half> kvCacheDevice(kvCache);
    thrust::device_vector<int32_t> seqLensDevice(fullSeqLens);
    thrust::device_vector<int32_t> customSeqLensDevice(customSeqLens);

    int32_t const tokenToProcess = batchSize * qLen;
    if (!isTreeAttention)
    {
        launchApplyRopeWriteKVDecode(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(kvCacheDevice.data()), thrust::raw_pointer_cast(qOutDevice.data()),
            thrust::raw_pointer_cast(cosSinCacheDevice.data()), thrust::raw_pointer_cast(seqLensDevice.data()), qLen,
            tokenToProcess, kvCacheCapacity, numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize,
            cosSinCacheSeqLen, stream);
    }
    else
    {
        launchApplyRopeWriteKVTreeDecode(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(kvCacheDevice.data()), thrust::raw_pointer_cast(qOutDevice.data()),
            thrust::raw_pointer_cast(cosSinCacheDevice.data()), thrust::raw_pointer_cast(seqLensDevice.data()),
            thrust::raw_pointer_cast(customSeqLensDevice.data()), qLen, tokenToProcess, kvCacheCapacity, numQHeads,
            numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    thrust::host_vector<half> qOut(batchSize * qLen * numQHeads * headDim);
    thrust::copy(qOutDevice.begin(), qOutDevice.end(), qOut.begin());
    thrust::host_vector<half> kvCacheOut(kvCache.size());
    thrust::copy(kvCacheDevice.begin(), kvCacheDevice.end(), kvCacheOut.begin());

    // Directly compare the output of Q since output and reference have the same layout.
    EXPECT_EQ(qOut.size(), qreference.size());
    for (int32_t i = 0; i < qOut.size(); ++i)
    {
        ASSERT_TRUE(isclose(qOut[i], qreference[i], 1e-3, 4e-3));
    }

    KvCacheIndexer kvIndexer(batchSize, numKVHeads, kvCacheCapacity, headDim);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const qStartIdx = fullSeqLens[b] - qLen;
        for (int32_t s = 0; s < qLen; ++s)
        {
            int32_t const inCacheIdx = qStartIdx + s;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                int32_t const kvRefOffset = b * qLen * numKVHeads * headDim + s * numKVHeads * headDim + hkv * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const kVal = kvCacheOut[kvIndexer.indexK(b, hkv, inCacheIdx, d)];
                    half const kRefVal = kreference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 4e-3));
                    half const vVal = kvCacheOut[kvIndexer.indexV(b, hkv, inCacheIdx, d)];
                    half const vRefVal = vreference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 4e-3));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvDecode "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " QLength: " << qLen << " Total Sequence Lengths (including past KVcache): " << fullSeqLens
              << " RopeScale: " << ropeScale << " RopeTheta: " << ropeTheta
              << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << " cosSinCacheSeqLen: " << cosSinCacheSeqLen
              << std::endl;
}

void BenchmarkRopeWriteKv(
    uint32_t const batchSize, AttnParams const& attnParams, int32_t const qSeqLen, int32_t cosSinCacheBatchSize = 1)
{
    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const kvCacheCapacity = 1024 + qSeqLen;

    std::vector<half> qkvInput(batchSize * qSeqLen * (numQHeads + 2 * numKVHeads) * headDim);
    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    std::vector<float> cosSinCache(cosSinCacheBatchSize * kvCacheCapacity * rotaryDim);

    uniformFloatinitialization(cosSinCache, -1, 1);
    uniformFloatinitialization(qkvInput);

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<float> cosSinCacheDevice(cosSinCache);
    thrust::device_vector<half> kvCacheDevice(batchSize * (numKVHeads + numKVHeads) * kvCacheCapacity * headDim);

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * qSeqLen;

    auto launchPrefill = [&]() {
        launchApplyRopeWriteKV(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(kvCacheDevice.data()), nullptr, thrust::raw_pointer_cast(cosSinCacheDevice.data()),
            nullptr, nullptr, qSeqLen, tokenToProcess, kvCacheCapacity, numQHeads, numKVHeads, headDim, rotaryDim,
            cosSinCacheBatchSize, kvCacheCapacity, stream);
    };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launchPrefill();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchPrefill();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Bench Perf: BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " qSeqLen: " << qSeqLen
              << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << std::endl;
    std::cout << "RopeWriteKv(non-interleave) time: " << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(RopeWriteKvPrefill, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(1, {32, 8, 128, 128}, 2048, 512);
    // QheadNum = 24, kvHeadNum = 3, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvPrefill(2, {24, 3, 128, 128}, 4096, 512);
    // QheadNum = 28, kvHeadNum = 7, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(1, {28, 7, 128, 128}, 2048, 512);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, rotaryDim = 64, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(4, {16, 4, 64, 64}, 2048, 512);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvPrefill(2, {24, 8, 128, 96}, 4096, 512);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512,
    // cosSinCacheBatchSize = 2, cosSinCacheSeqLen = 8192
    TestRopeWriteKvPrefill(2, {24, 8, 128, 96}, 4096, 512, 10000.0f, 2, 8192);
}

TEST(RopeWriteKvDecodeVanilla, Accuracy)
{
    // qHeadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 1, 10000.0f, false);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {28, 4, 128, 128}, 4096, 1, 500000.0f, false);
    // QheadNum = 16, kvHeadNum = 2, headSize = 64, rotaryDim = 64, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false
    TestRopeWriteKvDecode(1, {16, 2, 64, 64}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {24, 4, 128, 128}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false, cosSinCacheBatchSize = 2, cosSinCacheSeqLen = 8192
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 1, 10000.0f, false, 2, 8192);
}

TEST(RopeWriteKvDecodeTreeAttention, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 4, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 4, 10000.0f, true);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 32, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {28, 4, 128, 128}, 4096, 32, 500000.0f, true);
    // QheadNum = 24, kvHeadNum = 6, headSize = 64, rotaryDim = 64, kvCacheCapacity = 4096, qLen = 64, isTreeAttention =
    // true
    TestRopeWriteKvDecode(1, {24, 6, 64, 64}, 4096, 64, 10000.0f, true);
    // QheadNum = 16, kvHeadNum = 2, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 50, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {16, 2, 128, 128}, 4096, 50, 10000.0f, true);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512, isTreeAttention
    // = true
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 32, 10000.0f, true);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512, isTreeAttention
    // = true, cosSinCacheBatchSize = 2, cosSinCacheSeqLen = 8192
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 32, 10000.0f, true, 2, 8192);
}

TEST(RopeWriteKvPrefill, Benchmark)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 1024
    BenchmarkRopeWriteKv(1, {32, 8, 128, 128}, 1024);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 2048
    BenchmarkRopeWriteKv(2, {24, 3, 128, 128}, 2048);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 4096
    BenchmarkRopeWriteKv(1, {28, 7, 128, 128}, 4096);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, rotaryDim = 64, qLen = 1024
    BenchmarkRopeWriteKv(4, {16, 4, 64, 64}, 1024);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 512, cosSinCacheBatchSize = 2
    BenchmarkRopeWriteKv(2, {32, 8, 128, 128}, 512, 2);
}

void TestLongRopeCosSin(int32_t rotaryDim, int32_t kvCacheCapacity, int32_t maxPositionEmbeddings = 131072,
    int32_t originalMaxPositionEmbeddings = 4096, float rotaryBaseFrequency = 10000.0f)
{
    // Generate random extension factors
    std::vector<float> shortReference(kvCacheCapacity * rotaryDim);
    std::vector<float> longReference(kvCacheCapacity * rotaryDim);
    std::vector<float> shortFactor(rotaryDim / 2, 1.0f);
    std::vector<float> longFactor(rotaryDim / 2);
    uniformFloatinitialization(longFactor, 1.0f, float(rotaryDim / 2 - 1));

    computeLongRopeReference(shortReference, longReference, shortFactor, longFactor, rotaryBaseFrequency, rotaryDim,
        kvCacheCapacity, maxPositionEmbeddings, originalMaxPositionEmbeddings);

    // Allocate device memory
    thrust::device_vector<float> shortCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> longCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> shortFactorDevice(shortFactor);
    thrust::device_vector<float> longFactorDevice(longFactor);

    cudaStream_t stream{nullptr};

    // Launch kernel
    initializeLongRopeCosSin(thrust::raw_pointer_cast(shortCosSinCacheDevice.data()),
        thrust::raw_pointer_cast(longCosSinCacheDevice.data()), thrust::raw_pointer_cast(shortFactorDevice.data()),
        thrust::raw_pointer_cast(longFactorDevice.data()), rotaryBaseFrequency, rotaryDim, kvCacheCapacity,
        maxPositionEmbeddings, originalMaxPositionEmbeddings, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back to host
    thrust::host_vector<float> shortCosSinCacheHost(shortCosSinCacheDevice);
    thrust::host_vector<float> longCosSinCacheHost(longCosSinCacheDevice);

    // Verify short cache results
    for (int32_t i = 0; i < kvCacheCapacity * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(shortCosSinCacheHost[i], shortReference[i], 1e-3, 1e-3))
            << "Short cache mismatch at index " << i << ": got " << shortCosSinCacheHost[i] << ", expected "
            << shortReference[i];
    }

    // Verify long cache results
    for (int32_t i = 0; i < kvCacheCapacity * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(longCosSinCacheHost[i], longReference[i], 1e-3, 1e-3))
            << "Long cache mismatch at index " << i << ": got " << longCosSinCacheHost[i] << ", expected "
            << longReference[i];
    }

    std::cout << "TestLongRopeCosSin passed: rotaryDim=" << rotaryDim << ", kvCacheCapacity=" << kvCacheCapacity
              << ", maxPositionEmbeddings=" << maxPositionEmbeddings
              << ", originalMaxPositionEmbeddings=" << originalMaxPositionEmbeddings
              << ", rotaryBaseFrequency=" << rotaryBaseFrequency << std::endl;
}

void BenchmarkLongRopeCosSin(int32_t rotaryDim, int32_t kvCacheCapacity, int32_t maxPositionEmbeddings = 131072,
    int32_t originalMaxPositionEmbeddings = 4096)
{
    std::vector<float> shortFactor(rotaryDim / 2, 1.0f);
    std::vector<float> longFactor(rotaryDim / 2);
    uniformFloatinitialization(longFactor, 1.0f, float(rotaryDim / 2 - 1));

    thrust::device_vector<float> shortCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> longCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> shortFactorDevice(shortFactor);
    thrust::device_vector<float> longFactorDevice(longFactor);

    cudaStream_t stream{nullptr};

    auto launch = [&]() {
        initializeLongRopeCosSin(thrust::raw_pointer_cast(shortCosSinCacheDevice.data()),
            thrust::raw_pointer_cast(longCosSinCacheDevice.data()), thrust::raw_pointer_cast(shortFactorDevice.data()),
            thrust::raw_pointer_cast(longFactorDevice.data()), 10000.0f, rotaryDim, kvCacheCapacity,
            maxPositionEmbeddings, originalMaxPositionEmbeddings, stream);
    };

    // Warmup
    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "LongRopeCosSin Benchmark: rotaryDim=" << rotaryDim << ", kvCacheCapacity=" << kvCacheCapacity
              << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(InitializeLongRopeCosSin, Accuracy)
{
    TestLongRopeCosSin(96, 8192);
    TestLongRopeCosSin(128, 4096);
}

TEST(InitializeLongRopeCosSin, Benchmark)
{
    BenchmarkLongRopeCosSin(96, 8192);
    BenchmarkLongRopeCosSin(128, 4096);
}

void TestMRopeCosSin(
    int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize, float rotaryBaseFrequency = 10000.0f)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    std::vector<float> reference(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    computeMRopeReference(
        reference, mropePositionIds, rotaryBaseFrequency, rotaryDim, rotaryEmbeddingMaxPositions, batchSize);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    // Launch kernel
    initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
        thrust::raw_pointer_cast(mropePositionIdsDevice.data()), rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back to host
    thrust::host_vector<float> cosSinCacheHost(cosSinCacheDevice);

    // Verify results
    for (int32_t i = 0; i < batchSize * rotaryEmbeddingMaxPositions * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(cosSinCacheHost[i], reference[i], 1e-3, 1e-3))
            << "MRope cache mismatch at index " << i << ": got " << cosSinCacheHost[i] << ", expected " << reference[i];
    }

    std::cout << "TestMRopeCosSin passed: rotaryDim=" << rotaryDim
              << ", rotaryEmbeddingMaxPositions=" << rotaryEmbeddingMaxPositions << ", batchSize=" << batchSize
              << ", rotaryBaseFrequency=" << rotaryBaseFrequency << std::endl;
}

void BenchmarkMRopeCosSin(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    auto launch = [&]() {
        initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
            thrust::raw_pointer_cast(mropePositionIdsDevice.data()), 10000.0f, rotaryDim, rotaryEmbeddingMaxPositions,
            batchSize, stream);
    };

    // Warmup
    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "MRopeCosSin Benchmark: rotaryDim=" << rotaryDim
              << ", rotaryEmbeddingMaxPositions=" << rotaryEmbeddingMaxPositions << ", batchSize=" << batchSize
              << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(InitializeMRopeCosSin, Accuracy)
{
    TestMRopeCosSin(128, 4096, 2);
    TestMRopeCosSin(128, 8192, 1);
}

TEST(InitializeMRopeCosSin, Benchmark)
{
    BenchmarkMRopeCosSin(128, 4096, 2);
    BenchmarkMRopeCosSin(128, 8192, 1);
}
