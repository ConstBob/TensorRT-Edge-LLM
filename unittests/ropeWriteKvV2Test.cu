#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "attentionPlugin/posEnc/applyRopeWriteKV.h"
#include "common/cudaUtils.h"
#include "references.h"
#include "testUtils.h"

struct AttnParams
{
    uint32_t numQHeads;
    uint32_t numKVHeads;
    uint32_t headDim;
};

void TestRopeWriteKvV2Prefill(uint32_t const batchSize, AttnParams const& attnParams, int64_t const kvCacheCapacity,
    int64_t const qSeqLen, bool const useInterleaveRope)
{
    uint32_t const headDim = attnParams.headDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int64_t const kvCacheVolume = batchSize * (numKVHeads + numKVHeads) * kvCacheCapacity * headDim;

    std::vector<half> qkvInput;
    std::vector<half> qkvReference;
    std::vector<float> cosSinCache(kvCacheCapacity * headDim);

    // Randomly initialize the cosSinCache instead of explicitly compute them.
    // Compute reference based on random input and cosSinCache.
    uniformFloatinitialization(cosSinCache, -1, 1);
    bool const permuteRope = !useInterleaveRope;
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

            auto const cosVec = std::vector<float>(
                cosSinCache.begin() + j * headDim, cosSinCache.begin() + j * headDim + headDim / 2);
            auto const sinVec = std::vector<float>(
                cosSinCache.begin() + j * headDim + headDim / 2, cosSinCache.begin() + j * headDim + headDim);

            auto qRoped = ropeRefCosSin(qij, numQHeads, headDim, cosVec, sinVec, permuteRope);
            auto kRoped = ropeRefCosSin(kij, numKVHeads, headDim, cosVec, sinVec, permuteRope);

            qkvReference.insert(qkvReference.end(), qRoped.begin(), qRoped.end());
            qkvReference.insert(qkvReference.end(), kRoped.begin(), kRoped.end());
            qkvReference.insert(qkvReference.end(), vij.begin(), vij.end());
        }
    }

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<half> kvCacheDevice(kvCacheVolume);
    thrust::device_vector<float> cosSinCacheDevice(cosSinCache);

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * qSeqLen;

    // Set qOut, kvCacheStartIds, tokenPosIds to nullptr since they are not used in prefill case.
    launchApplyRopeWriteKV(thrust::raw_pointer_cast(qkvDevice.data()), thrust::raw_pointer_cast(kvCacheDevice.data()),
        nullptr, thrust::raw_pointer_cast(cosSinCacheDevice.data()), nullptr, nullptr, qSeqLen, tokenToProcess,
        kvCacheCapacity, numQHeads, numKVHeads, headDim, useInterleaveRope, stream);
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

    std::cout << "TestRopeWriteKvPrefillV2 "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " KVCacheCapacity: " << kvCacheCapacity << " qSeqLen: " << qSeqLen
              << " useInterleaveRope: " << useInterleaveRope << std::endl;
}
void BenchmarkRopeWriteKvV2(uint32_t const batchSize, AttnParams const& attnParams, int64_t const qSeqLen)
{
    uint32_t const headDim = attnParams.headDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int64_t const kvCacheCapacity = 1024 + qSeqLen;

    std::vector<half> qkvInput(batchSize * qSeqLen * (numQHeads + 2 * numKVHeads) * headDim);
    std::vector<float> cosSinCache(kvCacheCapacity * headDim);

    uniformFloatinitialization(cosSinCache, -1, 1);
    uniformFloatinitialization(qkvInput);

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<float> cosSinCacheDevice(cosSinCache);
    thrust::device_vector<half> kvCacheDevice(batchSize * (numKVHeads + numKVHeads) * kvCacheCapacity * headDim);

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * qSeqLen;

    auto launchPrefill = [&](bool const useInterleaveRope) {
        launchApplyRopeWriteKV(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(kvCacheDevice.data()), nullptr, thrust::raw_pointer_cast(cosSinCacheDevice.data()),
            nullptr, nullptr, qSeqLen, tokenToProcess, kvCacheCapacity, numQHeads, numKVHeads, headDim,
            useInterleaveRope, stream);
    };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launchPrefill(false);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchPrefill(false);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Bench Perf: BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " qSeqLen: " << qSeqLen << std::endl;
    std::cout << "RopeWriteKvV2(non-interleave) time: " << elapsedTime / numBenchIter << " ms" << std::endl;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchPrefill(true);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, stream);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "RopeWriteKvV2(interleave) time: " << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(RopeWriteKvPrefillV2, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvV2Prefill(1, {32, 8, 128}, 2048, 512, false);
    // QheadNum = 24, kvHeadNum = 3, headSize = 128, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvV2Prefill(2, {24, 3, 128}, 4096, 512, false);
    // QheadNum = 28, kvHeadNum = 7, headSize = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvV2Prefill(1, {28, 7, 128}, 2048, 512, true);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvV2Prefill(4, {16, 4, 64}, 2048, 512, true);
}

TEST(RopeWriteKvPrefillV2, Benchmark)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, qLen = 1024
    BenchmarkRopeWriteKvV2(1, {32, 8, 128}, 1024);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, qLen = 2048
    BenchmarkRopeWriteKvV2(2, {24, 3, 128}, 2048);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, qLen = 4096
    BenchmarkRopeWriteKvV2(1, {28, 7, 128}, 4096);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, qLen = 1024
    BenchmarkRopeWriteKvV2(4, {16, 4, 64}, 1024);
}
