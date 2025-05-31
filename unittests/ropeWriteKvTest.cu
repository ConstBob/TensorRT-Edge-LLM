#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "attentionPlugin/utilKernels.h"
#include "common/cudaUtils.h"
#include "references.h"
#include "testUtils.h"

struct RopeParams
{
    PositionEmbeddingType posEmbedType;
    float rotaryEmbeddingTheta;
    float rotaryEmbeddingScale;
};

class KvCacheIndexer
{
public:
    KvCacheIndexer(
        int32_t const batchSize, int32_t const kvHeadNum, int32_t const kvCacheCapacity, int32_t const headSize)
    {
        mBatchSize = batchSize;
        mKvHeadNum = kvHeadNum;
        mKvCacheCapacity = kvCacheCapacity;
        mHeadSize = headSize;
    }

    int32_t indexK(int32_t const b, int32_t const hk, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + hk * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

    int32_t indexV(int32_t const b, int32_t const hv, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        // V cache need to offset the whole kCache buffer for the sequence.
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + (mKvHeadNum + hv) * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

private:
    int32_t mBatchSize;
    int32_t mKvHeadNum;
    int32_t mKvCacheCapacity;
    int32_t mHeadSize;
};

void TestRopeWriteKvPrefill(int32_t const batchSize, int32_t const qHeadNum, int32_t const kvHeadNum,
    int32_t const headSize, int32_t const kvCacheCapacity, int32_t const paddedSeqlen, RopeParams const& ropeParams)
{
    // Not tested for MROPE which supply positional encoding coefficients as input tensor.
    EXPECT_NE(ropeParams.posEmbedType, PositionEmbeddingType::kMROPE);

    std::vector<half> qkvInput;
    std::vector<half> kvCache(batchSize * 2 * kvHeadNum * kvCacheCapacity * headSize, 0);
    std::vector<int32_t> seqLens(batchSize, paddedSeqlen);

    std::vector<half> qkvReference;

    bool const permuteRope = ropeParams.posEmbedType == PositionEmbeddingType::kROPE_ROTATE_NEOX;
    float const ropeScale = ropeParams.rotaryEmbeddingScale;
    float const ropeTheta = ropeParams.rotaryEmbeddingTheta;
    RopeInitType const ropeInitType = RopeInitType::kDEFAULT;

    for (int32_t i = 0; i < batchSize; i++)
    {
        for (int32_t j = 0; j < paddedSeqlen; j++)
        {
            std::vector<half> qij(qHeadNum * headSize);
            std::vector<half> kij(kvHeadNum * headSize);
            std::vector<half> vij(kvHeadNum * headSize);

            uniformFloatinitialization(qij);
            uniformFloatinitialization(kij);
            uniformFloatinitialization(vij);

            // QKV tensor has layout [B, S, H, D]
            qkvInput.insert(qkvInput.end(), qij.begin(), qij.end());
            qkvInput.insert(qkvInput.end(), kij.begin(), kij.end());
            qkvInput.insert(qkvInput.end(), vij.begin(), vij.end());

            auto qRefij = ropeRef(qij, qHeadNum, headSize, j, ropeScale, ropeTheta, permuteRope);
            auto kRefij = ropeRef(kij, kvHeadNum, headSize, j, ropeScale, ropeTheta, permuteRope);

            qkvReference.insert(qkvReference.end(), qRefij.begin(), qRefij.end());
            qkvReference.insert(qkvReference.end(), kRefij.begin(), kRefij.end());
            qkvReference.insert(qkvReference.end(), vij.begin(), vij.end());
        }
    }

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<half> kvCacheDevice(kvCache);
    thrust::device_vector<int32_t> seqLensDevice(seqLens);

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * paddedSeqlen;
    int32_t const rotaryEmbeddingMaxPositions = 0; // not used.
    invokeContextApplyRopeUpdateKVFP16(thrust::raw_pointer_cast(qkvDevice.data()),
        thrust::raw_pointer_cast(kvCacheDevice.data()), thrust::raw_pointer_cast(seqLensDevice.data()), qHeadNum,
        kvHeadNum, headSize, kvCacheCapacity, paddedSeqlen, ropeParams.posEmbedType, ropeTheta, ropeScale, ropeInitType,
        tokenToProcess, rotaryEmbeddingMaxPositions, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    thrust::host_vector<half> qkvOut(qkvInput.size());
    thrust::host_vector<half> kvCacheOut(kvCache.size());
    thrust::copy(qkvDevice.begin(), qkvDevice.end(), qkvOut.begin());
    thrust::copy(kvCacheDevice.begin(), kvCacheDevice.end(), kvCacheOut.begin());

    KvCacheIndexer kvIndexer(batchSize, kvHeadNum, kvCacheCapacity, headSize);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        int32_t const batchOffset = i * paddedSeqlen * (qHeadNum + 2 * kvHeadNum) * headSize;
        for (int32_t j = 0; j < paddedSeqlen; ++j)
        {
            int32_t const tokenOffset = j * (qHeadNum + 2 * kvHeadNum) * headSize;
            for (int32_t hq = 0; hq < qHeadNum; ++hq)
            {
                int32_t const qOffset = batchOffset + tokenOffset + hq * headSize;
                for (int32_t d = 0; d < headSize; ++d)
                {
                    half const qVal = qkvOut[qOffset + d];
                    half const qRefVal = qkvReference[qOffset + d];
                    EXPECT_TRUE(isclose(qVal, qRefVal, 1e-3, 1e-3));
                }
            }
            for (int32_t hkv = 0; hkv < kvHeadNum; ++hkv)
            {
                int32_t const kOffset = batchOffset + tokenOffset + qHeadNum * headSize + hkv * headSize;
                int32_t const vOffset
                    = batchOffset + tokenOffset + qHeadNum * headSize + kvHeadNum * headSize + hkv * headSize;
                for (int32_t d = 0; d < headSize; ++d)
                {
                    half const kVal = qkvOut[kOffset + d];
                    half const kCacheVal = kvCacheOut[kvIndexer.indexK(i, hkv, j, d)];
                    half const kRefVal = qkvReference[kOffset + d];
                    half const vVal = qkvOut[vOffset + d];
                    half const vCacheVal = kvCacheOut[kvIndexer.indexV(i, hkv, j, d)];
                    half const vRefVal = qkvReference[vOffset + d];
                    EXPECT_TRUE(isclose(kVal, kRefVal, 1e-3, 1e-3));
                    EXPECT_TRUE(isclose(vVal, vRefVal, 1e-3, 1e-3));
                    EXPECT_TRUE(isclose(kCacheVal, kVal, 1e-5, 1e-5));
                    EXPECT_TRUE(isclose(vCacheVal, vVal, 1e-5, 1e-5));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvPrefill "
              << "BatchSize: " << batchSize << " QHeadNum: " << qHeadNum << " KVHeadNum: " << kvHeadNum
              << " HeadSize: " << headSize << " KVCacheCapacity: " << kvCacheCapacity
              << " PaddedSeqLen: " << paddedSeqlen << " PosEmbedType: " << static_cast<int>(ropeParams.posEmbedType)
              << " RopeScale: " << ropeScale << " RopeTheta: " << ropeTheta << std::endl;
}

void TestRopeWriteKvDecode(int32_t const batchSize, int32_t const qHeadNum, int32_t const kvHeadNum,
    int32_t const headSize, int32_t const kvCacheCapacity, int32_t const qLen, RopeParams const& ropeParams,
    bool const isTreeAttention)
{
    // Not tested for MROPE which supply positional encoding coefficients as input tensor.
    EXPECT_NE(ropeParams.posEmbedType, PositionEmbeddingType::kMROPE);
    EXPECT_TRUE(qLen == 1 || isTreeAttention);

    // QKV tensor has layout [B, S, Hq+Hk+Hv, D]. KV cache has layout [B, 2, S, Hkv, D].
    std::vector<half> qkvInput;
    std::vector<half> kvCache(batchSize * 2 * kvHeadNum * kvCacheCapacity * headSize, 0);

    // Reference output of Q, K, V all have layout [B, S, H, D].
    std::vector<half> qreference;
    std::vector<half> kreference;
    std::vector<half> vreference;

    // Random initialized the total length which is committed kv-cache length + new tokens length.
    std::vector<int32_t> fullSeqLens(batchSize);
    uniformIntInitialization(fullSeqLens, kvCacheCapacity / 4, kvCacheCapacity);
    std::vector<int32_t> customSeqLens;

    bool const permuteRope = ropeParams.posEmbedType == PositionEmbeddingType::kROPE_ROTATE_NEOX;
    float const ropeScale = ropeParams.rotaryEmbeddingScale;
    float const ropeTheta = ropeParams.rotaryEmbeddingTheta;
    RopeInitType const ropeInitType = RopeInitType::kDEFAULT;

    for (int32_t i = 0; i < batchSize; i++)
    {
        int32_t const qStartIdx = fullSeqLens[i] - qLen;
        // With speculative decoding, the sequence index is not identical to kvcache index.
        std::vector<int32_t> customSeqLen(qLen);
        uniformIntInitialization(customSeqLen, qStartIdx, qStartIdx + qLen - 1);
        customSeqLens.insert(customSeqLens.end(), customSeqLen.begin(), customSeqLen.end());

        for (int32_t j = 0; j < qLen; j++)
        {
            std::vector<half> qi(qHeadNum * headSize);
            std::vector<half> ki(kvHeadNum * headSize);
            std::vector<half> vi(kvHeadNum * headSize);

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
            auto qRefij = ropeRef(qi, qHeadNum, headSize, seqIdx, ropeScale, ropeTheta, permuteRope);
            auto kRefij = ropeRef(ki, kvHeadNum, headSize, seqIdx, ropeScale, ropeTheta, permuteRope);

            qreference.insert(qreference.end(), qRefij.begin(), qRefij.end());
            kreference.insert(kreference.end(), kRefij.begin(), kRefij.end());
            vreference.insert(vreference.end(), vi.begin(), vi.end());
        }
    }

    thrust::device_vector<half> qkvDevice(qkvInput);
    thrust::device_vector<half> qOutDevice(batchSize * qLen * qHeadNum * headSize);
    thrust::device_vector<half> kvCacheDevice(kvCache);
    thrust::device_vector<int32_t> seqLensDevice(fullSeqLens);
    thrust::device_vector<int32_t> customSeqLensDevice(customSeqLens);

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * qLen;
    int32_t const rotaryEmbeddingMaxPositions = 0; // not used.
    if (!isTreeAttention)
    {
        invokeGenerationApplyRopeUpdateKVFP16(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(qOutDevice.data()), thrust::raw_pointer_cast(kvCacheDevice.data()),
            thrust::raw_pointer_cast(seqLensDevice.data()), qHeadNum, kvHeadNum, headSize, kvCacheCapacity, qLen,
            ropeParams.posEmbedType, ropeTheta, ropeScale, ropeInitType, tokenToProcess, rotaryEmbeddingMaxPositions,
            nullptr, stream);
    }
    else
    {
        invokeSpecDecodeGenerationApplyRopeUpdateKVFP16(thrust::raw_pointer_cast(qkvDevice.data()),
            thrust::raw_pointer_cast(qOutDevice.data()), thrust::raw_pointer_cast(kvCacheDevice.data()),
            thrust::raw_pointer_cast(seqLensDevice.data()), thrust::raw_pointer_cast(customSeqLensDevice.data()),
            qHeadNum, kvHeadNum, headSize, kvCacheCapacity, qLen, ropeParams.posEmbedType, ropeTheta, ropeScale,
            ropeInitType, tokenToProcess, rotaryEmbeddingMaxPositions, nullptr, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    thrust::host_vector<half> qOut(batchSize * qLen * qHeadNum * headSize);
    thrust::copy(qOutDevice.begin(), qOutDevice.end(), qOut.begin());
    thrust::host_vector<half> kvCacheOut(kvCache.size());
    thrust::copy(kvCacheDevice.begin(), kvCacheDevice.end(), kvCacheOut.begin());

    // Directly compare the output of Q since output and reference have the same layout.
    EXPECT_EQ(qOut.size(), qreference.size());
    for (int32_t i = 0; i < qOut.size(); ++i)
    {
        EXPECT_TRUE(isclose(qOut[i], qreference[i], 1e-3, 4e-3));
    }

    KvCacheIndexer kvIndexer(batchSize, kvHeadNum, kvCacheCapacity, headSize);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const qStartIdx = fullSeqLens[b] - qLen;
        for (int32_t s = 0; s < qLen; ++s)
        {
            int32_t const inCacheIdx = qStartIdx + s;
            for (int32_t hkv = 0; hkv < kvHeadNum; ++hkv)
            {
                int32_t const kvRefOffset = b * qLen * kvHeadNum * headSize + s * kvHeadNum * headSize + hkv * headSize;
                for (int32_t d = 0; d < headSize; ++d)
                {
                    half const kVal = kvCacheOut[kvIndexer.indexK(b, hkv, inCacheIdx, d)];
                    half const kRefVal = kreference[kvRefOffset + d];
                    EXPECT_TRUE(isclose(kVal, kRefVal, 1e-3, 4e-3));
                    half const vVal = kvCacheOut[kvIndexer.indexV(b, hkv, inCacheIdx, d)];
                    half const vRefVal = vreference[kvRefOffset + d];
                    EXPECT_TRUE(isclose(vVal, vRefVal, 1e-3, 4e-3));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvDecode "
              << "BatchSize: " << batchSize << " QHeadNum: " << qHeadNum << " KVHeadNum: " << kvHeadNum
              << " HeadSize: " << headSize << " KVCacheCapacity: " << kvCacheCapacity << " QLength: " << qLen
              << " Total Sequence Lengths (including past KVcache): " << fullSeqLens
              << " PosEmbedType: " << static_cast<int>(ropeParams.posEmbedType) << " RopeScale: " << ropeScale
              << " RopeTheta: " << ropeTheta << std::endl;
}

TEST(RopeWriteKvPrefill, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, kvCacheCapacity = 2048, paddedSeqLen = 512
    TestRopeWriteKvPrefill(1, 32, 8, 128, 2048, 512, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 10000.0f, 1.0f});
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, kvCacheCapacity = 1024, paddedSeqLen = 800
    TestRopeWriteKvPrefill(1, 28, 4, 128, 1024, 800, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 500000.0f, 1.0f});
    // QheadNum = 16, kvHeadNum = 2, headSize = 64, kvCacheCapacity = 4096, paddedSeqLen = 1024
    TestRopeWriteKvPrefill(1, 16, 2, 64, 4096, 1024, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 400000.0f, 1.0f});
    // QheadNum = 24, kvHeadNum = 4, headSize = 128, kvCacheCapacity = 4096, paddedSeqLen = 512
    TestRopeWriteKvPrefill(1, 24, 4, 128, 4096, 512, {PositionEmbeddingType::kROPE_ROTATE_GPTJ, 400000.0f, 1.0f});
}

TEST(RopeWriteKvDecodeVanilla, Accuracy)
{
    // qHeadNum = 32, kvHeadNum = 8, headSize = 128, kvCacheCapacity = 2048, qLen = 1, isTreeAttention = false
    TestRopeWriteKvDecode(1, 32, 8, 128, 2048, 1, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 10000.0f, 1.0f}, false);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention = false
    TestRopeWriteKvDecode(1, 28, 4, 128, 4096, 1, {PositionEmbeddingType::kROPE_ROTATE_GPTJ, 500000.0f, 1.0f}, false);
    // QheadNum = 16, kvHeadNum = 2, headSize = 64, kvCacheCapacity = 4096, qLen = 1, isTreeAttention = false
    TestRopeWriteKvDecode(1, 16, 2, 64, 4096, 1, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 400000.0f, 1.0f}, false);
    // QheadNum = 24, kvHeadNum = 4, headSize = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention = false
    TestRopeWriteKvDecode(1, 24, 4, 128, 4096, 1, {PositionEmbeddingType::kROPE_ROTATE_GPTJ, 400000.0f, 1.0f}, false);
}

TEST(RopeWriteKvDecodeTreeAttention, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, kvCacheCapacity = 2048, qLen = 4, isTreeAttention = true
    TestRopeWriteKvDecode(1, 32, 8, 128, 2048, 4, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 10000.0f, 1.0f}, true);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, kvCacheCapacity = 4096, qLen = 32, isTreeAttention = true
    TestRopeWriteKvDecode(1, 28, 4, 128, 4096, 32, {PositionEmbeddingType::kROPE_ROTATE_GPTJ, 500000.0f, 1.0f}, true);
    // QheadNum = 24, kvHeadNum = 6, headSize = 64, kvCacheCapacity = 4096, qLen = 64, isTreeAttention = true
    TestRopeWriteKvDecode(1, 24, 6, 64, 4096, 64, {PositionEmbeddingType::kROPE_ROTATE_NEOX, 400000.0f, 1.0f}, true);
    // QheadNum = 16, kvHeadNum = 2, headSize = 128, kvCacheCapacity = 4096, qLen = 50, isTreeAttention = true
    TestRopeWriteKvDecode(1, 16, 2, 128, 4096, 50, {PositionEmbeddingType::kROPE_ROTATE_GPTJ, 400000.0f, 1.0f}, true);
}
