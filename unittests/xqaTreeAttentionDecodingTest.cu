#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/common.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"

using namespace nvinfer1;

void TestXQATreeAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvSequenceLength, int32_t qSequenceLength)
{
    int32_t const smVersion = getSMVersion();

    std::vector<int32_t> kvCacheLength(batchSize, kvSequenceLength);
    std::vector<half> qInput;
    std::vector<half> kvInput;
    std::vector<half> outReference;
    std::vector<int32_t> packedTreeMaskInput;

    for (int32_t i = 0; i < batchSize; i++)
    {
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        std::vector<half> ki(numKVHeads * headSize * kvSequenceLength);
        std::vector<half> vi(numKVHeads * headSize * kvSequenceLength);
        std::vector<int32_t> treeMaski(qSequenceLength * qSequenceLength);

        uniformFloatinitialization(qi);
        uniformFloatinitialization(ki);
        uniformFloatinitialization(vi);
        uniformIntInitialization(treeMaski, 0, 1);

        auto ref = casualAttentionRef(qi, ki, vi, qSequenceLength, kvSequenceLength, numQHeads, numKVHeads, headSize,
            std::make_optional(treeMaski));

        // Add data from batch to input Tensors
        qInput.insert(qInput.end(), qi.begin(), qi.end());

        // KVcache layout assumed to have layout of [B, 2n H, S, D]
        kvInput.insert(kvInput.end(), ki.begin(), ki.end());
        kvInput.insert(kvInput.end(), vi.begin(), vi.end());
        outReference.insert(outReference.end(), ref.begin(), ref.end());

        // Prepare packed tree mask. The layout of mask is [qSeqLen, qSeqLen]. Which represent whether two tokens
        // will attend to each other.
        int32_t const numBitsPerPackedMask = 32;
        int32_t const numPackedMasksPerToken = divUp(qSequenceLength, numBitsPerPackedMask);
        std::vector<int32_t> packedMaski(numPackedMasksPerToken * qSequenceLength, 0);
        for (int32_t i = 0; i < qSequenceLength; i++)
        {
            for (int32_t j = 0; j < numPackedMasksPerToken; j++)
            {
                int32_t mask = 0;
                for (int32_t k = 0; k < numBitsPerPackedMask; k++)
                {
                    int32_t const bitIndex = j * numBitsPerPackedMask + k;
                    int32_t maskFlag = 0;
                    if (j * numBitsPerPackedMask + k < qSequenceLength)
                    {
                        maskFlag = treeMaski[i * qSequenceLength + bitIndex];
                    }
                    mask |= maskFlag << k;
                }
                packedMaski[i * numPackedMasksPerToken + j] = mask;
            }
        }
        packedTreeMaskInput.insert(packedTreeMaskInput.end(), packedMaski.begin(), packedMaski.end());
    }
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 1.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> packedTreeMaskDevice(packedTreeMaskInput);

    drivellm::DecoderXQARunner runner(DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qSeqLen = qSequenceLength;
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvSequenceLength;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.treeAttnMask = thrust::raw_pointer_cast(packedTreeMaskDevice.data());
    // Use default stream .
    cudaStream_t stream{nullptr};
    runner.dispatchSpecDecodeXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Check accuracy.
    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * qSequenceLength * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (__hisnan(outHost[i]))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * qSequenceLength * numQHeads * headSize);

    std::cout << "XQA Tree Attention Decoding test. batch_size: " << batchSize << " num_Q_heads: " << numQHeads
              << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " kvcache seq_len: " << kvSequenceLength << " q_seq_len: " << qSequenceLength
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio4HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 512, 10);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 256, 32);
    /// KVSequence 64, QSequence 128
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 320, 60);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio8HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 64);
    /// KVSequence 192, QSequence 60 KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 192, 60);
    /// KVSequence 512, QSequence 20， KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 512, 20);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio7HeadDim64)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 128, 64);
    /// KVSequence 192, QSequence 60
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 192, 60);
    /// KVSequence 512, QSequence 20
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 512, 20);
}
