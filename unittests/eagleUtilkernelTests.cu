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

#include <gtest/gtest.h>
#include <random>

#include "common/cudaUtils.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "references.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

void TestPrepareEagleDraftProposalInput(
    int32_t const batchSize, int32_t const paddedDraftTreeSize, int32_t const selectTokenLength)
{
    cudaStream_t stream{nullptr};
    std::random_device dev;
    std::mt19937 rng(dev());

    int32_t const packedTreeMaskLen = divUp(paddedDraftTreeSize, 32);

    // CPU reference
    // Inputs
    std::vector<int8_t> draftTreeMask(batchSize * paddedDraftTreeSize * paddedDraftTreeSize);
    std::uniform_int_distribution<std::mt19937::result_type> draftTreeMaskDist(0, 1);
    std::generate(
        draftTreeMask.begin(), draftTreeMask.end(), [&draftTreeMaskDist, &rng]() { return draftTreeMaskDist(rng); });
    std::vector<int32_t> draftTreeLength(batchSize);
    std::uniform_int_distribution<std::mt19937::result_type> draftTreeLengthDist(0, paddedDraftTreeSize - 1);
    std::generate(draftTreeLength.begin(), draftTreeLength.end(),
        [&draftTreeLengthDist, &rng]() { return draftTreeLengthDist(rng); });
    std::vector<int32_t> sequenceStartIndex(batchSize);
    std::uniform_int_distribution<std::mt19937::result_type> sequenceStartIndexDist(128, 1024);
    std::generate(sequenceStartIndex.begin(), sequenceStartIndex.end(),
        [&sequenceStartIndexDist, &rng]() { return sequenceStartIndexDist(rng); });
    // Outputs
    std::vector<int32_t> packedDraftTreeMaskReference(batchSize * paddedDraftTreeSize * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesReference(batchSize * paddedDraftTreeSize);
    std::vector<int64_t> selectTokenIndicesReference(batchSize * selectTokenLength);
    std::vector<int32_t> sequenceContextLengthsReference(batchSize);

    // Call reference function
    assembleDraftTreeDescReference(draftTreeMask, draftTreeLength, sequenceStartIndex, packedDraftTreeMaskReference,
        tensorPositionIndicesReference, paddedDraftTreeSize);
    prepareEagleDraftProposalMiscInputReference(draftTreeLength, sequenceStartIndex, sequenceContextLengthsReference,
        selectTokenIndicesReference, selectTokenLength, paddedDraftTreeSize);

    // GPU test
    // Inputs
    auto draftTreeMaskDevice
        = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
    auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto sequenceStartIndexDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(draftTreeMaskDevice.rawPointer(), draftTreeMask.data(),
        draftTreeMask.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(draftTreeLengthDevice.rawPointer(), draftTreeLength.data(),
        draftTreeLength.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(sequenceStartIndexDevice.rawPointer(), sequenceStartIndex.data(),
        sequenceStartIndex.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    // Outputs
    auto packedDraftTreeMaskDevice
        = rt::Tensor({batchSize, paddedDraftTreeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto tensorPositionIndicesDevice
        = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectTokenIndicesDevice = rt::Tensor({batchSize * selectTokenLength}, rt::DeviceType::kGPU, DataType::kINT64);
    auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    // Call kernel
    prepareEagleDraftProposalInputs(draftTreeMaskDevice, draftTreeLengthDevice, sequenceStartIndexDevice,
        packedDraftTreeMaskDevice, tensorPositionIndicesDevice, selectTokenIndicesDevice, sequenceContextLengthsDevice,
        stream);

    // Copy back to host
    std::vector<int32_t> packedDraftTreeMaskHost(batchSize * paddedDraftTreeSize * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesHost(batchSize * paddedDraftTreeSize);
    std::vector<int64_t> selectTokenIndicesHost(batchSize * selectTokenLength);
    std::vector<int32_t> sequenceContextLengthsHost(batchSize);
    CUDA_CHECK(cudaMemcpyAsync(packedDraftTreeMaskHost.data(), packedDraftTreeMaskDevice.rawPointer(),
        packedDraftTreeMaskHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tensorPositionIndicesHost.data(), tensorPositionIndicesDevice.rawPointer(),
        tensorPositionIndicesHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(selectTokenIndicesHost.data(), selectTokenIndicesDevice.rawPointer(),
        selectTokenIndicesHost.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(sequenceContextLengthsHost.data(), sequenceContextLengthsDevice.rawPointer(),
        sequenceContextLengthsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify results
    for (int i = 0; i < batchSize; i++)
    {
        for (int j = 0; j < paddedDraftTreeSize; j++)
        {
            // Verify packed tree mask
            for (int k = 0; k < packedTreeMaskLen; k++)
            {
                int32_t maskVal
                    = packedDraftTreeMaskHost[i * paddedDraftTreeSize * packedTreeMaskLen + j * packedTreeMaskLen + k];
                int32_t maskRefVal = packedDraftTreeMaskReference[i * paddedDraftTreeSize * packedTreeMaskLen
                    + j * packedTreeMaskLen + k];
                EXPECT_EQ(maskVal, maskRefVal);
            }
            // Verify tensor position indices
            int32_t positionVal = tensorPositionIndicesHost[i * paddedDraftTreeSize + j];
            int32_t positionRefVal = tensorPositionIndicesReference[i * paddedDraftTreeSize + j];
            EXPECT_EQ(positionVal, positionRefVal);
        }

        // Verify select token indices
        for (int j = 0; j < selectTokenLength; j++)
        {
            int64_t selectTokenVal = selectTokenIndicesHost[i * selectTokenLength + j];
            int64_t selectTokenRefVal = selectTokenIndicesReference[i * selectTokenLength + j];
            EXPECT_EQ(selectTokenVal, selectTokenRefVal);
        }
        // Verify sequence context lengths
        int32_t sequenceContextLengthVal = sequenceContextLengthsHost[i];
        int32_t sequenceContextLengthRefVal = sequenceContextLengthsReference[i];
        EXPECT_EQ(sequenceContextLengthVal, sequenceContextLengthRefVal);
    }

    std::cout << "TestPrepareEagleDraftProposalInput "
              << "BatchSize: " << batchSize << " PaddedDraftTreeSize: " << paddedDraftTreeSize
              << " SelectTokenLength: " << selectTokenLength << std::endl;
}

TEST(PrepareEagle, PrepareEagleDraftProposalInput)
{
    // Constrains batch size to 1 for now.
    TestPrepareEagleDraftProposalInput(1, 32, 8);
    TestPrepareEagleDraftProposalInput(1, 60, 10);
    TestPrepareEagleDraftProposalInput(1, 100, 12);
}

void TestPrepareEaglePrefillInput(int32_t const batchSize, int32_t const sequenceLength)
{
    cudaStream_t stream{nullptr};
    std::random_device dev;
    std::mt19937 rng(dev());

    // CPU reference
    std::vector<int32_t> sequenceContextLengthsReference(batchSize);
    std::vector<int64_t> selectTokenIndicesReference(batchSize);

    // Call reference function
    prepareEaglePrefillInputReference(sequenceContextLengthsReference, selectTokenIndicesReference, sequenceLength);

    // GPU test
    auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectTokenIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);

    // Call kernel
    prepareEaglePrefillInputs(sequenceContextLengthsDevice, selectTokenIndicesDevice, sequenceLength, stream);

    // Copy back to host
    std::vector<int32_t> sequenceContextLengthsHost(batchSize);
    std::vector<int64_t> selectTokenIndicesHost(batchSize);
    CUDA_CHECK(cudaMemcpyAsync(sequenceContextLengthsHost.data(), sequenceContextLengthsDevice.rawPointer(),
        sequenceContextLengthsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(selectTokenIndicesHost.data(), selectTokenIndicesDevice.rawPointer(),
        selectTokenIndicesHost.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify results
    for (int i = 0; i < batchSize; i++)
    {
        EXPECT_EQ(sequenceContextLengthsHost[i], sequenceContextLengthsReference[i]);
        EXPECT_EQ(selectTokenIndicesHost[i], selectTokenIndicesReference[i]);
    }

    std::cout << "TestPrepareEaglePrefillInput "
              << "BatchSize: " << batchSize << " SequenceLength: " << sequenceLength << std::endl;
}

TEST(PrepareEagle, PrepareEaglePrefillInput)
{
    TestPrepareEaglePrefillInput(1, 512);
    TestPrepareEaglePrefillInput(4, 1024);
    TestPrepareEaglePrefillInput(8, 2048);
}

void TestPrepareEagleAcceptDecodeTokenInput(int32_t const batchSize, int32_t const acceptedTokenNum)
{
    cudaStream_t stream{nullptr};
    std::random_device dev;
    std::mt19937 rng(dev());

    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(acceptedTokenNum, 32));

    // CPU reference
    std::vector<int32_t> sequenceStartIndices(batchSize);
    std::uniform_int_distribution<std::mt19937::result_type> sequenceStartIndexDist(128, 1024);
    std::generate(sequenceStartIndices.begin(), sequenceStartIndices.end(),
        [&sequenceStartIndexDist, &rng]() { return sequenceStartIndexDist(rng); });

    std::vector<int32_t> packedTreeMaskReference(batchSize * acceptedTokenNum * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesReference(batchSize * acceptedTokenNum);
    std::vector<int64_t> selectTokenIndicesReference(batchSize);
    std::vector<int32_t> sequenceContextLengthsReference(batchSize);

    // Call reference function
    prepareEagleAcceptDecodeTokenInputReference(sequenceStartIndices, packedTreeMaskReference,
        tensorPositionIndicesReference, selectTokenIndicesReference, sequenceContextLengthsReference, acceptedTokenNum);

    // GPU test
    auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(sequenceStartIndicesDevice.rawPointer(), sequenceStartIndices.data(),
        sequenceStartIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto packedTreeMaskDevice
        = rt::Tensor({batchSize, acceptedTokenNum, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto tensorPositionIndicesDevice
        = rt::Tensor({batchSize, acceptedTokenNum}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectTokenIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    // Initialize output tensors to zero
    CUDA_CHECK(cudaMemsetAsync(packedTreeMaskDevice.rawPointer(), 0, packedTreeMaskDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        tensorPositionIndicesDevice.rawPointer(), 0, tensorPositionIndicesDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        selectTokenIndicesDevice.rawPointer(), 0, selectTokenIndicesDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        sequenceContextLengthsDevice.rawPointer(), 0, sequenceContextLengthsDevice.getMemoryCapacity(), stream));

    // Call kernel
    prepareEagleAcceptDecodeTokenInputs(sequenceStartIndicesDevice, packedTreeMaskDevice, tensorPositionIndicesDevice,
        selectTokenIndicesDevice, sequenceContextLengthsDevice, acceptedTokenNum, stream);

    // Copy back to host
    std::vector<int32_t> packedTreeMaskHost(batchSize * acceptedTokenNum * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesHost(batchSize * acceptedTokenNum);
    std::vector<int64_t> selectTokenIndicesHost(batchSize);
    std::vector<int32_t> sequenceContextLengthsHost(batchSize);
    CUDA_CHECK(cudaMemcpyAsync(packedTreeMaskHost.data(), packedTreeMaskDevice.rawPointer(),
        packedTreeMaskHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tensorPositionIndicesHost.data(), tensorPositionIndicesDevice.rawPointer(),
        tensorPositionIndicesHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(selectTokenIndicesHost.data(), selectTokenIndicesDevice.rawPointer(),
        selectTokenIndicesHost.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(sequenceContextLengthsHost.data(), sequenceContextLengthsDevice.rawPointer(),
        sequenceContextLengthsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify results
    for (int i = 0; i < batchSize; i++)
    {
        for (int j = 0; j < acceptedTokenNum; j++)
        {
            for (int k = 0; k < packedTreeMaskLen; k++)
            {
                int32_t packedTreeMaskVal
                    = packedTreeMaskHost[i * acceptedTokenNum * packedTreeMaskLen + j * packedTreeMaskLen];
                int32_t packedTreeMaskRefVal
                    = packedTreeMaskReference[i * acceptedTokenNum * packedTreeMaskLen + j * packedTreeMaskLen + k];
                EXPECT_EQ(packedTreeMaskVal, packedTreeMaskRefVal);
            }

            int32_t tensorPositionVal = tensorPositionIndicesHost[i * acceptedTokenNum + j];
            int32_t tensorPositionRefVal = tensorPositionIndicesReference[i * acceptedTokenNum + j];
            EXPECT_EQ(tensorPositionVal, tensorPositionRefVal);
        }

        int64_t selectTokenVal = selectTokenIndicesHost[i];
        int64_t selectTokenRefVal = selectTokenIndicesReference[i];
        EXPECT_EQ(selectTokenVal, selectTokenRefVal);

        int32_t sequenceContextLengthVal = sequenceContextLengthsHost[i];
        int32_t sequenceContextLengthRefVal = sequenceContextLengthsReference[i];
        EXPECT_EQ(sequenceContextLengthVal, sequenceContextLengthRefVal);
    }

    std::cout << "TestPrepareEagleAcceptDecodeTokenInput "
              << "BatchSize: " << batchSize << " AcceptedTokenNum: " << acceptedTokenNum << std::endl;
}

TEST(PrepareEagle, PrepareEagleAcceptDecodeTokenInput)
{
    TestPrepareEagleAcceptDecodeTokenInput(1, 5);
    TestPrepareEagleAcceptDecodeTokenInput(2, 8);
    TestPrepareEagleAcceptDecodeTokenInput(4, 16);
}

void TestPrepareEagleBaseTreeDecodingInput(int32_t const batchSize, int32_t const treeSize)
{
    cudaStream_t stream{nullptr};
    std::random_device dev;
    std::mt19937 rng(dev());

    int32_t const packedTreeMaskLen = divUp(treeSize, 32);

    // CPU reference
    // Inputs
    std::vector<int8_t> treeMask(batchSize * treeSize * treeSize);
    std::uniform_int_distribution<std::mt19937::result_type> treeMaskDist(0, 1);
    std::generate(treeMask.begin(), treeMask.end(), [&treeMaskDist, &rng]() { return treeMaskDist(rng); });
    std::vector<int32_t> sequenceStartIndex(batchSize);
    std::uniform_int_distribution<std::mt19937::result_type> sequenceStartIndexDist(128, 1024);
    std::generate(sequenceStartIndex.begin(), sequenceStartIndex.end(),
        [&sequenceStartIndexDist, &rng]() { return sequenceStartIndexDist(rng); });
    // Outputs
    std::vector<int32_t> packedTreeMaskReference(batchSize * treeSize * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesReference(batchSize * treeSize);
    std::vector<int64_t> selectTokenIndicesReference(batchSize * treeSize);
    std::vector<int32_t> sequenceContextLengthsReference(batchSize);

    // Call reference function
    prepareEagleBaseTreeDecodingInputReference(treeMask, sequenceStartIndex, packedTreeMaskReference,
        tensorPositionIndicesReference, sequenceContextLengthsReference, selectTokenIndicesReference, treeSize);

    // GPU test
    // Inputs
    auto treeMaskDevice = rt::Tensor({batchSize, treeSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT8);
    auto sequenceStartIndexDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(treeMaskDevice.rawPointer(), treeMask.data(), treeMask.size() * sizeof(int8_t),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(sequenceStartIndexDevice.rawPointer(), sequenceStartIndex.data(),
        sequenceStartIndex.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    // Outputs
    auto packedTreeMaskDevice
        = rt::Tensor({batchSize, treeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto tensorPositionIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectTokenIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT64);
    auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    // Call kernel
    prepareEagleBaseTreeDecodingInputs(treeMaskDevice, sequenceStartIndexDevice, packedTreeMaskDevice,
        tensorPositionIndicesDevice, selectTokenIndicesDevice, sequenceContextLengthsDevice, stream);

    // Copy back to host
    std::vector<int32_t> packedTreeMaskHost(batchSize * treeSize * packedTreeMaskLen);
    std::vector<int32_t> tensorPositionIndicesHost(batchSize * treeSize);
    std::vector<int64_t> selectTokenIndicesHost(batchSize * treeSize);
    std::vector<int32_t> sequenceContextLengthsHost(batchSize);
    CUDA_CHECK(cudaMemcpyAsync(packedTreeMaskHost.data(), packedTreeMaskDevice.rawPointer(),
        packedTreeMaskHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tensorPositionIndicesHost.data(), tensorPositionIndicesDevice.rawPointer(),
        tensorPositionIndicesHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(selectTokenIndicesHost.data(), selectTokenIndicesDevice.rawPointer(),
        selectTokenIndicesHost.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(sequenceContextLengthsHost.data(), sequenceContextLengthsDevice.rawPointer(),
        sequenceContextLengthsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify results
    for (int i = 0; i < batchSize; i++)
    {
        for (int j = 0; j < treeSize; j++)
        {
            // Verify packed tree mask
            for (int k = 0; k < packedTreeMaskLen; k++)
            {
                int32_t maskVal = packedTreeMaskHost[i * treeSize * packedTreeMaskLen + j * packedTreeMaskLen + k];
                int32_t maskRefVal
                    = packedTreeMaskReference[i * treeSize * packedTreeMaskLen + j * packedTreeMaskLen + k];
                EXPECT_EQ(maskVal, maskRefVal);
            }
            // Verify tensor position indices
            int32_t positionVal = tensorPositionIndicesHost[i * treeSize + j];
            int32_t positionRefVal = tensorPositionIndicesReference[i * treeSize + j];
            EXPECT_EQ(positionVal, positionRefVal);
        }

        // Verify select token indices
        for (int j = 0; j < treeSize; j++)
        {
            int64_t selectTokenVal = selectTokenIndicesHost[i * treeSize + j];
            int64_t selectTokenRefVal = selectTokenIndicesReference[i * treeSize + j];
            EXPECT_EQ(selectTokenVal, selectTokenRefVal);
        }
        // Verify sequence context lengths
        int32_t sequenceContextLengthVal = sequenceContextLengthsHost[i];
        int32_t sequenceContextLengthRefVal = sequenceContextLengthsReference[i];
        EXPECT_EQ(sequenceContextLengthVal, sequenceContextLengthRefVal);
    }

    std::cout << "TestPrepareEagleBaseTreeDecodingInput "
              << "BatchSize: " << batchSize << " TreeSize: " << treeSize << std::endl;
}

TEST(PrepareEagle, PrepareEagleBaseTreeDecodingInput)
{
    TestPrepareEagleBaseTreeDecodingInput(1, 32);
    TestPrepareEagleBaseTreeDecodingInput(2, 60);
    TestPrepareEagleBaseTreeDecodingInput(4, 100);
}

struct KVCacheParameters
{
    int32_t numDecoderLayers;
    int32_t maxBatchSize;
    int32_t maxSequenceLength;
    int32_t numKVHead;
    int32_t headDim;
};

void TestEagleBaseCommitKVCache(KVCacheParameters const& cacheParams, int32_t const maxDepth = 6,
    int32_t const draftTreeSize = 60, int32_t const baseHiddenDim = 512)
{
    cudaStream_t stream{nullptr};
    static std::random_device dev;
    static std::mt19937 rng(dev());

    // Generate random inputs
    std::uniform_int_distribution<int32_t> dist(1, cacheParams.maxBatchSize);
    int32_t const batchSize = dist(rng);

    std::vector<half> kvCacheBuffer(cacheParams.numDecoderLayers * cacheParams.maxBatchSize * 2 * cacheParams.numKVHead
        * cacheParams.maxSequenceLength * cacheParams.headDim);
    uniformFloatInitialization<half>(kvCacheBuffer);
    std::vector<half> hiddenState(batchSize * draftTreeSize * baseHiddenDim);
    uniformFloatInitialization<half>(hiddenState);
    std::vector<int32_t> acceptedIndices(batchSize * maxDepth);
    uniformIntInitialization(acceptedIndices, 0, draftTreeSize - 1);
    std::vector<int32_t> acceptLengths(batchSize);
    uniformIntInitialization(acceptLengths, 0, maxDepth);
    std::vector<int32_t> kvCacheLengths(batchSize);
    uniformIntInitialization(kvCacheLengths, 128, 1024);

    // CPU reference, copy input to output first
    std::vector<half> kvCacheBufferRef(kvCacheBuffer);
    std::vector<half> hiddenStateRef(hiddenState);

    eagleBaseCommitKVCacheAndAssembleHiddenStateReference(acceptedIndices, acceptLengths, kvCacheBuffer, kvCacheLengths,
        hiddenState, kvCacheBufferRef, hiddenStateRef, cacheParams.numDecoderLayers, cacheParams.maxBatchSize,
        cacheParams.numKVHead, cacheParams.maxSequenceLength, cacheParams.headDim, maxDepth, draftTreeSize,
        baseHiddenDim);

    // Create GPU tensors
    rt::Tensor kvCacheBufferDevice({cacheParams.numDecoderLayers, cacheParams.maxBatchSize, 2, cacheParams.numKVHead,
                                       cacheParams.maxSequenceLength, cacheParams.headDim},
        rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheLengthsDevice({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor acceptedIndicesDevice({batchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor acceptLengthsDevice({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    // WAR for the current implementation. Need to modify when enable multi-batch for eagle3.
    rt::Tensor hiddenStateDevice({batchSize * draftTreeSize, baseHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(acceptedIndicesDevice.rawPointer(), acceptedIndices.data(),
        acceptedIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(acceptLengthsDevice.rawPointer(), acceptLengths.data(),
        acceptLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(kvCacheBufferDevice.rawPointer(), kvCacheBuffer.data(),
        kvCacheBuffer.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(kvCacheLengthsDevice.rawPointer(), kvCacheLengths.data(),
        kvCacheLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(hiddenStateDevice.rawPointer(), hiddenState.data(), hiddenState.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));

    // Launch kernel
    eagleBaseCommitKVCacheAndAssembleHiddenState(acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice,
        kvCacheBufferDevice, hiddenStateDevice, stream);

    // Copy results back to CPU
    std::vector<half> kvCacheBufferHost(kvCacheBuffer.size());
    std::vector<half> hiddenStateHost(hiddenState.size());

    CUDA_CHECK(cudaMemcpyAsync(kvCacheBufferHost.data(), kvCacheBufferDevice.rawPointer(),
        kvCacheBufferHost.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hiddenStateHost.data(), hiddenStateDevice.rawPointer(),
        hiddenStateHost.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify results
    for (int b = 0; b < batchSize; b++)
    {
        int32_t kvCacheLength = kvCacheLengths[b];
        int32_t acceptLength = acceptLengths[b];

        // Verify kvCacheBuffer
        for (int l = 0; l < cacheParams.numDecoderLayers; l++)
        {
            for (int k = 0; k < 2; k++)
            {
                for (int h = 0; h < cacheParams.numKVHead; h++)
                {
                    for (int s = 0; s < acceptLength; s++)
                    {
                        for (int d = 0; d < cacheParams.headDim; d++)
                        {
                            int32_t offset = l * cacheParams.maxBatchSize * 2 * cacheParams.numKVHead
                                    * cacheParams.maxSequenceLength * cacheParams.headDim
                                + b * 2 * cacheParams.numKVHead * cacheParams.maxSequenceLength * cacheParams.headDim
                                + k * cacheParams.numKVHead * cacheParams.maxSequenceLength * cacheParams.headDim
                                + h * cacheParams.maxSequenceLength * cacheParams.headDim
                                + (kvCacheLength + s) * cacheParams.headDim + d;
                            ASSERT_TRUE(isclose(kvCacheBufferHost[offset], kvCacheBufferRef[offset], 1e-5, 1e-5));
                        }
                    }
                }
            }
        }

        // Verify hiddenState
        for (int s = 0; s < acceptLength; s++)
        {
            for (int d = 0; d < baseHiddenDim; d++)
            {
                int32_t offset = b * draftTreeSize * baseHiddenDim + s * baseHiddenDim + d;
                ASSERT_TRUE(isclose(hiddenStateHost[offset], hiddenStateRef[offset], 1e-5, 1e-5));
            }
        }
    }

    std::cout << "TestEagleBaseCommitKVCache "
              << "numDecoderLayers: " << cacheParams.numDecoderLayers << " MaxBatchSize: " << cacheParams.maxBatchSize
              << " numKVHead: " << cacheParams.numKVHead << " maxSequenceLength: " << cacheParams.maxSequenceLength
              << " HeadDim: " << cacheParams.headDim << " MaxDepth: " << maxDepth << " DraftTreeSize: " << draftTreeSize
              << " BaseHiddenDim: " << baseHiddenDim << std::endl;
}

TEST(EagleBaseCommitKVCache, BasicTest)
{
    // Constrains batch size to 1 for now.
    TestEagleBaseCommitKVCache({8, 1, 4096, 4, 128}, 6, 60, 256);
    TestEagleBaseCommitKVCache({3, 1, 4096, 4, 128}, 6, 24, 256);
    TestEagleBaseCommitKVCache({3, 1, 4096, 8, 128}, 4, 16, 128);
}