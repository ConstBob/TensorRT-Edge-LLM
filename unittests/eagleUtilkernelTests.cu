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
#include "kernels/speculative/newEagleUtilKernels.h"
#include "references.h"
#include "testUtils.h"

using namespace drivellm;
using namespace drivellm::kernel;
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
    auto selectTokenIndicesDevice = rt::Tensor({batchSize, selectTokenLength}, rt::DeviceType::kGPU, DataType::kINT64);
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
    TestPrepareEagleDraftProposalInput(1, 32, 8);
    TestPrepareEagleDraftProposalInput(2, 60, 10);
    TestPrepareEagleDraftProposalInput(4, 100, 12);
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