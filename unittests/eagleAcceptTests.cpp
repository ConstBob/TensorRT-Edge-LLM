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

#include "common/checkMacros.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

using namespace drivellm;

class EagleAcceptTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        // Cleanup is handled by individual tests
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // Common test runner that handles GPU memory management, kernel execution, and result validation
    void runEagleAcceptTest(std::vector<int32_t> const& tokenIds, std::vector<int8_t> const& attentionMask,
        std::vector<float> const& logits, int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth,
        std::string const& testName,
        std::function<void(std::vector<int32_t> const&, std::vector<int32_t> const&, std::vector<int32_t> const&,
            EagleAcceptResult const&)>
            validator
        = nullptr)
    {
        // Create GPU tensors with proper shapes
        rt::Tensor logitsTensor(
            {batchSize, numTokens, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
        rt::Tensor tokenIdsTensor({batchSize, numTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
        rt::Tensor attentionMaskTensor(
            {batchSize, numTokens, numTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "attentionMask");
        rt::Tensor acceptedTokenIdsTensor(
            {batchSize, maxDepth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedTokenIds");
        rt::Tensor acceptedIndicesTensor(
            {batchSize, maxDepth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedIndices");
        rt::Tensor acceptLengthTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptLength");

        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpy(logitsTensor.rawPointer(), logits.data(),
            batchSize * numTokens * vocabSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tokenIdsTensor.rawPointer(), tokenIds.data(), batchSize * numTokens * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(attentionMaskTensor.rawPointer(), attentionMask.data(),
            batchSize * numTokens * numTokens * sizeof(int8_t), cudaMemcpyHostToDevice));

        // Allocate workspace for kernel temporary storage
        size_t workspaceSize = kernel::getEagleAcceptWorkspaceSize(batchSize, numTokens);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        // Execute kernel with timing
        auto start = std::chrono::high_resolution_clock::now();
        EXPECT_NO_THROW({
            kernel::eagleAccept(logitsTensor, tokenIdsTensor, attentionMaskTensor, acceptedTokenIdsTensor,
                acceptedIndicesTensor, acceptLengthTensor, maxDepth, workspace, workspaceSize, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
        });
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Cleanup workspace
        CUDA_CHECK(cudaFree(workspace));

        // Run reference implementation for comparison
        EagleAcceptResult refResult
            = eagleAcceptRef(logits, tokenIds, attentionMask, batchSize, numTokens, vocabSize, maxDepth);

        // Copy results back to host for validation
        std::vector<int32_t> hostAcceptedTokenIds(batchSize * maxDepth);
        std::vector<int32_t> hostAcceptedIndices(batchSize * maxDepth);
        std::vector<int32_t> hostAcceptLengths(batchSize);

        CUDA_CHECK(cudaMemcpy(hostAcceptedTokenIds.data(), acceptedTokenIdsTensor.rawPointer(),
            batchSize * maxDepth * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostAcceptedIndices.data(), acceptedIndicesTensor.rawPointer(),
            batchSize * maxDepth * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostAcceptLengths.data(), acceptLengthTensor.rawPointer(), batchSize * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

        // Basic performance check
        EXPECT_LT(duration.count(), 1000) << testName << ": Kernel took too long, possible infinite loop";

        // Validate results against reference implementation
        for (int32_t b = 0; b < batchSize; ++b)
        {
            EXPECT_EQ(hostAcceptLengths[b], refResult.acceptLengths[b])
                << testName << ": Accept length mismatch at batch " << b;

            // Check accepted tokens and indices
            for (int32_t i = 0; i < hostAcceptLengths[b]; ++i)
            {
                int32_t idx = b * maxDepth + i;
                EXPECT_EQ(hostAcceptedTokenIds[idx], refResult.acceptedTokenIds[b * refResult.maxAcceptLength + i])
                    << testName << ": Token ID mismatch at batch " << b << " position " << i;
                EXPECT_EQ(hostAcceptedIndices[idx], refResult.acceptedIndices[b * refResult.maxAcceptLength + i])
                    << testName << ": Index mismatch at batch " << b << " position " << i;
            }

            // Check that unused positions are properly initialized to -1
            for (int32_t i = hostAcceptLengths[b]; i < maxDepth; ++i)
            {
                int32_t idx = b * maxDepth + i;
                EXPECT_EQ(hostAcceptedTokenIds[idx], -1)
                    << testName << ": Unused token ID position should be -1 at batch " << b << " position " << i;
                EXPECT_EQ(hostAcceptedIndices[idx], -1)
                    << testName << ": Unused index position should be -1 at batch " << b << " position " << i;
            }
        }

        // Run custom validation if provided
        if (validator)
        {
            validator(hostAcceptedTokenIds, hostAcceptedIndices, hostAcceptLengths, refResult);
        }
        std::cout << testName << " - Duration: " << duration.count() << "ms" << std::endl;
        std::cout << "Accept lengths: ";
        for (int32_t b = 0; b < batchSize; ++b)
        {
            std::cout << hostAcceptLengths[b] << " ";
        }
        std::cout << std::endl;
        std::cout << "Accepted path: ";
        for (int32_t i = 0; i < hostAcceptLengths[0]; ++i)
        {
            std::cout << hostAcceptedTokenIds[i] << " ";
        }
        std::cout << std::endl;
    }
    cudaStream_t stream;
};

// Test basic multi-batch functionality with simple chains
TEST_F(EagleAcceptTest, MultiBatchSimple)
{
    constexpr int32_t numTokens = 3;
    constexpr int32_t batchSize = 2;
    constexpr int32_t vocabSize = 10;
    constexpr int32_t maxDepth = 3;

    // Setup token IDs: batch 0 = [1,2,3], batch 1 = [4,5,6]
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0 * numTokens + 0] = 1;
    tokenIds[0 * numTokens + 1] = 2;
    tokenIds[0 * numTokens + 2] = 3;
    tokenIds[1 * numTokens + 0] = 4;
    tokenIds[1 * numTokens + 1] = 5;
    tokenIds[1 * numTokens + 2] = 6;

    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: full chain attention pattern (1->2->3)
    attentionMask[0 * numTokens * numTokens + 0 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 1 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 1 * numTokens + 1] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 1] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 2] = 1;

    // Batch 1: partial chain (4->5, no continuation to 6)
    attentionMask[1 * numTokens * numTokens + 0 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 1] = 1;

    // Setup logits to favor the expected path
    std::vector<float> logits(batchSize * numTokens * vocabSize, -10.0f);
    logits[0 * numTokens * vocabSize + 0 * vocabSize + 2] = 10.0f; // pos 0 -> token 2
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 3] = 10.0f; // pos 1 -> token 3
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 5] = 10.0f; // pos 0 -> token 5

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth, "PerBatchDesignTest",
        [](auto const& acceptedTokenIds, auto const& acceptedIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Batch 0 should accept 3 tokens";
            EXPECT_EQ(acceptLengths[1], 2) << "Batch 1 should accept 2 tokens";

            EXPECT_EQ(acceptedTokenIds[0 * 3 + 0], 1) << "Batch 0 token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 1], 2) << "Batch 0 token 1";
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 2], 3) << "Batch 0 token 2";
            EXPECT_EQ(acceptedIndices[0 * 3 + 0], 0) << "Batch 0 index 0";
            EXPECT_EQ(acceptedIndices[0 * 3 + 1], 1) << "Batch 0 index 1";
            EXPECT_EQ(acceptedIndices[0 * 3 + 2], 2) << "Batch 0 index 2";

            EXPECT_EQ(acceptedTokenIds[1 * 3 + 0], 4) << "Batch 1 token 0";
            EXPECT_EQ(acceptedTokenIds[1 * 3 + 1], 5) << "Batch 1 token 1";
            EXPECT_EQ(acceptedIndices[1 * 3 + 0], 0) << "Batch 1 index 0";
            EXPECT_EQ(acceptedIndices[1 * 3 + 1], 1) << "Batch 1 index 1";

            EXPECT_EQ(acceptedTokenIds[1 * 3 + 2], -1) << "Batch 1 unused token position should be -1";
            EXPECT_EQ(acceptedIndices[1 * 3 + 2], -1) << "Batch 1 unused index position should be -1";
        });
}

// Test complex multi-batch scenario with varying tree structures and early termination
TEST_F(EagleAcceptTest, MultiBatchAsymmetricTree)
{
    constexpr int32_t numTokens = 5;
    constexpr int32_t batchSize = 6;
    constexpr int32_t vocabSize = 150;
    constexpr int32_t maxDepth = 5;

    std::vector<int32_t> tokenIds(batchSize * numTokens);
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: long chain [90->91->92->93->94]
    tokenIds[0 * numTokens + 0] = 90;
    tokenIds[0 * numTokens + 1] = 91;
    tokenIds[0 * numTokens + 2] = 92;
    tokenIds[0 * numTokens + 3] = 93;
    tokenIds[0 * numTokens + 4] = 94;

    // Batch 1: short chain [90->95->96] with early termination
    tokenIds[1 * numTokens + 0] = 90;
    tokenIds[1 * numTokens + 1] = 95;
    tokenIds[1 * numTokens + 2] = 96;
    tokenIds[1 * numTokens + 3] = 97;
    tokenIds[1 * numTokens + 4] = 98;

    // Batch 2: different long chain [90->97->98->99->89]
    tokenIds[2 * numTokens + 0] = 90;
    tokenIds[2 * numTokens + 1] = 97;
    tokenIds[2 * numTokens + 2] = 98;
    tokenIds[2 * numTokens + 3] = 99;
    tokenIds[2 * numTokens + 4] = 89;

    // Batch 3: minimal chain [90->88] with early termination
    tokenIds[3 * numTokens + 0] = 90;
    tokenIds[3 * numTokens + 1] = 88;
    tokenIds[3 * numTokens + 2] = 87;
    tokenIds[3 * numTokens + 3] = 86;
    tokenIds[3 * numTokens + 4] = 85;

    // Batch 4: full chain [90->87->86->85->84]
    tokenIds[4 * numTokens + 0] = 90;
    tokenIds[4 * numTokens + 1] = 87;
    tokenIds[4 * numTokens + 2] = 86;
    tokenIds[4 * numTokens + 3] = 85;
    tokenIds[4 * numTokens + 4] = 84;

    // Batch 5: no continuation [90] (logits favor non-existent tokens)
    tokenIds[5 * numTokens + 0] = 90;
    tokenIds[5 * numTokens + 1] = 83;
    tokenIds[5 * numTokens + 2] = 82;
    tokenIds[5 * numTokens + 3] = 81;
    tokenIds[5 * numTokens + 4] = 80;

    // Define valid token count per batch for attention mask setup
    int32_t validTokensPerBatch[] = {5, 3, 5, 2, 5, 5};

    // Create triangular attention masks for each batch based on valid token count
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < validTokensPerBatch[b]; ++i)
        {
            for (int32_t j = 0; j <= i; ++j)
            {
                attentionMask[b * numTokens * numTokens + i * numTokens + j] = 1;
            }
        }
    }

    // Setup logits to control which tokens are selected at each position
    std::vector<float> logits(batchSize * numTokens * vocabSize, -10.0f);

    // Batch 0: favor path [90->91->92->93->94]
    logits[0 * numTokens * vocabSize + 0 * vocabSize + 91] = 10.0f;
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 92] = 10.0f;
    logits[0 * numTokens * vocabSize + 2 * vocabSize + 93] = 10.0f;
    logits[0 * numTokens * vocabSize + 3 * vocabSize + 94] = 10.0f;

    // Batch 1: favor path [90->95->96], then non-existent token for early termination
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 95] = 10.0f;
    logits[1 * numTokens * vocabSize + 1 * vocabSize + 96] = 10.0f;
    logits[1 * numTokens * vocabSize + 2 * vocabSize + 149] = 10.0f; // non-existent token

    // Batch 2: favor path [90->97->98->99->89]
    logits[2 * numTokens * vocabSize + 0 * vocabSize + 97] = 10.0f;
    logits[2 * numTokens * vocabSize + 1 * vocabSize + 98] = 10.0f;
    logits[2 * numTokens * vocabSize + 2 * vocabSize + 99] = 10.0f;
    logits[2 * numTokens * vocabSize + 3 * vocabSize + 89] = 10.0f;

    // Batch 3: favor path [90->88], then non-existent token for early termination
    logits[3 * numTokens * vocabSize + 0 * vocabSize + 88] = 10.0f;
    logits[3 * numTokens * vocabSize + 1 * vocabSize + 149] = 10.0f; // non-existent token

    // Batch 4: favor path [90->87->86->85->84]
    logits[4 * numTokens * vocabSize + 0 * vocabSize + 87] = 10.0f;
    logits[4 * numTokens * vocabSize + 1 * vocabSize + 86] = 10.0f;
    logits[4 * numTokens * vocabSize + 2 * vocabSize + 85] = 10.0f;
    logits[4 * numTokens * vocabSize + 3 * vocabSize + 84] = 10.0f;

    // Batch 5: favor non-existent token immediately for no continuation
    logits[5 * numTokens * vocabSize + 0 * vocabSize + 149] = 15.0f; // non-existent token
    logits[5 * numTokens * vocabSize + 0 * vocabSize + 83] = 5.0f;   // valid but lower priority

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "AsymmetricTreeMultiBatch",
        [](auto const& acceptedTokenIds, auto const& acceptedIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 5) << "Batch 0: long path A";
            EXPECT_EQ(acceptLengths[1], 3) << "Batch 1: short path B";
            EXPECT_EQ(acceptLengths[2], 5) << "Batch 2: longest path C";
            EXPECT_EQ(acceptLengths[3], 2) << "Batch 3: isolated path";
            EXPECT_EQ(acceptLengths[4], 5) << "Batch 4: full chain";
            EXPECT_EQ(acceptLengths[5], 1) << "Batch 5: no valid continuation";

            EXPECT_EQ(acceptedTokenIds[0 * 5 + 1], 91) << "Batch 0 path A token";
            EXPECT_EQ(acceptedIndices[0 * 5 + 1], 1) << "Batch 0 path A index";
            EXPECT_EQ(acceptedTokenIds[1 * 5 + 1], 95) << "Batch 1 path B token";
            EXPECT_EQ(acceptedIndices[1 * 5 + 1], 1) << "Batch 1 path B index";
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 1], 97) << "Batch 2 path C token";
            EXPECT_EQ(acceptedIndices[2 * 5 + 1], 1) << "Batch 2 path C index";
            EXPECT_EQ(acceptedTokenIds[3 * 5 + 1], 88) << "Batch 3 isolated token";
            EXPECT_EQ(acceptedIndices[3 * 5 + 1], 1) << "Batch 3 isolated index";
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 1], 87) << "Batch 4 chain token";
            EXPECT_EQ(acceptedIndices[4 * 5 + 1], 1) << "Batch 4 chain index";

            EXPECT_EQ(acceptedTokenIds[1 * 5 + 3], -1) << "Batch 1 unused position should be -1";
            EXPECT_EQ(acceptedIndices[1 * 5 + 3], -1) << "Batch 1 unused position should be -1";
            EXPECT_EQ(acceptedTokenIds[3 * 5 + 2], -1) << "Batch 3 unused position should be -1";
            EXPECT_EQ(acceptedIndices[3 * 5 + 2], -1) << "Batch 3 unused position should be -1";
            EXPECT_EQ(acceptedTokenIds[5 * 5 + 1], -1) << "Batch 5 unused position should be -1";
            EXPECT_EQ(acceptedIndices[5 * 5 + 1], -1) << "Batch 5 unused position should be -1";
        });
}

// Test multi-batch trees with different path lengths and attention patterns
TEST_F(EagleAcceptTest, ComplexMultiBranchTree)
{
    constexpr int32_t numTokens = 4;
    constexpr int32_t vocabSize = 1000;
    constexpr int32_t maxDepth = 4;
    constexpr int32_t batchSize = 3;

    std::vector<int32_t> tokenIds(batchSize * numTokens);
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: full chain [100->200->300->400]
    tokenIds[0 * numTokens + 0] = 100;
    tokenIds[0 * numTokens + 1] = 200;
    tokenIds[0 * numTokens + 2] = 300;
    tokenIds[0 * numTokens + 3] = 400;

    // Create full triangular attention mask for batch 0
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[0 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Batch 1: different full chain [100->202->302->402]
    tokenIds[1 * numTokens + 0] = 100;
    tokenIds[1 * numTokens + 1] = 202;
    tokenIds[1 * numTokens + 2] = 302;
    tokenIds[1 * numTokens + 3] = 402;

    // Create full triangular attention mask for batch 1
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[1 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Batch 2: partial chain [100->201->304] with early termination
    tokenIds[2 * numTokens + 0] = 100;
    tokenIds[2 * numTokens + 1] = 201;
    tokenIds[2 * numTokens + 2] = 304;
    tokenIds[2 * numTokens + 3] = 999; // invalid token

    // Create partial triangular attention mask for batch 2 (only first 3 tokens)
    for (int32_t i = 0; i < 3; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[2 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Setup logits to guide token selection
    std::vector<float> logits(batchSize * numTokens * vocabSize, -10.0f);

    // Batch 0: favor path [100->200->300->400]
    logits[0 * numTokens * vocabSize + 0 * vocabSize + 200] = 10.0f;
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 300] = 10.0f;
    logits[0 * numTokens * vocabSize + 2 * vocabSize + 400] = 10.0f;

    // Batch 1: favor path [100->202->302->402]
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 202] = 10.0f;
    logits[1 * numTokens * vocabSize + 1 * vocabSize + 302] = 10.0f;
    logits[1 * numTokens * vocabSize + 2 * vocabSize + 402] = 10.0f;

    // Batch 2: favor path [100->201->304], then non-existent token
    logits[2 * numTokens * vocabSize + 0 * vocabSize + 201] = 10.0f;
    logits[2 * numTokens * vocabSize + 1 * vocabSize + 304] = 10.0f;
    logits[2 * numTokens * vocabSize + 2 * vocabSize + 999] = 10.0f; // non-existent token

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "ComplexMultiBranchTree",
        [](auto const& acceptedTokenIds, auto const& acceptedIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 4) << "Batch 0 should complete full path";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 0], 100) << "Batch 0 token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 1], 200) << "Batch 0 token 1";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 2], 300) << "Batch 0 token 2";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 3], 400) << "Batch 0 token 3";
            EXPECT_EQ(acceptedIndices[0 * 4 + 0], 0) << "Batch 0 index 0";
            EXPECT_EQ(acceptedIndices[0 * 4 + 1], 1) << "Batch 0 index 1";
            EXPECT_EQ(acceptedIndices[0 * 4 + 2], 2) << "Batch 0 index 2";
            EXPECT_EQ(acceptedIndices[0 * 4 + 3], 3) << "Batch 0 index 3";

            EXPECT_EQ(acceptLengths[1], 4) << "Batch 1 should complete different path";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 0], 100) << "Batch 1 token 0";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 1], 202) << "Batch 1 token 1";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 2], 302) << "Batch 1 token 2";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 3], 402) << "Batch 1 token 3";
            EXPECT_EQ(acceptedIndices[1 * 4 + 0], 0) << "Batch 1 index 0";
            EXPECT_EQ(acceptedIndices[1 * 4 + 1], 1) << "Batch 1 index 1";
            EXPECT_EQ(acceptedIndices[1 * 4 + 2], 2) << "Batch 1 index 2";
            EXPECT_EQ(acceptedIndices[1 * 4 + 3], 3) << "Batch 1 index 3";

            EXPECT_EQ(acceptLengths[2], 3) << "Batch 2 should terminate early";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 0], 100) << "Batch 2 token 0";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 1], 201) << "Batch 2 token 1";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 2], 304) << "Batch 2 token 2";
            EXPECT_EQ(acceptedIndices[2 * 4 + 0], 0) << "Batch 2 index 0";
            EXPECT_EQ(acceptedIndices[2 * 4 + 1], 1) << "Batch 2 index 1";
            EXPECT_EQ(acceptedIndices[2 * 4 + 2], 2) << "Batch 2 index 2";

            EXPECT_EQ(acceptedTokenIds[2 * 4 + 3], -1) << "Batch 2 unused token position should be -1";
            EXPECT_EQ(acceptedIndices[2 * 4 + 3], -1) << "Batch 2 unused index position should be -1";
        });
}

// Test single batch with logit-controlled early termination
TEST_F(EagleAcceptTest, SingleBatchLogitTermination)
{
    constexpr int32_t numTokens = 5;
    constexpr int32_t batchSize = 1;
    constexpr int32_t vocabSize = 50;
    constexpr int32_t maxDepth = 5;

    // Setup a complete tree but use logits to cause early termination
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0] = 15; // root
    tokenIds[1] = 25; // depth 2
    tokenIds[2] = 35; // depth 3
    tokenIds[3] = 45; // depth 4
    tokenIds[4] = 49; // depth 5

    // Create full triangular attention mask (all tokens can attend to previous ones)
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[i * numTokens + j] = 1;
        }
    }

    // Setup logits to cause termination at position 2
    std::vector<float> logits(batchSize * numTokens * vocabSize, -10.0f);

    logits[0 * vocabSize + 25] = 10.0f; // pos 0 -> token 25 (valid)
    logits[1 * vocabSize + 35] = 10.0f; // pos 1 -> token 35 (valid)
    logits[2 * vocabSize + 48] = 15.0f; // pos 2 -> token 48 (not in tree, higher priority)
    logits[2 * vocabSize + 45] = 5.0f;  // pos 2 -> token 45 (valid but lower priority)

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "SingleBatchLogitTermination",
        [](auto const& acceptedTokenIds, auto const& acceptedIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Should terminate at position 2 due to logits";
            EXPECT_EQ(acceptedTokenIds[0], 15) << "Root token";
            EXPECT_EQ(acceptedTokenIds[1], 25) << "Second token";
            EXPECT_EQ(acceptedTokenIds[2], 35) << "Third token";
            EXPECT_EQ(acceptedIndices[0], 0) << "Root index";
            EXPECT_EQ(acceptedIndices[1], 1) << "Second index";
            EXPECT_EQ(acceptedIndices[2], 2) << "Third index";

            EXPECT_EQ(acceptedTokenIds[3], -1) << "Unused token position should be -1";
            EXPECT_EQ(acceptedIndices[3], -1) << "Unused index position should be -1";
            EXPECT_EQ(acceptedTokenIds[4], -1) << "Unused token position should be -1";
            EXPECT_EQ(acceptedIndices[4], -1) << "Unused index position should be -1";
        });
}

// Test device validation - should reject CPU tensors
TEST_F(EagleAcceptTest, DeviceValidation)
{
    // Create one CPU tensor (logits) while others are on GPU - should cause validation failure
    rt::Tensor logitsTensor({2, 4, 10}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "logits");
    rt::Tensor tokenIdsTensor({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
    rt::Tensor attentionMaskTensor({2, 4, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "attentionMask");
    rt::Tensor acceptedTokenIdsTensor({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedTokenIds");
    rt::Tensor acceptedIndicesTensor({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedIndices");
    rt::Tensor acceptLengthTensor({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptLength");

    // Allocate workspace (still needed for function call)
    size_t workspaceSize = kernel::getEagleAcceptWorkspaceSize(2, 4);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Kernel should throw due to CPU tensor
    EXPECT_THROW(kernel::eagleAccept(logitsTensor, tokenIdsTensor, attentionMaskTensor, acceptedTokenIdsTensor,
                     acceptedIndicesTensor, acceptLengthTensor, 4, workspace, workspaceSize, stream),
        std::runtime_error)
        << "Should reject CPU tensor";

    CUDA_CHECK(cudaFree(workspace));
}
