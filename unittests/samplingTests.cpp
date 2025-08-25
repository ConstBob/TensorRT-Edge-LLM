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

#include "common/common.h"
#include "references.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <vector>

using namespace drivellm;

// Test configuration
int32_t const ACCURACY_BATCH_SIZE = 4;
int32_t const ACCURACY_VOCAB_SIZE = 20;
uint64_t const TEST_SEED = 42;

// Test fixture for sampling tests
class SamplingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
    }

    void TearDown() override
    {
        // Cleanup is handled by individual tests
    }

    // Generate deterministic test logits (FP32 only)
    void generateTestLogits(float* dLogits, std::vector<std::vector<float>>& hostLogits, int batchSize, int vocabSize)
    {
        hostLogits.resize(batchSize);
        std::vector<float> flatHostLogits(batchSize * vocabSize);

        // Generate deterministic but varied logits for testing using testUtils
        for (int b = 0; b < batchSize; ++b)
        {
            hostLogits[b].resize(vocabSize);
            uniformFloatinitialization(hostLogits[b], -2.0f, 2.0f);

            // Add some structure to make testing more interesting
            if (vocabSize <= 20)
            {
                hostLogits[b][0] += 3.0f; // Make token 0 very likely
                hostLogits[b][1] += 2.0f; // Make token 1 second most likely
                for (int v = 0; v < 5; ++v)
                {
                    hostLogits[b][v] += 1.0f; // Make first 5 tokens more likely
                }
            }
            else
            {
                hostLogits[b][0] += 5.0f; // Make token 0 very likely
                for (int v = 0; v < 10; ++v)
                {
                    hostLogits[b][v] += 1.0f; // Make first 10 tokens more likely
                }
            }

            // Ensure minimum differences between logits to prevent numerical instability
            std::sort(hostLogits[b].begin(), hostLogits[b].end(), std::greater<float>());
            for (int v = 1; v < vocabSize; ++v)
            {
                if (std::abs(hostLogits[b][v] - hostLogits[b][v - 1]) < 0.01f)
                {
                    hostLogits[b][v] = hostLogits[b][v - 1] - 0.01f;
                }
            }

            // Shuffle the logits to ensure we're not testing with sorted data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(hostLogits[b].begin(), hostLogits[b].end(), gen);

            // Copy to flat array (FP32 only)
            for (int v = 0; v < vocabSize; ++v)
            {
                flatHostLogits[b * vocabSize + v] = hostLogits[b][v];
            }
        }

        // Copy host data to device memory
        CUDA_CHECK(
            cudaMemcpy(dLogits, flatHostLogits.data(), batchSize * vocabSize * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Validate sampling results (FP32 only)
    bool validateSamplingResults(std::vector<int32_t> const& gpuResults,
        std::vector<std::vector<float>> const& hostLogits, SamplingParams const& params)
    {
        bool allValid = true;
        for (int b = 0; b < static_cast<int>(gpuResults.size()); ++b)
        {
            std::set<int32_t> allowedTokens;

            if (params.useTopK && params.useTopP)
            {
                // Combined top-k and top-p
                allowedTokens
                    = getCombinedAllowedTokensRef(hostLogits[b], params.topK, params.topP, params.temperature);
            }
            else if (params.useTopK)
            {
                // Top-k only
                allowedTokens = getTopKAllowedTokensRef(hostLogits[b], params.topK);
            }
            else if (params.useTopP)
            {
                // Top-p only
                allowedTokens = getTopPAllowedTokensRef(hostLogits[b], params.topP, params.temperature);
            }

            // Check if token is in allowed set
            if (allowedTokens.count(gpuResults[b]) == 0)
            {
                // Output detailed debug info without throwing
                std::cout << "=== SAMPLING VALIDATION FAILED ===" << std::endl;
                std::cout << "Batch " << b << ": Token " << gpuResults[b] << " not in allowed set" << std::endl;
                std::cout << "Allowed tokens: ";
                for (auto token : allowedTokens)
                {
                    std::cout << token << " ";
                }
                std::cout << std::endl;
                std::cout << "Logits for batch " << b << ": ";
                for (size_t i = 0; i < hostLogits[b].size(); ++i)
                {
                    std::cout << hostLogits[b][i];
                    if (i < hostLogits[b].size() - 1)
                        std::cout << ", ";
                }
                std::cout << std::endl;
                std::cout << "=================================" << std::endl;

                allValid = false;
            }
        }
        return allValid;
    }

    // Simplified validate selectAllTopK results (FP32 only)
    bool validateSelectAllTopKResults(std::vector<int32_t> const& gpuIndices,
        std::vector<std::vector<float>> const& hostInput, int topK, int batchSize)
    {
        bool allValid = true;
        for (int b = 0; b < batchSize; ++b)
        {
            auto expectedResults = returnAllTopKReference(hostInput[b], topK, false, false, false);

            // Check that we got the right number of elements
            int expectedSize = std::min(topK, static_cast<int>(hostInput[b].size()));
            if (static_cast<int>(expectedResults.size()) != expectedSize)
            {
                std::cout << "Wrong number of elements - expected " << expectedSize << ", got "
                          << expectedResults.size() << std::endl;
                allValid = false;
                continue;
            }

            // Check that GPU indices match expected indices
            for (int k = 0; k < expectedSize; ++k)
            {
                if (b * topK + k >= static_cast<int>(gpuIndices.size()))
                {
                    std::cout << "Index out of bounds for gpuIndices at batch " << b << " position " << k << std::endl;
                    allValid = false;
                    continue;
                }

                int32_t gpuIdx = gpuIndices[b * topK + k];

                // Check if the index is within valid range
                if (gpuIdx < 0 || gpuIdx >= static_cast<int>(hostInput[b].size()))
                {
                    std::cout << "Invalid index " << gpuIdx << " at batch " << b << " position " << k << std::endl;
                    allValid = false;
                    continue;
                }

                // Check if this index is in the expected top-K results
                bool found = false;
                for (auto const& expected : expectedResults)
                {
                    if (expected.second == gpuIdx)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    std::cout << "Index " << gpuIdx << " not found in expected top-K results at batch " << b
                              << " position " << k << std::endl;
                    allValid = false;
                }
            }
        }
        return allValid;
    }
};

// Test error handling for returnLogProbs=true with nullptr topKValues
TEST_F(SamplingTest, SelectAllTopKErrorHandlingReturnLogProbsWithNullptr)
{
    int const batchSize = 2;
    int const vocabSize = 10;
    int const topK = 5;

    // Allocate device memory directly instead of using Thrust
    float* dInput;
    int32_t* dTopKIndices;
    CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * topK * sizeof(int32_t)));

    std::vector<std::vector<float>> hostLogits;
    generateTestLogits(dInput, hostLogits, batchSize, vocabSize);

    // Calculate workspace size and allocate workspace
    size_t workspaceSize = getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Test that calling with returnLogProbs=true and nullptr topKValues throws an exception
    EXPECT_THROW(
        {
            selectAllTopKFromLogits(dInput, nullptr, dTopKIndices, batchSize, vocabSize, topK, workspace, workspaceSize,
                0, true, false, false);
        },
        std::invalid_argument)
        << "Should throw exception when returnLogProbs=true and topKValues=nullptr";

    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dTopKIndices));
}

// Test error handling for returnLogProbs=false with non-null topKValues
TEST_F(SamplingTest, SelectAllTopKErrorHandlingReturnLogProbsFalseWithNonNullTopKValues)
{
    int const batchSize = 2;
    int const vocabSize = 10;
    int const topK = 5;

    // Allocate device memory directly instead of using Thrust
    float* dInput;
    float* dTopKValues;
    int32_t* dTopKIndices;
    CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * topK * sizeof(int32_t)));

    std::vector<std::vector<float>> hostLogits;
    generateTestLogits(dInput, hostLogits, batchSize, vocabSize);

    // Calculate workspace size and allocate workspace
    size_t workspaceSize = getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Test that calling with returnLogProbs=false and non-null topKValues throws an exception
    EXPECT_THROW(
        {
            selectAllTopKFromLogits(dInput, dTopKValues, dTopKIndices, batchSize, vocabSize, topK, workspace,
                workspaceSize, 0, false, false, false);
        },
        std::invalid_argument)
        << "Should throw exception when returnLogProbs=false and topKValues is not nullptr";

    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dTopKValues));
    CUDA_CHECK(cudaFree(dTopKIndices));
}

// Unified sampling tests (accuracy only)
class SamplingTestSuites : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        float topP;
        float temperature;
        bool accuracyPassed;
        std::string errorMessage;
    };

    TestResult runSamplingAccuracyTest(
        std::string const& methodName, int batchSize, int vocabSize, int topK, float topP, float temperature)
    {
        TestResult result;
        result.methodName = methodName;
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.topP = topP;
        result.temperature = temperature;
        result.accuracyPassed = true;
        result.errorMessage = "";

        // Allocate device memory directly instead of using Thrust
        float* dLogits;
        int32_t* dSelectedIndices;
        CUDA_CHECK(cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dSelectedIndices, batchSize * sizeof(int32_t)));

        std::vector<std::vector<float>> hostLogits;
        generateTestLogits(dLogits, hostLogits, batchSize, vocabSize);

        // Run accuracy test
        SamplingParams params(batchSize, vocabSize, temperature, topK, topP);
        size_t workspaceSize = getTopKtopPSamplingWorkspaceSize(batchSize, vocabSize, params);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        topKtopPSamplingFromLogits(dLogits, dSelectedIndices, params, workspace, workspaceSize, 0, TEST_SEED, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        std::vector<int32_t> gpuResults(batchSize);
        CUDA_CHECK(
            cudaMemcpy(gpuResults.data(), dSelectedIndices, batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));

        // Run validation and get result
        bool validationPassed = validateSamplingResults(gpuResults, hostLogits, params);

        // Set result based on validation
        result.accuracyPassed = validationPassed;
        if (!validationPassed)
        {
            result.errorMessage = "Sampling validation failed - check output for details";
        }

        // Single Google Test assertion for comprehensive validation
        EXPECT_TRUE(validationPassed) << "Sampling validation failed for " << methodName
                                      << " with batchSize=" << batchSize << ", vocabSize=" << vocabSize
                                      << ", topK=" << topK << ", topP=" << topP << ", temperature=" << temperature;

        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(dLogits));
        CUDA_CHECK(cudaFree(dSelectedIndices));

        return result;
    }
};

// Unified returnAllTopK tests (accuracy only)
class ReturnAllTopKTests : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        bool returnLogProbs;
        bool normalizeLogProbs;
        bool inputHasProbs;
        bool accuracyPassed;
        std::string errorMessage;
    };

    TestResult runReturnAllTopKAccuracyTest(
        int batchSize, int vocabSize, int topK, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs)
    {
        TestResult result;
        result.methodName = "SelectAllTopK";
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.returnLogProbs = returnLogProbs;
        result.normalizeLogProbs = normalizeLogProbs;
        result.inputHasProbs = inputHasProbs;
        result.accuracyPassed = true;
        result.errorMessage = "";

        // Allocate device memory directly instead of using Thrust
        float* dInput;
        float* dTopKValues = nullptr;
        int32_t* dTopKIndices;
        CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * topK * sizeof(int32_t)));

        std::vector<std::vector<float>> hostLogits;
        std::vector<std::vector<float>> hostProbs;
        std::vector<float> flatHostProbs;

        // Generate test data
        generateTestLogits(dInput, hostLogits, batchSize, vocabSize);

        if (inputHasProbs)
        {
            // Convert logits to probabilities
            hostProbs.resize(batchSize);
            flatHostProbs.resize(batchSize * vocabSize);

            for (int b = 0; b < batchSize; ++b)
            {
                hostProbs[b].resize(vocabSize);
                auto probs = softmaxRef(hostLogits[b]);

                for (int v = 0; v < vocabSize; ++v)
                {
                    hostProbs[b][v] = probs[v];
                    flatHostProbs[b * vocabSize + v] = hostProbs[b][v];
                }
            }

            // Copy probabilities to device memory
            CUDA_CHECK(cudaMemcpy(
                dInput, flatHostProbs.data(), batchSize * vocabSize * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (returnLogProbs)
        {
            CUDA_CHECK(cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float)));
        }

        // Run accuracy test
        size_t workspaceSize = getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        selectAllTopKFromLogits(dInput, returnLogProbs ? dTopKValues : nullptr, dTopKIndices, batchSize, vocabSize,
            topK, workspace, workspaceSize, 0, returnLogProbs, normalizeLogProbs, inputHasProbs);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        std::vector<int32_t> gpuIndices(batchSize * topK);
        CUDA_CHECK(
            cudaMemcpy(gpuIndices.data(), dTopKIndices, batchSize * topK * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::vector<std::vector<float>>& hostInput = inputHasProbs ? hostProbs : hostLogits;
        bool validationPassed = validateSelectAllTopKResults(gpuIndices, hostInput, topK, batchSize);

        // Set result based on validation
        result.accuracyPassed = validationPassed;
        if (!validationPassed)
        {
            result.errorMessage = "SelectAllTopK validation failed - check output for details";
        }

        // Single Google Test assertion for comprehensive validation
        EXPECT_TRUE(validationPassed) << "SelectAllTopK validation failed for batchSize=" << batchSize
                                      << ", vocabSize=" << vocabSize << ", topK=" << topK
                                      << ", returnLogProbs=" << returnLogProbs
                                      << ", normalizeLogProbs=" << normalizeLogProbs
                                      << ", inputHasProbs=" << inputHasProbs;

        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(dInput));
        CUDA_CHECK(cudaFree(dTopKIndices));
        if (dTopKValues != nullptr)
        {
            CUDA_CHECK(cudaFree(dTopKValues));
        }

        return result;
    }
};

// Sampling tests
TEST_F(SamplingTestSuites, SamplingAccuracy)
{
    std::vector<SamplingTestSuites::TestResult> accuracyResults;

    // Test configurations
    struct SamplingConfig
    {
        std::string methodName;
        int topK;
        float topP;
        float temperature;
    };

    std::vector<SamplingConfig> configs = {
        {"TopK", 20, 1.0f, 1.0f},
        {"TopK", 50, 1.0f, 1.0f},
        {"TopK", 100, 1.0f, 1.0f},
        {"TopP", 0, 0.9f, 1.0f},
        {"TopP", 0, 0.95f, 1.0f},
        {"TopP", 0, 0.99f, 1.0f},
        {"TopKTopP", 20, 0.9f, 1.0f},
        {"TopKTopP", 50, 0.95f, 1.0f},
        {"TopKTopP", 100, 0.99f, 1.0f},
        {"TopK", 20, 1.0f, 0.5f},
        {"TopK", 20, 1.0f, 1.5f},
        {"TopP", 0, 0.9f, 0.5f},
        {"TopP", 0, 0.9f, 1.5f},
    };

    // Run accuracy tests with small vocab size
    for (int batchSize : {1, 4})
    {
        for (auto const& config : configs)
        {
            auto result = runSamplingAccuracyTest(
                config.methodName, batchSize, ACCURACY_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            accuracyResults.push_back(result);
        }
    }

    // Print accuracy results table
    std::cout << "\nSampling Accuracy Results (FP32 only):" << std::endl;
    std::cout << "Method   | Batch | AccVocabSize | TopK | TopP  | Temp  | Accuracy" << std::endl;
    std::cout << "---------|-------|--------------|------|-------|-------|----------" << std::endl;

    bool allAccuracyTestsPassed = true;
    std::vector<std::string> accuracyErrorMessages;

    for (auto const& result : accuracyResults)
    {
        std::string topKStr = (result.topK == 0) ? "N/A" : std::to_string(result.topK);

        std::string topPStr;
        if (result.topP == 1.0f)
        {
            topPStr = "N/A";
        }
        else
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << result.topP;
            topPStr = oss.str();
        }

        std::ostringstream tempOss;
        tempOss << std::fixed << std::setprecision(2) << result.temperature;
        std::string tempStr = tempOss.str();

        std::string accuracyStr = result.accuracyPassed ? "PASS" : "FAIL";

        if (!result.accuracyPassed)
        {
            allAccuracyTestsPassed = false;
            accuracyErrorMessages.push_back(result.errorMessage);
        }

        std::cout << std::setw(8) << result.methodName << " | " << std::setw(5) << result.batchSize << " | "
                  << std::setw(12) << result.vocabSize << " | " << std::setw(4) << topKStr << " | " << std::setw(5)
                  << topPStr << " | " << std::setw(5) << tempStr << " | " << std::setw(8) << accuracyStr << std::endl;
    }

    // Print summary
    if (allAccuracyTestsPassed)
    {
        std::cout << "\nAll sampling accuracy tests PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\nSome sampling accuracy tests FAILED!" << std::endl;
        std::cout << "Error details:" << std::endl;
        for (auto const& error : accuracyErrorMessages)
        {
            std::cout << "  - " << error << std::endl;
        }
    }
}

// SelectAllTopK tests
TEST_F(ReturnAllTopKTests, SelectAllTopKAccuracy)
{
    std::vector<ReturnAllTopKTests::TestResult> accuracyResults;

    // Test configurations using booleans
    struct TopKConfig
    {
        int topK;
        bool returnLogProbs;
        bool normalizeLogProbs;
        bool inputHasProbs;
    };

    std::vector<TopKConfig> configs = {
        {10, false, false, false}, // NoLogProbs_Logits
        {10, false, false, true},  // NoLogProbs_Probs
        {10, true, false, false},  // LogProbs_NonNorm_Logits
        {10, true, true, false},   // LogProbs_Norm_Logits
        {10, true, false, true},   // LogProbs_NonNorm_Probs
        {10, true, true, true},    // LogProbs_Norm_Probs
        {5, false, false, false},  // NoLogProbs_Logits_TopK5
        {20, false, false, false}, // NoLogProbs_Logits_TopK20
        {5, true, false, false},   // LogProbs_NonNorm_Logits_TopK5
        {20, true, false, false},  // LogProbs_NonNorm_Logits_TopK20
    };

    // Run accuracy tests with small vocab size
    for (int batchSize : {1, 4})
    {
        for (auto const& config : configs)
        {
            auto result = runReturnAllTopKAccuracyTest(batchSize, ACCURACY_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            accuracyResults.push_back(result);
        }
    }

    // Print accuracy results table
    std::cout << "\nSelectAllTopK Accuracy Results (FP32 only):" << std::endl;
    std::cout << "Batch | TopK | ReturnLogProbs | NormalizeLogProbs | InputHasProbs | AccVocabSize | Accuracy"
              << std::endl;
    std::cout << "------|------|----------------|-------------------|---------------|--------------|----------"
              << std::endl;

    bool allAccuracyTestsPassed = true;
    std::vector<std::string> accuracyErrorMessages;

    for (auto const& result : accuracyResults)
    {
        std::string returnLogProbsStr = result.returnLogProbs ? "true" : "false";
        std::string normalizeLogProbsStr = result.normalizeLogProbs ? "true" : "false";
        std::string inputHasProbsStr = result.inputHasProbs ? "true" : "false";
        std::string accuracyStr = result.accuracyPassed ? "PASS" : "FAIL";

        if (!result.accuracyPassed)
        {
            allAccuracyTestsPassed = false;
            accuracyErrorMessages.push_back(result.errorMessage);
        }

        std::cout << std::setw(5) << result.batchSize << " | " << std::setw(4) << result.topK << " | " << std::setw(14)
                  << returnLogProbsStr << " | " << std::setw(17) << normalizeLogProbsStr << " | " << std::setw(13)
                  << inputHasProbsStr << " | " << std::setw(12) << result.vocabSize << " | " << std::setw(8)
                  << accuracyStr << std::endl;
    }

    // Print summary
    if (allAccuracyTestsPassed)
    {
        std::cout << "\nAll SelectAllTopK accuracy tests PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\nSome SelectAllTopK accuracy tests FAILED!" << std::endl;
        std::cout << "Error details:" << std::endl;
        for (auto const& error : accuracyErrorMessages)
        {
            std::cout << "  - " << error << std::endl;
        }
    }
}