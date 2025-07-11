#include "common/common.h"
#include "references.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

using namespace drivellm;

// Test configuration
const int32_t ACCURACY_BATCH_SIZE = 4;
const int32_t ACCURACY_VOCAB_SIZE = 20;
const std::vector<int32_t> PERFORMANCE_BATCH_SIZES = {1, 4};
const int32_t PERFORMANCE_VOCAB_SIZE = 100000;
const uint64_t TEST_SEED = 42;

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

    // Generate deterministic test logits
    template <typename T>
    void generateTestLogits(T* dLogits, std::vector<std::vector<float>>& hostLogits, int batchSize, int vocabSize)
    {
        hostLogits.resize(batchSize);
        std::vector<T> flatHostLogits(batchSize * vocabSize);

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

            // Convert to appropriate type
            for (int v = 0; v < vocabSize; ++v)
            {
                if constexpr (std::is_same_v<T, half>)
                {
                    flatHostLogits[b * vocabSize + v] = __float2half(hostLogits[b][v]);
                }
                else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                {
                    flatHostLogits[b * vocabSize + v] = __float2bfloat16(hostLogits[b][v]);
                }
                else
                {
                    flatHostLogits[b * vocabSize + v] = static_cast<T>(hostLogits[b][v]);
                }
            }
        }

        CUDA_CHECK(
            cudaMemcpy(dLogits, flatHostLogits.data(), batchSize * vocabSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Validate sampling results
    template <typename T>
    bool validateSamplingResults(std::vector<int64_t> const& gpuResults,
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
            if (allowedTokens.count(static_cast<int32_t>(gpuResults[b])) == 0)
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

    // Validate selectAllTopK results
    template <typename T>
    bool validateSelectAllTopKResults(std::vector<float> const& gpuValues, std::vector<int64_t> const& gpuIndices,
        std::vector<std::vector<float>> const& hostInput, int topK, int batchSize, bool returnLogProbs = false,
        bool normalizeLogProbs = false, bool inputHasProbs = false)
    {
        bool allValid = true;
        for (int b = 0; b < batchSize; ++b)
        {
            auto expectedResults
                = returnAllTopKReference(hostInput[b], topK, returnLogProbs, normalizeLogProbs, inputHasProbs);

            // Check that we got the right number of elements
            if (static_cast<int>(expectedResults.size()) != std::min(topK, static_cast<int>(hostInput[b].size())))
            {
                std::cout << "Wrong number of elements - expected "
                          << std::min(topK, static_cast<int>(hostInput[b].size())) << ", got " << expectedResults.size()
                          << std::endl;
                allValid = false;
                continue;
            }

            // Check that GPU results match expected elements
            for (int k = 0; k < static_cast<int>(expectedResults.size()); ++k)
            {
                // Bounds checking
                if (!checkBounds(b * topK + k, static_cast<int>(gpuIndices.size()), "gpuIndices", b, k))
                {
                    allValid = false;
                    continue;
                }

                int32_t gpuIdx = static_cast<int32_t>(gpuIndices[b * topK + k]);

                // Only check gpuValues if the vector is not empty (returnLogProbs=true case)
                float gpuVal = 0.0f;
                if (!gpuValues.empty())
                {
                    if (!checkBounds(b * topK + k, static_cast<int>(gpuValues.size()), "gpuValues", b, k))
                    {
                        allValid = false;
                        continue;
                    }
                    gpuVal = gpuValues[b * topK + k];
                }

                // Find matching element in expected results
                bool found = false;
                for (auto const& expected : expectedResults)
                {
                    if (expected.second == static_cast<int32_t>(gpuIdx))
                    {
                        // Only check value if gpuValues is not empty
                        if (!gpuValues.empty())
                        {
                            if (!validateValue<T>(gpuVal, expected.first, gpuIdx, b, k, "SelectAllTopK"))
                            {
                                allValid = false;
                            }
                        }
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    std::cout << "Index " << gpuIdx << " not found in expected results at batch " << b << " position "
                              << k << std::endl;
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

    float* dInput;
    int64_t* dTopKIndices;

    CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * vocabSize * sizeof(int64_t)));

    std::vector<std::vector<float>> hostLogits;
    this->template generateTestLogits<float>(dInput, hostLogits, batchSize, vocabSize);

    // Calculate workspace size and allocate workspace
    size_t workspaceSize = getSelectAllTopKWorkspaceSize<float>(batchSize, vocabSize, topK);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Test that calling with returnLogProbs=true and nullptr topKValues throws an exception
    EXPECT_THROW(
        {
            selectAllTopKFromLogits<float>(dInput, nullptr, dTopKIndices, batchSize, vocabSize, topK, workspace,
                workspaceSize, 0, true, false, false);
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

    float* dInput;
    float* dTopKValues;
    int64_t* dTopKIndices;

    CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * vocabSize * sizeof(int64_t)));

    std::vector<std::vector<float>> hostLogits;
    this->template generateTestLogits<float>(dInput, hostLogits, batchSize, vocabSize);

    // Calculate workspace size and allocate workspace
    size_t workspaceSize = getSelectAllTopKWorkspaceSize<float>(batchSize, vocabSize, topK);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Test that calling with returnLogProbs=false and non-null topKValues throws an exception
    EXPECT_THROW(
        {
            selectAllTopKFromLogits<float>(dInput, dTopKValues, dTopKIndices, batchSize, vocabSize, topK, workspace,
                workspaceSize, 0, false, false, false);
        },
        std::invalid_argument)
        << "Should throw exception when returnLogProbs=false and topKValues is not nullptr";

    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dTopKValues));
    CUDA_CHECK(cudaFree(dTopKIndices));
}

// Unified sampling tests (accuracy + performance)
class SamplingTests : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string typeName;
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        float topP;
        float temperature;
        double avgTimeMs;
        double throughputSamplesPerSec;
        bool accuracyPassed;
        std::string errorMessage;
    };

    template <typename T>
    TestResult runSamplingAccuracyTest(
        std::string const& methodName, int batchSize, int vocabSize, int topK, float topP, float temperature)
    {
        TestResult result;
        result.typeName = std::is_same_v<T, float> ? "FP32" : std::is_same_v<T, half> ? "FP16" : "BF16";
        result.methodName = methodName;
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.topP = topP;
        result.temperature = temperature;
        result.accuracyPassed = true;
        result.errorMessage = "";
        result.avgTimeMs = 0.0;
        result.throughputSamplesPerSec = 0.0;

        T* dLogits;
        int64_t* dSelectedIndices;

        CUDA_CHECK(cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dSelectedIndices, batchSize * sizeof(int64_t)));

        std::vector<std::vector<float>> hostLogits;
        this->template generateTestLogits<T>(dLogits, hostLogits, batchSize, vocabSize);

        // Run accuracy test
        SamplingParams params(batchSize, vocabSize, temperature, topK, topP);
        size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<T>(batchSize, vocabSize, params);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        topKtopPSamplingFromLogits<T>(dLogits, dSelectedIndices, params, workspace, workspaceSize, 0, TEST_SEED, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int64_t> gpuResults(batchSize);
        CUDA_CHECK(
            cudaMemcpy(gpuResults.data(), dSelectedIndices, batchSize * sizeof(int64_t), cudaMemcpyDeviceToHost));

        // Run validation and get result
        bool validationPassed = this->template validateSamplingResults<T>(gpuResults, hostLogits, params);

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

    template <typename T>
    TestResult runSamplingPerformanceTest(
        std::string const& methodName, int batchSize, int vocabSize, int topK, float topP, float temperature)
    {
        TestResult result;
        result.typeName = std::is_same_v<T, float> ? "FP32" : std::is_same_v<T, half> ? "FP16" : "BF16";
        result.methodName = methodName;
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.topP = topP;
        result.temperature = temperature;
        result.accuracyPassed = true;
        result.errorMessage = "";

        T* dLogits;
        int64_t* dSelectedIndices;

        CUDA_CHECK(cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dSelectedIndices, batchSize * sizeof(int64_t)));

        std::vector<std::vector<float>> hostLogits;
        this->template generateTestLogits<T>(dLogits, hostLogits, batchSize, vocabSize);

        // Run performance test
        int const numIterations = 5;
        int const warmupIterations = 1;

        // Warmup
        for (int i = 0; i < warmupIterations; ++i)
        {
            SamplingParams params(batchSize, vocabSize, temperature, topK, topP);
            size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<T>(batchSize, vocabSize, params);
            void* workspace;
            cudaMalloc(&workspace, workspaceSize);
            topKtopPSamplingFromLogits<T>(dLogits, dSelectedIndices, params, workspace, workspaceSize, 0, TEST_SEED, 0);
            cudaFree(workspace);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numIterations; ++i)
        {
            SamplingParams params(batchSize, vocabSize, temperature, topK, topP);
            size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<T>(batchSize, vocabSize, params);
            void* workspace;
            cudaMalloc(&workspace, workspaceSize);
            topKtopPSamplingFromLogits<T>(dLogits, dSelectedIndices, params, workspace, workspaceSize, 0, TEST_SEED, 0);
            cudaFree(workspace);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        double totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.avgTimeMs = totalTimeMs / numIterations;
        result.throughputSamplesPerSec = (batchSize * numIterations) / (totalTimeMs / 1000.0);

        CUDA_CHECK(cudaFree(dLogits));
        CUDA_CHECK(cudaFree(dSelectedIndices));

        return result;
    }
};

// Unified returnAllTopK tests (accuracy + performance)
class ReturnAllTopKTests : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string typeName;
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        bool returnLogProbs;
        bool normalizeLogProbs;
        bool inputHasProbs;
        double avgTimeMs;
        double throughputSamplesPerSec;
        bool accuracyPassed;
        std::string errorMessage;
    };

    template <typename T>
    TestResult runReturnAllTopKAccuracyTest(
        int batchSize, int vocabSize, int topK, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs)
    {
        TestResult result;
        result.typeName = std::is_same_v<T, float> ? "FP32" : std::is_same_v<T, half> ? "FP16" : "BF16";
        result.methodName = "SelectAllTopK";
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.returnLogProbs = returnLogProbs;
        result.normalizeLogProbs = normalizeLogProbs;
        result.inputHasProbs = inputHasProbs;
        result.accuracyPassed = true;
        result.errorMessage = "";
        result.avgTimeMs = 0.0;
        result.throughputSamplesPerSec = 0.0;

        T* dInput;
        float* dTopKValues = nullptr;
        int64_t* dTopKIndices;

        CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * vocabSize * sizeof(int64_t)));

        if (returnLogProbs)
        {
            CUDA_CHECK(cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float)));
        }

        std::vector<std::vector<float>> hostLogits;
        std::vector<std::vector<float>> hostProbs;
        std::vector<T> flatHostProbs;

        // Generate test data
        this->template generateTestLogits<T>(dInput, hostLogits, batchSize, vocabSize);

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

                    if constexpr (std::is_same_v<T, half>)
                    {
                        flatHostProbs[b * vocabSize + v] = __float2half(hostProbs[b][v]);
                    }
                    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                    {
                        flatHostProbs[b * vocabSize + v] = __float2bfloat16(hostProbs[b][v]);
                    }
                    else
                    {
                        flatHostProbs[b * vocabSize + v] = static_cast<T>(hostProbs[b][v]);
                    }
                }
            }

            CUDA_CHECK(
                cudaMemcpy(dInput, flatHostProbs.data(), batchSize * vocabSize * sizeof(T), cudaMemcpyHostToDevice));
        }

        // Run accuracy test
        size_t workspaceSize = getSelectAllTopKWorkspaceSize<T>(batchSize, vocabSize, topK);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        selectAllTopKFromLogits<T>(dInput, dTopKValues, dTopKIndices, batchSize, vocabSize, topK, workspace,
            workspaceSize, 0, returnLogProbs, normalizeLogProbs, inputHasProbs);
        CUDA_CHECK(cudaDeviceSynchronize());

        bool validationPassed = false;
        if (returnLogProbs)
        {
            std::vector<float> gpuValues(batchSize * topK);
            std::vector<int64_t> gpuIndices(batchSize * topK);
            CUDA_CHECK(
                cudaMemcpy(gpuValues.data(), dTopKValues, batchSize * topK * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                gpuIndices.data(), dTopKIndices, batchSize * topK * sizeof(int64_t), cudaMemcpyDeviceToHost));

            std::vector<std::vector<float>>& hostInput = inputHasProbs ? hostProbs : hostLogits;
            validationPassed = this->template validateSelectAllTopKResults<T>(
                gpuValues, gpuIndices, hostInput, topK, batchSize, returnLogProbs, normalizeLogProbs, inputHasProbs);
        }
        else
        {
            std::vector<int64_t> gpuIndices(batchSize * topK);
            CUDA_CHECK(cudaMemcpy(
                gpuIndices.data(), dTopKIndices, batchSize * topK * sizeof(int64_t), cudaMemcpyDeviceToHost));

            std::vector<std::vector<float>>& hostInput = inputHasProbs ? hostProbs : hostLogits;
            validationPassed = this->template validateSelectAllTopKResults<T>(
                std::vector<float>(), gpuIndices, hostInput, topK, batchSize);
        }

        // Set result based on validation
        result.accuracyPassed = validationPassed;
        if (!validationPassed)
        {
            result.errorMessage = "SelectAllTopK validation failed - check output for details";
        }

        // Single Google Test assertion for comprehensive validation
        if (returnLogProbs)
        {
            EXPECT_TRUE(validationPassed)
                << "SelectAllTopK log probs validation failed for batchSize=" << batchSize
                << ", vocabSize=" << vocabSize << ", topK=" << topK << ", returnLogProbs=" << returnLogProbs
                << ", normalizeLogProbs=" << normalizeLogProbs << ", inputHasProbs=" << inputHasProbs;
        }
        else
        {
            EXPECT_TRUE(validationPassed)
                << "SelectAllTopK indices validation failed for batchSize=" << batchSize << ", vocabSize=" << vocabSize
                << ", topK=" << topK << ", returnLogProbs=" << returnLogProbs
                << ", normalizeLogProbs=" << normalizeLogProbs << ", inputHasProbs=" << inputHasProbs;
        }

        CUDA_CHECK(cudaFree(workspace));

        CUDA_CHECK(cudaFree(dInput));
        CUDA_CHECK(cudaFree(dTopKIndices));
        if (dTopKValues != nullptr)
        {
            CUDA_CHECK(cudaFree(dTopKValues));
        }

        return result;
    }

    template <typename T>
    TestResult runReturnAllTopKPerformanceTest(
        int batchSize, int vocabSize, int topK, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs)
    {
        TestResult result;
        result.typeName = std::is_same_v<T, float> ? "FP32" : std::is_same_v<T, half> ? "FP16" : "BF16";
        result.methodName = "SelectAllTopK";
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.returnLogProbs = returnLogProbs;
        result.normalizeLogProbs = normalizeLogProbs;
        result.inputHasProbs = inputHasProbs;
        result.accuracyPassed = true;
        result.errorMessage = "";

        T* dInput;
        float* dTopKValues = nullptr;
        int64_t* dTopKIndices;

        CUDA_CHECK(cudaMalloc(&dInput, batchSize * vocabSize * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dTopKIndices, batchSize * vocabSize * sizeof(int64_t)));

        if (returnLogProbs)
        {
            CUDA_CHECK(cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float)));
        }

        std::vector<std::vector<float>> hostLogits;
        std::vector<std::vector<float>> hostProbs;
        std::vector<T> flatHostProbs;

        // Generate test data
        this->template generateTestLogits<T>(dInput, hostLogits, batchSize, vocabSize);

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

                    if constexpr (std::is_same_v<T, half>)
                    {
                        flatHostProbs[b * vocabSize + v] = __float2half(hostProbs[b][v]);
                    }
                    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                    {
                        flatHostProbs[b * vocabSize + v] = __float2bfloat16(hostProbs[b][v]);
                    }
                    else
                    {
                        flatHostProbs[b * vocabSize + v] = static_cast<T>(hostProbs[b][v]);
                    }
                }
            }

            CUDA_CHECK(
                cudaMemcpy(dInput, flatHostProbs.data(), batchSize * vocabSize * sizeof(T), cudaMemcpyHostToDevice));
        }

        // Run performance test
        int const numIterations = 5;
        int const warmupIterations = 1;

        // Warmup
        for (int i = 0; i < warmupIterations; ++i)
        {
            size_t workspaceSize = getSelectAllTopKWorkspaceSize<T>(batchSize, vocabSize, topK);
            void* workspace;
            cudaMalloc(&workspace, workspaceSize);
            selectAllTopKFromLogits<T>(dInput, dTopKValues, dTopKIndices, batchSize, vocabSize, topK, workspace,
                workspaceSize, 0, returnLogProbs, normalizeLogProbs, inputHasProbs);
            cudaFree(workspace);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numIterations; ++i)
        {
            size_t workspaceSize = getSelectAllTopKWorkspaceSize<T>(batchSize, vocabSize, topK);
            void* workspace;
            cudaMalloc(&workspace, workspaceSize);
            selectAllTopKFromLogits<T>(dInput, dTopKValues, dTopKIndices, batchSize, vocabSize, topK, workspace,
                workspaceSize, 0, returnLogProbs, normalizeLogProbs, inputHasProbs);
            cudaFree(workspace);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        double totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.avgTimeMs = totalTimeMs / numIterations;
        result.throughputSamplesPerSec = (batchSize * numIterations) / (totalTimeMs / 1000.0);

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
TEST_F(SamplingTests, SamplingAccuracyAndPerformance)
{
    std::vector<SamplingTests::TestResult> accuracyResults;
    std::vector<SamplingTests::TestResult> performanceResults;

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
    for (int batchSize : PERFORMANCE_BATCH_SIZES)
    {
        for (auto const& config : configs)
        {
            // FP32
            auto resultFp32 = runSamplingAccuracyTest<float>(
                config.methodName, batchSize, ACCURACY_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            accuracyResults.push_back(resultFp32);

            // FP16
            auto resultFp16 = runSamplingAccuracyTest<half>(
                config.methodName, batchSize, ACCURACY_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            accuracyResults.push_back(resultFp16);

            // BF16
            auto resultBf16 = runSamplingAccuracyTest<__nv_bfloat16>(
                config.methodName, batchSize, ACCURACY_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            accuracyResults.push_back(resultBf16);
        }
    }

    // Run performance tests with large vocab size
    for (int batchSize : PERFORMANCE_BATCH_SIZES)
    {
        for (auto const& config : configs)
        {
            // FP32
            auto resultFp32 = runSamplingPerformanceTest<float>(
                config.methodName, batchSize, PERFORMANCE_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            performanceResults.push_back(resultFp32);

            // FP16
            auto resultFp16 = runSamplingPerformanceTest<half>(
                config.methodName, batchSize, PERFORMANCE_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            performanceResults.push_back(resultFp16);

            // BF16
            auto resultBf16 = runSamplingPerformanceTest<__nv_bfloat16>(
                config.methodName, batchSize, PERFORMANCE_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            performanceResults.push_back(resultBf16);
        }
    }

    // Print accuracy results table
    std::cout << "\nSampling Accuracy Results:" << std::endl;
    std::cout << "Type | Method   | Batch | AccVocabSize | TopK | TopP  | Temp  | Accuracy" << std::endl;
    std::cout << "-----|----------|-------|--------------|------|-------|-------|----------" << std::endl;

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

        std::cout << std::setw(4) << result.typeName << " | " << std::setw(8) << result.methodName << " | "
                  << std::setw(5) << result.batchSize << " | " << std::setw(12) << result.vocabSize << " | "
                  << std::setw(4) << topKStr << " | " << std::setw(5) << topPStr << " | " << std::setw(5) << tempStr
                  << " | " << std::setw(8) << accuracyStr << std::endl;
    }

    // Print performance results table
    std::cout << "\nSampling Performance Results:" << std::endl;
    std::cout
        << "Type | Method   | Batch | PerfVocabSize | TopK | TopP  | Temp  | Avg Time (ms) | Throughput (samples/sec)"
        << std::endl;
    std::cout
        << "-----|----------|-------|---------------|------|-------|-------|---------------|-------------------------"
        << std::endl;

    for (auto const& result : performanceResults)
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

        std::cout << std::setw(4) << result.typeName << " | " << std::setw(8) << result.methodName << " | "
                  << std::setw(5) << result.batchSize << " | " << std::setw(13) << result.vocabSize << " | "
                  << std::setw(4) << topKStr << " | " << std::setw(5) << topPStr << " | " << std::setw(5) << tempStr
                  << " | " << std::setw(13) << std::fixed << std::setprecision(4) << result.avgTimeMs << " | "
                  << std::setw(23) << std::fixed << std::setprecision(1) << result.throughputSamplesPerSec << std::endl;
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
TEST_F(ReturnAllTopKTests, SelectAllTopKAccuracyAndPerformance)
{
    std::vector<ReturnAllTopKTests::TestResult> accuracyResults;
    std::vector<ReturnAllTopKTests::TestResult> performanceResults;

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
    for (int batchSize : PERFORMANCE_BATCH_SIZES)
    {
        for (auto const& config : configs)
        {
            // FP32
            auto resultFp32 = runReturnAllTopKAccuracyTest<float>(batchSize, ACCURACY_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            accuracyResults.push_back(resultFp32);

            // FP16
            auto resultFp16 = runReturnAllTopKAccuracyTest<half>(batchSize, ACCURACY_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            accuracyResults.push_back(resultFp16);

            // BF16
            auto resultBf16 = runReturnAllTopKAccuracyTest<__nv_bfloat16>(batchSize, ACCURACY_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            accuracyResults.push_back(resultBf16);
        }
    }

    // Run performance tests with large vocab size
    for (int batchSize : PERFORMANCE_BATCH_SIZES)
    {
        for (auto const& config : configs)
        {
            // FP32
            auto resultFp32 = runReturnAllTopKPerformanceTest<float>(batchSize, PERFORMANCE_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            performanceResults.push_back(resultFp32);

            // FP16
            auto resultFp16 = runReturnAllTopKPerformanceTest<half>(batchSize, PERFORMANCE_VOCAB_SIZE, config.topK,
                config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            performanceResults.push_back(resultFp16);

            // BF16
            auto resultBf16 = runReturnAllTopKPerformanceTest<__nv_bfloat16>(batchSize, PERFORMANCE_VOCAB_SIZE,
                config.topK, config.returnLogProbs, config.normalizeLogProbs, config.inputHasProbs);
            performanceResults.push_back(resultBf16);
        }
    }

    // Print accuracy results table
    std::cout << "\nSelectAllTopK Accuracy Results:" << std::endl;
    std::cout << "Type | Batch | TopK | ReturnLogProbs | NormalizeLogProbs | InputHasProbs | AccVocabSize | Accuracy"
              << std::endl;
    std::cout << "-----|-------|------|----------------|-------------------|---------------|--------------|----------"
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

        std::cout << std::setw(4) << result.typeName << " | " << std::setw(5) << result.batchSize << " | "
                  << std::setw(4) << result.topK << " | " << std::setw(14) << returnLogProbsStr << " | "
                  << std::setw(17) << normalizeLogProbsStr << " | " << std::setw(13) << inputHasProbsStr << " | "
                  << std::setw(12) << result.vocabSize << " | " << std::setw(8) << accuracyStr << std::endl;
    }

    // Print performance results table
    std::cout << "\nSelectAllTopK Performance Results:" << std::endl;
    std::cout << "Type | Batch | TopK | ReturnLogProbs | NormalizeLogProbs | InputHasProbs | PerfVocabSize | Avg Time "
                 "(ms) | Throughput (samples/sec)"
              << std::endl;
    std::cout << "-----|-------|------|----------------|-------------------|---------------|---------------|-----------"
                 "----|-------------------------"
              << std::endl;

    for (auto const& result : performanceResults)
    {
        std::string returnLogProbsStr = result.returnLogProbs ? "true" : "false";
        std::string normalizeLogProbsStr = result.normalizeLogProbs ? "true" : "false";
        std::string inputHasProbsStr = result.inputHasProbs ? "true" : "false";

        std::cout << std::setw(4) << result.typeName << " | " << std::setw(5) << result.batchSize << " | "
                  << std::setw(4) << result.topK << " | " << std::setw(14) << returnLogProbsStr << " | "
                  << std::setw(17) << normalizeLogProbsStr << " | " << std::setw(13) << inputHasProbsStr << " | "
                  << std::setw(13) << result.vocabSize << " | " << std::setw(13) << std::fixed << std::setprecision(3)
                  << result.avgTimeMs << " | " << std::setw(23) << std::fixed << std::setprecision(1)
                  << result.throughputSamplesPerSec << std::endl;
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