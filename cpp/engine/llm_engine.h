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

#pragma once

#include "common/benchmarkProfiler.h"
#include "decoder/decoder.h"
#include "eagle/eagle.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>

struct GenerationConfig;

struct EngineConfig
{
    std::string baseEnginePath;
    std::string eagleEnginePath{""};
    int32_t maxPathLen{6};
    int32_t topK{10};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
    bool useCudaGraph{true};
    EngineConfig() = default;
    EngineConfig(std::string const& base_engine_path);
    EngineConfig(std::string const& base_engine_path, std::string const& eagle_engine_path, int32_t max_path_len,
        int32_t top_k, bool is_eagle3, int32_t max_decoding_tokens, bool use_cuda_graph = true);
};

template <typename T>
class LLMEngine
{
public:
    using ModelPtr = std::variant<std::unique_ptr<Decoder<T>>, std::unique_ptr<Eagle<T>>>;

    explicit LLMEngine(EngineConfig const& config, cudaStream_t stream);

    LLMEngine(LLMEngine const&) = delete;
    LLMEngine& operator=(LLMEngine const&) = delete;
    LLMEngine(LLMEngine&&) = default;
    LLMEngine& operator=(LLMEngine&&) = default;

    ~LLMEngine() = default;
    bool isEagleModel() const;

    int32_t getMaxContextLength() const;
    int64_t getBatchSize() const;
    std::unique_ptr<Decoder<T>>& getDecoder();
    std::unique_ptr<Eagle<T>>& getEagle();
    int64_t getDeviceMemorySize();
    void getLastHostLogits(std::vector<T>& hostLogits);
    void generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> const& contextLengths,
        std::vector<std::vector<int64_t>>& outputIds, GenerationConfig const& generationConfig,
        std::vector<int32_t>* newTokensNumbers = nullptr, std::vector<int32_t>* iterNumbers = nullptr,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr, Tokenizer* tokenizer = nullptr,
        bool autoDecode = false);

    // Input processing methods
    void processInputSequence(std::string const& inputString, Tokenizer* tokenizer,
        std::vector<int32_t>& contextLengths, std::vector<int64_t>& inputIds, int64_t batchIdx, int64_t padId,
        bool truncate = true, bool padding = true);

private:
    ModelPtr mModel;
    int32_t mMaxContextLength;
    int64_t mBatchSize;
    bool mIsEagle3;

    int64_t calculatePrefixSum(std::vector<int32_t> const& contextLengths, int32_t i) const;
    ModelPtr createModel(EngineConfig const& config, cudaStream_t stream);
    void updateModelDimensions();
};

using LLMEngineHalf = LLMEngine<half>;
