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
    std::string engineDir;
    std::string baseModelDir{""};
    std::string draftModelDir{""};
    int32_t maxPathLen{6};
    int32_t topK{10};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
    bool useCudaGraph{true};
    int32_t batchSize{1};
    EngineConfig() = default;
    EngineConfig(std::string const& engine_dir, bool useCudaGraph = true, int32_t batch_size = 1);
    EngineConfig(std::string const& engine_dir, std::string const& base_model_dir, std::string const& draft_model_dir,
        int32_t max_path_len, int32_t top_k, bool is_eagle3, int32_t max_decoding_tokens, bool use_cuda_graph = true);
};

class LLMEngine
{
public:
    using LogitsType = float;
    using ModelPtr = std::variant<std::unique_ptr<Decoder>, std::unique_ptr<Eagle>>;

    explicit LLMEngine(EngineConfig const& config, cudaStream_t stream);

    LLMEngine(LLMEngine const&) = delete;
    LLMEngine& operator=(LLMEngine const&) = delete;
    LLMEngine(LLMEngine&&) = default;
    LLMEngine& operator=(LLMEngine&&) = default;

    ~LLMEngine() = default;
    bool isEagleModel() const;

    int32_t getMinSupportedInputLength() const;
    int32_t getMaxSupportedInputLength() const;
    int64_t getBatchSize() const;
    std::unique_ptr<Decoder>& getDecoder();
    std::unique_ptr<Eagle>& getEagle();
    int64_t getDeviceMemorySize();
    ModelConfig getBaseModelConfig();
    void getLastHostLogits(std::vector<LogitsType>& hostLogits);
    void setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs);
    void setupRopeCosSin();

    void generate(std::vector<int32_t> const& inputIds, std::vector<int32_t> const& contextLengths,
        std::vector<std::vector<int32_t>>& outputIds, GenerationConfig const& generationConfig,
        std::vector<int32_t>* newTokensNumbers = nullptr, std::vector<int32_t>* iterNumbers = nullptr,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr, Tokenizer* tokenizer = nullptr,
        bool autoDecode = false);

    // Input processing methods
    std::vector<int32_t> processInputSequence(std::vector<std::string> const& inputStrings, Tokenizer* tokenizer,
        std::vector<int32_t>& contextLengths, int32_t padId);

private:
    ModelPtr mModel;
    int32_t mMaxSupportedInputLength;
    int32_t mMinSupportedInputLength;
    int64_t mBatchSize;
    bool mIsEagle3;

    ModelPtr createModel(EngineConfig const& config, cudaStream_t stream);
    void updateModelDimensions();
};
