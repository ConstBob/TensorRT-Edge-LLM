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

#pragma once

#include "common/benchmarkProfiler.h"
#include "decoder/decoder.h"
#include "eagle/eagle.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace drivellm
{
namespace rt
{
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
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr,
        drivellm::tokenizer::Tokenizer* tokenizer = nullptr, bool autoDecode = false);

    // Input processing methods
    std::vector<int32_t> processInputSequence(std::vector<std::string> const& inputStrings,
        drivellm::tokenizer::Tokenizer* tokenizer, std::vector<int32_t>& contextLengths, int32_t padId);

private:
    ModelPtr mModel;
    int32_t mMaxSupportedInputLength;
    int32_t mMinSupportedInputLength;
    int64_t mBatchSize;
    bool mIsEagle3;

    ModelPtr createModel(EngineConfig const& config, cudaStream_t stream);
    void updateModelDimensions();
};

} // namespace rt
} // namespace drivellm