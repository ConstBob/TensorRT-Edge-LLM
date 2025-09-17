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

#include "multimodal/multimodalRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <string>
#include <unordered_map>

namespace drivellm
{
namespace rt
{

struct SystemPromptKVCache
{
    std::string systemPrompt;
    std::vector<tokenizer::Rank> tokenizedPrompt;
    rt::Tensor kvCacheContent;
};

class LLMInferenceRuntime
{
public:
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);
    ~LLMInferenceRuntime() = default;

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    bool captureDecodingCUDAGraph(cudaStream_t stream);

    //! Execute the prefill step generation of the KVCache for the prompt and save for later usage.
    //! Input:
    //! - prompt: The system prompt to generate the KVCache.
    //! - loraWeightsName: The name of the LoRA weights.
    //! - stream: The CUDA stream used for the generation.
    //! Output:
    //! - true if the KVCache is generated and saved successfully, false otherwise.
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

private:
    std::unique_ptr<LLMEngineRunner> mLLMEngineRunner{nullptr};
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};
    std::unordered_map<size_t, SystemPromptKVCache> mSystemPromptKVCache{};

    rt::Tensor mSamplingWorkspace{};
    rt::Tensor mInputIds{};
    rt::Tensor mHostContextLengths{};
    rt::Tensor mOutputLogits{};
    rt::Tensor mSelectedIndices{};
    std::string mEmptyLoraWeightsName{""};

    LLMEngineRunnerConfig mEngineConfig{};

    bool examineRequest(LLMGenerationRequest const& request);

    bool setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream);
};
} // namespace rt
} // namespace drivellm
