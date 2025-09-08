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

#include "multimodal/multimodalRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

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
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir, cudaStream_t stream);
    ~LLMInferenceRuntime() = default;

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    bool captureDecodingCUDAGraph(cudaStream_t stream);

    //! Execute the prefill step generation of the KVCache for the prompt and save for later usage.
    //! Input:
    //! - prompt: The system prompt to generate the KVCache.
    //! - stream: The CUDA stream used for the generation.
    //! Output:
    //! - true if the KVCache is generated and saved successfully, false otherwise.
    bool genAndSaveSystemPromptKVCache(std::string const& prompt, cudaStream_t stream);

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

    LLMEngineRunnerConfig mEngineConfig{};

    bool examineRequest(LLMGenerationRequest const& request);

    bool setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, cudaStream_t stream);
};
} // namespace rt
} // namespace drivellm
