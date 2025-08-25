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
#include "tokenizer/tokenizer.h"

namespace drivellm
{
namespace rt
{

struct LLMGenerationRequest
{
    struct Prompt
    {
        std::string systemPrompt;
        std::string userPrompt;
    };
    std::vector<Prompt> prompts;
    std::vector<std::vector<rt::imageUtils::ImageData>> imageBuffers;
    float temperature;
    float topP;
    int64_t topK;
    int64_t maxGenerateLength; // Max length of the generated tokens.
};

struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds;
    std::vector<std::string> outputTexts;
};

class LLMInferenceRuntime
{
public:
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir, cudaStream_t stream);
    ~LLMInferenceRuntime() = default;

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    bool captureDecodingCUDAGraph(cudaStream_t stream);

private:
    std::unique_ptr<LLMEngineRunner> mLLMEngineRunner{nullptr};
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};

    rt::Tensor mSamplingWorkspace{};
    rt::Tensor mInputIds{};
    rt::Tensor mHostContextLengths{};
    rt::Tensor mOutputLogits{};
    rt::Tensor mSelectedIndices{};

    LLMEngineRunnerConfig mEngineConfig{};

    bool prepareInputIds(LLMGenerationRequest const& request, std::vector<int32_t>& packedInputIds,
        std::vector<int32_t>& inputIdsLengths);
    bool packInputIds(std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& inputIdsLengths,
        std::vector<int32_t>& packedInputIds);
    bool getInputTexts(LLMGenerationRequest const& request, std::vector<std::string>& inputTexts);
};
} // namespace rt
} // namespace drivellm
