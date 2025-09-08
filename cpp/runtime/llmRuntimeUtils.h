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

#include "common/tensor.h"
#include "runtime/imageUtils.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace drivellm
{
namespace rt
{

// LLM Generation Request/Response types
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

enum class RopeType
{
    // Default 1-D RoPE that specified by the original paper.
    kDefault,
    // Dynamic RoPE type used by InternVL-3.
    kDynamic,
    // Long RoPE type used by Phi-4.
    kLongRope,
    // MRope type used by Qwen2-VL
    kMRope,
};

struct RopeCommonConfig
{
    RopeType type{};
    // Instantiate the RopeConfig with common default values.
    float rotaryScale{1.0F};
    float rotaryTheta{100000.0F};
    int32_t maxPositionEmbeddings{32768};
};

// Collect basic rope configuration from the model config.
// Input:
//     config [JSON]: The model config file supplied with the model.
// Output:
//     The parsed rope configuration. Default values are used if certain fields are not
//     specified in the model config.
RopeCommonConfig collectBaseRopeConfig(nlohmann::json const& config);

// Initialize the rope cos/sin cache tensor for persistent type of RoPE (default, longrope)
// Input:
//     cosSinCache [GPU]: The tensor to store the rope cos/sin cache.
//     config [RopeCommonConfig]: The basic rope configuration.
//     modelConfig [JSON]: Model config json that can supply additional information for the rope initialization.
//     stream [CUDA stream]: The stream to execute the initialization.
// Returns:
//     True if the initialization is successful, false otherwise.
bool initializeRopeCosSinCache(
    rt::Tensor& cosSinCache, RopeCommonConfig const& config, nlohmann::json const& modelConfig, cudaStream_t stream);

} // namespace rt
} // namespace drivellm