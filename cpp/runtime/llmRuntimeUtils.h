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

namespace trt_edgellm
{
namespace rt
{

/*! \brief LLM Generation Request structure
 */
struct LLMGenerationRequest
{
    /*! \brief Prompt structure containing system prompt, user prompt, and optional images
     */
    struct Prompt
    {
        std::string systemPrompt;                            //!< System prompt text
        std::string userPrompt;                              //!< User prompt text
        std::vector<rt::imageUtils::ImageData> imageBuffers; //!< Optional image data for multimodal inputs
    };
    std::vector<Prompt> prompts;      //!< Vector of prompts for batched requests
    float temperature;                //!< Temperature parameter for sampling
    float topP;                       //!< Top-p (nucleus) sampling parameter
    int64_t topK;                     //!< Top-k sampling parameter
    int64_t maxGenerateLength;        //!< Max length of the generated tokens
    std::string loraWeightsName = ""; //!< Name of the LoRA weights. Default to empty string for no LoRA weights
};

/*! \brief LLM Generation Response structure
 */
struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds; //!< Generated token IDs for each request in the batch
    std::vector<std::string> outputTexts;        //!< Generated text strings for each request in the batch
};

/*! \brief RoPE (Rotary Position Embedding) type enumeration
 */
enum class RopeType
{
    kDefault,  //!< Default 1-D RoPE that specified by the original paper
    kDynamic,  //!< Dynamic RoPE type used by InternVL-3
    kLongRope, //!< Long RoPE type used by Phi-4
    kMRope,    //!< MRope type used by Qwen2-VL
};

/*! \brief Common configuration structure for RoPE (Rotary Position Embedding)
 *
 *  Instantiate the RopeConfig with common default values.
 */
struct RopeCommonConfig
{
    RopeType type{};                      //!< Type of RoPE to use
    float rotaryScale{1.0F};              //!< Scaling factor for rotary embeddings
    float rotaryTheta{100000.0F};         //!< Base frequency for rotary embeddings
    int32_t maxPositionEmbeddings{32768}; //!< Maximum position embeddings supported
};

/*! \brief Collect basic rope configuration from the model config
 *
 *  The parsed rope configuration. Default values are used if certain fields are not
 *  specified in the model config.
 *
 *  \param config [JSON] The model config file supplied with the model
 *  \return The parsed rope configuration
 */
RopeCommonConfig collectBaseRopeConfig(nlohmann::json const& config);

/*! \brief Initialize the rope cos/sin cache tensor for persistent type of RoPE (default, longrope)
 *
 *  \param cosSinCache [GPU] The tensor to store the rope cos/sin cache
 *  \param config [RopeCommonConfig] The basic rope configuration
 *  \param modelConfig [JSON] Model config json that can supply additional information for the rope initialization
 *  \param stream [CUDA stream] The stream to execute the initialization
 *  \return True if the initialization is successful, false otherwise
 */
bool initializeRopeCosSinCache(
    rt::Tensor& cosSinCache, RopeCommonConfig const& config, nlohmann::json const& modelConfig, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm