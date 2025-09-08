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

#include "exampleUtils.h"
#include "multimodal/multimodalRunner.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace drivellm;

std::unique_ptr<rt::LLMEngine> getLLMEngine(int32_t batchSize, BaseParams const& baseParams,
    EagleParams const& eagleParams, LoraWeights const& loraWeights, cudaStream_t stream)
{
    rt::EngineConfig engineConfig;
    bool eagleMode = !eagleParams.baseModelDir.empty() || !eagleParams.draftModelDir.empty();
    if (eagleMode && batchSize != 1)
    {
        throw std::runtime_error("Eagle only supports batch size 1 currently!");
    }

    if (eagleMode)
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = rt::EngineConfig(baseParams.engineDir, eagleParams.baseModelDir, eagleParams.draftModelDir,
            eagleParams.maxPathLen, eagleParams.topK, eagleParams.isEagle3, eagleParams.maxDecodingTokens,
            !baseParams.noCudaGraph);
    }
    else
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = rt::EngineConfig(baseParams.engineDir, !baseParams.noCudaGraph, batchSize);
    }

    auto llmEngine = std::make_unique<rt::LLMEngine>(engineConfig, stream);

    // Load and switch to LoRA weights if provided
    if (loraWeights.hasWeights() && !eagleMode)
    {
        auto& decoderPtr = llmEngine->getDecoder();
        auto loraPair = loraWeights.getFirst();
        if (!decoderPtr->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return nullptr;
        }
        if (!decoderPtr->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return nullptr;
        }
    }

    llmEngine->setupRopeCosSin();

    return llmEngine;
}