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