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
#include "multimodal/internViTRunner.h"
#include "multimodal/qwenViTRunner.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace drivellm;
using namespace drivellm::rt;

// Ensure STB implementation is only defined once
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

#include <stb_image.h>
#include <stb_image_resize2.h>

// Image loading and resizing helper functions
ImageData loadImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (data == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));
    }
    return ImageData(data, width, height, channels);
}

ImageData loadImageFromMemory(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    if (imageData == nullptr)
    {
        throw std::runtime_error("Failed to load image from memory: " + std::string(stbi_failure_reason()));
    }
    return ImageData(imageData, width, height, desiredChannels);
}

ImageData resizeImage(ImageData const& image, int newWidth, int newHeight, bool isThumbnail)
{
    if (newWidth <= 0 || newHeight <= 0)
    {
        throw std::invalid_argument("New dimensions must be positive");
    }

    // Allocate memory for resized image
    unsigned char* resizedData = (unsigned char*) malloc(newWidth * newHeight * image.channels);
    if (resizedData == nullptr)
    {
        throw std::runtime_error("Failed to allocate memory for resized image");
    }

    // Resize the image
    stbir_resize_uint8_linear(
        image.data(), image.width, image.height, 0, resizedData, newWidth, newHeight, 0, STBIR_RGB);

    return ImageData(resizedData, newWidth, newHeight, image.channels, isThumbnail);
}

std::unique_ptr<MultimodalRunner> getMultimodalRunner(VLMRunParams const& vlmRunParams, cudaStream_t stream)
{
    std::unique_ptr<MultimodalRunner> multimodalRunner;

    // Read config.json to determine model type
    std::string configPath = vlmRunParams.visualEngineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        throw std::runtime_error("Failed to open config file: " + configPath);
    }

    nlohmann::json jsonConfig;
    try
    {
        jsonConfig = nlohmann::json::parse(configFileStream);
        configFileStream.close();
    }
    catch (nlohmann::json::parse_error const& e)
    {
        throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
    }

    std::string modelType = jsonConfig["model_type"].get<std::string>();

    if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
    {
        multimodalRunner = std::make_unique<QwenViTRunner>(vlmRunParams.visualEngineDir, stream);
    }
    else if (modelType == "internvl")
    {
        multimodalRunner = std::make_unique<InternViTRunner>(vlmRunParams.visualEngineDir, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported model type: " + modelType);
    }

    return multimodalRunner;
}

std::unique_ptr<LLMEngine> getLLMEngine(int32_t batchSize, BaseParams const& baseParams, EagleParams const& eagleParams,
    LoraWeights const& loraWeights, cudaStream_t stream)
{
    EngineConfig engineConfig;
    bool eagleMode = !eagleParams.baseModelDir.empty() || !eagleParams.draftModelDir.empty();
    if (eagleMode && batchSize != 1)
    {
        throw std::runtime_error("Eagle only supports batch size 1 currently!");
    }

    if (eagleMode)
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = EngineConfig(baseParams.engineDir, eagleParams.baseModelDir, eagleParams.draftModelDir,
            eagleParams.maxPathLen, eagleParams.topK, eagleParams.isEagle3, eagleParams.maxDecodingTokens,
            !baseParams.noCudaGraph);
    }
    else
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = EngineConfig(baseParams.engineDir, !baseParams.noCudaGraph, batchSize);
    }

    auto llmEngine = std::make_unique<LLMEngine>(engineConfig, stream);

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