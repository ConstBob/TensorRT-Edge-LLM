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

#include "internViTRunner.h"
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <tuple>

using Json = nlohmann::json;

namespace drivellm
{
namespace rt
{

InternViTRunner::InternViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    std::string configPath = engineDir + "/config.json";
    validateAndFillConfig(configPath);
    allocateBuffer();
}

void InternViTRunner::validateAndFillConfig(std::string const& configPath)
{
    Json jsonConfig;

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        throw std::runtime_error("InternViTRunner::validateAndFillConfig(): Failed to open config file: " + configPath);
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        throw std::runtime_error(
            "InternViTRunner::validateAndFillConfig(): Failed to parse config file: " + configPath);
    }

    mModelType = jsonConfig["model_type"].get<std::string>();
    if (mModelType != "internvl")
    {
        throw std::invalid_argument("InternViTRunner::validateAndFillConfig(): Invalid model type: " + mModelType);
    }

    auto textConfig = jsonConfig["text_config"];
    mConfig.vocabSize = textConfig["vocab_size"].get<int32_t>();

    auto visionConfig = jsonConfig["vision_config"];
    mConfig.patchSizeH = visionConfig["patch_size"][0].get<int32_t>();
    mConfig.patchSizeW = visionConfig["patch_size"][1].get<int32_t>();
    mConfig.blockImageSizeH = visionConfig["image_size"][0].get<int32_t>();
    mConfig.blockImageSizeW = visionConfig["image_size"][1].get<int32_t>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    mConfig.inputDim = mContext->getTensorShape("input").d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape("output").d[1];
}

void* InternViTRunner::getConfig()
{
    return &mConfig;
}

void InternViTRunner::allocateBuffer()
{
    LOG_INFO(
        "InternViTRunner::allocateBuffer() mConfig.maxHW: %d, mConfig.inputDim: %d", mConfig.maxHW, mConfig.inputDim);
    mVitInput = rt::Tensor({mConfig.maxHW, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mContext->setTensorAddress("input", mVitInput.rawPointer());
    // In InternVL3, VIT output is downsampled by 4x, so output size is maxHW/4
    LOG_INFO("InternViTRunner::allocateBuffer() mConfig.maxHW: %d, mConfig.outHiddenSize: %d", mConfig.maxHW,
        mConfig.outHiddenSize);
    mVitOutput
        = rt::Tensor({mConfig.maxHW / 4, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mContext->setTensorAddress("output", mVitOutput.rawPointer());
}

std::vector<EngineInputDesc> InternViTRunner::getComputedEmbeddings()
{
    std::vector<EngineInputDesc> extraInputs;

    extraInputs.emplace_back(EngineInputDesc{"image_embeds", mVitOutput.rawPointer(), mVitOutput.rawPointer(),
        {2, {mConfig.maxHW / 4, mConfig.outHiddenSize}}, {2, {1, mConfig.outHiddenSize}}});

    return extraInputs;
}

void InternViTRunner::formatPatch(ImageData const& image, std::vector<half>& patches,
    std::vector<int64_t>& imageTokenLengths, int64_t& numImagePerBatch, int64_t& totalSeqLength)
{
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    int64_t curSeqLength = (height / mConfig.patchSizeH) * (width / mConfig.patchSizeW);
    totalSeqLength += curSeqLength;

    int64_t curTokenLength = curSeqLength / 4; // Image token length here is seq/4 because of the downsampling
    if (image.isThumbnail)
    {
        // Add to the last image token length, instead of considered as a new image
        imageTokenLengths.back() += curTokenLength;
    }
    else
    {
        imageTokenLengths.push_back(curTokenLength);
        ++numImagePerBatch;
    }

    std::vector<half> curPatch(height * width * channels);

    // Normalize and transpose into patches of 448x448.
    // TODO: Simplify to [N, 3, 448, 448], no need to follow qwen style
    for (int gridH = 0; gridH < height / mConfig.blockImageSizeH; ++gridH)
    {
        for (int gridW = 0; gridW < width / mConfig.blockImageSizeW; ++gridW)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int mergeH = 0; mergeH < mConfig.blockImageSizeH / mConfig.patchSizeH; ++mergeH)
                {
                    for (int mergeW = 0; mergeW < mConfig.blockImageSizeW / mConfig.patchSizeW; ++mergeW)
                    {
                        for (int patchH = 0; patchH < mConfig.patchSizeH; ++patchH)
                        {
                            for (int patchW = 0; patchW < mConfig.patchSizeW; ++patchW)
                            {

                                // src dimensions: (H, W, C) => (gridH, blockSize, patchSize, gridW, blockSize,
                                // patchSize, C)
                                int originalH = gridH * mConfig.blockImageSizeH + mergeH * mConfig.patchSizeH + patchH;
                                int originalW = gridW * mConfig.blockImageSizeW + mergeW * mConfig.patchSizeW + patchW;

                                unsigned char value
                                    = imageData[originalH * width * channels + originalW * channels + c];
                                half normalized
                                    = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);
                                // dst dimensions: (gridH, gridW, channels) x (blockSize/patchSize, blockSize/patchSize,

                                // patchSize, patchSize)
                                int dstHW = gridH * (width / mConfig.blockImageSizeW) * channels + gridW * channels + c;
                                int dstDim = mergeH * mConfig.blockImageSizeH * mConfig.patchSizeH
                                    + patchH * mConfig.blockImageSizeH + mergeW * mConfig.patchSizeW + patchW;
                                curPatch[dstHW * mConfig.blockImageSizeH * mConfig.blockImageSizeW + dstDim]
                                    = normalized;
                            }
                        }
                    }
                }
            }
        }
    }

    patches.insert(patches.end(), curPatch.begin(), curPatch.end());
}

std::vector<std::pair<int, int>> InternViTRunner::getAllSupportedAspectRatios(
    int const minImageTiles, int const maxImageTiles)
{
    std::vector<std::pair<int, int>> aspectRatios;
    for (int width = 1; width <= maxImageTiles; ++width)
    {
        for (int height = 1; height <= maxImageTiles; ++height)
        {
            if (width * height <= maxImageTiles && width * height >= minImageTiles)
            {
                aspectRatios.emplace_back(width, height);
            }
        }
    }
    std::sort(aspectRatios.begin(), aspectRatios.end(), [](std::pair<int, int> const& a, std::pair<int, int> const& b) {
        return a.first * a.second < b.first * b.second;
    });
    return aspectRatios;
}

std::tuple<int, int> InternViTRunner::resizeImage(int const height, int const width, int const targetTileHeight,
    int const targetTileWidth, int const minImageTiles, int const maxImageTiles)
{
    auto targetRatios = getAllSupportedAspectRatios(minImageTiles, maxImageTiles);
    double const aspectRatio = static_cast<double>(width) / height;
    int const area = width * height;

    double bestRatioDiff = HUGE_VAL;
    std::pair<int, int> bestRatio = {1, 1};
    for (auto const& ratio : targetRatios)
    {
        double const targetAspectRatio = static_cast<double>(ratio.first) / ratio.second;
        double const ratioDiff = std::abs(aspectRatio - targetAspectRatio);

        if (ratioDiff < bestRatioDiff)
        {
            bestRatioDiff = ratioDiff;
            bestRatio = ratio;
        }
        else if (ratioDiff == bestRatioDiff)
        {
            if (area > 0.5 * targetTileHeight * targetTileWidth * ratio.first * ratio.second)
            {
                bestRatio = ratio;
            }
        }
    }
    // return (height, width)
    return {bestRatio.second * targetTileHeight, bestRatio.first * targetTileWidth};
}

void InternViTRunner::imagePreprocess(std::vector<std::vector<ImageData>> const& imageBuffers,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImagePerBatch, cudaStream_t stream)
{
    std::vector<half> patches;
    int64_t totalSeqLength = 0;

    for (auto const& imageBuffer : imageBuffers)
    {
        int64_t numImage{0};
        for (auto const& image : imageBuffer)
        {
            formatPatch(image, patches, imageTokenLengths, numImage, totalSeqLength);
        }
        numImagePerBatch.emplace_back(numImage);
    }

    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    mContext->setInputShape("input", {2, {totalSeqLength, mConfig.inputDim}});
    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
}

std::string InternViTRunner::applyChatTemplate(std::string const& inputString, int const& numImages,
    std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt)
{
    // System prefix
    std::string prompt
        = "<|im_start|>"
          "system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语"
          "言模型。<|im_end|>\n<|im_start|>user\n";

    // Images
    for (int i = 0; i < numImages; ++i)
    {
        int imagePadLen = imageTokenLengths[totalImageIdx++];

        prompt += "<img>";
        for (int j = 0; j < imagePadLen; ++j)
        {
            prompt += "<IMG_CONTEXT>";
        }

        prompt += "</img>\n";
    }

    prompt += inputString;
    prompt += "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }
    return prompt;
}

void InternViTRunner::textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds,
    std::vector<int32_t>& batchInputLengths, std::vector<std::string> const& inputStrings,
    std::vector<int64_t> const& numImagePerBatch, std::vector<int64_t> const& imageTokenLengths,
    drivellm::tokenizer::Tokenizer* tokenizer)
{
    int totalImageIdx = 0;
    int value = mConfig.vocabSize;

    for (size_t i = 0; i < inputStrings.size(); ++i)
    {
        std::string prompt = applyChatTemplate(inputStrings[i], numImagePerBatch[i], imageTokenLengths, totalImageIdx);
        std::vector<int32_t> ids = tokenizer->encode(prompt);

        // replace vis tokens
        for (size_t j = 0; j < ids.size(); ++j)
        {
            // <IMG_CONTEXT>
            if (ids[j] == 151667)
            {
                ids[j] = value;
                ++value;
            }
        }
        batchInputLengths.emplace_back(static_cast<int32_t>(ids.size()));
        batchInputIds.emplace_back(std::move(ids));
    }
}

void InternViTRunner::preprocess(std::vector<std::string> const& inputStrings,
    std::vector<std::vector<ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
    std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer, int const maxSupportedInputLength,
    bool enableDynamicShape, void* ropeRotaryCosSinDevice [[maybe_unused]],
    int const maxPositionEmbeddings [[maybe_unused]], int const rotaryDim [[maybe_unused]], cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImagePerBatch;
    imagePreprocess(imageBuffers, imageTokenLengths, numImagePerBatch, stream);

    std::vector<std::vector<int32_t>> batchInputIds;
    std::vector<int32_t> batchInputLengths;
    textPreprocess(batchInputIds, batchInputLengths, inputStrings, numImagePerBatch, imageTokenLengths, tokenizer);

    flattenBatch(inputIds, contextLengths, batchInputIds, batchInputLengths, tokenizer->getPadId(),
        maxSupportedInputLength, enableDynamicShape);
}

void InternViTRunner::initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
    int const inputLength, cudaStream_t stream)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init visual inputs
    // HW is always 4ximageTokens because downsampling ratio is 0.5 (factor = (1/0.5)^2 = 4)
    int const totalSeqLength = 4 * imageTokenLength;
    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    std::vector<half> patches(totalSeqLength * mConfig.inputDim);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(patches.begin(), patches.end(), [&rng, &dist]() { return __float2half(dist(rng)); });
    mContext->setInputShape("input", {2, {totalSeqLength, mConfig.inputDim}});
    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    // Init input ids
    std::uniform_int_distribution<std::mt19937::result_type> intDist(0, 10000);
    int value = mConfig.vocabSize;
    for (int i = 0; i < batchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * inputLength;
        std::generate(beginIter, beginIter + inputLength, [&rng, &intDist]() { return intDist(rng); });
        // Replace image tokens at the beginning of each batch
        for (int j = 0; j < imageTokenLength; ++j)
        {
            *(beginIter + j) = value;
            ++value;
        }
    }
}

} // namespace rt
} // namespace drivellm