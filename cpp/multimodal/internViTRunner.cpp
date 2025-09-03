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
    mConfig.numChannels = visionConfig["num_channels"].get<int32_t>();
    mConfig.patchSizeH = visionConfig["patch_size"][0].get<int32_t>();
    mConfig.patchSizeW = visionConfig["patch_size"][1].get<int32_t>();
    mConfig.blockImageSizeH = visionConfig["image_size"][0].get<int32_t>();
    mConfig.blockImageSizeW = visionConfig["image_size"][1].get<int32_t>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxNumBlocks = inputShapeMax.d[0];
    mConfig.minNumBlocks = inputShapeMin.d[0];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape("output").d[1];
}

void* InternViTRunner::getConfig()
{
    return &mConfig;
}

void InternViTRunner::allocateBuffer()
{
    LOG_INFO(
        "InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.numChannels: %d, mConfig.blockImageSizeH: "
        "%d, mConfig.blockImageSizeW: %d",
        mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mContext->setTensorAddress("input", mVitInput.rawPointer());
    // In InternVL3, each block generates 256 tokens, so output size is maxNumBlocks*256
    LOG_INFO("InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.outHiddenSize: %d",
        mConfig.maxNumBlocks * 256, mConfig.outHiddenSize);
    mOutputEmbedding = rt::Tensor(
        {mConfig.maxNumBlocks * 256, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mContext->setTensorAddress("output", mOutputEmbedding.rawPointer());
}

std::vector<EngineInputDesc> InternViTRunner::getComputedEmbeddings()
{
    std::vector<EngineInputDesc> extraInputs;

    extraInputs.emplace_back(
        EngineInputDesc{"image_embeds", mOutputEmbedding.rawPointer(), mOutputEmbedding.rawPointer(),
            {2, {mConfig.maxNumBlocks * 256, mConfig.outHiddenSize}}, {2, {1, mConfig.outHiddenSize}}});

    return extraInputs;
}

void InternViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<half>& patches,
    std::vector<int64_t>& imageTokenLengths, int64_t& numImagePerBatch, int64_t& totalNumBlocks)
{
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    int64_t curNumBlocks = (height / mConfig.blockImageSizeH) * (width / mConfig.blockImageSizeW);
    totalNumBlocks += curNumBlocks;

    int64_t curTokenLength = curNumBlocks * 256;
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

    // Normalize and transpose into patches of 448x448
    for (int gridH = 0; gridH < height / mConfig.blockImageSizeH; ++gridH)
    {
        for (int gridW = 0; gridW < width / mConfig.blockImageSizeW; ++gridW)
        {
            for (int blockH = 0; blockH < mConfig.blockImageSizeH; ++blockH)
            {
                for (int blockW = 0; blockW < mConfig.blockImageSizeW; ++blockW)
                {
                    for (int c = 0; c < channels; ++c)
                    {

                        // src dimensions: (H, W, C) => (gridH, blockImageSizeH, gridW, blockImageSizeW, C)
                        int originalH = gridH * mConfig.blockImageSizeH + blockH;
                        int originalW = gridW * mConfig.blockImageSizeW + blockW;
                        unsigned char value = imageData[originalH * width * channels + originalW * channels + c];
                        half normalized = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);

                        // dst dimensions: (gridH*gridW, C, blockImageSizeH, blockImageSizeW)
                        int dstNumBlocks = gridH * (width / mConfig.blockImageSizeW) + gridW;
                        curPatch[dstNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW
                            + c * mConfig.blockImageSizeH * mConfig.blockImageSizeW + blockH * mConfig.blockImageSizeW
                            + blockW]
                            = normalized;
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

std::tuple<int, int> InternViTRunner::getResizedImageSize(int const height, int const width, int const targetTileHeight,
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

void InternViTRunner::imagePreprocess(std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImagePerBatch, bool doResize, cudaStream_t stream)
{
    std::vector<half> patches;
    int64_t totalNumBlocks = 0;

    for (auto const& imageBuffer : imageBuffers)
    {
        int64_t numImage{0};
        for (auto const& image : imageBuffer)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width,
                    mConfig.blockImageSizeH, mConfig.blockImageSizeW, mConfig.minImageTiles, mConfig.maxImageTiles);
                auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
                formatPatch(resizedImage, patches, imageTokenLengths, numImage, totalNumBlocks);

                auto thumbnailImage
                    = rt::imageUtils::resizeImage(image, mConfig.blockImageSizeW, mConfig.blockImageSizeH, true);
                formatPatch(thumbnailImage, patches, imageTokenLengths, numImage, totalNumBlocks);
            }
            else
            {
                formatPatch(image, patches, imageTokenLengths, numImage, totalNumBlocks);
            }
        }
        numImagePerBatch.emplace_back(numImage);
    }

    if (totalNumBlocks < mConfig.minNumBlocks || totalNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + ", min = " + std::to_string(mConfig.minNumBlocks) + " of VIT engine.");
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
    mOutputEmbedding.reshape({totalNumBlocks * 256, mConfig.outHiddenSize});
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
    std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
    std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer, int const maxSupportedInputLength,
    bool enableDynamicShape, void* ropeRotaryCosSinDevice [[maybe_unused]],
    int const maxPositionEmbeddings [[maybe_unused]], int const rotaryDim [[maybe_unused]], cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImagePerBatch;
    imagePreprocess(imageBuffers, imageTokenLengths, numImagePerBatch, false, stream);

    std::vector<std::vector<int32_t>> batchInputIds;
    std::vector<int32_t> batchInputLengths;
    textPreprocess(batchInputIds, batchInputLengths, inputStrings, numImagePerBatch, imageTokenLengths, tokenizer);

    flattenBatch(inputIds, contextLengths, batchInputIds, batchInputLengths, tokenizer->getPadId(),
        maxSupportedInputLength, enableDynamicShape);
}

bool InternViTRunner::preprocess(std::vector<std::string> const& inputStrings,
    std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers,
    std::vector<std::vector<int32_t>>& batchInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImagePerBatch;

    // TODO: Clean out the field, only put here for compatibility with old API.
    std::vector<int32_t> inputIdsLengths;

    try
    {
        imagePreprocess(imageBuffers, imageTokenLengths, numImagePerBatch, true, stream);
        textPreprocess(batchInputIds, inputIdsLengths, inputStrings, numImagePerBatch, imageTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("InternViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

bool InternViTRunner::infer(cudaStream_t stream)
{
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mContext->setInputShape("input", mVitInput.getShape().getTRTDims());
    if (!setEngineIOStatus)
    {
        LOG_ERROR("InternViTRunner::infer(): Failed to bind engine input tensors.");
        return false;
    }

    mContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void InternViTRunner::initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
    int const inputLength, cudaStream_t stream)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init visual inputs
    // In InternVL3, each block generates 256 tokens, so totalNumBlocks is imageTokenLength/256
    if (imageTokenLength % 256 != 0)
    {
        throw std::runtime_error("imageTokenLength " + std::to_string(imageTokenLength)
            + " must be divisible by 256 for InternVL ViT model.");
    }
    int const totalNumBlocks = imageTokenLength / 256;
    if (totalNumBlocks < mConfig.minNumBlocks || totalNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + ", min = " + std::to_string(mConfig.minNumBlocks) + " of VIT engine.");
    }

    std::vector<half> patches(totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(patches.begin(), patches.end(), [&rng, &dist]() { return __float2half(dist(rng)); });
    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
    mOutputEmbedding.reshape({totalNumBlocks * 256, mConfig.outHiddenSize});

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