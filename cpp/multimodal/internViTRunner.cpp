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

#include "internViTRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
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
    if (!validateAndFillConfig(configPath))
    {
        LOG_ERROR("InternViTRunner::InternViTRunner(): Failed to validate and fill config");
        throw std::runtime_error("InternViTRunner::InternViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer())
    {
        LOG_ERROR("InternViTRunner::InternViTRunner(): Failed to allocate buffer");
        throw std::runtime_error("InternViTRunner::InternViTRunner(): Failed to allocate buffer");
    }
}

bool InternViTRunner::validateAndFillConfig(std::string const& configPath)
{
    Json jsonConfig;

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        return false;
    }

    mModelType = jsonConfig["model_type"].get<std::string>();
    if (mModelType != "internvl")
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Invalid model type: %s", mModelType.c_str());
        return false;
    }

    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
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

    return true;
}

void* InternViTRunner::getConfig()
{
    return &mConfig;
}

bool InternViTRunner::allocateBuffer()
{
    bool setTensorAddressStatus{true};
    LOG_INFO(
        "InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.numChannels: %d, mConfig.blockImageSizeH: "
        "%d, mConfig.blockImageSizeW: %d",
        mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress("input", mVitInput.rawPointer());
    // In InternVL3, each block generates 256 tokens, so output size is maxNumBlocks*256
    LOG_INFO("InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.outHiddenSize: %d",
        mConfig.maxNumBlocks * 256, mConfig.outHiddenSize);
    mOutputEmbedding = rt::Tensor(
        {mConfig.maxNumBlocks * 256, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress("output", mOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }
    return true;
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
    std::vector<int64_t>& imageTokenLengths, int64_t& numImages, int64_t& totalNumBlocks)
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
        ++numImages;
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
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
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
        numImages.emplace_back(numImage);
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

void InternViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    std::vector<half> patches;
    int64_t totalNumBlocks = 0;
    int32_t totalImageTokens = 0;

    for (auto const& prompt : request.prompts)
    {
        int64_t numImage = 0;
        for (auto const& image : prompt.imageBuffers)
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
        numImages.emplace_back(numImage);
    }

    // Calculate total image tokens for profiling (InternVL: each block generates 256 tokens)
    totalImageTokens = static_cast<int32_t>(totalNumBlocks * 256);

    // Record performance data (always count metrics regardless of profiler state)
    int32_t actualImageCount = 0;
    for (auto const& prompt : request.prompts)
    {
        actualImageCount += static_cast<int32_t>(prompt.imageBuffers.size());
    }
    if (actualImageCount > 0 && totalImageTokens > 0)
    {
        mMultimodalMetrics.recordRun(actualImageCount, totalImageTokens);
    }

    if (totalNumBlocks == 0)
    {
        mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
        return;
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
        int imagePadLen = imageTokenLengths.at(totalImageIdx++);

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
    std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
    drivellm::tokenizer::Tokenizer* tokenizer)
{
    int totalImageIdx = 0;
    int32_t imageTokenId = mConfig.vocabSize;

    for (size_t i = 0; i < inputStrings.size(); ++i)
    {
        std::string prompt = applyChatTemplate(inputStrings[i], numImages[i], imageTokenLengths, totalImageIdx);
        std::vector<int32_t> ids = tokenizer->encode(prompt);

        // replace vis tokens
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId)
            {
                ids[j] = imageTokenId;
                ++imageTokenId;
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
    std::vector<int64_t> numImages;
    imagePreprocess(imageBuffers, imageTokenLengths, numImages, false, stream);

    std::vector<std::vector<int32_t>> batchInputIds;
    std::vector<int32_t> batchInputLengths;
    textPreprocess(batchInputIds, batchInputLengths, inputStrings, numImages, imageTokenLengths, tokenizer);

    flattenBatch(inputIds, contextLengths, batchInputIds, batchInputLengths, tokenizer->getPadId(),
        maxSupportedInputLength, enableDynamicShape);
}

std::string InternViTRunner::applyChatTemplateSystem(std::string const& systemPrompt)
{
    return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
}

std::string InternViTRunner::applyChatTemplateUser(
    std::string const& userPrompt, int const& numImage, bool addGenerationPrompt)
{
    std::string prompt = "<|im_start|>user\n";
    for (int i = 0; i < numImage; ++i)
    {
        prompt += "<img><IMG_CONTEXT></img>\n";
    }
    prompt += userPrompt + "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}

void InternViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, drivellm::tokenizer::Tokenizer* tokenizer)
{
    if (numImages.size() != request.prompts.size())
    {
        std::string errorMsg = "InternViTRunner::textPreprocess() numImages.size() != request.prompts.size(), "
            + std::to_string(numImages.size()) + " != " + std::to_string(request.prompts.size());
        LOG_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    int imageIndex = 0;
    // Image token id will start from vocabSize and increment for each image token position
    int32_t imageTokenId = mConfig.vocabSize;

    for (size_t i = 0; i < request.prompts.size(); ++i)
    {
        // Direct concate to avoid extra copy
        std::string prompt = applyChatTemplateSystem(request.prompts[i].systemPrompt)
            + applyChatTemplateUser(request.prompts[i].userPrompt, numImages[i], true);
        std::vector<int32_t> ids = tokenizer->encode(prompt);

        // replace vis tokens
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId)
            {
                int64_t numImageTokens = imageTokenLengths.at(imageIndex);
                for (int k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(imageTokenId);
                    ++imageTokenId;
                }
                ++imageIndex;
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }
        batchInputIds.emplace_back(std::move(newIds));
    }
}

bool InternViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageTokenLengths, numImages, true, stream);
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("InternViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

std::string InternViTRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    return applyChatTemplateSystem(systemPrompt);
}

bool InternViTRunner::infer(cudaStream_t stream)
{
    // Skip VIT inference if there are no images to process
    // Check if the first dimension (sequence length) is 0, indicating no images
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }

    // Profile ViT inference with automatic cleanup
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus &= mContext->setInputShape("input", mVitInput.getShape().getTRTDims());
        if (!setEngineIOStatus)
        {
            LOG_ERROR("InternViTRunner::infer(): Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mContext->enqueueV3(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!enqueueStatus)
        {
            LOG_ERROR("InternViTRunner::infer(): Failed to enqueue engine.");
            return false;
        }
    }

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
    std::uniform_int_distribution<std::mt19937::result_type> intDist(0, mConfig.vocabSize - 1);
    int32_t imageTokenId = mConfig.vocabSize;
    for (int i = 0; i < batchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * inputLength;
        std::generate(beginIter, beginIter + inputLength, [&rng, &intDist]() { return intDist(rng); });
        // Replace image tokens at the beginning of each batch
        for (int j = 0; j < imageTokenLength; ++j)
        {
            *(beginIter + j) = imageTokenId;
            ++imageTokenId;
        }
    }
}

} // namespace rt
} // namespace drivellm