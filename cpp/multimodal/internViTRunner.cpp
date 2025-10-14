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
#include "common/bindingNames.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
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
    if (!allocateBuffer(stream))
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

    auto builderConfig = jsonConfig["builder_config"];
    // Min token per image is the same as min token per batch (minimum number of images = 1).
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int32_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int32_t>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxNumBlocks = inputShapeMax.d[0];
    mConfig.minNumBlocks = inputShapeMin.d[0];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

void* InternViTRunner::getConfig()
{
    return &mConfig;
}

bool InternViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    LOG_INFO(
        "InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.numChannels: %d, mConfig.blockImageSizeH: "
        "%d, mConfig.blockImageSizeW: %d",
        mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());
    // In InternVL3, each block generates 256 tokens, so output size is maxNumBlocks*256
    LOG_INFO("InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.outHiddenSize: %d",
        mConfig.maxNumBlocks * 256, mConfig.outHiddenSize);
    mOutputEmbedding = rt::Tensor(
        {mConfig.maxNumBlocks * 256, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    auto channels = mConfig.imageMean.size();
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    return true;
}

void InternViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
    int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream)
{
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    if (channels != mConfig.numChannels)
    {
        throw std::runtime_error("Image channels mismatch, got " + std::to_string(channels) + ", expected "
            + std::to_string(mConfig.numChannels));
    }
    if (height % mConfig.blockImageSizeH != 0 || width % mConfig.blockImageSizeW != 0)
    {
        throw std::runtime_error(
            "Image height or width is not divisible by blockImageSizeH or blockImageSizeW, "
            "got height: "
            + std::to_string(height) + ", width: " + std::to_string(width)
            + ", blockImageSizeH: " + std::to_string(mConfig.blockImageSizeH)
            + ", blockImageSizeW: " + std::to_string(mConfig.blockImageSizeW));
    }

    int64_t curNumBlocks = (height / mConfig.blockImageSizeH) * (width / mConfig.blockImageSizeW);
    if (totalNumBlocks + curNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks) + " + curNumBlocks "
            + std::to_string(curNumBlocks) + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + " of VIT engine.");
    }

    int64_t curTokenLength = curNumBlocks * 256;
    if (isThumbnail)
    {
        // Add to the last image token length, instead of considered as a new image
        imageTokenLengths.back() += curTokenLength;
    }
    else
    {
        imageTokenLengths.push_back(curTokenLength);
        ++numImages;
    }

    // Copy image to device.
    auto imageDevice = rt::Tensor({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    CUDA_CHECK(cudaMemcpyAsync(imageDevice.rawPointer(), imageData, height * width * channels * sizeof(unsigned char),
        cudaMemcpyHostToDevice, stream));

    // Normalize image
    auto normalizedImageDevice
        = rt::Tensor({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::normalizeImage(imageDevice, mImageMean, mImageStd, normalizedImageDevice, stream);

    // Transpose to patch
    int64_t offset = totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW;
    kernel::transposeToPatchInternVL(normalizedImageDevice, mVitInput, offset, stream);

    // Update numBlocks
    totalNumBlocks += curNumBlocks;
}

std::vector<std::pair<int32_t, int32_t>> InternViTRunner::getAllSupportedAspectRatios(
    int32_t const minImageTiles, int32_t const maxImageTiles)
{
    std::vector<std::pair<int32_t, int32_t>> aspectRatios;
    for (int32_t width = 1; width <= maxImageTiles; ++width)
    {
        for (int32_t height = 1; height <= maxImageTiles; ++height)
        {
            if (width * height <= maxImageTiles && width * height >= minImageTiles)
            {
                aspectRatios.emplace_back(width, height);
            }
        }
    }
    std::sort(aspectRatios.begin(), aspectRatios.end(),
        [](std::pair<int32_t, int32_t> const& a, std::pair<int32_t, int32_t> const& b) {
            return a.first * a.second < b.first * b.second;
        });
    return aspectRatios;
}

std::tuple<int32_t, int32_t> InternViTRunner::getResizedImageSize(int32_t const height, int32_t const width)
{
    // -1 because we add a thumbnail image for each image
    int32_t const minImageTiles = std::max(1, mConfig.minImageTokensPerImage / 256 - 1);
    int32_t const maxImageTiles = std::max(1, mConfig.maxImageTokensPerImage / 256 - 1);
    auto targetRatios = getAllSupportedAspectRatios(minImageTiles, maxImageTiles);
    double const aspectRatio = static_cast<double>(width) / height;
    int64_t const area = width * height;

    double bestRatioDiff = HUGE_VAL;
    std::pair<int32_t, int32_t> bestRatio = {1, 1};
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
            if (area > 0.5 * mConfig.blockImageSizeH * mConfig.blockImageSizeW * ratio.first * ratio.second)
            {
                bestRatio = ratio;
            }
        }
    }
    // return (height, width)
    return {bestRatio.second * mConfig.blockImageSizeH, bestRatio.first * mConfig.blockImageSizeW};
}

void InternViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    int64_t totalNumBlocks = 0;

    for (auto const& prompt : request.prompts)
    {
        int64_t numImage = 0;
        for (auto const& image : prompt.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width);
                auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
                formatPatch(resizedImage, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
            else
            {
                formatPatch(image, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
            // Add thumbnail image by default
            auto thumbnailImage = rt::imageUtils::resizeImage(image, mConfig.blockImageSizeW, mConfig.blockImageSizeH);
            formatPatch(thumbnailImage, imageTokenLengths, numImage, totalNumBlocks, true, stream);
        }
        numImages.emplace_back(numImage);
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

    // Calculate total image tokens for profiling (InternVL: each block generates 256 tokens)
    int64_t totalImageTokens = totalNumBlocks * 256;

    // Record performance data
    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), 0);
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
    mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize});
}

std::string InternViTRunner::applyChatTemplateSystem(std::string const& systemPrompt)
{
    return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
}

std::string InternViTRunner::applyChatTemplateUser(
    std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt)
{
    std::string prompt = "<|im_start|>user\n";
    for (int64_t i = 0; i < numImage; ++i)
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
        setEngineIOStatus &= mContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());
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

} // namespace rt
} // namespace drivellm