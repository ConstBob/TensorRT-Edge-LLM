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

#include "qwenViTRunner.h"
#include "common/bindingNames.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
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

QwenViTRunner::QwenViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    std::string configPath = engineDir + "/config.json";
    if (!validateAndFillConfig(configPath))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer())
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
    }
}

bool QwenViTRunner::validateAndFillConfig(std::string const& configPath)
{
    Json jsonConfig;

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        return false;
    }

    mModelType = jsonConfig["model_type"].get<std::string>();
    if (mModelType != "qwen2_5_vl" && mModelType != "qwen2_vl")
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Invalid model type: %s", mModelType.c_str());
        return false;
    }

    mConfig.vocabSize = jsonConfig["vocab_size"].get<int32_t>();
    mConfig.visionStartTokenId = jsonConfig["vision_start_token_id"].get<int32_t>();
    mConfig.visionTokenId = jsonConfig["vision_token_id"].get<int32_t>();
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    mConfig.videoTokenId = jsonConfig["video_token_id"].get<int32_t>();
    mConfig.mropeTheta = jsonConfig["rope_theta"].get<float>();
    auto visionConfig = jsonConfig["vision_config"];
    mConfig.patchSize = visionConfig["spatial_patch_size"].get<int32_t>();
    mConfig.temporalPatchSize = visionConfig["temporal_patch_size"].get<int32_t>();
    mConfig.mergeSize = visionConfig["spatial_merge_size"].get<int32_t>();
    if (mModelType == "qwen2_5_vl")
    {
        mConfig.windowSize = visionConfig["window_size"].get<int32_t>();
    }

    // Get config from engine shapes
    // TODO: use json config to get the shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    mConfig.inputDim = mContext->getTensorShape(binding_names::kVisualInput).d[1];
    mConfig.vitPosEmbDim = mContext->getTensorShape(binding_names::kRotaryPosEmb).d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

void* QwenViTRunner::getConfig()
{
    return &mConfig;
}

bool QwenViTRunner::allocateBuffer()
{
    bool setTensorAddressStatus{true};
    mVitInput = rt::Tensor({mConfig.maxHW, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mAttentionMask = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kAttentionMask, mAttentionMask.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxHW, mConfig.vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kRotaryPosEmb, mRotaryPosEmb.rawPointer());

    // In Qwen2-VL, VIT input mHW is always 4*numImageTokens because it equals to spatial_merge_size ** 2.
    mOutputEmbedding
        = rt::Tensor({mConfig.maxHW / 4, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    if (mModelType == "qwen2_5_vl")
    {
        mWindowAttentionMask
            = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kWindowAttentionMask, mWindowAttentionMask.rawPointer());

        mWindowIndex = rt::Tensor({mConfig.maxHW / 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
        setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kWindowIndex, mWindowIndex.rawPointer());

        mReverseWindowIndex = rt::Tensor({mConfig.maxHW / 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kReverseWindowIndex, mReverseWindowIndex.rawPointer());
    }
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }
    return true;
}

void QwenViTRunner::formatPatch(rt::imageUtils::ImageData const& image,
    std::vector<std::vector<int32_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int32_t>& cuSeqlens, cudaStream_t stream)
{
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    if (height % (mConfig.patchSize * mConfig.mergeSize) != 0 || width % (mConfig.patchSize * mConfig.mergeSize) != 0)
    {
        throw std::invalid_argument("Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));
    }

    std::vector<int32_t> curGrid{1, (height / mConfig.patchSize), (width / mConfig.patchSize)};
    imageGridTHWs.emplace_back(curGrid);
    int64_t curSeqLength = (height / mConfig.patchSize) * (width / mConfig.patchSize);
    imageTokenLengths.emplace_back(curSeqLength / mConfig.mergeSize / mConfig.mergeSize);

    // Copy image to device. Repeat for T = temporalPatchSize
    auto imageSize = height * width * channels;
    auto imageDevice = rt::Tensor(
        {mConfig.temporalPatchSize, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    for (int i = 0; i < mConfig.temporalPatchSize; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(imageDevice.rawPointer() + i * imageSize * sizeof(unsigned char), imageData,
            imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
    }

    // Normalize image
    auto normalizedImageDevice = rt::Tensor(
        {mConfig.temporalPatchSize, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    auto imageMeanDevice = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    auto imageStdDevice = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpyAsync(imageMeanDevice.rawPointer(), mConfig.imageMean.data(),
        mConfig.imageMean.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(imageStdDevice.rawPointer(), mConfig.imageStd.data(),
        mConfig.imageStd.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    kernel::normalizeImage(imageDevice, imageMeanDevice, imageStdDevice, normalizedImageDevice, stream);

    // Transpose to patch
    kernel::transposeToPatchQwenViT(normalizedImageDevice, mVitInput, cuSeqlens.back() * mConfig.inputDim,
        mConfig.temporalPatchSize, mConfig.patchSize, mConfig.mergeSize, stream);

    // Update sequence length
    cuSeqlens.emplace_back(cuSeqlens.back() + curSeqLength);
}

void QwenViTRunner::computeRotaryPosEmb(
    std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t const totalSeqLength, cudaStream_t stream)
{
    // Get position ids
    int32_t* posIdsPtr;
    CUDA_CHECK(cudaMallocHost(&posIdsPtr, totalSeqLength * 2 * sizeof(int32_t)));
    int32_t posIdsOffset = 0;

    for (auto const& grid : imageGridTHWs)
    {
        int32_t T = grid[0], H = grid[1], W = grid[2];

        for (int i = 0; i < H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                // (H, W) => (H / mergeSize, mergeSize, W / mergeSize, mergeSize)
                // => (H / mergeSize, W / mergeSize, mergeSize, mergeSize)
                int dstHW = (i / mConfig.mergeSize) * W * mConfig.mergeSize
                    + (j / mConfig.mergeSize) * mConfig.mergeSize * mConfig.mergeSize
                    + (i % mConfig.mergeSize) * mConfig.mergeSize + (j % mConfig.mergeSize);

                // duplicate for T
                for (int t = 0; t < T; ++t)
                {
                    int baseIdx = t * H * W * 2 + dstHW * 2;
                    posIdsPtr[posIdsOffset + baseIdx] = i;
                    posIdsPtr[posIdsOffset + baseIdx + 1] = j;
                }
            }
        }

        posIdsOffset += T * H * W * 2;
    }

    auto posIdsDevice = rt::Tensor({totalSeqLength * 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(
        posIdsDevice.rawPointer(), posIdsPtr, totalSeqLength * 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(posIdsPtr));

    kernel::initRotaryPosEmbQwenViT(posIdsDevice, mRotaryPosEmb, 10000.0f, 1.0f, stream);
}

std::tuple<int, int> QwenViTRunner::getResizedImageSize(
    int const height, int const width, int const factor, int const minPixels, int const maxPixels, int const maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    auto roundByFactor
        = [](int value, int factor) -> int { return std::round(static_cast<double>(value) / factor) * factor; };
    auto floorByFactor
        = [](int value, int factor) -> int { return std::floor(static_cast<double>(value) / factor) * factor; };
    auto ceilByFactor
        = [](int value, int factor) -> int { return std::ceil(static_cast<double>(value) / factor) * factor; };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::invalid_argument("absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));
    }

    int hBar = std::max(factor, roundByFactor(height, factor));
    int wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        hBar = floorByFactor(static_cast<int>(height / beta), factor);
        wBar = floorByFactor(static_cast<int>(width / beta), factor);
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int>(width * beta), factor);
    }

    return {hBar, wBar};
}

void QwenViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    std::vector<int32_t> cuSeqlens{0};

    for (auto const& prompt : request.prompts)
    {
        int32_t numImage = 0;
        for (auto const& image : prompt.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width,
                    mConfig.patchSize * mConfig.mergeSize, mConfig.minPixels, mConfig.maxPixels);
                auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
                formatPatch(resizedImage, imageGridTHWs, imageTokenLengths, cuSeqlens, stream);
            }
            else
            {
                formatPatch(image, imageGridTHWs, imageTokenLengths, cuSeqlens, stream);
            }
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    int32_t totalSeqLength = cuSeqlens.back();
    if (totalSeqLength == 0)
    {
        mVitInput.reshape({totalSeqLength, mConfig.inputDim});
        return;
    }

    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    // Reshape tensors
    int64_t totalImageTokens = totalSeqLength / 4;
    mVitInput.reshape({totalSeqLength, mConfig.inputDim});
    mAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
    mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim});
    mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize});

    // Record performance data
    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), 0);
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    // Compute attention mask
    auto cuSeqlensDevice = rt::Tensor({cuSeqlens.size()}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(cuSeqlensDevice.rawPointer(), cuSeqlens.data(), cuSeqlens.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    kernel::initAttentionMaskQwenViT(cuSeqlensDevice, mAttentionMask, stream);

    // Compute rotary position embeddings
    computeRotaryPosEmb(imageGridTHWs, totalSeqLength, stream);

    // For Qwen2.5-VL, compute additional inputs
    if (mModelType == "qwen2_5_vl")
    {
        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndex.reshape({totalSeqLength / 4});
        mReverseWindowIndex.reshape({totalSeqLength / 4});

        getWindowIndex(imageGridTHWs, totalSeqLength, stream);
    }
}

void QwenViTRunner::getRopeIdx(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t* mropePositionIdsPtr,
    int const maxPositionEmbeddings)
{
    // According to transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.get_rope_index
    // mropePositionIds: (bs, 3, maxPositionEmbeddings), 3 is for T, H, W
    int32_t totalImageIdx = 0;
    int32_t batchOffset = 0;

    for (auto const& inputIds : batchInputIds)
    {
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int startIdx = 0;
        int remainingStartPos = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
            // Text part
            int textLen = it + 1 - start;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < textLen; ++j)
                {
                    mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
                }
            }

            // Visual part
            int32_t T = imageGridTHWs[totalImageIdx][0];
            int32_t H = imageGridTHWs[totalImageIdx][1] / mConfig.mergeSize;
            int32_t W = imageGridTHWs[totalImageIdx][2] / mConfig.mergeSize;
            ++totalImageIdx;

            for (int t = 0; t < T; ++t)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        int32_t idx = remainingStartPos + textLen + t * H * W + h * W + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx] = t + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + T * H * W;
            startIdx += std::max(T, std::max(H, W)) + textLen;
            remainingStartPos = start - inputIds.begin();
        }

        // Remaining text part till maxPositionEmbeddings. Treat all generated tokens as text tokens.
        int textLen = maxPositionEmbeddings - remainingStartPos;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < textLen; ++j)
            {
                mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
            }
        }

        batchOffset += 3 * maxPositionEmbeddings;
    }
}

void QwenViTRunner::generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int32_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    auto ropeRotaryCosSinDim = ropeRotaryCosSinDevice.getShape();
    int const maxPositionEmbeddings = ropeRotaryCosSinDim[1];
    int const rotaryDim = ropeRotaryCosSinDim[2];

    // Get [T, H, W] information for each token position
    // mropePositionIds: (bs, 3, maxPositionEmbeddings)
    int32_t mropePositionIdsSize = batchInputIds.size() * 3 * maxPositionEmbeddings;
    int32_t* mropePositionIdsPtr;
    CUDA_CHECK(cudaMallocHost(&mropePositionIdsPtr, mropePositionIdsSize * sizeof(int32_t)));
    getRopeIdx(batchInputIds, imageGridTHWs, mropePositionIdsPtr, maxPositionEmbeddings);

    auto mropePositionIdsDevice = rt::Tensor({mropePositionIdsSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(mropePositionIdsDevice.rawPointer(), mropePositionIdsPtr,
        mropePositionIdsSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(mropePositionIdsPtr));

    // Initialize mrope cosSinCacheDevice
    ropeRotaryCosSinDevice.reshape({batchInputIds.size(), maxPositionEmbeddings, rotaryDim});
    kernel::initializeMRopeCosSin(reinterpret_cast<float*>(const_cast<void*>(ropeRotaryCosSinDevice.rawPointer())),
        reinterpret_cast<int32_t*>(mropePositionIdsDevice.rawPointer()), mConfig.mropeTheta, rotaryDim,
        maxPositionEmbeddings, batchInputIds.size(), stream);
}

void QwenViTRunner::getWindowIndex(
    std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t const curHW, cudaStream_t stream)
{
    // Init windowIndex and cuWindowSeqlens
    int32_t windowIndexSize = curHW / mConfig.mergeSize / mConfig.mergeSize;
    int vitMergerWindowSize = mConfig.windowSize / mConfig.mergeSize / mConfig.patchSize;

    int64_t* windowIndexPtr;
    CUDA_CHECK(cudaMallocHost(&windowIndexPtr, windowIndexSize * sizeof(int64_t)));
    int32_t windowIndexPos = 0;
    int32_t windowIndexValue = 0;

    std::vector<int32_t> cuWindowSeqlens{0};

    for (auto const& grid : imageGridTHWs)
    {
        int32_t T = grid[0], H = grid[1], W = grid[2];
        int32_t llmGridH = H / mConfig.mergeSize;
        int32_t llmGridW = W / mConfig.mergeSize;
        int numWindowsH = (llmGridH + vitMergerWindowSize - 1) / vitMergerWindowSize;
        int numWindowsW = (llmGridW + vitMergerWindowSize - 1) / vitMergerWindowSize;

        for (int i = 0; i < numWindowsH; ++i)
        {
            for (int j = 0; j < numWindowsW; ++j)
            {
                int cnt{0};
                for (int m = 0; m < vitMergerWindowSize; ++m)
                {
                    for (int n = 0; n < vitMergerWindowSize; ++n)
                    {
                        int64_t idxH = i * vitMergerWindowSize + m;
                        int64_t idxW = j * vitMergerWindowSize + n;
                        if (idxH < llmGridH && idxW < llmGridW)
                        {
                            windowIndexPtr[windowIndexPos++] = idxH * llmGridW + idxW + windowIndexValue;
                            ++cnt;
                        }
                    }
                }

                cuWindowSeqlens.emplace_back(cnt * mConfig.mergeSize * mConfig.mergeSize + cuWindowSeqlens.back());
            }
        }

        windowIndexValue += T * llmGridH * llmGridW;
    }

    if (windowIndexPos * 4 != curHW)
    {
        throw std::runtime_error("windowIndex size * 4 does not match curHW. Got windowIndex size: "
            + std::to_string(windowIndexPos) + ", curHW: " + std::to_string(curHW));
    }

    int64_t* reverseWindowIndexPtr;
    CUDA_CHECK(cudaMallocHost(&reverseWindowIndexPtr, windowIndexSize * sizeof(int64_t)));
    std::iota(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize, 0);
    std::sort(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize,
        [windowIndexPtr](size_t left, size_t right) { return windowIndexPtr[left] < windowIndexPtr[right]; });

    CUDA_CHECK(cudaMemcpyAsync(
        mWindowIndex.rawPointer(), windowIndexPtr, windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(windowIndexPtr));
    CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndex.rawPointer(), reverseWindowIndexPtr,
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(reverseWindowIndexPtr));

    // Init window attention mask
    auto cuWindowSeqlensDevice = rt::Tensor({cuWindowSeqlens.size()}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(cuWindowSeqlensDevice.rawPointer(), cuWindowSeqlens.data(),
        cuWindowSeqlens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    kernel::initAttentionMaskQwenViT(cuWindowSeqlensDevice, mWindowAttentionMask, stream);
}

std::string QwenViTRunner::applyChatTemplateSystem(std::string const& systemPrompt)
{
    return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
}

std::string QwenViTRunner::applyChatTemplateUser(
    std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt)
{
    std::string prompt = "<|im_start|>user\n";
    for (int64_t i = 0; i < numImage; ++i)
    {
        prompt += "<|vision_start|><|image_pad|><|vision_end|>";
    }
    prompt += userPrompt + "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}

void QwenViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, drivellm::tokenizer::Tokenizer* tokenizer)
{
    if (numImages.size() != request.prompts.size())
    {
        std::string errorMsg = "QwenViTRunner::textPreprocess() numImages.size() != request.prompts.size(), "
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

        // insert image tokens
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.visionTokenId || ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
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

bool QwenViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> imageGridTHWs;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageGridTHWs, imageTokenLengths, numImages, true, stream);
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
        generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QwenViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

std::string QwenViTRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::string prompt = applyChatTemplateSystem(systemPrompt);

    std::vector<int32_t> ids = tokenizer->encode(prompt);
    std::vector<std::vector<int32_t>> batchedInputIds;
    batchedInputIds.emplace_back(std::move(ids));
    std::vector<std::vector<int32_t>> imageGridTHWs;
    generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);

    return prompt;
}

bool QwenViTRunner::infer(cudaStream_t stream)
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
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kAttentionMask, mAttentionMask.getShape().getTRTDims());
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kRotaryPosEmb, mRotaryPosEmb.getShape().getTRTDims());
        if (mModelType == "qwen2_5_vl")
        {
            setEngineIOStatus &= mContext->setInputShape(
                binding_names::kWindowAttentionMask, mWindowAttentionMask.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kWindowIndex, mWindowIndex.getShape().getTRTDims());
            setEngineIOStatus &= mContext->setInputShape(
                binding_names::kReverseWindowIndex, mReverseWindowIndex.getShape().getTRTDims());
        }

        if (!setEngineIOStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mContext->enqueueV3(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!enqueueStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to enqueue engine.");
            return false;
        }
    }

    return true;
}

void QwenViTRunner::initRandomInputs(std::vector<int32_t>& inputIds, int32_t const batchSize,
    int32_t const imageTokenLength, int const inputLength, cudaStream_t stream)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init visual inputs
    // HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
    int const totalSeqLength = 4 * imageTokenLength;
    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    mVitInput.reshape({totalSeqLength, mConfig.inputDim});
    mAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
    mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim});
    mOutputEmbedding.reshape({totalSeqLength / 4, mConfig.outHiddenSize});

    // Use pinned memory for the patches
    half* patchesPtr;
    int32_t patchesSize = totalSeqLength * mConfig.inputDim;
    CUDA_CHECK(cudaMallocHost(&patchesPtr, patchesSize * sizeof(half)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(patchesPtr, patchesPtr + patchesSize, [&rng, &dist]() { return __float2half(dist(rng)); });
    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patchesPtr, patchesSize * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(patchesPtr));

    float* rotaryPosEmbPtr;
    int32_t rotaryPosEmbSize = totalSeqLength * mConfig.vitPosEmbDim;
    CUDA_CHECK(cudaMallocHost(&rotaryPosEmbPtr, rotaryPosEmbSize * sizeof(float)));
    std::generate(rotaryPosEmbPtr, rotaryPosEmbPtr + rotaryPosEmbSize, [&rng, &dist]() { return dist(rng); });
    CUDA_CHECK(cudaMemcpyAsync(
        mRotaryPosEmb.rawPointer(), rotaryPosEmbPtr, rotaryPosEmbSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaFreeHost(rotaryPosEmbPtr));

    // directly initialize attention mask with zero in cuda
    CUDA_CHECK(cudaMemsetAsync(mAttentionMask.rawPointer(), 0, totalSeqLength * totalSeqLength * sizeof(half), stream));

    // For Qwen2.5-VL, initialize window attention parameters
    if (mModelType == "qwen2_5_vl")
    {
        // Assume grid (T,H,W) = (1, 2, imageTokenLength * 2) for simplicity
        std::vector<std::vector<int32_t>> imageGridTHWs = {{1, 2, imageTokenLength * 2}};

        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndex.reshape({totalSeqLength / 4});
        mReverseWindowIndex.reshape({totalSeqLength / 4});

        getWindowIndex(imageGridTHWs, totalSeqLength, stream);
    }

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