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
#include "kernels/posEncoding/initializeCosSinCache.h"
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
    nvinfer1::Dims const inputShapeMax = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    mConfig.inputDim = mContext->getTensorShape("input").d[1];
    mConfig.vitPosEmbDim = mContext->getTensorShape("rotary_pos_emb").d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape("output").d[1];

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
    setTensorAddressStatus &= mContext->setTensorAddress("input", mVitInput.rawPointer());

    mAttentionMask = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress("attention_mask", mAttentionMask.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxHW, mConfig.vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    setTensorAddressStatus &= mContext->setTensorAddress("rotary_pos_emb", mRotaryPosEmb.rawPointer());

    // In Qwen2-VL, VIT input mHW is always 4*numImageTokens because it equals to spatial_merge_size ** 2.
    mOutputEmbedding
        = rt::Tensor({mConfig.maxHW / 4, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    setTensorAddressStatus &= mContext->setTensorAddress("output", mOutputEmbedding.rawPointer());

    if (mModelType == "qwen2_5_vl")
    {
        mWindowAttentionMask
            = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        setTensorAddressStatus
            &= mContext->setTensorAddress("window_attention_mask", mWindowAttentionMask.rawPointer());

        mWindowIndex = rt::Tensor({mConfig.maxHW / 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
        setTensorAddressStatus &= mContext->setTensorAddress("window_index", mWindowIndex.rawPointer());

        mReverseWindowIndex = rt::Tensor({mConfig.maxHW / 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
        setTensorAddressStatus &= mContext->setTensorAddress("reverse_window_index", mReverseWindowIndex.rawPointer());
    }
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }
    return true;
}

std::vector<EngineInputDesc> QwenViTRunner::getComputedEmbeddings()
{
    std::vector<EngineInputDesc> extraInputs;

    extraInputs.emplace_back(EngineInputDesc{"image_embeds", mOutputEmbedding.rawPointer(),
        mOutputEmbedding.rawPointer(), mOutputEmbedding.getTRTDims(), {2, {1, mConfig.outHiddenSize}}});

    return extraInputs;
}

void QwenViTRunner::initRotaryEmbedding(
    int numPos, int dim, float theta, std::vector<std::vector<float>>& sinusoidInp, float scale)
{
    std::vector<float> invFreq;
    for (int i = 0; i < dim; i += 2)
    {
        float value = scale / pow(theta, (static_cast<float>(i) / dim));
        invFreq.emplace_back(value);
    }

    for (int i = 0; i < numPos; ++i)
    {
        for (int j = 0; j < (dim / 2); ++j)
        {
            sinusoidInp[i][j] = i * invFreq[j];
        }
    }

    return;
}

void QwenViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<half>& patches,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths, int64_t& totalSeqLength)
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

    std::vector<int64_t> curGrid{1, (height / mConfig.patchSize), (width / mConfig.patchSize)};
    imageGridTHWs.emplace_back(curGrid);
    int64_t curSeqLength = (height / mConfig.patchSize) * (width / mConfig.patchSize);
    totalSeqLength += curSeqLength;
    imageTokenLengths.emplace_back(curSeqLength / mConfig.mergeSize / mConfig.mergeSize);

    int curSize = mConfig.temporalPatchSize * height * width * channels;
    std::vector<half> curPatch(curSize);

    // Normalize and store to patches. Reorder dimensions according to:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L299
    // TODO: optimize transpose with CUDA
    for (int gridH = 0; gridH < curGrid[1] / mConfig.mergeSize; ++gridH)
    {
        for (int gridW = 0; gridW < curGrid[2] / mConfig.mergeSize; ++gridW)
        {
            for (int mergeH = 0; mergeH < mConfig.mergeSize; ++mergeH)
            {
                for (int mergeW = 0; mergeW < mConfig.mergeSize; ++mergeW)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        for (int patchH = 0; patchH < mConfig.patchSize; ++patchH)
                        {
                            for (int patchW = 0; patchW < mConfig.patchSize; ++patchW)
                            {

                                // src dimensions: (H, W, C) => (gridH, mergeSize, patchSize, gridW, mergeSize,
                                // patchSize, C)
                                int originalH = gridH * mConfig.mergeSize * mConfig.patchSize
                                    + mergeH * mConfig.patchSize + patchH;
                                int originalW = gridW * mConfig.mergeSize * mConfig.patchSize
                                    + mergeW * mConfig.patchSize + patchW;
                                unsigned char value
                                    = imageData[originalH * width * channels + originalW * channels + c];
                                half normalized
                                    = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);

                                // duplicate pixels to temporalPatchSize
                                // dst dimensions: (gridH, gridW, mergeSize, mergeSize) x (channels,
                                // temporalPatchSize, patchSize, patchSize)
                                for (int t = 0; t < mConfig.temporalPatchSize; ++t)
                                {
                                    int dstHW = gridH * curGrid[2] * mConfig.mergeSize
                                        + gridW * mConfig.mergeSize * mConfig.mergeSize + mergeH * mConfig.mergeSize
                                        + mergeW;
                                    int dstDim = c * mConfig.temporalPatchSize * mConfig.patchSize * mConfig.patchSize
                                        + t * mConfig.patchSize * mConfig.patchSize + patchH * mConfig.patchSize
                                        + patchW;
                                    curPatch[dstHW * channels * mConfig.temporalPatchSize * mConfig.patchSize
                                            * mConfig.patchSize
                                        + dstDim]
                                        = normalized;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    patches.insert(patches.end(), curPatch.begin(), curPatch.end());
}

void QwenViTRunner::computeRotaryPosEmb(
    std::vector<std::vector<int64_t>> const& imageGridTHWs, std::vector<float>& rotaryPosEmb)
{
    // TODO: optimize with CUDA
    int64_t maxGridSize{0};
    for (auto const& grid : imageGridTHWs)
    {
        maxGridSize = std::max(maxGridSize, std::max(grid[1], grid[2]));
    }

    std::vector<std::vector<float>> rotaryPosEmbFull(maxGridSize, std::vector<float>(mConfig.vitPosEmbDim / 2));
    initRotaryEmbedding(maxGridSize, mConfig.vitPosEmbDim, 10000.0f, rotaryPosEmbFull);

    std::vector<int> posIds;
    for (auto const& grid : imageGridTHWs)
    {
        int64_t T = grid[0], H = grid[1], W = grid[2];
        std::vector<int> curPosIds(T * H * W * 2);

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
                    curPosIds[baseIdx] = i;
                    curPosIds[baseIdx + 1] = j;
                }
            }
        }

        posIds.insert(posIds.end(), curPosIds.begin(), curPosIds.end());
    }

    rotaryPosEmb.resize(posIds.size() * (mConfig.vitPosEmbDim / 2));
    for (size_t i = 0; i < posIds.size(); ++i)
    {
        auto const& emb = rotaryPosEmbFull[posIds[i]];
        std::copy(emb.begin(), emb.end(), rotaryPosEmb.begin() + i * (mConfig.vitPosEmbDim / 2));
    }
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

void QwenViTRunner::imagePreprocess(std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    std::vector<half> patches;
    int64_t totalSeqLength = 0;

    for (auto const& imageBuffer : imageBuffers)
    {
        numImages.emplace_back(imageBuffer.size());
        for (auto const& image : imageBuffer)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width,
                    mConfig.patchSize * mConfig.mergeSize, mConfig.minPixels, mConfig.maxPixels);
                auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
                formatPatch(resizedImage, patches, imageGridTHWs, imageTokenLengths, totalSeqLength);
            }
            else
            {
                formatPatch(image, patches, imageGridTHWs, imageTokenLengths, totalSeqLength);
            }
        }
    }

    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    // Set attention mask
    std::vector<half> attentionMask(totalSeqLength * totalSeqLength, -CUDART_MAX_NORMAL_FP16);
    int start = 0;
    for (auto const& grid : imageGridTHWs)
    {
        int64_t len = grid[1] * grid[2];
        for (int t = 0; t < grid[0]; ++t)
        {
            for (int i = start; i < start + len; ++i)
            {
                for (int j = start; j < start + len; ++j)
                {
                    attentionMask[i * totalSeqLength + j] = CUDART_ZERO_FP16;
                }
            }
            start += len;
        }
    }

    // Compute rotary position embeddings
    std::vector<float> rotaryPosEmb;
    computeRotaryPosEmb(imageGridTHWs, rotaryPosEmb);

    // Copy to device
    mVitInput.reshape({totalSeqLength, mConfig.inputDim});
    mAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
    mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim});
    mOutputEmbedding.reshape({totalSeqLength / 4, mConfig.outHiddenSize});

    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mAttentionMask.rawPointer(), attentionMask.data(), attentionMask.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mRotaryPosEmb.rawPointer(), rotaryPosEmb.data(), rotaryPosEmb.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    // For Qwen2.5-VL, compute additional inputs
    if (mModelType == "qwen2_5_vl")
    {
        std::vector<half> windowAttentionMask;
        std::vector<int64_t> windowIndex;
        std::vector<int64_t> reverseWindowIndex;

        getWindowIndex(imageGridTHWs, windowAttentionMask, windowIndex, reverseWindowIndex, totalSeqLength);

        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndex.reshape({totalSeqLength / 4});
        mReverseWindowIndex.reshape({totalSeqLength / 4});

        CUDA_CHECK(cudaMemcpyAsync(mWindowAttentionMask.rawPointer(), windowAttentionMask.data(),
            windowAttentionMask.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mWindowIndex.rawPointer(), windowIndex.data(), windowIndex.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndex.rawPointer(), reverseWindowIndex.data(),
            reverseWindowIndex.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    }
}

void QwenViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    std::vector<half> patches;
    int64_t totalSeqLength = 0;

    int32_t totalImageTokens = 0;

    for (auto const& prompt : request.prompts)
    {
        int64_t numImage = 0;
        for (auto const& image : prompt.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width,
                    mConfig.patchSize * mConfig.mergeSize, mConfig.minPixels, mConfig.maxPixels);
                auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
                formatPatch(resizedImage, patches, imageGridTHWs, imageTokenLengths, totalSeqLength);
            }
            else
            {
                formatPatch(image, patches, imageGridTHWs, imageTokenLengths, totalSeqLength);
            }
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    // Calculate total image tokens for profiling
    for (auto const& tokenLength : imageTokenLengths)
    {
        totalImageTokens += static_cast<int32_t>(tokenLength);
    }

    // Record performance data (always count metrics regardless of profiler state)
    int32_t imageCount = 0;
    for (auto const& prompt : request.prompts)
    {
        imageCount += static_cast<int32_t>(prompt.imageBuffers.size());
    }
    if (imageCount > 0 && totalImageTokens > 0)
    {
        mMultimodalMetrics.recordRun(imageCount, totalImageTokens);
    }

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

    // Set attention mask
    std::vector<half> attentionMask(totalSeqLength * totalSeqLength, -CUDART_MAX_NORMAL_FP16);
    int start = 0;
    for (auto const& grid : imageGridTHWs)
    {
        int64_t len = grid[1] * grid[2];
        for (int t = 0; t < grid[0]; ++t)
        {
            for (int i = start; i < start + len; ++i)
            {
                for (int j = start; j < start + len; ++j)
                {
                    attentionMask[i * totalSeqLength + j] = CUDART_ZERO_FP16;
                }
            }
            start += len;
        }
    }

    // Compute rotary position embeddings
    std::vector<float> rotaryPosEmb;
    computeRotaryPosEmb(imageGridTHWs, rotaryPosEmb);

    // Copy to device
    mVitInput.reshape({totalSeqLength, mConfig.inputDim});
    mAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
    mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim});
    mOutputEmbedding.reshape({totalSeqLength / 4, mConfig.outHiddenSize});

    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mAttentionMask.rawPointer(), attentionMask.data(), attentionMask.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mRotaryPosEmb.rawPointer(), rotaryPosEmb.data(), rotaryPosEmb.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    // For Qwen2.5-VL, compute additional inputs
    if (mModelType == "qwen2_5_vl")
    {
        std::vector<half> windowAttentionMask;
        std::vector<int64_t> windowIndex;
        std::vector<int64_t> reverseWindowIndex;

        getWindowIndex(imageGridTHWs, windowAttentionMask, windowIndex, reverseWindowIndex, totalSeqLength);

        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndex.reshape({totalSeqLength / 4});
        mReverseWindowIndex.reshape({totalSeqLength / 4});

        CUDA_CHECK(cudaMemcpyAsync(mWindowAttentionMask.rawPointer(), windowAttentionMask.data(),
            windowAttentionMask.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mWindowIndex.rawPointer(), windowIndex.data(), windowIndex.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndex.rawPointer(), reverseWindowIndex.data(),
            reverseWindowIndex.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    }
}

void QwenViTRunner::getRopeIdx(std::vector<int64_t>& mropePositionIds,
    std::vector<std::vector<int32_t>> const& batchInputIds, std::vector<std::vector<int64_t>> const& imageGridTHWs,
    int const maxPositionEmbeddings)
{
    // According to transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.get_rope_index
    int totalImageIdx = 0;

    for (auto const& inputIds : batchInputIds)
    {
        std::vector<std::vector<int64_t>> positionIds(3);

        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int startIdx = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
            // Text part
            int textLen = it + 1 - start;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < textLen; ++j)
                {
                    positionIds[i].emplace_back(j + startIdx);
                }
            }

            // Visual part
            int64_t T = imageGridTHWs[totalImageIdx][0];
            int64_t H = imageGridTHWs[totalImageIdx][1] / mConfig.mergeSize;
            int64_t W = imageGridTHWs[totalImageIdx][2] / mConfig.mergeSize;
            ++totalImageIdx;

            for (int t = 0; t < T; ++t)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        positionIds[0].emplace_back(t + textLen + startIdx);
                        positionIds[1].emplace_back(h + textLen + startIdx);
                        positionIds[2].emplace_back(w + textLen + startIdx);
                    }
                }
            }

            start = it + 1 + T * H * W;
            startIdx += std::max(T, std::max(H, W)) + textLen;
        }

        // Remaining text part till maxPositionEmbeddings. Treat all generated tokens as text tokens.
        int textLen = maxPositionEmbeddings - positionIds[0].size();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < textLen; ++j)
            {
                positionIds[i].emplace_back(j + startIdx);
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            assert(static_cast<int>(positionIds[i].size()) == maxPositionEmbeddings);
            mropePositionIds.insert(mropePositionIds.end(), positionIds[i].begin(), positionIds[i].end());
        }
    }
}

void QwenViTRunner::generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, void* cosSinCacheDevice, int const maxPositionEmbeddings,
    int const rotaryDim, cudaStream_t stream)
{
    // Get [T, H, W] information for each token position
    // mropePositionIds: (bs, 3, maxPositionEmbeddings)
    std::vector<int64_t> mropePositionIds;
    getRopeIdx(mropePositionIds, batchInputIds, imageGridTHWs, maxPositionEmbeddings);

    auto mropePositionIdsDevice
        = rt::Tensor({static_cast<int64_t>(mropePositionIds.size())}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    CUDA_CHECK(cudaMemcpyAsync(mropePositionIdsDevice.rawPointer(), mropePositionIds.data(),
        mropePositionIds.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Initialize mrope cosSinCacheDevice
    kernel::initializeMRopeCosSin(reinterpret_cast<float*>(cosSinCacheDevice),
        reinterpret_cast<int64_t*>(mropePositionIdsDevice.rawPointer()), mConfig.mropeTheta, rotaryDim,
        maxPositionEmbeddings, batchInputIds.size(), stream);
}

void QwenViTRunner::generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    auto ropeRotaryCosSinDim = ropeRotaryCosSinDevice.getShape();
    int const maxPositionEmbeddings = ropeRotaryCosSinDim[1];
    int const rotaryDim = ropeRotaryCosSinDim[2];

    // Get [T, H, W] information for each token position
    // mropePositionIds: (bs, 3, maxPositionEmbeddings)
    std::vector<int64_t> mropePositionIds;
    getRopeIdx(mropePositionIds, batchInputIds, imageGridTHWs, maxPositionEmbeddings);

    auto mropePositionIdsDevice
        = rt::Tensor({static_cast<int64_t>(mropePositionIds.size())}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    CUDA_CHECK(cudaMemcpyAsync(mropePositionIdsDevice.rawPointer(), mropePositionIds.data(),
        mropePositionIds.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Initialize mrope cosSinCacheDevice
    ropeRotaryCosSinDevice.reshape({batchInputIds.size(), maxPositionEmbeddings, rotaryDim});
    kernel::initializeMRopeCosSin(reinterpret_cast<float*>(const_cast<void*>(ropeRotaryCosSinDevice.rawPointer())),
        reinterpret_cast<int64_t*>(mropePositionIdsDevice.rawPointer()), mConfig.mropeTheta, rotaryDim,
        maxPositionEmbeddings, batchInputIds.size(), stream);
}

std::string QwenViTRunner::applyChatTemplate(std::string const& inputString, int const& numImage,
    std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt)
{
    // System prefix
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";

    // Images
    for (int i = 0; i < numImage; ++i)
    {
        int imagePadLen = imageTokenLengths.at(totalImageIdx++);

        prompt += "<|vision_start|>";
        for (int j = 0; j < imagePadLen; ++j)
        {
            prompt += "<|image_pad|>";
        }

        prompt += "<|vision_end|>";
    }

    prompt += inputString;
    prompt += "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}

void QwenViTRunner::textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds,
    std::vector<int32_t>& batchInputLengths, std::vector<std::string> const& inputStrings,
    std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
    drivellm::tokenizer::Tokenizer* tokenizer)
{
    int totalImageIdx = 0;
    // Image token id will start from vocabSize and increment for each image token position
    int32_t imageTokenId = mConfig.vocabSize;

    for (size_t i = 0; i < inputStrings.size(); ++i)
    {
        std::string prompt = applyChatTemplate(inputStrings[i], numImages[i], imageTokenLengths, totalImageIdx);
        std::vector<int32_t> ids = tokenizer->encode(prompt);

        // replace vis tokens
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.visionTokenId || ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                ids[j] = imageTokenId;
                ++imageTokenId;
            }
        }
        batchInputLengths.emplace_back(static_cast<int32_t>(ids.size()));
        batchInputIds.emplace_back(std::move(ids));
    }
}

void QwenViTRunner::getWindowIndex(std::vector<std::vector<int64_t>> const& imageGridTHWs,
    std::vector<half>& windowAttentionMask, std::vector<int64_t>& windowIndex, std::vector<int64_t>& reverseWindowIndex,
    int const curHW)
{
    // Init windowIndex and cuWindowSeqlens
    std::vector<int64_t> cuWindowSeqlens{0};
    int windowIndexId = 0;
    int vitMergerWindowSize = mConfig.windowSize / mConfig.mergeSize / mConfig.patchSize;

    for (auto const& grid : imageGridTHWs)
    {
        int64_t T = grid[0], H = grid[1], W = grid[2];
        int64_t llmGridH = H / mConfig.mergeSize;
        int64_t llmGridW = W / mConfig.mergeSize;
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
                            windowIndex.emplace_back(idxH * llmGridW + idxW + windowIndexId);
                            ++cnt;
                        }
                    }
                }

                cuWindowSeqlens.emplace_back(cnt * mConfig.mergeSize * mConfig.mergeSize + cuWindowSeqlens.back());
            }
        }

        windowIndexId += T * llmGridH * llmGridW;
    }

    if (windowIndex.size() * 4 != static_cast<size_t>(curHW))
    {
        throw std::runtime_error("windowIndex size * 4 does not match curHW. Got windowIndex size: "
            + std::to_string(windowIndex.size()) + ", curHW: " + std::to_string(curHW));
    }

    reverseWindowIndex.resize(windowIndex.size());
    std::iota(reverseWindowIndex.begin(), reverseWindowIndex.end(), 0);
    std::sort(reverseWindowIndex.begin(), reverseWindowIndex.end(),
        [&windowIndex](size_t left, size_t right) { return windowIndex[left] < windowIndex[right]; });

    windowAttentionMask.resize(curHW * curHW, -CUDART_MAX_NORMAL_FP16);
    for (size_t s = 1; s < cuWindowSeqlens.size(); ++s)
    {
        for (int i = cuWindowSeqlens[s - 1]; i < cuWindowSeqlens[s]; ++i)
        {
            for (int j = cuWindowSeqlens[s - 1]; j < cuWindowSeqlens[s]; ++j)
            {
                windowAttentionMask[i * curHW + j] = CUDART_ZERO_FP16;
            }
        }
    }
}

void QwenViTRunner::preprocess(std::vector<std::string> const& inputStrings,
    std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
    std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer, int const maxSupportedInputLength,
    bool enableDynamicShape, void* ropeRotaryCosSinDevice, int const maxPositionEmbeddings, int const rotaryDim,
    cudaStream_t stream)
{
    std::vector<std::vector<int64_t>> imageGridTHWs;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;
    imagePreprocess(imageBuffers, imageGridTHWs, imageTokenLengths, numImages, false, stream);

    std::vector<std::vector<int32_t>> batchInputIds;
    std::vector<int32_t> batchInputLengths;
    textPreprocess(batchInputIds, batchInputLengths, inputStrings, numImages, imageTokenLengths, tokenizer);

    generateMropeParams(batchInputIds, imageGridTHWs, ropeRotaryCosSinDevice, maxPositionEmbeddings, rotaryDim, stream);

    flattenBatch(inputIds, contextLengths, batchInputIds, batchInputLengths, tokenizer->getPadId(),
        maxSupportedInputLength, enableDynamicShape);
}

std::string QwenViTRunner::applyChatTemplateSystem(std::string const& systemPrompt)
{
    return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
}

std::string QwenViTRunner::applyChatTemplateUser(
    std::string const& userPrompt, int const& numImage, bool addGenerationPrompt)
{
    std::string prompt = "<|im_start|>user\n";
    for (int i = 0; i < numImage; ++i)
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
    std::vector<std::vector<int64_t>> imageGridTHWs;
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
    std::vector<std::vector<int64_t>> imageGridTHWs;
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
        setEngineIOStatus &= mContext->setInputShape("input", mVitInput.getShape().getTRTDims());
        setEngineIOStatus &= mContext->setInputShape("attention_mask", mAttentionMask.getShape().getTRTDims());
        setEngineIOStatus &= mContext->setInputShape("rotary_pos_emb", mRotaryPosEmb.getShape().getTRTDims());
        if (mModelType == "qwen2_5_vl")
        {
            setEngineIOStatus
                &= mContext->setInputShape("window_attention_mask", mWindowAttentionMask.getShape().getTRTDims());
            setEngineIOStatus &= mContext->setInputShape("window_index", mWindowIndex.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape("reverse_window_index", mReverseWindowIndex.getShape().getTRTDims());
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

void QwenViTRunner::initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
    int const inputLength, cudaStream_t stream)
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

    std::vector<half> patches(totalSeqLength * mConfig.inputDim);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(patches.begin(), patches.end(), [&rng, &dist]() { return __float2half(dist(rng)); });
    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    std::vector<float> rotaryPosEmb(totalSeqLength * mConfig.vitPosEmbDim);
    std::generate(rotaryPosEmb.begin(), rotaryPosEmb.end(), [&rng, &dist]() { return dist(rng); });
    CUDA_CHECK(cudaMemcpyAsync(mRotaryPosEmb.rawPointer(), rotaryPosEmb.data(), rotaryPosEmb.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    std::vector<half> attentionMask(totalSeqLength * totalSeqLength, CUDART_ZERO_FP16);
    CUDA_CHECK(cudaMemcpyAsync(mAttentionMask.rawPointer(), attentionMask.data(), attentionMask.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));

    // For Qwen2.5-VL, initialize window attention parameters
    if (mModelType == "qwen2_5_vl")
    {
        // Assume grid (T,H,W) = (1, 2, imageTokenLength * 2) for simplicity
        std::vector<std::vector<int64_t>> imageGridTHWs = {{1, 2, imageTokenLength * 2}};
        std::vector<half> windowAttentionMask;
        std::vector<int64_t> windowIndex;
        std::vector<int64_t> reverseWindowIndex;

        getWindowIndex(imageGridTHWs, windowAttentionMask, windowIndex, reverseWindowIndex, totalSeqLength);

        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndex.reshape({totalSeqLength / 4});
        mReverseWindowIndex.reshape({totalSeqLength / 4});

        CUDA_CHECK(cudaMemcpyAsync(mWindowAttentionMask.rawPointer(), windowAttentionMask.data(),
            windowAttentionMask.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mWindowIndex.rawPointer(), windowIndex.data(), windowIndex.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndex.rawPointer(), reverseWindowIndex.data(),
            reverseWindowIndex.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
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