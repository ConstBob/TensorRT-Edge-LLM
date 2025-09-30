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

#include "multimodalRunner.h"
#include <cuda_fp16.h>
#include <vector>

namespace drivellm
{
namespace rt
{

struct QwenViTConfig
{
    int32_t maxHW{0};
    int32_t minHW{0};
    int32_t inputDim{0};
    int32_t vitPosEmbDim{0};
    int32_t outHiddenSize{0};
    int32_t vocabSize;
    int32_t visionStartTokenId;
    int32_t visionTokenId;
    int32_t imageTokenId;
    int32_t videoTokenId;
    float mropeTheta;
    int32_t patchSize;
    int32_t temporalPatchSize;
    int32_t mergeSize;
    int32_t windowSize; // window attention size used by Qwen2.5-VL
    std::vector<float> imageMean{0.48145466, 0.4578275, 0.40821073};
    std::vector<float> imageStd{0.26862954, 0.26130258, 0.27577711};

    // Resize configuration. TODO: add to json config
    int32_t minPixels{128 * 28 * 28};
    int32_t maxPixels{512 * 28 * 28};
};

class QwenViTRunner : public MultimodalRunner
{
public:
    QwenViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~QwenViTRunner() = default;

    // TODO: Clean Old API
    void preprocess(std::vector<std::string> const& inputStrings,
        std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
        std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer,
        int const maxSupportedInputLength, bool enableDynamicShape, void* ropeRotaryCosSinDevice,
        int const maxPositionEmbeddings, int const rotaryDim, cudaStream_t stream) override;

    // TODO: Clean Old API
    std::vector<EngineInputDesc> getComputedEmbeddings() override;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    std::string preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
        rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    bool infer(cudaStream_t stream) override;

    void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
        int const inputLength, cudaStream_t stream) override;

    bool validateAndFillConfig(std::string const& configPath) override;
    bool allocateBuffer() override;
    void* getConfig() override;

    // QwenVL-specific methods
    static std::tuple<int, int> getResizedImageSize(int const height, int const width, int const factor,
        int const minPixels, int const maxPixels, int const maxRatio = 200);

private:
    // TODO: Clean Old API
    void textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& batchInputLengths,
        std::vector<std::string> const& inputStrings, std::vector<int64_t> const& numImages,
        std::vector<int64_t> const& imageTokenLengths, drivellm::tokenizer::Tokenizer* tokenizer);

    // TODO: Clean Old API
    std::string applyChatTemplate(std::string const& inputString, int64_t const& numImage,
        std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt = true);

    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
        drivellm::tokenizer::Tokenizer* tokenizer);

    std::string applyChatTemplateSystem(std::string const& systemPrompt);

    std::string applyChatTemplateUser(std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt);

    // QwenVL-specific methods
    void getWindowIndex(std::vector<std::vector<int32_t>> const& imageGridTHWs, int const curHW, cudaStream_t stream);

    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<std::vector<int32_t>>& imageGridTHWs,
        std::vector<int64_t>& imageTokenLengths, std::vector<int32_t>& cuSeqlens, cudaStream_t stream);

    void computeRotaryPosEmb(
        std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t const totalSeqLength, cudaStream_t stream);

    void getRopeIdx(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t* mropePositionIdsPtr,
        int const maxPositionEmbeddings);

    // TODO: Clean Old API
    void generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int32_t>> const& imageGridTHWs, void* cosSinCacheDevice,
        int const maxPositionEmbeddings, int const rotaryDim, cudaStream_t stream);

    void generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int32_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice,
        cudaStream_t stream);

    // TODO: Clean Old API
    void imagePreprocess(std::vector<std::vector<rt::imageUtils::ImageData>> const& imageBuffers,
        std::vector<std::vector<int32_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream);

    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& imageGridTHWs,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream);

    QwenViTConfig mConfig{};
    rt::Tensor mVitInput{};
    rt::Tensor mAttentionMask{};
    rt::Tensor mRotaryPosEmb{};
    rt::Tensor mWindowAttentionMask{};
    rt::Tensor mWindowIndex{};
    rt::Tensor mReverseWindowIndex{};
};

} // namespace rt
} // namespace drivellm