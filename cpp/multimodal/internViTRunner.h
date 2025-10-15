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
#include <common/tensor.h>
#include <cuda_fp16.h>
#include <vector>

namespace drivellm
{
namespace rt
{

struct InternViTConfig
{
    int32_t maxNumBlocks{0};
    int32_t minNumBlocks{0};
    int32_t numChannels{0};
    int32_t outHiddenSize{0};
    int32_t vocabSize;
    int32_t patchSizeH;
    int32_t patchSizeW;
    int32_t blockImageSizeH;
    int32_t blockImageSizeW;
    int32_t imageTokenId;
    std::vector<float> imageMean{0.485, 0.456, 0.406};
    std::vector<float> imageStd{0.229, 0.224, 0.225};

    // Resize configuration. TODO: add to json config
    int32_t minImageTiles{1};
    int32_t maxImageTiles{6};
};

class InternViTRunner : public MultimodalRunner
{
public:
    InternViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~InternViTRunner() = default;

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

    // InternVL-specific methods
    static std::tuple<int, int> getResizedImageSize(int const height, int const width, int const targetTileHeight,
        int const targetTileWidth, int const minImageTiles, int const maxImageTiles);

    static std::vector<std::pair<int, int>> getAllSupportedAspectRatios(
        int const minImageTiles = 1, int const maxImageTiles = 12);

private:
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
        drivellm::tokenizer::Tokenizer* tokenizer);

    std::string applyChatTemplateSystem(std::string const& systemPrompt);

    std::string applyChatTemplateUser(std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt);

    // InternVL-specific methods
    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
        int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream);

    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream);

    InternViTConfig mConfig;
    rt::Tensor mVitInput{};
};

} // namespace rt
} // namespace drivellm