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
    int64_t maxHW{0};
    int64_t minHW{0};
    int64_t inputDim{0};
    int64_t outHiddenSize{0};
    int64_t vocabSize;
    int64_t patchSizeH;
    int64_t patchSizeW;
    int64_t blockImageSizeH;
    int64_t blockImageSizeW;
    std::vector<double> imageMean{0.485, 0.456, 0.406};
    std::vector<double> imageStd{0.229, 0.224, 0.225};
};

class InternViTRunner : public MultimodalRunner
{
public:
    InternViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~InternViTRunner() = default;

    void preprocess(std::vector<std::string> const& inputStrings,
        std::vector<std::vector<ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
        std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer,
        int const maxSupportedInputLength, bool enableDynamicShape, void* ropeRotaryCosSinDevice [[maybe_unused]],
        int const maxPositionEmbeddings [[maybe_unused]], int const rotaryDim [[maybe_unused]],
        cudaStream_t stream) override;

    std::vector<EngineInputDesc> getComputedEmbeddings() override;

    void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
        int const inputLength, cudaStream_t stream) override;

    void validateAndFillConfig(std::string const& configPath) override;
    void allocateBuffer() override;
    void* getConfig() override;

    // InternVL-specific methods
    static std::tuple<int, int> resizeImage(int const height, int const width, int const targetTileHeight,
        int const targetTileWidth, int const minImageTiles, int const maxImageTiles);

    static std::vector<std::pair<int, int>> getAllSupportedAspectRatios(
        int const minImageTiles = 1, int const maxImageTiles = 12);

private:
    void textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& batchInputLengths,
        std::vector<std::string> const& inputStrings, std::vector<int64_t> const& numImagePerBatch,
        std::vector<int64_t> const& imageTokenLengths, drivellm::tokenizer::Tokenizer* tokenizer) override;

    std::string applyChatTemplate(std::string const& inputString, int const& numImages,
        std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt = true) override;

    // InternVL-specific methods
    void formatPatch(ImageData const& image, std::vector<half>& patches, std::vector<int64_t>& imageTokenLengths,
        int64_t& numImagePerBatch, int64_t& totalSeqLength);

    void imagePreprocess(std::vector<std::vector<ImageData>> const& imageBuffers,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImagePerBatch, cudaStream_t stream);

    InternViTConfig mConfig;
    rt::Tensor mVitInput{};
    rt::Tensor mVitOutput{};
};

} // namespace rt
} // namespace drivellm