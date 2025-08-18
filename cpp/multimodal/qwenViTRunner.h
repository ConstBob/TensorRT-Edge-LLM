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

struct QwenViTConfig
{
    int64_t maxHW{0};
    int64_t minHW{0};
    int64_t inputDim{0};
    int64_t vitPosEmbDim{0};
    int64_t outHiddenSize{0};
    int64_t vocabSize;
    int64_t visionStartTokenId;
    float mropeTheta;
    int64_t patchSize;
    int64_t temporalPatchSize;
    int64_t mergeSize;
    int64_t windowSize; // window attention size used by Qwen2.5-VL
    std::vector<double> imageMean{0.48145466, 0.4578275, 0.40821073};
    std::vector<double> imageStd{0.26862954, 0.26130258, 0.27577711};
};

class QwenViTRunner : public MultimodalRunner
{
public:
    QwenViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~QwenViTRunner() = default;

    void preprocess(std::vector<std::string> const& inputStrings,
        std::vector<std::vector<ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
        std::vector<int32_t>& contextLengths, drivellm::tokenizer::Tokenizer* tokenizer,
        int const maxSupportedInputLength, bool enableDynamicShape, void* ropeRotaryCosSinDevice,
        int const maxPositionEmbeddings, int const rotaryDim, cudaStream_t stream) override;

    std::vector<EngineInputDesc> getComputedEmbeddings() override;

    void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
        int const inputLength, cudaStream_t stream) override;

    void validateAndFillConfig(std::string const& configPath) override;
    void allocateBuffer() override;
    void* getConfig() override;

    // QwenVL-specific methods
    static std::tuple<int, int> resizeImage(int const height, int const width, int const factor, int const minPixels,
        int const maxPixels, int const maxRatio = 200);

private:
    void textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& batchInputLengths,
        std::vector<std::string> const& inputStrings, std::vector<int64_t> const& numImagePerBatch,
        std::vector<int64_t> const& imageTokenLengths, drivellm::tokenizer::Tokenizer* tokenizer) override;

    std::string applyChatTemplate(std::string const& inputString, int const& numImage,
        std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt = true) override;

    // QwenVL-specific methods
    void setInputShape(int const curHW);

    void getWindowIndex(std::vector<std::vector<int64_t>> const& imageGridTHWs, std::vector<half>& windowAttentionMask,
        std::vector<int64_t>& windowIndex, std::vector<int64_t>& reverseWindowIndex, int const curHW);

    void initRotaryEmbedding(
        int numPos, int dim, float theta, std::vector<std::vector<float>>& sinusoidInp, float scale = 1.0f);

    void formatPatch(ImageData const& image, std::vector<half>& patches,
        std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
        int64_t& totalSeqLength);

    void computeRotaryPosEmb(std::vector<std::vector<int64_t>> const& imageGridTHWs, std::vector<float>& rotaryPosEmb);

    void getRopeIdx(std::vector<int64_t>& mropePositionIds, std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int64_t>> const& imageGridTHWs, int const maxPositionEmbeddings);

    void generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int64_t>> const& imageGridTHWs, void* cosSinCacheDevice,
        int const maxPositionEmbeddings, int const rotaryDim, cudaStream_t stream);

    void imagePreprocess(std::vector<std::vector<ImageData>> const& imageBuffers,
        std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& numImagePerBatch, cudaStream_t stream);

    QwenViTConfig mConfig{};
    rt::Tensor mVitInput{};
    rt::Tensor mVitOutput{};
    rt::Tensor mAttentionMask{};
    rt::Tensor mRotaryPosEmb{};
    rt::Tensor mWindowAttentionMask{};
    rt::Tensor mWindowIndex{};
    rt::Tensor mReverseWindowIndex{};
};

} // namespace rt
} // namespace drivellm