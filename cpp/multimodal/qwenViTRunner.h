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

//! \brief Configuration for Qwen-VL vision encoder
struct QwenViTConfig
{
    int32_t maxHW{0};           //!< Maximum height * width
    int32_t minHW{0};           //!< Minimum height * width
    int32_t inputDim{0};        //!< Input dimension
    int32_t vitPosEmbDim{0};    //!< Vision transformer position embedding dimension
    int32_t outHiddenSize{0};   //!< Output hidden dimension size
    int32_t vocabSize;          //!< Vocabulary size
    int32_t visionStartTokenId; //!< Token ID for vision start
    int32_t visionTokenId;      //!< Token ID for vision content
    int32_t imageTokenId;       //!< Token ID for image placeholder
    int32_t videoTokenId;       //!< Token ID for video placeholder
    float mropeTheta;           //!< Multi-dimensional RoPE theta parameter
    int32_t patchSize;          //!< Patch size in pixels
    int32_t temporalPatchSize;  //!< Temporal patch size for video
    int32_t mergeSize;          //!< Merge size for patches
    int32_t windowSize;         //!< Window attention size used by Qwen2.5-VL
    std::vector<float> imageMean{0.48145466, 0.4578275, 0.40821073}; //!< Image normalization mean values (RGB)
    std::vector<float> imageStd{
        0.26862954, 0.26130258, 0.27577711}; //!< Image normalization standard deviation values (RGB)

    //! Resize configuration. TODO: add to json config
    int32_t minPixels{128 * 28 * 28}; //!< Minimum number of pixels for dynamic resolution
    int32_t maxPixels{512 * 28 * 28}; //!< Maximum number of pixels for dynamic resolution
};

//! \brief Runner for Qwen-VL vision encoder
//!
//! This class handles the preprocessing and inference of Qwen-VL vision encoder,
//! which is part of the Qwen-VL and Qwen2-VL multimodal models.
class QwenViTRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for QwenViTRunner
    //! \param[in] engineDir Directory containing the TensorRT engine files
    //! \param[in] stream CUDA stream for execution
    QwenViTRunner(std::string const& engineDir, cudaStream_t stream);

    ~QwenViTRunner() = default;

    //! \brief Preprocess multimodal input including images and text
    //! \param[in] request LLM generation request containing images and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    //! \brief Preprocess system prompt for Qwen-VL chat template
    //! \param[in] systemPrompt System prompt string
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache
    //! \param[in] stream CUDA stream for execution
    //! \return Formatted system prompt string
    std::string preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
        rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    //! \brief Run inference on the vision encoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream) override;

    //! \brief Initialize random inputs for testing
    //! \param[out] inputIds Input token IDs
    //! \param[in] batchSize Batch size
    //! \param[in] imageTokenLength Number of image tokens
    //! \param[in] inputLength Total input length
    //! \param[in] stream CUDA stream for execution
    void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
        int const inputLength, cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    //! \param[in] configPath Path to configuration file
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& configPath) override;

    //! \brief Allocate buffers for inference
    //! \return True if allocation succeeded, false otherwise
    bool allocateBuffer() override;

    //! \brief Get pointer to configuration
    //! \return Pointer to QwenViTConfig
    void* getConfig() override;

    //! \name QwenVL-specific methods
    //! @{

    //! \brief Calculate resized image dimensions based on dynamic resolution constraints
    //! \param[in] height Input image height
    //! \param[in] width Input image width
    //! \param[in] factor Patch size factor
    //! \param[in] minPixels Minimum number of pixels
    //! \param[in] maxPixels Maximum number of pixels
    //! \param[in] maxRatio Maximum aspect ratio (default: 200)
    //! \return Tuple of (resized_height, resized_width)
    static std::tuple<int, int> getResizedImageSize(int const height, int const width, int const factor,
        int const minPixels, int const maxPixels, int const maxRatio = 200);

    //! @}

private:
    //! \brief Preprocess text portion of the request
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batch of input token IDs
    //! \param[in] numImages Number of images per request
    //! \param[in] imageTokenLengths Token lengths for each image
    //! \param[in] tokenizer Tokenizer for text processing
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
        drivellm::tokenizer::Tokenizer* tokenizer);

    //! \brief Apply Qwen-VL chat template to system prompt
    //! \param[in] systemPrompt System prompt string
    //! \return Formatted system prompt
    std::string applyChatTemplateSystem(std::string const& systemPrompt);

    //! \brief Apply Qwen-VL chat template to user prompt
    //! \param[in] userPrompt User prompt string
    //! \param[in] numImage Number of images in the prompt
    //! \param[in] addGenerationPrompt Whether to add generation prompt
    //! \return Formatted user prompt
    std::string applyChatTemplateUser(std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt);

    //! \name QwenVL-specific private methods
    //! @{

    //! \brief Compute window indices for window attention (Qwen2.5-VL)
    //! \param[in] imageGridTHWs Image grid dimensions (Temporal, Height, Width)
    //! \param[in] curHW Current height * width
    //! \param[in] stream CUDA stream for execution
    void getWindowIndex(std::vector<std::vector<int32_t>> const& imageGridTHWs, int const curHW, cudaStream_t stream);

    //! \brief Format and process a single image patch
    //! \param[in] image Input image data
    //! \param[out] imageGridTHWs Image grid dimensions for each image
    //! \param[out] imageTokenLengths Token lengths for each image
    //! \param[out] cuSeqlens Cumulative sequence lengths
    //! \param[in] stream CUDA stream for execution
    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<std::vector<int32_t>>& imageGridTHWs,
        std::vector<int64_t>& imageTokenLengths, std::vector<int32_t>& cuSeqlens, cudaStream_t stream);

    //! \brief Compute rotary position embeddings for multi-dimensional RoPE
    //! \param[in] imageGridTHWs Image grid dimensions (Temporal, Height, Width)
    //! \param[in] totalSeqLength Total sequence length
    //! \param[in] stream CUDA stream for execution
    void computeRotaryPosEmb(
        std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t const totalSeqLength, cudaStream_t stream);

    //! \brief Get multi-dimensional RoPE position indices
    //! \param[in] batchInputIds Batch of input token IDs
    //! \param[in] imageGridTHWs Image grid dimensions (Temporal, Height, Width)
    //! \param[out] mropePositionIdsPtr Pointer to multi-dimensional RoPE position IDs
    //! \param[in] maxPositionEmbeddings Maximum position embeddings
    void getRopeIdx(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int32_t>> const& imageGridTHWs, int32_t* mropePositionIdsPtr,
        int const maxPositionEmbeddings);

    //! \brief Generate multi-dimensional RoPE parameters
    //! \param[in] batchInputIds Batch of input token IDs
    //! \param[in] imageGridTHWs Image grid dimensions (Temporal, Height, Width)
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache
    //! \param[in] stream CUDA stream for execution
    void generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<std::vector<int32_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice,
        cudaStream_t stream);

    //! \brief Preprocess all images in the request
    //! \param[in] request LLM generation request containing images
    //! \param[out] imageGridTHWs Image grid dimensions for each image
    //! \param[out] imageTokenLengths Token lengths for each image
    //! \param[out] numImages Number of images per request
    //! \param[in] doResize Whether to resize images
    //! \param[in] stream CUDA stream for execution
    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& imageGridTHWs,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream);

    //! @}

    QwenViTConfig mConfig{};           //!< Qwen-VL configuration
    rt::Tensor mVitInput{};            //!< Vision encoder input tensor
    rt::Tensor mAttentionMask{};       //!< Attention mask tensor
    rt::Tensor mRotaryPosEmb{};        //!< Rotary position embeddings tensor (multi-dimensional RoPE)
    rt::Tensor mWindowAttentionMask{}; //!< Window attention mask for Qwen2.5-VL
    rt::Tensor mWindowIndex{};         //!< Window index mapping for window attention
    rt::Tensor mReverseWindowIndex{};  //!< Reverse window index mapping
};

} // namespace rt
} // namespace drivellm