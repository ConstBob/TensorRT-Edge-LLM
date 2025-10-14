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

//! \brief Configuration for InternViT vision encoder
struct InternViTConfig
{
    int32_t maxNumBlocks{0};                           //!< Maximum number of image blocks
    int32_t minNumBlocks{0};                           //!< Minimum number of image blocks
    int32_t numChannels{0};                            //!< Number of image channels (typically 3 for RGB)
    int32_t outHiddenSize{0};                          //!< Output hidden dimension size
    int32_t vocabSize;                                 //!< Vocabulary size
    int32_t patchSizeH;                                //!< Patch height in pixels
    int32_t patchSizeW;                                //!< Patch width in pixels
    int32_t blockImageSizeH;                           //!< Block image height
    int32_t blockImageSizeW;                           //!< Block image width
    int32_t imageTokenId;                              //!< Token ID for image placeholder
    std::vector<float> imageMean{0.485, 0.456, 0.406}; //!< Image normalization mean values (RGB)
    std::vector<float> imageStd{0.229, 0.224, 0.225};  //!< Image normalization standard deviation values (RGB)

    //! Resize configuration. TODO: add to json config
    int32_t minImageTiles{1}; //!< Minimum number of image tiles for dynamic resolution
    int32_t maxImageTiles{6}; //!< Maximum number of image tiles for dynamic resolution
};

//! \brief Runner for InternViT vision encoder
//!
//! This class handles the preprocessing and inference of InternViT vision encoder,
//! which is part of the InternVL multimodal model.
class InternViTRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for InternViTRunner
    //! \param[in] engineDir Directory containing the TensorRT engine files
    //! \param[in] stream CUDA stream for execution
    InternViTRunner(std::string const& engineDir, cudaStream_t stream);

    ~InternViTRunner() = default;

    //! \brief Preprocess multimodal input including images and text
    //! \param[in] request LLM generation request containing images and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream) override;

    //! \brief Preprocess system prompt for InternVL chat template
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
    //! \return Pointer to InternViTConfig
    void* getConfig() override;

    //! \name InternVL-specific methods
    //! @{

    //! \brief Calculate resized image dimensions based on dynamic resolution constraints
    //! \param[in] height Input image height
    //! \param[in] width Input image width
    //! \param[in] targetTileHeight Target tile height
    //! \param[in] targetTileWidth Target tile width
    //! \param[in] minImageTiles Minimum number of tiles
    //! \param[in] maxImageTiles Maximum number of tiles
    //! \return Tuple of (resized_height, resized_width)
    static std::tuple<int, int> getResizedImageSize(int const height, int const width, int const targetTileHeight,
        int const targetTileWidth, int const minImageTiles, int const maxImageTiles);

    //! \brief Get all supported aspect ratios for dynamic resolution
    //! \param[in] minImageTiles Minimum number of tiles (default: 1)
    //! \param[in] maxImageTiles Maximum number of tiles (default: 12)
    //! \return Vector of (width_ratio, height_ratio) pairs
    static std::vector<std::pair<int, int>> getAllSupportedAspectRatios(
        int const minImageTiles = 1, int const maxImageTiles = 12);

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

    //! \brief Apply InternVL chat template to system prompt
    //! \param[in] systemPrompt System prompt string
    //! \return Formatted system prompt
    std::string applyChatTemplateSystem(std::string const& systemPrompt);

    //! \brief Apply InternVL chat template to user prompt
    //! \param[in] userPrompt User prompt string
    //! \param[in] numImage Number of images in the prompt
    //! \param[in] addGenerationPrompt Whether to add generation prompt
    //! \return Formatted user prompt
    std::string applyChatTemplateUser(std::string const& userPrompt, int64_t const& numImage, bool addGenerationPrompt);

    //! \name InternVL-specific private methods
    //! @{

    //! \brief Format and process a single image patch
    //! \param[in] image Input image data
    //! \param[out] imageTokenLengths Token lengths for each image
    //! \param[out] numImages Number of images processed
    //! \param[out] totalNumBlocks Total number of image blocks
    //! \param[in] isThumbnail Whether the image is a thumbnail
    //! \param[in] stream CUDA stream for execution
    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
        int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream);

    //! \brief Preprocess all images in the request
    //! \param[in] request LLM generation request containing images
    //! \param[out] imageTokenLengths Token lengths for each image
    //! \param[out] numImages Number of images per request
    //! \param[in] doResize Whether to resize images
    //! \param[in] stream CUDA stream for execution
    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream);

    //! @}

    InternViTConfig mConfig; //!< InternViT configuration
    rt::Tensor mVitInput{};  //!< Vision encoder input tensor
};

} // namespace rt
} // namespace drivellm