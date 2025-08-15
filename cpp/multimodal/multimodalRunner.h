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

#include "common/common.h"
#include "common/trtUtils.h"
#include "engine/llm_engine.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace drivellm
{
namespace rt
{

// Image data structure for managing loaded images
class ImageData
{
public:
    std::shared_ptr<unsigned char[]> buffer;
    int width;
    int height;
    int channels;
    bool isThumbnail;

    ImageData(unsigned char* data, int w, int h, int c, bool thumbnail = false)
        : buffer(data)
        , width(w)
        , height(h)
        , channels(c)
        , isThumbnail(thumbnail)
    {
    }

    unsigned char* data() const
    {
        return buffer.get();
    }
};

class MultimodalRunner
{
public:
    MultimodalRunner() = default;
    MultimodalRunner(std::string const& engineDir, cudaStream_t stream);
    virtual ~MultimodalRunner() = default;

    // Preprocess all inputs for multimodal runner and LLM runner
    virtual void preprocess(std::vector<std::string> const& inputStrings,
        std::vector<std::vector<ImageData>> const& imageBuffers, std::vector<int32_t>& inputIds,
        std::vector<int32_t>& contextLengths, Tokenizer* tokenizer, int const maxSupportedInputLength,
        bool enableDynamicShape, void* ropeRotaryCosSinDevice, int const maxPositionEmbeddings, int const rotaryDim,
        cudaStream_t stream)
        = 0;

    // Multimodal inference
    virtual void infer(cudaStream_t stream);

    // Get multimodal output embeddings. Used to setup extra inputs for LLM.
    virtual std::vector<EngineInputDesc> getComputedEmbeddings() = 0;

    // Initialize random inputs for benchmark purpose
    virtual void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const textTokenLength,
        int const imageTokenLength, int const inputLength, cudaStream_t stream)
        = 0;

    // Parse and fill config from config file and engine
    virtual void validateAndFillConfig(std::string const& configPath) = 0;

    // Allocate device buffer
    virtual void allocateBuffer() = 0;

    virtual void* getConfig() = 0;

    // Get model type at runtime
    virtual std::string getModelType() const
    {
        return mModelType;
    }

protected:
    // Flatten batch inputs ids to 1D array with padding and initialize context lengths
    virtual void flattenBatch(std::vector<int32_t>& inputIds, std::vector<int32_t>& contextLengths,
        std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& batchInputLengths, int32_t const padId,
        int const maxSupportedInputLength, bool enableDynamicShape);

    // Get batch input ids and lengths from input strings
    virtual void textPreprocess(std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int32_t>& batchInputLengths, std::vector<std::string> const& inputStrings,
        std::vector<int64_t> const& numImagePerBatch, std::vector<int64_t> const& imageTokenLengths,
        Tokenizer* tokenizer)
        = 0;

    // Apply chat template to prompt and insert multimodal token placeholders
    virtual std::string applyChatTemplate(std::string const& inputString, int const& numImage,
        std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt = true)
        = 0;

    // Common members
    std::string mModelType;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
};

} // namespace rt
} // namespace drivellm