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
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct InternVLVisualPreprocessorConfig
{
    std::string modelType;
    int64_t llmBatchSize;
    int64_t curHW;
    int64_t maxHW;
    int64_t inputDim;
    int64_t hiddenDim;
    int64_t patchSize{14};
    float theta = 1000000.0f;
    std::vector<double> imageMean{0.485, 0.456, 0.406};
    std::vector<double> imageStd{0.229, 0.224, 0.225};
    int64_t vocabSize = 151674;
    int64_t visionStartTokenId = 151652;
    int64_t downsampleFactor = 2;
    int64_t blockImageSize = 448;
};

class InternVLViTRunner
{
public:
    InternVLViTRunner(std::string modelType)
        : mStream{nullptr}
        , mVisualEngine{nullptr}
        , mDeviceBuffer{}
        , isSetup{false}
        , mConfig{modelType}
    {
    }

    /**
     * @brief Sets up the InternVLViTRunner with the necessary configurations and resources.
     *
     * @param fp The file path to visual TensorRT engine file.
     * @param stream The CUDA stream to be used for GPU operations.
     * @param llmBatchSize The batch size of LLM.
     *
     * @return True if the setup is successful, false otherwise.
     */
    bool setup(std::filesystem::path const& fp, cudaStream_t& stream, int llmBatchSize);

    void visualPreprocess(std::vector<unsigned char*> const& imageBuffers, std::vector<unsigned char*> const& thumbnailImageBuffers,
        std::vector<std::vector<int>> const& imageSizes, std::vector<half>& patches,
        std::vector<int64_t>& imageTokenLengths, bool useThumbnail);

    void textPreprocess(std::vector<std::string> const& inputStrings, std::vector<int> const& numImages,
        std::vector<int64_t> const& imageTokenLengths, Tokenizer* tokenizer, std::vector<int64_t>& inputIds,
        std::vector<int32_t>& contextLengths, int const maxContextLength);

    void internVLViTInfer(std::vector<half> const& input);

    std::vector<EngineInputDesc> getExtraLLMInputs();

    void initRandomInputs(std::vector<half>& visualInput, std::vector<int64_t>& inputIds,
        int const textTokenLength, int const imageTokenLength, int const maxContextLength);

    void allocateBuffer();

    std::tuple<int, int> adjustImageSize(int const height, int const width, std::vector<std::pair<int, int>> const& targetRatios);

    ~InternVLViTRunner()
    {
        for (auto deviceMem : mDeviceBuffer)
        {
            cudaFree(deviceMem.second);
        }
        mDeviceBuffer.clear();
        isSetup = false;
    }

private:
    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;
    std::map<std::string, void*> mDeviceBuffer;
    bool isSetup;
    InternVLVisualPreprocessorConfig mConfig;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;

    void validateAndFillConfig(int llmBatchSize);

    int setInputShape();
    /**
     * Apply chat template according to chat_template.json
     * As an example, we assume putting images first and then texts, and combining into a single prompt message.
     */
    std::string applyChatTemplate(std::string const& inputString, int const& numImages,
        std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt = true);

    void preprocessImage(unsigned char* image, unsigned char* thumbnailImage, int const& width, int const& height, int const& channels,
        std::vector<half>& patches, int64_t& totalSeqLength, bool useThumbnail, std::vector<int64_t>& imageTokenLengths);
}; 