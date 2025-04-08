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

struct VisualPreprocessorConfig
{
    VisualPreprocessorConfig(std::string modelType)
        : modelType{modelType}
    {
        if (modelType == "qwen2_5_vl")
        {
            maxPositionEmbeddings = 128000;
        }
    }

    std::string modelType;
    int64_t batchSize;
    int64_t minPixels{4 * 28 * 28};
    int64_t maxPixels{16384 * 28 * 28};
    int64_t patchSize{14};
    int64_t temporalPatchSize{2};
    int64_t mergeSize{2};
    int64_t embedDim{1280};
    int64_t numHeads{16};
    int64_t maxPositionEmbeddings{32768};
    int rotaryEmbedDim{128};
    float theta = 1000000.0f;
    std::vector<double> imageMean{0.48145466, 0.4578275, 0.40821073};
    std::vector<double> imageStd{0.26862954, 0.26130258, 0.27577711};
    int64_t windowSize{112};  // window attention size used by Qwen2.5-VL
};

class Qwen2ViTRunner
{
public:
    Qwen2ViTRunner(std::string modelType)
        : mStream{nullptr}
        , mVisualEngine{nullptr}
        , mDeviceBuffer{}
        , isSetup{false}
        , mConfig{modelType}
    {
    }

    bool setup(std::filesystem::path const& fp, cudaStream_t& stream, int batchSize = 1, int minPixels = 4 * 28 * 28,
        int maxPixels = 16384 * 28 * 28);

    void visualPreprocess(std::vector<unsigned char*> const& imageBuffers,
        std::vector<std::vector<int>> const& imageSizes, std::vector<half>& patches, std::vector<half>& attentionMask,
        std::vector<float>& rotaryPosEmb, std::vector<std::vector<int64_t>>& grids);

    void textPreprocess(std::vector<std::string> const& inputStrings, std::vector<int> const& numImages,
        std::vector<std::vector<int64_t>> const& visualGridTHWs, Tokenizer* tokenizer, std::vector<int64_t>& inputIds,
        std::vector<int32_t>& contextLengths, int const maxContextLength, int const vocabSize = 152064);

    void getWindowIndex(std::vector<std::vector<int64_t>> const& grids, std::vector<half>& windowAttentionMask, 
        std::vector<int64_t>& windowIndex, std::vector<int64_t>& reverseWindowIndex);
    
    void qwen2ViTInfer(
        std::vector<half> const& input, std::vector<half> const& attentionMask, std::vector<float> const& rotaryPosEmb);

    void qwen2_5ViTInfer(
        std::vector<half> const& input, std::vector<half> const& attentionMask, std::vector<float> const& rotaryPosEmb, 
        std::vector<half> const& windowAttentionMask, std::vector<int64_t> const& windowIndex,
        std::vector<int64_t> const& reverseWindowIndex);

    std::vector<EngineInputDesc> getExtraLLMInputs();

    void initRandomInputs(std::vector<half>& visualInput, std::vector<half>& visualAttentionMask,
        std::vector<float>& visualRotaryPosEmb, std::vector<half>& windowAttentionMask, std::vector<int64_t>& windowIndex,
        std::vector<int64_t>& reverseWindowIndex, std::vector<int64_t>& inputIds, int const textTokenLength,
        int const imageTokenLength, int const maxContextLength, int const vocabSize = 152064);

    void allocateBuffer();
    void freeBuffer();

    ~Qwen2ViTRunner()
    {
        freeBuffer();
        isSetup = false;
    }

private:
    std::map<std::string, void*> mDeviceBuffer;
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    cudaStream_t mStream;
    bool isSetup;
    int64_t mHW;

    std::tuple<int, int> smartResize(int const height, int const width, int const maxRatio = 200);
    void initRotaryEmbedding(
        int numPos, int dim, float theta, std::vector<std::vector<float>>& sinusoidInp, float scale = 1.0f);
    /**
     * Apply chat template according to chat_template.json
     * As an example, we assume putting images first and then texts, and combining into a single prompt message.
     */
    std::string applyChatTemplate(std::string const& inputString, int const& numImages,
        std::vector<std::vector<int64_t>> const& visualGridTHWs, int& totalImageIdx, int64_t imageMergeSize = 2,
        bool addGenerationPrompt = true);

    void preprocessImage(unsigned char* image, int const& width, int const& height, int const& channels,
        std::vector<half>& patches, std::vector<std::vector<int64_t>>& grids, int64_t& totalSeqLength);
    void computeRotaryPosEmb(std::vector<std::vector<int64_t>> const& grids, std::vector<float>& rotaryPosEmb);
    /**
     * Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
     */
    void getRopeIdx(std::vector<std::vector<int64_t>> const& batchInputIds,
        std::vector<std::vector<int64_t>> const& imageGridTHWs, std::vector<int64_t>& mropePositionIds,
        std::vector<int64_t>& mropePositionDeltas, int64_t maxPositionEmbeddings, int64_t visionStartTokenId = 151652,
        int64_t spacialMergeSize = 2);
    void generateMropeParams(std::vector<std::vector<int64_t>> const& batchInputIds,
        std::vector<std::vector<int64_t>> const& visualGridTHWs);

    VisualPreprocessorConfig mConfig;
};