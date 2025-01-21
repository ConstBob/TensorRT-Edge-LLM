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
    // TODO: parse config from json or user-defined from args
    std::vector<double> imageMean{0.48145466, 0.4578275, 0.40821073};
    std::vector<double> imageStd{0.26862954, 0.26130258, 0.27577711};
    int64_t minPixels{3136};
    int64_t maxPixels{12845056};
    int64_t patchSize{14};
    int64_t temporalPatchSize{2};
    int64_t mergeSize{2};
    int64_t embedDim{1280};
    int64_t numHeads{16};
    int64_t maxPositionEmbeddings{32768};
    int rotaryEmbedDim{128};
    float theta = 1000000.0f;
};

class Qwen2ViTRunner
{
public:
    Qwen2ViTRunner()
        : mStream{nullptr}
        , mVisualEngine{nullptr}
        , mDeviceBuffer{}
        , mHostBuffer{}
        , isSetup{false}
    {
    }
    bool setup(std::filesystem::path const& fp, cudaStream_t& stream, int batchSize);

    void visualPreprocess(std::vector<std::vector<std::string>> const& imagePaths, std::vector<half>& patches,
        std::vector<half>& attentionMask, std::vector<float>& rotaryPosEmb, std::vector<std::vector<int64_t>>& grids);

    void textPreprocess(std::vector<std::string> const& inputStrings,
        std::vector<std::vector<std::string>> const& imagePaths,
        std::vector<std::vector<int64_t>> const& visualGridTHWs, Tokenizer* tokenizer, std::vector<int64_t>& inputIds,
        std::vector<int32_t>& contextLengths, int maxContextLength, int vocabSize = 152064);

    void visualInfer(
        std::vector<half> const& input, std::vector<half> const& attentionMask, std::vector<float> const& rotaryPosEmb);
    TensorInfo getImageEmbeds();
    TensorInfo getMropeRotaryCosSin();
    TensorInfo getMropePositionDeltas();
    void allocateBuffer();

    ~Qwen2ViTRunner()
    {
        for (auto deviceMem : mDeviceBuffer)
        {
            cudaFree(deviceMem.second);
        }
        for (auto hostMem : mHostBuffer)
        {
            free(hostMem.second);
        }
        mDeviceBuffer.clear();
        isSetup = false;
    }

private:
    std::map<std::string, void*> mDeviceBuffer;
    std::map<std::string, void*> mHostBuffer;
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    cudaStream_t mStream;
    int mBatchSize;
    bool isSetup;
    int64_t mHW;

    std::tuple<int, int> smartResize(
        int const height, int const width, int const factor, int const minPixels, int const maxPixels);
    void initRotaryEmbedding(
        int numPos, int dim, float theta, std::vector<std::vector<float>>& sinusoidInp, float scale = 1.0f);
    /**
     * Apply chat template according to chat_template.json
     * As an example, we assume putting images first and then texts, and combining into a single prompt message.
     */
    std::string applyChatTemplate(std::string const& inputString, std::vector<std::string> const& imagePaths,
        std::vector<std::vector<int64_t>> const& visualGridTHWs, int& totalImageIdx, int64_t imageMergeSize = 2,
        bool addVisionId = false, bool addGenerationPrompt = true);

    void preprocessImage(std::string const& imagePath, std::vector<half>& patches,
        std::vector<std::vector<int64_t>>& grids, int64_t& totalSeqLength);
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
