#pragma once
#ifndef EAGLE_H
#define EAGLE_H
#include "common/benchmarkProfiler.h"
#include "common/common.h"
#include "common/json.h"
#include "decoder/decoder.h"
#include "utils/eagleUtilKernels.h"
#include <NvInferRuntime.h>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

template <typename T>
class Eagle
{
public:
    Eagle(std::unique_ptr<Decoder<T>> baseModel, std::unique_ptr<Decoder<T>> draftModel, cudaStream_t stream,
        std::string eagleEnginePath, int32_t maxPathLen = 6, int32_t topK = 10, bool isEagle3 = false,
        int32_t maxDecodingTokens = 60)
        : mBaseModel(std::move(baseModel))
        , mDraftModel(std::move(draftModel))
        , mStream(stream)
    {

        auto modelConfig = mBaseModel->getModelConfig();
        mBatchSize = modelConfig.batchSize;
        // for eagle plus 1
        mMaxSeqLen = modelConfig.maxLength + 1;
        mVocabSize = modelConfig.vocabSize;
        mMaxInputLength = modelConfig.maxInputLength;
        // don't contain root node
        mMaxDraftTokens = maxDecodingTokens - 1;
        mMaxDecodingTokens = maxDecodingTokens;
        mMaxPathLen = maxPathLen;
        mTopK = topK;
        mIsEagle3 = isEagle3;
        mEagleEnginePath = eagleEnginePath;
        mMaxDraftTokensPerStep = mMaxPathLen * mTopK;

        drivellm::JsonRoot root;
        std::string folderPath = extractFolderName(mEagleEnginePath);
        std::string configPath = folderPath + "/config.json";
        root.parseFromPath(configPath);
        auto rootNode = root.getRoot();
        mHiddenDim = rootNode["hidden_size"].getInteger();
        mTargetOutputHiddenDim = isEagle3 ? mHiddenDim * 3 : mHiddenDim;
        if (isEagle3 && rootNode.hasMember("draft_vocab_size"))
        {
            mDraftVocabSize = rootNode["draft_vocab_size"].getInteger();
        }
        else
        {
            mDraftVocabSize = rootNode["vocab_size"].getInteger();
        }
        eagleCommonParamsInit();
        allocateEagleBuffer();
        addNewBufferForModelIO();
        setupExtraInputsForBaseModel();
    };

    void generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
        std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds = -1,
        bool isEagle3 = false, std::shared_ptr<BenchmarkProfiler> const profiler = nullptr,
        std::vector<int32_t>* newTokens = nullptr, std::vector<int32_t>* iterNumbers = nullptr);
    size_t getDeviceMemorySize() const noexcept;
    void getLastHostLogits(std::vector<T>& hostLogits);
    int64_t getModelBatchSize() const noexcept;
    int64_t getMaxContextLength() const noexcept;
    void setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs);

    ~Eagle()
    {
        for (auto deviceMem : mEagleDeviceBuffer)
        {
            cudaFree(deviceMem.second);
        }
        for (auto hostMem : mEagleHostBuffer)
        {
            free(hostMem.second);
        }
        mEagleDeviceBuffer.clear();
        mEagleHostBuffer.clear();
    }

private:
    void addNewBufferForModelIO();
    void invokeSamplingAndAccept(int64_t* draftIds, const int32_t curTokensPerStep, int64_t endIds);
    void invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(int32_t layerIdx, T* hs_draft);
    void invokeUpdateCumScoresAndParentsIds(int32_t layerIdx);
    void invokeAssembleDraftIdsAndPathAndMaskAndPositionIds();
    void invokeInitializeAttentionMaskCausal();
    void draftDecodePostProcess(int32_t layerIdx);
    void draftModelDecodeInfer(std::vector<int32_t> ContextLengthForDraft);
    void updateGenerationStatus(
        int32_t& generationIter, int64_t& unfinishedBatchNum, std::vector<int32_t>& contextLengths);
    void allocateEagleBuffer();
    void invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds();

    void eagleCommonParamsInit();
    void initDraftVoc();
    void setupExtraInputsForBaseModel();
    void setupExtraInputsForDraftModel(std::vector<int32_t> const& contextLengths);
    std::unique_ptr<Decoder<T>> mBaseModel;
    std::unique_ptr<Decoder<T>> mDraftModel;

    std::map<std::string, void*> mEagleDeviceBuffer;
    std::map<std::string, void*> mEagleHostBuffer;
    EagleCommonParams mEagleCommonParams;
    std::string mEagleEnginePath;
    std::vector<int64_t> acceptedLengthsHost{mBatchSize};
    std::vector<int64_t> lastLogitsOffsetHost{mBatchSize};

    int64_t mBatchSize;
    int32_t mMaxSeqLen;
    int32_t mHiddenDim;
    int32_t mVocabSize;
    int32_t mMaxInputLength;
    int32_t mTargetOutputHiddenDim;
    int32_t mMaxDraftTokensPerStep;
    int32_t mMaxDraftTokens;
    int32_t mMaxDecodingTokens;
    // depth+1
    int32_t mMaxPathLen;
    int32_t mTopK;
    int32_t mDraftVocabSize;
    bool mIsEagle3;

    cudaStream_t mStream;
};

#endif