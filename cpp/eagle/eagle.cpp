#include "eagle.h"
#include "common/common.h"
#include "eagle/utils/eagleUtilKernels.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>

template <typename T>
void Eagle<T>::eagleCommonParamsInit()
{
    mEagleCommonParams.batchSize = mBatchSize;
    mEagleCommonParams.maxDecodingTokens = mMaxDecodingTokens;
    mEagleCommonParams.maxPathLen = mMaxPathLen;
    mEagleCommonParams.hiddenDim = mHiddenDim;
    mEagleCommonParams.targetHiddenDim = mTargetOutputHiddenDim;
    mEagleCommonParams.topK = mTopK;
    mEagleCommonParams.maxDraftTokens = mMaxDraftTokens;
    mEagleCommonParams.numLayers = mBaseModel->getModelConfig().numLayers;
    mEagleCommonParams.numHead = mBaseModel->getModelConfig().numHead;
    mEagleCommonParams.hiddenSizePerHead = mBaseModel->getModelConfig().hiddenSizePerHead;
    mEagleCommonParams.stream = mStream;
    mEagleCommonParams.maxSeqLen = mMaxSeqLen;
    // less 1 than mMaxSeqLen
    mEagleCommonParams.kvCacheSeqLen = mBaseModel->getModelConfig().maxLength;
}

template <typename T>
void Eagle<T>::invokeSamplingAndAccept(int64_t* draftIds, const int32_t curTokensPerStep, int64_t endIds)
{
    auto const logits_last_token = mBaseModel->getDeviceBuffer("logits");

    TopKSamplingKernelParams<T> params;
    params.logProbs = reinterpret_cast<T*>(logits_last_token);
    if (draftIds == nullptr)
    {
        params.batchSize = 1 * mBatchSize;
        params.maxBatchSize = 1 * mBatchSize;
        params.maxSeqLen = 1;
    }
    else
    {
        params.batchSize = mMaxDecodingTokens * mBatchSize;
        params.maxBatchSize = mMaxDecodingTokens * mBatchSize;
        params.maxSeqLen = 1;
    }
    params.workspace = mEagleDeviceBuffer["workspaceForVerification"];
    params.maxTokensPerStep = 1;
    params.maxTopK = 1;
    params.vocabSizePadded = mVocabSize;
    params.outputIds = static_cast<int64_t*>(mEagleDeviceBuffer["targetIds"]);
    // fist sample
    invokeBatchTopKSampling(params, mStream);
    auto const hiddenStates = static_cast<T*>(mBaseModel->getDeviceBuffer("hidden_states"));

    AcceptDraftTokensByIdsWithPathsParams accparms;
    accparms.outputIds = static_cast<int64_t*>(mBaseModel->getDeviceBuffer("input_ids"));
    accparms.inputIdsDraftDecode = static_cast<int64_t*>(mEagleDeviceBuffer["selectedOutputIdsDraft"]);
    accparms.draftIds = draftIds;
    accparms.targetIds = static_cast<int64_t*>(mEagleDeviceBuffer["targetIds"]);
    accparms.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    accparms.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    accparms.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    accparms.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    accparms.curTokensPerStep = curTokensPerStep;
    accparms.endIds = endIds;
    accparms.finishedFinal = static_cast<int64_t*>(mEagleDeviceBuffer["finishedFinal"]);
    dispatchAcceptDraftTokensByIdsWithPaths(accparms, mEagleCommonParams);
}
template <typename T>
void Eagle<T>::invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds()
{
    void* KVCache = mBaseModel->getDeviceBuffer("kv_cache");
    void* hiddenStates = mBaseModel->getDeviceBuffer("hidden_states");

    UpdateKVCacheParams<T> params;
    params.KVCache = static_cast<T*>(KVCache);
    params.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    params.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    params.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    params.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    params.hiddenStates = static_cast<T*>(hiddenStates);
    params.hiddenStatesInputs = static_cast<T*>(mEagleDeviceBuffer["hiddenStatesDraftDecode"]);
    params.treePositionIds = static_cast<int32_t*>(mEagleDeviceBuffer["treePositionIds"]);
    dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds<T>(params, mEagleCommonParams);
}

template <typename T>
void Eagle<T>::invokeInitializeAttentionMaskCausal()
{
    InitCausalAttentionMaskParams params;
    params.mask = static_cast<bool*>(mEagleDeviceBuffer["attentionMaskCausal"]);
    params.packedMask = static_cast<int32_t*>(mEagleDeviceBuffer["packedAttentionMaskCausal"]);
    dispatchInitializeAttentionMaskCausal(params, mEagleCommonParams);
}

template <typename T>
void Eagle<T>::initDraftVoc()
{
    std::vector<int64_t> draftVocHost(mDraftVocabSize);
    std::filesystem::path engineFilePath(mEagleEnginePath);
    auto const draftVocPath = engineFilePath.parent_path().string() + "/d2t.bin";
    std::ifstream fin(draftVocPath, std::ios::binary);
    if (!fin)
    {
        printf("Error: Failed to open d2t.bin in path %s, it must be provided for Eagle3.\n", draftVocPath.c_str());
        exit(EXIT_FAILURE);
    }
    fin.seekg(0, std::ios::end);
    std::streamsize fileSize = fin.tellg();
    if (fileSize == 0)
    {
        printf("Error: d2t.bin file is empty in path %s\n", draftVocPath.c_str());
        exit(EXIT_FAILURE);
    }
    if (fileSize != mDraftVocabSize * sizeof(int64_t))
    {
        printf("Error: d2t.bin file size mismatch. Got: %ld, Expected: %ld\n", (long) fileSize,
            (long) (mDraftVocabSize * sizeof(int64_t)));
        exit(EXIT_FAILURE);
    }
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(draftVocHost.data()), mDraftVocabSize * sizeof(int64_t));
    if (!fin)
    {
        printf("Error: Failed to read d2t.bin in path %s, it must be provided for Eagle3.\n", draftVocPath.c_str());
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpyAsync(mEagleDeviceBuffer["draftVoc"], draftVocHost.data(), mDraftVocabSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, mStream));
}

template <typename T>
void Eagle<T>::allocateEagleBuffer()
{
    //[bs,maxDraftTokens,mMaxDraftTokens]
    void* allDraftIdsDevice;
    CUDA_CHECK(cudaMalloc(&allDraftIdsDevice, mBatchSize * mMaxDraftTokens * mMaxDraftTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["allDraftIds"] = allDraftIdsDevice;

    // target model inputids:[maxDraftTokens,maxDraftTokens]
    void* decodingInputIdsDevice;
    CUDA_CHECK(cudaMalloc(&decodingInputIdsDevice, mBatchSize * mMaxDecodingTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["draftIds"] = decodingInputIdsDevice;

    void* targetIdsDevice;
    CUDA_CHECK(cudaMalloc(&targetIdsDevice, mBatchSize * mMaxDecodingTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["targetIds"] = targetIdsDevice;

    void* draftIdsAncestorsDevice;
    CUDA_CHECK(cudaMalloc(&draftIdsAncestorsDevice, mBatchSize * mMaxDraftTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["draftIdsAncestors"] = draftIdsAncestorsDevice;

    // draft model input ids:[mBatchSize,mMaxPathLen,mTopK]
    void* selectedOutputIdsDraftDevice;
    CUDA_CHECK(cudaMalloc(&selectedOutputIdsDraftDevice, mBatchSize * mMaxDraftTokensPerStep * sizeof(int64_t)));
    mEagleDeviceBuffer["selectedOutputIdsDraft"] = selectedOutputIdsDraftDevice;

    // for update kv cache
    void* bestPathIdsDevice;
    CUDA_CHECK(cudaMalloc(&bestPathIdsDevice, mBatchSize * sizeof(int32_t)));
    mEagleDeviceBuffer["bestPathIds"] = bestPathIdsDevice;
    // for update kv cache
    void* acceptedLengthsDevice;
    CUDA_CHECK(cudaMalloc(&acceptedLengthsDevice, mBatchSize * sizeof(int64_t)));
    mEagleDeviceBuffer["acceptedLengths"] = acceptedLengthsDevice;

    void* finishedFinalDevice;
    CUDA_CHECK(cudaMalloc(&finishedFinalDevice, mBatchSize * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(finishedFinalDevice, 0, mBatchSize * sizeof(int64_t)));
    mEagleDeviceBuffer["finishedFinal"] = finishedFinalDevice;
    mEagleHostBuffer["finishedFinal"] = malloc(mBatchSize * sizeof(int64_t));

    // draft model input:[mBatchSize,mMaxPathLen,mTargetOutputHiddenDim]
    void* hiddenStatesDraftDecodeDevice;
    CUDA_CHECK(cudaMalloc(
        &hiddenStatesDraftDecodeDevice, mBatchSize * mMaxDraftTokensPerStep * mTargetOutputHiddenDim * sizeof(T)));
    mEagleDeviceBuffer["hiddenStatesDraftDecode"] = hiddenStatesDraftDecodeDevice;

    // draft model input:[mBatchSize,mMaxPathLen,mTargetOutputHiddenDim]
    void* hiddenStatesDraftDecodeFromDraftDevice;
    CUDA_CHECK(cudaMalloc(
        &hiddenStatesDraftDecodeFromDraftDevice, mBatchSize * mMaxDraftTokensPerStep * mHiddenDim * sizeof(T)));
    mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"] = hiddenStatesDraftDecodeFromDraftDevice;

    void* pathsDevice;
    CUDA_CHECK(cudaMalloc(&pathsDevice, mBatchSize * mMaxDecodingTokens * (mMaxPathLen + 1) * sizeof(int32_t)));
    mEagleDeviceBuffer["paths"] = pathsDevice;

    void* validPathNumDevice;
    CUDA_CHECK(cudaMalloc(&validPathNumDevice, mBatchSize * mMaxDecodingTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["validPathNum"] = validPathNumDevice;

    // [mBatchSize,mMaxPathLen,topk*topk] mMaxPathLen = depth+1
    void* allScoresDevice;
    CUDA_CHECK(cudaMalloc(&allScoresDevice, mBatchSize * mMaxPathLen * mTopK * mTopK * sizeof(float)));
    const size_t totalElements = mBatchSize * mMaxPathLen * mTopK * mTopK;
    std::vector<float> hostScores(totalElements, -INFINITY);
    CUDA_CHECK(cudaMemcpyAsync(
        allScoresDevice, hostScores.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mEagleDeviceBuffer["allScores"] = allScoresDevice;

    //[mBatchSize,topk*topk]
    void* cumScoresDevice;
    CUDA_CHECK(cudaMalloc(&cumScoresDevice, mBatchSize * mTopK * mTopK * sizeof(float)));
    mEagleDeviceBuffer["cumScores"] = cumScoresDevice;

    //[mBatchSize,topk]
    void* intermediateScoresDevice;
    CUDA_CHECK(cudaMalloc(&intermediateScoresDevice, mBatchSize * mTopK * sizeof(float)));
    mEagleDeviceBuffer["intermediateScores"] = intermediateScoresDevice;

    //[mBatchSize,mMaxPathLen,topk*topk]
    void* allTokensDevice;
    CUDA_CHECK(cudaMalloc(&allTokensDevice, mBatchSize * mMaxPathLen * mTopK * mTopK * sizeof(int64_t)));
    mEagleDeviceBuffer["allTokens"] = allTokensDevice;

    //[mBatchSize,mMaxPathLen,topk] [6,10]
    void* parantsIdsDevice;
    CUDA_CHECK(cudaMalloc(&parantsIdsDevice, mBatchSize * mMaxPathLen * mTopK * sizeof(int64_t)));
    mEagleDeviceBuffer["parantsIds"] = parantsIdsDevice;

    //[mBatchSize,topk,topk] for eagle draft every step for second topk in whole process: 100
    void* outputIdsAllDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputIdsAllDraftDevice, mBatchSize * mTopK * mTopK * sizeof(int64_t)));
    mEagleDeviceBuffer["outputIdsAllDraft"] = outputIdsAllDraftDevice;

    //[mBatchSize,topk,topk]
    void* outputLogProbsAllDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputLogProbsAllDraftDevice, mBatchSize * mTopK * mTopK * sizeof(float)));
    mEagleDeviceBuffer["outputLogProbsAllDraft"] = outputLogProbsAllDraftDevice;

    //[mBatchSize,topk] for third topk in whole process: 10 out of 100
    void* outputIdsCurrentDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputIdsCurrentDraftDevice, mBatchSize * mTopK * sizeof(int64_t)));
    mEagleDeviceBuffer["outputIdsCurrentDraft"] = outputIdsCurrentDraftDevice;
    //[mBatchSize,topk]
    void* outputLogProbsCurrentDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputLogProbsCurrentDraftDevice, mBatchSize * mTopK * sizeof(float)));
    mEagleDeviceBuffer["outputLogProbsCurrentDraft"] = outputLogProbsCurrentDraftDevice;

    //[mBatchSize,topk,topk]
    void* treeMaskInitDevice;
    CUDA_CHECK(cudaMalloc(&treeMaskInitDevice, mBatchSize * mTopK * mTopK * sizeof(bool)));
    mEagleDeviceBuffer["treeMaskInit"] = treeMaskInitDevice;

    //[mBatchSize,topk,mMaxDraftTokensPerStep]
    void* treeMaskInputDevice;
    CUDA_CHECK(cudaMalloc(&treeMaskInputDevice, mBatchSize * mTopK * mMaxDraftTokensPerStep * sizeof(bool)));
    mEagleDeviceBuffer["treeMaskInput"] = treeMaskInputDevice;
    //[mBatchSize,topk,mMaxDraftTokensPerStep]
    void* treeMaskUpdateDevice;
    CUDA_CHECK(cudaMalloc(&treeMaskUpdateDevice, mBatchSize * mTopK * mMaxDraftTokensPerStep * sizeof(bool)));
    mEagleDeviceBuffer["treeMaskUpdate"] = treeMaskUpdateDevice;
    //[mBatchSize,mMaxDraftTokensPerStep,mMaxDraftTokensPerStep]
    void* treeMaskUpdateforAttentionDevice;
    CUDA_CHECK(cudaMalloc(&treeMaskUpdateforAttentionDevice,
        mBatchSize * mMaxDraftTokensPerStep * mMaxDraftTokensPerStep * sizeof(bool)));
    CUDA_CHECK(cudaMemsetAsync(treeMaskUpdateforAttentionDevice, 0,
        mBatchSize * mMaxDraftTokensPerStep * mMaxDraftTokensPerStep * sizeof(bool), mStream));
    mEagleDeviceBuffer["treeMaskUpdateforAttention"] = treeMaskUpdateforAttentionDevice;
    //[bs,maxLength,ceil(maxLength/32)]
    void* packedTreeMaskUpdateforAttentionDevice;
    CUDA_CHECK(cudaMalloc(&packedTreeMaskUpdateforAttentionDevice,
        mBatchSize * mMaxDraftTokensPerStep * divUp(mMaxDraftTokensPerStep, 32) * sizeof(int32_t)));
    mEagleDeviceBuffer["packedTreeMaskUpdateforAttention"] = packedTreeMaskUpdateforAttentionDevice;

    //[bs,maxLength,ceil(maxLength/32)]
    void* packedTreeMaskUpdateforAttentionNoPaddingDevice;
    CUDA_CHECK(cudaMalloc(&packedTreeMaskUpdateforAttentionNoPaddingDevice,
        mBatchSize * mMaxDraftTokensPerStep * divUp(mMaxDraftTokensPerStep, 32) * sizeof(int32_t)));
    mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"] = packedTreeMaskUpdateforAttentionNoPaddingDevice;

    //[mBatchSize,topk*(depth+1),topk]
    void* treePositionIdsDevice;
    CUDA_CHECK(cudaMalloc(&treePositionIdsDevice, mBatchSize * mMaxDraftTokensPerStep * sizeof(int32_t)));
    mEagleDeviceBuffer["treePositionIds"] = treePositionIdsDevice;

    //[mBatchSize,maxDecodingTokens,maxDecodingTokens] for the verification
    void* treeMaskVerificationDevice;
    CUDA_CHECK(
        cudaMalloc(&treeMaskVerificationDevice, mBatchSize * mMaxDecodingTokens * mMaxDecodingTokens * sizeof(bool)));
    mEagleDeviceBuffer["treeMaskVerification"] = treeMaskVerificationDevice;

    //[bs,maxLength,ceil(maxLength/32)]
    void* packedTreeMaskVerificationDevice;
    CUDA_CHECK(cudaMalloc(&packedTreeMaskVerificationDevice,
        mBatchSize * mMaxDecodingTokens * divUp(mMaxDecodingTokens, 32) * sizeof(int32_t)));
    mEagleDeviceBuffer["packedTreeMaskVerification"] = packedTreeMaskVerificationDevice;

    //[mBatchSize,maxDecodingTokens]
    void* positionIdsVerificationDevice;
    CUDA_CHECK(cudaMalloc(&positionIdsVerificationDevice, mBatchSize * mMaxDecodingTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["positionIdsVerification"] = positionIdsVerificationDevice;

    // for eagle_0   for all draft model:[bs,mMaxDecodingTokens]
    void* attentionMaskCausalDevice;
    CUDA_CHECK(
        cudaMalloc(&attentionMaskCausalDevice, mBatchSize * mMaxDecodingTokens * mMaxDecodingTokens * sizeof(bool)));
    mEagleDeviceBuffer["attentionMaskCausal"] = attentionMaskCausalDevice;

    //[bs,maxLength,ceil(maxLength/32)]
    void* packedAttentionMaskCausalDevice;
    CUDA_CHECK(cudaMalloc(&packedAttentionMaskCausalDevice,
        mBatchSize * mMaxDecodingTokens * divUp(mMaxDecodingTokens, 32) * sizeof(int32_t)));
    mEagleDeviceBuffer["packedAttentionMaskCausal"] = packedAttentionMaskCausalDevice;
    invokeInitializeAttentionMaskCausal();

    //[mBatchSize,mMaxDraftTokens]
    void* fourthTopKIdsDevice;
    CUDA_CHECK(cudaMalloc(&fourthTopKIdsDevice, mBatchSize * mMaxDraftTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["fourthTopKIds"] = fourthTopKIdsDevice;

    //[mBatchSize,mMaxDraftTokens]
    void* fourthTopkProbsDevice;
    CUDA_CHECK(cudaMalloc(&fourthTopkProbsDevice, mBatchSize * mMaxDraftTokens * sizeof(float)));
    mEagleDeviceBuffer["fourthTopkProbs"] = fourthTopkProbsDevice;

    // for topk1: torch.Size([1, 152064])
    auto const workspaceSize1 = getTopKWorkspaceSize<T>(mBatchSize, /* maxTokensPerStep */ 1, mTopK, mVocabSize);
    void* topk1WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk1WorkspaceDevice, workspaceSize1));
    mEagleDeviceBuffer["topk1Workspace"] = topk1WorkspaceDevice;

    // for topk2
    auto const workspaceSize2
        = getTopKWorkspaceSize<T>(mTopK * mBatchSize, /* maxTokensPerStep */ 1, mTopK, mVocabSize);
    void* topk2WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk2WorkspaceDevice, workspaceSize2));
    mEagleDeviceBuffer["topk2Workspace"] = topk2WorkspaceDevice;

    // for topk3: [bs*topk,topk]
    auto const workspaceSize3 = getTopKWorkspaceSize<float>(mBatchSize, /* maxTokensPerStep */ 1, mTopK, mTopK * mTopK);
    void* topk3WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk3WorkspaceDevice, workspaceSize3));
    mEagleDeviceBuffer["topk3Workspace"] = topk3WorkspaceDevice;

    // for topk4:[bs*topk, mMaxPathLen*mTopK]
    auto const workspaceSize4 = getTopKWorkspaceSize<float>(
        mBatchSize, /* maxTokensPerStep */ 1, mMaxDraftTokens, mMaxPathLen * mTopK * mTopK);
    void* topk4WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk4WorkspaceDevice, workspaceSize4));
    mEagleDeviceBuffer["topk4Workspace"] = topk4WorkspaceDevice;

    auto const workspaceSizeForVerification
        = getTopKWorkspaceSize<float>(mBatchSize * mMaxDecodingTokens, /* maxTokensPerStep */ 1, 1, mVocabSize);
    void* workspaceForVerificationDevice;
    CUDA_CHECK(cudaMalloc(&workspaceForVerificationDevice, workspaceSizeForVerification));
    mEagleDeviceBuffer["workspaceForVerification"] = workspaceForVerificationDevice;

    void* draftVocDevice;
    CUDA_CHECK(cudaMalloc(&draftVocDevice, mDraftVocabSize * sizeof(int64_t)));
    mEagleDeviceBuffer["draftVoc"] = draftVocDevice;
    if (mIsEagle3)
    {
        initDraftVoc();
    }

    void* lastLogitsOffsetDevice;
    CUDA_CHECK(cudaMalloc(&lastLogitsOffsetDevice, mBatchSize * sizeof(int64_t)));
    mEagleDeviceBuffer["lastLogitsOffset"] = lastLogitsOffsetDevice;
}

template <typename T>
int64_t Eagle<T>::getModelBatchSize() const noexcept
{
    return mBatchSize;
}
template <typename T>
int64_t Eagle<T>::getMaxContextLength() const noexcept
{
    return mBaseModel->getMaxContextLength();
}

template <typename T>
void Eagle<T>::addNewBufferForModelIO()
{
    // add new buffer for base model and draft model IO
    mBaseModel->addNewBuffer("input_ids", {2, {mBatchSize, mMaxInputLength}}, sizeof(int64_t));
    mBaseModel->addNewBuffer("hidden_states", {3, {mBatchSize, mMaxInputLength, mTargetOutputHiddenDim}}, sizeof(T));
    mBaseModel->addNewBuffer("logits", {2, {mBatchSize * mMaxDecodingTokens, mVocabSize}}, sizeof(T));
    mDraftModel->addNewBuffer("hidden_states", {3, {mBatchSize, mMaxInputLength, mHiddenDim}}, sizeof(T));
    mDraftModel->addNewBuffer("logits", {2, {mBatchSize * mMaxDraftTokensPerStep, mVocabSize}}, sizeof(T));
}
template <typename T>
void Eagle<T>::setupExtraInputsForBaseModel()
{

    // for base model context:
    std::vector<EngineInputDesc> extraInputsForBaseModelContext;
    extraInputsForBaseModelContext.push_back(EngineInputDesc(
        "attention_mask", mEagleDeviceBuffer["packedAttentionMaskCausal"], nullptr, {3, {mBatchSize, 1, 1}}, {}));
    extraInputsForBaseModelContext.push_back(
        EngineInputDesc("attention_pos_id", mEagleDeviceBuffer["treePositionIds"], nullptr, {2, {mBatchSize, 1}}, {}));
    mBaseModel->setupExtraInputs(extraInputsForBaseModelContext);

    // for base model decode:
    std::vector<EngineInputDesc> extraInputsForBaseModelDecode;
    extraInputsForBaseModelDecode.push_back(EngineInputDesc(
        "input_ids", nullptr, mEagleDeviceBuffer["draftIds"], {}, {2, {mBatchSize, mMaxDecodingTokens}}));
    extraInputsForBaseModelDecode.push_back(
        EngineInputDesc("attention_mask", nullptr, mEagleDeviceBuffer["packedTreeMaskVerification"], {},
            {3, {mBatchSize, mMaxDecodingTokens, divUp(mMaxDecodingTokens, 32)}}));
    extraInputsForBaseModelDecode.push_back(EngineInputDesc("attention_pos_id", nullptr,
        mEagleDeviceBuffer["positionIdsVerification"], {}, {2, {mBatchSize, mMaxDecodingTokens}}));

    auto const last_token_ids_device = mBaseModel->getDeviceBuffer("last_token_ids");
    extraInputsForBaseModelDecode.push_back(
        EngineInputDesc("last_token_ids", nullptr, last_token_ids_device, {}, {1, {mBatchSize * mMaxDecodingTokens}}));
    mBaseModel->setupExtraInputs(extraInputsForBaseModelDecode);
}

template <typename T>
void Eagle<T>::setupExtraInputsForDraftModel(std::vector<int32_t> const& contextLengths)
{

    // for draft model context,only for bs=1
    std::vector<EngineInputDesc> extraInputsForDraftModelContext;
    auto const hiddenStates = mBaseModel->getDeviceBuffer("hidden_states");
    extraInputsForDraftModelContext.push_back(EngineInputDesc("hidden_states_input", hiddenStates, nullptr,
        {3, {mBatchSize, contextLengths[0], mTargetOutputHiddenDim}}, {}));

    auto const hiddenStatesFromDraft = mDraftModel->getDeviceBuffer("hidden_states");
    extraInputsForDraftModelContext.push_back(EngineInputDesc("hidden_states_from_draft", hiddenStatesFromDraft,
        nullptr, {3, {mBatchSize, contextLengths[0], mHiddenDim}}, {}));
    // no need for context phase
    extraInputsForDraftModelContext.push_back(EngineInputDesc(
        "attention_mask", mEagleDeviceBuffer["packedAttentionMaskCausal"], nullptr, {3, {mBatchSize, 1, 1}}, {}));
    extraInputsForDraftModelContext.push_back(
        EngineInputDesc("attention_pos_id", mEagleDeviceBuffer["treePositionIds"], nullptr, {2, {mBatchSize, 1}}, {}));
    auto last_token_ids_device = mDraftModel->getDeviceBuffer("last_token_ids");
    extraInputsForDraftModelContext.push_back(
        EngineInputDesc("last_token_ids", nullptr, last_token_ids_device, {}, {1, {mBatchSize}}));
    mDraftModel->setupExtraInputs(extraInputsForDraftModelContext);

    // for draft model decode:
    std::vector<EngineInputDesc> extraInputsForDraftModelDecode;
    extraInputsForDraftModelDecode.push_back(EngineInputDesc("input_ids", nullptr,
        mEagleDeviceBuffer["selectedOutputIdsDraft"], {}, {2, {mBatchSize, mMaxDraftTokensPerStep}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("attention_mask", nullptr, mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, divUp(mMaxDraftTokensPerStep, 32)}}));
    extraInputsForDraftModelDecode.push_back(EngineInputDesc("attention_pos_id", nullptr,
        mEagleDeviceBuffer["treePositionIds"], {}, {2, {mBatchSize, mMaxDraftTokensPerStep}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("hidden_states_input", nullptr, mEagleDeviceBuffer["hiddenStatesDraftDecode"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, mTargetOutputHiddenDim}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("hidden_states_from_draft", nullptr, mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, mHiddenDim}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("last_token_ids", nullptr, last_token_ids_device, {}, {1, {mBatchSize * mTopK}}));

    mDraftModel->setupExtraInputs(extraInputsForDraftModelDecode);
}

template <typename T>
size_t Eagle<T>::getDeviceMemorySize() const noexcept
{
    return mBaseModel->getDeviceMemorySize() + mDraftModel->getDeviceMemorySize();
}

template <typename T>
void Eagle<T>::getLastHostLogits(std::vector<T>& hostLogits)
{
    size_t totalLogitSize = mBatchSize * mVocabSize;
    hostLogits.resize(totalLogitSize);
    GetLastLogitsOffsetParams params;
    params.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    params.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    params.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    params.lastLogitsOffset = static_cast<int64_t*>(mEagleDeviceBuffer["lastLogitsOffset"]);
    dispatchGetLastLogitsOffset(params, mEagleCommonParams);
    CUDA_CHECK(cudaMemcpyAsync(lastLogitsOffsetHost.data(), mEagleDeviceBuffer["lastLogitsOffset"],
        mBatchSize * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    for (int i = 0; i < mBatchSize; i++)
    {
        auto const offset = i * mMaxDecodingTokens * mVocabSize + lastLogitsOffsetHost[i] * mVocabSize;
        CUDA_CHECK(cudaMemcpy(hostLogits.data() + i * mVocabSize,
            static_cast<T*>(mBaseModel->getDeviceBuffer("logits")) + offset, mVocabSize * sizeof(T),
            cudaMemcpyDeviceToHost));
    }
    return;
}

template <typename T>
void Eagle<T>::generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
    std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds, bool isEagle3,
    std::shared_ptr<BenchmarkProfiler> const profiler, std::vector<int32_t>* newTokensNumbers,
    std::vector<int32_t>* iterNumbers)
{

    std::vector<int32_t> initContextLengths = contextLengths;
    std::vector<int64_t> lastTokenIds(mBatchSize);
    for (int i = 0; i < mBatchSize; i++)
    {
        lastTokenIds[i] = contextLengths[i] - 1;
    }
    auto inputIdsDevice = mBaseModel->getDeviceBuffer("input_ids");
    CUDA_CHECK(cudaMemcpyAsync(inputIdsDevice, inputIds.data(), mBatchSize * contextLengths[0] * sizeof(int64_t),
        cudaMemcpyHostToDevice, mStream));
    setupExtraInputsForDraftModel(contextLengths);

    mBaseModel->generateForContext(
        inputIdsDevice, contextLengths, lastTokenIds, {2, {mBatchSize, contextLengths[0] /*mMaxInputLength*/}});
    if (profiler)
    {
        profiler->recordHostEnd("first token latency");
        profiler->recordDeviceStart("generation");
    }
    invokeSamplingAndAccept(nullptr, 1, endIds);
    std::vector<int32_t> contextLengthForDraft = contextLengths;
    auto iterNum = 1;
    int32_t generationIter = 0;
    int64_t unfinishedBatchNum = mBaseModel->getModelConfig().batchSize;
    updateGenerationStatus(generationIter, unfinishedBatchNum, contextLengths);
    std::for_each(contextLengths.begin(), contextLengths.end(), [this](int32_t& val) { val -= 1; });

    if (generationIter < generationConfig.maxLength && unfinishedBatchNum != 0)
    {
        // only for bs=1
        void* inputIdsDeviceForDraft
            = static_cast<void*>(static_cast<int64_t*>(mBaseModel->getDeviceBuffer("input_ids")) + 1);
        mDraftModel->generateForContext(
            inputIdsDeviceForDraft, contextLengthForDraft, lastTokenIds, {2, {mBatchSize, contextLengths[0]}});
        draftModelDecodeInfer(contextLengthForDraft);
        std::vector<int64_t> lastTokenIdsForVerification(mMaxDecodingTokens);
        std::iota(lastTokenIdsForVerification.begin(), lastTokenIdsForVerification.end(), 0);

        // tree verification
        std::vector<int32_t> tempContextLengths = contextLengths;
        for (size_t i = 0; i < tempContextLengths.size(); ++i)
        {
            tempContextLengths[i] += 1;
            contextLengths[i] += mMaxDecodingTokens;
        }
        mBaseModel->generateForDecode(contextLengths, lastTokenIdsForVerification);
        // reset the context_lengths
        CUDA_CHECK(cudaMemcpyAsync(mBaseModel->getDeviceBuffer("context_lengths"), tempContextLengths.data(),
            mBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));

        invokeSamplingAndAccept(static_cast<int64_t*>(mEagleDeviceBuffer["draftIds"]), mMaxDecodingTokens, endIds);
        invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds();
        std::for_each(
            contextLengths.begin(), contextLengths.end(), [this](int32_t& val) { val -= mMaxDecodingTokens; });
        contextLengthForDraft = contextLengths;

        updateGenerationStatus(generationIter, unfinishedBatchNum, contextLengths);
        // decode
        while (generationIter < generationConfig.maxLength && unfinishedBatchNum != 0)
        {
            iterNum++;
            std::vector<int64_t> lastTokenIdsForEagle0;
            for (int i = 0; i < mBatchSize; i++)
            {
                lastTokenIdsForEagle0.push_back(acceptedLengthsHost[i] - 1);
            }

            std::for_each(contextLengthForDraft.begin(), contextLengthForDraft.end(),
                [this](int32_t& val) { val += mMaxDraftTokensPerStep; });
            // eagle0:
            CUDA_CHECK(cudaMemcpyAsync(mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"],
                mEagleDeviceBuffer["packedAttentionMaskCausal"],
                mBatchSize * mMaxDecodingTokens * divUp(mMaxDecodingTokens, 32) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice, mStream));

            CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"], 0,
                mBatchSize * mMaxDraftTokensPerStep * mHiddenDim * sizeof(T), mStream));
            mDraftModel->generateForDecode(contextLengthForDraft, lastTokenIdsForEagle0);

            for (int i = 0; i < mBatchSize; i++)
            {
                contextLengthForDraft[i] += acceptedLengthsHost[i] - mMaxDraftTokensPerStep;
            }

            draftModelDecodeInfer(contextLengthForDraft);
            std::vector<int32_t> tempContextLengths = contextLengths;
            // contextLengths[i] contains the new sample token, so we need to minus 1
            std::for_each(contextLengths.begin(), contextLengths.end(), [this](int32_t& val) { val -= 1; });

            std::for_each(
                contextLengths.begin(), contextLengths.end(), [this](int32_t& val) { val += mMaxDecodingTokens; });

            mBaseModel->generateForDecode(contextLengths, lastTokenIdsForVerification);
            // reset the context_lengths
            CUDA_CHECK(cudaMemcpyAsync(mBaseModel->getDeviceBuffer("context_lengths"), tempContextLengths.data(),
                mBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));

            invokeSamplingAndAccept(static_cast<int64_t*>(mEagleDeviceBuffer["draftIds"]), mMaxDecodingTokens, endIds);
            invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds();
            std::for_each(
                contextLengths.begin(), contextLengths.end(), [this](int32_t& val) { val -= mMaxDecodingTokens; });
            contextLengthForDraft = contextLengths;
            updateGenerationStatus(generationIter, unfinishedBatchNum, contextLengths);
        }
    }

    if (profiler)
    {
        profiler->recordDeviceEnd("generation");
    }
    // copy output_ids to host
    for (int i = 0; i < mBatchSize; i++)
    {
        auto const newLen = contextLengths[i] - initContextLengths[i];
        outputIds[i].resize(newLen);
        auto const acception_rate = (float) newLen / iterNum;
        LOG_DEBUG("Acception_rate: %f newLen: %d iterNum: %d\n", acception_rate, newLen, iterNum);
        if (newTokensNumbers)
        {
            newTokensNumbers->push_back(newLen);
        }
        if (iterNumbers)
        {
            iterNumbers->push_back(iterNum);
        }
        CUDA_CHECK(cudaMemcpyAsync(outputIds[i].data(),
            static_cast<int64_t*>(mBaseModel->getDeviceBuffer("input_ids")) + initContextLengths[i],
            newLen * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    return;
}
template <typename T>
void Eagle<T>::invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(int32_t layerIdx, T* hs_draft)
{

    UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<T> params;
    params.outputIdsAllDraft = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsAllDraft"]);
    params.selectedOutputIdsDraft = static_cast<int64_t*>(mEagleDeviceBuffer["selectedOutputIdsDraft"]);
    params.inputHiddenStatesDraft = hs_draft;
    params.outputHiddenStatesDraft = static_cast<T*>(mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"]);
    params.treeMaskInit = static_cast<bool*>(mEagleDeviceBuffer["treeMaskInit"]);
    params.treeMaskInput = static_cast<bool*>(mEagleDeviceBuffer["treeMaskInput"]);
    params.treeMaskUpdate = static_cast<bool*>(mEagleDeviceBuffer["treeMaskUpdate"]);
    params.treeMaskUpdateforAttention = static_cast<bool*>(mEagleDeviceBuffer["treeMaskUpdateforAttention"]);
    params.packedTreeMaskUpdateforAttention
        = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskUpdateforAttention"]);
    params.packedTreeMaskUpdateforAttentionNoPadding
        = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"]);
    params.treeIndices = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.treePositionIds = static_cast<int32_t*>(mEagleDeviceBuffer["treePositionIds"]);
    params.intermediateScores = static_cast<float*>(mEagleDeviceBuffer["intermediateScores"]);
    params.cumScoresForThirdTopk = static_cast<float*>(mEagleDeviceBuffer["cumScores"]);
    params.outputIdsForThirdTopk = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.allTokens = static_cast<int64_t*>(mEagleDeviceBuffer["allTokens"]);
    if (mIsEagle3)
    {
        params.draftVoc = static_cast<int64_t*>(mEagleDeviceBuffer["draftVoc"]);
    }
    else
    {
        params.draftVoc = nullptr;
    }
    params.layerIdx = layerIdx;
    params.maxLength = mMaxDraftTokensPerStep;
    params.curContextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores<T>(params, mEagleCommonParams);
}

template <typename T>
void Eagle<T>::invokeUpdateCumScoresAndParentsIds(int32_t layerIdx)
{
    auto const bias1 = layerIdx > 1 ? mTopK : 0;
    auto const bias2 = std::max(0, layerIdx - 2);
    auto const bias = 1 + mTopK * mTopK * bias2 + bias1;

    UpdateCumScoresAndParentsIdsParams params;
    params.outputLogProbsAllDraft = static_cast<float*>(mEagleDeviceBuffer["outputLogProbsAllDraft"]);
    params.intermediateScores = static_cast<float*>(mEagleDeviceBuffer["intermediateScores"]);
    // cu_scores = topk_p + params.intermediateScores
    params.cumScores = static_cast<float*>(mEagleDeviceBuffer["cumScores"]);
    params.outputIdsCurrentDraft = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.parantsIds = static_cast<int64_t*>(mEagleDeviceBuffer["parantsIds"]);
    params.bias = bias;
    params.layerIdx = layerIdx;
    dispatchUpdateCumScoresAndParentsIds(params, mEagleCommonParams);
}

template <typename T>
void Eagle<T>::draftDecodePostProcess(int32_t layerIdx)
{
    auto logits_last_token_draft = mDraftModel->getDeviceBuffer("logits");
    TopKSamplingKernelParams<T> params;
    params.logProbs = reinterpret_cast<T*>(logits_last_token_draft);
    if (layerIdx == 0)
    {
        params.batchSize = 1 * mBatchSize;
        params.maxBatchSize = 1 * mBatchSize;
        params.maxSeqLen = 1;
        params.workspace = mEagleDeviceBuffer["topk1Workspace"];
    }
    else
    {
        params.batchSize = mTopK * mBatchSize;
        params.maxBatchSize = mTopK * mBatchSize;
        params.maxSeqLen = mTopK;
        params.workspace = mEagleDeviceBuffer["topk2Workspace"];
    }

    params.maxTokensPerStep = 1;
    params.maxTopK = mTopK;

    params.vocabSizePadded = mDraftVocabSize;
    params.returnAllTopK = true;
    params.outputIds = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsAllDraft"]);
    params.outputLogProbs = static_cast<float*>(mEagleDeviceBuffer["outputLogProbsAllDraft"]);
    params.logitsHasProbs = true;
    // fist sample
    invokeBatchTopKSampling(params, mStream);
    auto const hiddenStatesDraft = mDraftModel->getDeviceBuffer("hidden_states");

    if (layerIdx == 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(mEagleDeviceBuffer["allScores"], mEagleDeviceBuffer["outputLogProbsAllDraft"],
            mTopK * sizeof(float), cudaMemcpyDeviceToDevice, mStream));
        invokeUpdateCumScoresAndParentsIds(layerIdx);
        CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["parantsIds"], 0, sizeof(int64_t), mStream)); // first = 0,init 0
    }
    else
    {

        invokeUpdateCumScoresAndParentsIds(layerIdx);
        auto const scoresDevicePtr
            = static_cast<float*>(mEagleDeviceBuffer["allScores"]) + (layerIdx - 1) * mTopK * mTopK + mTopK;
        CUDA_CHECK(cudaMemcpyAsync(scoresDevicePtr, mEagleDeviceBuffer["cumScores"], mTopK * mTopK * sizeof(float),
            cudaMemcpyDeviceToDevice, mStream));
        TopKSamplingKernelParams<float> paramsThirdTopk;
        paramsThirdTopk.logProbs = static_cast<float*>(mEagleDeviceBuffer["cumScores"]);
        paramsThirdTopk.batchSize = 1;
        paramsThirdTopk.maxBatchSize = 1;
        paramsThirdTopk.maxSeqLen = 1;
        paramsThirdTopk.maxTopK = mTopK;
        paramsThirdTopk.maxTokensPerStep = 1;
        paramsThirdTopk.vocabSizePadded = mTopK * mTopK;
        paramsThirdTopk.returnAllTopK = true;
        paramsThirdTopk.outputIds = static_cast<int64_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
        paramsThirdTopk.workspace = mEagleDeviceBuffer["topk3Workspace"];
        // 10 out of 100
        invokeBatchTopKSampling<float>(paramsThirdTopk, mStream);
    }
    // prepare for next layer
    invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(layerIdx, static_cast<T*>(hiddenStatesDraft));
}

template <typename T>
void Eagle<T>::invokeAssembleDraftIdsAndPathAndMaskAndPositionIds()
{
    TopKSamplingKernelParams<float> params;
    params.logProbs = static_cast<float*>(mEagleDeviceBuffer["allScores"]);
    params.batchSize = mBatchSize;
    params.maxBatchSize = mBatchSize;
    params.maxSeqLen = 1;
    params.maxTopK = mMaxDraftTokens;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mMaxPathLen * mTopK * mTopK;
    params.returnAllTopK = true;
    params.outputIds = static_cast<int64_t*>(mEagleDeviceBuffer["fourthTopKIds"]);
    params.workspace = mEagleDeviceBuffer["topk4Workspace"];
    invokeBatchTopKSampling<float>(params, mStream);

    AssembleDraftIdsAndPathAndMaskAndPositionIdsParams assembleParams;
    assembleParams.fourthTopKIds = static_cast<int64_t*>(mEagleDeviceBuffer["fourthTopKIds"]);
    assembleParams.allDraftIds = static_cast<int64_t*>(mEagleDeviceBuffer["allTokens"]);
    assembleParams.allDraftIdsAncestors = static_cast<int64_t*>(mEagleDeviceBuffer["parantsIds"]);
    assembleParams.modelInputIds = static_cast<int64_t*>(mBaseModel->getDeviceBuffer("input_ids"));
    assembleParams.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));

    assembleParams.treeMask = static_cast<bool*>(mEagleDeviceBuffer["treeMaskVerification"]);
    assembleParams.positionIds = static_cast<int32_t*>(mEagleDeviceBuffer["positionIdsVerification"]);
    assembleParams.draftIds = static_cast<int64_t*>(mEagleDeviceBuffer["draftIds"]);
    assembleParams.draftIdsAncestors = static_cast<int64_t*>(mEagleDeviceBuffer["draftIdsAncestors"]);
    assembleParams.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    assembleParams.validPathNum = static_cast<int32_t*>(mEagleDeviceBuffer["validPathNum"]);
    assembleParams.packedTreeMaskVerification = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskVerification"]);
    dispatchAssembleDraftIdsAndPathAndMaskAndPositionIds(assembleParams, mEagleCommonParams);
}

template <typename T>
void Eagle<T>::draftModelDecodeInfer(std::vector<int32_t> tempContextLengthForDraft)
{

    int layerIdx = 0;
    draftDecodePostProcess(0);
    std::vector<int64_t> lastTokenIds(mTopK);
    CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["hiddenStatesDraftDecode"], 0,
        mBatchSize * mMaxDraftTokensPerStep * mTargetOutputHiddenDim * sizeof(T), mStream));
    std::for_each(tempContextLengthForDraft.begin(), tempContextLengthForDraft.end(),
        [this](int& val) { val += mMaxDraftTokensPerStep; });
    for (size_t treeIdx = 0; treeIdx < mMaxPathLen - 1; treeIdx++)
    {
        std::iota(lastTokenIds.begin(), lastTokenIds.end(), layerIdx * mTopK);
        ++layerIdx;
        mDraftModel->generateForDecode(tempContextLengthForDraft, lastTokenIds);
        draftDecodePostProcess(layerIdx);
    }

    invokeAssembleDraftIdsAndPathAndMaskAndPositionIds();
}

template <typename T>
void Eagle<T>::updateGenerationStatus(
    int32_t& generationIter, int64_t& unfinishedBatchNum, std::vector<int32_t>& contextLengths)
{

    CUDA_CHECK(cudaMemcpyAsync(contextLengths.data(), mBaseModel->getDeviceBuffer("context_lengths"),
        mBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mEagleHostBuffer["finishedFinal"], mEagleDeviceBuffer["finishedFinal"],
        mBatchSize * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["finishedFinal"], 0, mBatchSize * sizeof(int64_t), mStream));
    CUDA_CHECK(cudaMemcpyAsync(acceptedLengthsHost.data(), mEagleDeviceBuffer["acceptedLengths"],
        mBatchSize * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    auto finishedStates = reinterpret_cast<int32_t*>(mEagleHostBuffer["finishedFinal"]);
    generationIter = *std::max_element(contextLengths.begin(), contextLengths.end());
    for (size_t bi = 0; bi < mBatchSize; bi++)
    {
        if (finishedStates[bi])
        {
            unfinishedBatchNum--;
        }
    }
}

template class Eagle<half>;
