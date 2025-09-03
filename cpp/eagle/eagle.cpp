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

#include "eagle.h"
#include "sampler/sampling.h"
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

namespace drivellm
{
namespace rt
{

void Eagle::eagleCommonParamsInit()
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

void Eagle::invokeSamplingAndAccept(int32_t* draftIds, int32_t const curTokensPerStep, int32_t endIds)
{
    auto const logits_last_token = mBaseModel->getDeviceBuffer("logits");

    int32_t batchSize;
    if (draftIds == nullptr)
    {
        batchSize = 1 * mBatchSize;
    }
    else
    {
        batchSize = mMaxDecodingTokens * mBatchSize;
    }

    // Create sampling parameters for greedy sampling (top_k=1)
    drivellm::SamplingParams samplingParams(batchSize, mVocabSize, 1.0f, 1);
    auto allocatedWorkspaceSize = drivellm::getTopKtopPSamplingWorkspaceSize(batchSize, mVocabSize, samplingParams);

    drivellm::topKtopPSamplingFromLogits(reinterpret_cast<LogitsType const*>(logits_last_token),
        static_cast<int32_t*>(mEagleDeviceBuffer["targetIds"]), samplingParams,
        mEagleDeviceBuffer["workspaceForVerification"], allocatedWorkspaceSize, mStream);

    drivellm::kernel::AcceptDraftTokensByIdsWithPathsParams accparms;
    accparms.outputIds = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("input_ids"));
    accparms.inputIdsDraftDecode = static_cast<int32_t*>(mEagleDeviceBuffer["selectedOutputIdsDraft"]);
    accparms.draftIds = draftIds;
    accparms.targetIds = static_cast<int32_t*>(mEagleDeviceBuffer["targetIds"]);
    accparms.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    accparms.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    accparms.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    accparms.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    accparms.curTokensPerStep = curTokensPerStep;
    accparms.endIds = endIds;
    accparms.finishedFinal = static_cast<int64_t*>(mEagleDeviceBuffer["finishedFinal"]);
    dispatchAcceptDraftTokensByIdsWithPaths(accparms, mEagleCommonParams);
}

void Eagle::invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds()
{
    void* KVCache = mBaseModel->getDeviceBuffer("kv_cache");
    void* hiddenStates = mBaseModel->getDeviceBuffer("hidden_states");
    static_assert(std::is_same_v<KVCacheType, HiddenStatesType>, "KVCacheType and HiddenStatesType must be the same");

    drivellm::kernel::UpdateKVCacheParams<KVCacheType> params;
    params.KVCache = static_cast<KVCacheType*>(KVCache);
    params.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    params.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    params.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    params.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    params.hiddenStates = static_cast<HiddenStatesType*>(hiddenStates);
    params.hiddenStatesInputs = static_cast<HiddenStatesType*>(mEagleDeviceBuffer["hiddenStatesDraftDecode"]);
    params.treePositionIds = static_cast<int32_t*>(mEagleDeviceBuffer["treePositionIds"]);
    drivellm::kernel::dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds<HiddenStatesType>(
        params, mEagleCommonParams);
}

void Eagle::invokeInitializeAttentionMaskCausal()
{
    drivellm::kernel::InitCausalAttentionMaskParams params;
    params.mask = static_cast<bool*>(mEagleDeviceBuffer["attentionMaskCausal"]);
    params.packedMask = static_cast<int32_t*>(mEagleDeviceBuffer["packedAttentionMaskCausal"]);
    drivellm::kernel::dispatchInitializeAttentionMaskCausal(params, mEagleCommonParams);
}

void Eagle::initDraftVoc()
{
    std::vector<int64_t> draftVocHost(mDraftVocabSize);
    std::string draftVocPath = mDraftModelDir + "/d2t.bin";
    std::ifstream fin(draftVocPath, std::ios::binary);
    if (!fin)
    {
        LOG_ERROR("Failed to open d2t.bin in path %s, it must be provided for Eagle3.", draftVocPath.c_str());
        throw std::runtime_error("d2t.bin is required for Eagle3 but not found in: " + draftVocPath);
    }
    fin.seekg(0, std::ios::end);
    std::streamsize fileSize = fin.tellg();
    if (fileSize == 0)
    {
        LOG_ERROR("d2t.bin file is empty in path %s", draftVocPath.c_str());
        throw std::runtime_error("d2t.bin file is empty in: " + draftVocPath);
    }
    if (static_cast<size_t>(fileSize) != mDraftVocabSize * sizeof(int32_t))
    {
        LOG_ERROR("d2t.bin file size mismatch. Got: %ld, Expected: %ld", (long) fileSize,
            (long) (mDraftVocabSize * sizeof(int32_t)));
        throw std::runtime_error("d2t.bin file size mismatch");
    }
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(draftVocHost.data()), mDraftVocabSize * sizeof(int32_t));
    if (!fin)
    {
        LOG_ERROR("Failed to read d2t.bin in path %s, it must be provided for Eagle3.", draftVocPath.c_str());
        throw std::runtime_error("Failed to read d2t.bin from: " + draftVocPath);
    }
    CUDA_CHECK(cudaMemcpyAsync(mEagleDeviceBuffer["draftVoc"], draftVocHost.data(), mDraftVocabSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream));
}

void Eagle::allocateEagleBuffer()
{

    //[bs,maxDraftTokens,mMaxDraftTokens]
    void* allDraftIdsDevice;
    CUDA_CHECK(cudaMalloc(&allDraftIdsDevice, mBatchSize * mMaxDraftTokens * mMaxDraftTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["allDraftIds"] = allDraftIdsDevice;

    // target model inputids:[maxDraftTokens,maxDraftTokens]
    void* decodingInputIdsDevice;
    CUDA_CHECK(cudaMalloc(&decodingInputIdsDevice, mBatchSize * mMaxDecodingTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["draftIds"] = decodingInputIdsDevice;

    void* targetIdsDevice;
    CUDA_CHECK(cudaMalloc(&targetIdsDevice, mBatchSize * mMaxDecodingTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["targetIds"] = targetIdsDevice;

    void* draftIdsAncestorsDevice;
    CUDA_CHECK(cudaMalloc(&draftIdsAncestorsDevice, mBatchSize * mMaxDraftTokens * sizeof(int64_t)));
    mEagleDeviceBuffer["draftIdsAncestors"] = draftIdsAncestorsDevice;

    // draft model input ids:[mBatchSize,mMaxPathLen,mTopK]
    void* selectedOutputIdsDraftDevice;
    CUDA_CHECK(cudaMalloc(&selectedOutputIdsDraftDevice, mBatchSize * mMaxDraftTokensPerStep * sizeof(int32_t)));
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
    CUDA_CHECK(cudaMalloc(&hiddenStatesDraftDecodeDevice,
        mBatchSize * mMaxDraftTokensPerStep * mTargetOutputHiddenDim * sizeof(HiddenStatesType)));
    mEagleDeviceBuffer["hiddenStatesDraftDecode"] = hiddenStatesDraftDecodeDevice;

    // draft model input:[mBatchSize,mMaxPathLen,mTargetOutputHiddenDim]
    void* hiddenStatesDraftDecodeFromDraftDevice;
    CUDA_CHECK(cudaMalloc(&hiddenStatesDraftDecodeFromDraftDevice,
        mBatchSize * mMaxDraftTokensPerStep * mHiddenDim * sizeof(HiddenStatesType)));
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
    size_t const totalElements = mBatchSize * mMaxPathLen * mTopK * mTopK;
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
    CUDA_CHECK(cudaMalloc(&allTokensDevice, mBatchSize * mMaxPathLen * mTopK * mTopK * sizeof(int32_t)));
    mEagleDeviceBuffer["allTokens"] = allTokensDevice;

    //[mBatchSize,mMaxPathLen,topk] [6,10]
    void* parentsIdsDevice;
    CUDA_CHECK(cudaMalloc(&parentsIdsDevice, mBatchSize * mMaxPathLen * mTopK * sizeof(int64_t)));
    mEagleDeviceBuffer["parentsIds"] = parentsIdsDevice;

    //[mBatchSize,topk,topk] for eagle draft every step for second topk in whole process: 100
    void* outputIdsAllDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputIdsAllDraftDevice, mBatchSize * mTopK * mTopK * sizeof(int32_t)));
    mEagleDeviceBuffer["outputIdsAllDraft"] = outputIdsAllDraftDevice;

    //[mBatchSize,topk,topk] - float version for eagle utility kernels
    void* outputLogProbsAllDraftFloatDevice;
    CUDA_CHECK(cudaMalloc(&outputLogProbsAllDraftFloatDevice, mBatchSize * mTopK * mTopK * sizeof(float)));
    mEagleDeviceBuffer["outputLogProbsAllDraftFloat"] = outputLogProbsAllDraftFloatDevice;

    //[mBatchSize,topk] for third topk in whole process: 10 out of 100
    void* outputIdsCurrentDraftDevice;
    CUDA_CHECK(cudaMalloc(&outputIdsCurrentDraftDevice, mBatchSize * mTopK * sizeof(int32_t)));
    mEagleDeviceBuffer["outputIdsCurrentDraft"] = outputIdsCurrentDraftDevice;

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
    CUDA_CHECK(cudaMemset(treePositionIdsDevice, 0, mBatchSize * mMaxDraftTokensPerStep * sizeof(int32_t)));
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
    CUDA_CHECK(cudaMemset(positionIdsVerificationDevice, 0, mBatchSize * mMaxDecodingTokens * sizeof(int32_t)));
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
    CUDA_CHECK(cudaMalloc(&fourthTopKIdsDevice, mBatchSize * mMaxDraftTokens * sizeof(int32_t)));
    mEagleDeviceBuffer["fourthTopKIds"] = fourthTopKIdsDevice;

    //[mBatchSize,mMaxDraftTokens]
    void* fourthTopkProbsDevice;
    CUDA_CHECK(cudaMalloc(&fourthTopkProbsDevice, mBatchSize * mMaxDraftTokens * sizeof(float)));
    mEagleDeviceBuffer["fourthTopkProbs"] = fourthTopkProbsDevice;

    // for topk1: torch.Size([1, 152064])
    auto const workspaceSize1 = drivellm::getSelectAllTopKWorkspaceSize(mBatchSize, mVocabSize, mTopK);
    void* topk1WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk1WorkspaceDevice, workspaceSize1));
    mEagleDeviceBuffer["topk1Workspace"] = topk1WorkspaceDevice;

    // for topk2
    auto const workspaceSize2 = drivellm::getSelectAllTopKWorkspaceSize(mTopK * mBatchSize, mVocabSize, mTopK);
    void* topk2WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk2WorkspaceDevice, workspaceSize2));
    mEagleDeviceBuffer["topk2Workspace"] = topk2WorkspaceDevice;

    // for topk3: [bs*topk,topk] - now using float type since cumScores is float
    auto const workspaceSize3 = drivellm::getSelectAllTopKWorkspaceSize(mBatchSize, mTopK * mTopK, mTopK);
    void* topk3WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk3WorkspaceDevice, workspaceSize3));
    mEagleDeviceBuffer["topk3Workspace"] = topk3WorkspaceDevice;

    // for topk4:[bs*topk, mMaxPathLen*mTopK] - now using float type since allScores is float
    auto const workspaceSize4
        = drivellm::getSelectAllTopKWorkspaceSize(mBatchSize, mMaxPathLen * mTopK * mTopK, mMaxDraftTokens);
    void* topk4WorkspaceDevice;
    CUDA_CHECK(cudaMalloc(&topk4WorkspaceDevice, workspaceSize4));
    mEagleDeviceBuffer["topk4Workspace"] = topk4WorkspaceDevice;

    // for verification - use sampling workspace since top_k=1
    drivellm::SamplingParams verificationParams(mBatchSize * mMaxDecodingTokens, mVocabSize, 1.0f, 1);
    auto const workspaceSizeForVerification
        = drivellm::getTopKtopPSamplingWorkspaceSize(mBatchSize * mMaxDecodingTokens, mVocabSize, verificationParams);
    void* workspaceForVerificationDevice;
    CUDA_CHECK(cudaMalloc(&workspaceForVerificationDevice, workspaceSizeForVerification));
    mEagleDeviceBuffer["workspaceForVerification"] = workspaceForVerificationDevice;

    void* draftVocDevice;
    CUDA_CHECK(cudaMalloc(&draftVocDevice, mDraftVocabSize * sizeof(int32_t)));
    mEagleDeviceBuffer["draftVoc"] = draftVocDevice;
    if (mIsEagle3)
    {
        initDraftVoc();
    }

    void* lastLogitsOffsetDevice;
    CUDA_CHECK(cudaMalloc(&lastLogitsOffsetDevice, mBatchSize * sizeof(int64_t)));
    mEagleDeviceBuffer["lastLogitsOffset"] = lastLogitsOffsetDevice;
}

int64_t Eagle::getModelBatchSize() const noexcept
{
    return mBatchSize;
}

ModelConfig Eagle::getBaseModelConfig() const noexcept
{
    return mBaseModel->getModelConfig();
}

int64_t Eagle::getMinSupportedInputLength() const noexcept
{
    return mBaseModel->getMinSupportedInputLength();
}

int64_t Eagle::getMaxSupportedInputLength() const noexcept
{
    return mBaseModel->getMaxSupportedInputLength();
}

void Eagle::addNewBufferForModelIO()
{
    // add new buffer for base model and draft model IO
    mBaseModel->addNewBuffer("input_ids", {2, {mBatchSize, mMaxInputLength}}, sizeof(int32_t));

    mBaseModel->addNewBuffer(
        "hidden_states", {3, {mBatchSize, mMaxInputLength, mTargetOutputHiddenDim}}, sizeof(HiddenStatesType));
    mBaseModel->addNewBuffer("logits", {2, {mBatchSize * mMaxDecodingTokens, mVocabSize}}, sizeof(LogitsType));
    mDraftModel->addNewBuffer(
        "hidden_states", {3, {mBatchSize, mMaxInputLength, mHiddenDim}}, sizeof(HiddenStatesType));
    mDraftModel->addNewBuffer("logits", {2, {mBatchSize * mMaxDraftTokensPerStep, mVocabSize}}, sizeof(LogitsType));
}

void Eagle::setupExtraInputsForBaseModel()
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
            {3, {mBatchSize, mMaxDecodingTokens, static_cast<int64_t>(divUp(mMaxDecodingTokens, 32))}}));
    extraInputsForBaseModelDecode.push_back(EngineInputDesc("attention_pos_id", nullptr,
        mEagleDeviceBuffer["positionIdsVerification"], {}, {2, {mBatchSize, mMaxDecodingTokens}}));

    auto const last_token_ids_device = mBaseModel->getDeviceBuffer("last_token_ids");
    extraInputsForBaseModelDecode.push_back(
        EngineInputDesc("last_token_ids", nullptr, last_token_ids_device, {}, {1, {mBatchSize * mMaxDecodingTokens}}));
    mBaseModel->setupExtraInputs(extraInputsForBaseModelDecode);
}

void Eagle::setupExtraInputsForDraftModelContext(std::vector<int32_t> const& contextLengths)
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
}

void Eagle::setupExtraInputsForDraftModelDecode()
{
    // for draft model decode:
    std::vector<EngineInputDesc> extraInputsForDraftModelDecode;
    extraInputsForDraftModelDecode.push_back(EngineInputDesc("input_ids", nullptr,
        mEagleDeviceBuffer["selectedOutputIdsDraft"], {}, {2, {mBatchSize, mMaxDraftTokensPerStep}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("attention_mask", nullptr, mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, static_cast<int64_t>(divUp(mMaxDraftTokensPerStep, 32))}}));
    extraInputsForDraftModelDecode.push_back(EngineInputDesc("attention_pos_id", nullptr,
        mEagleDeviceBuffer["treePositionIds"], {}, {2, {mBatchSize, mMaxDraftTokensPerStep}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("hidden_states_input", nullptr, mEagleDeviceBuffer["hiddenStatesDraftDecode"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, mTargetOutputHiddenDim}}));
    extraInputsForDraftModelDecode.push_back(
        EngineInputDesc("hidden_states_from_draft", nullptr, mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"], {},
            {3, {mBatchSize, mMaxDraftTokensPerStep, mHiddenDim}}));
    extraInputsForDraftModelDecode.push_back(EngineInputDesc(
        "last_token_ids", nullptr, mEagleDeviceBuffer["last_token_ids"], {}, {1, {mBatchSize * mTopK}}));

    mDraftModel->setupExtraInputs(extraInputsForDraftModelDecode);
}
void Eagle::setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs)
{
    mBaseModel->setupExtraInputs(extraInputs);
    mDraftModel->setupExtraInputs(extraInputs);
}

void Eagle::setupRopeCosSin()
{
    mBaseModel->setupRopeCosSin();
    mDraftModel->setupRopeCosSin();
}

size_t Eagle::getDeviceMemorySize() const noexcept
{
    return mBaseModel->getDeviceMemorySize() + mDraftModel->getDeviceMemorySize();
}

void Eagle::getLastHostLogits(std::vector<LogitsType>& hostLogits)
{
    size_t totalLogitSize = mBatchSize * mVocabSize;
    hostLogits.resize(totalLogitSize);
    drivellm::kernel::GetLastLogitsOffsetParams params;
    params.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    params.bestPathIds = static_cast<int32_t*>(mEagleDeviceBuffer["bestPathIds"]);
    params.acceptedLengths = static_cast<int64_t*>(mEagleDeviceBuffer["acceptedLengths"]);
    params.lastLogitsOffset = static_cast<int64_t*>(mEagleDeviceBuffer["lastLogitsOffset"]);
    drivellm::kernel::dispatchGetLastLogitsOffset(params, mEagleCommonParams);
    CUDA_CHECK(cudaMemcpyAsync(lastLogitsOffsetHost.data(), mEagleDeviceBuffer["lastLogitsOffset"],
        mBatchSize * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    for (int i = 0; i < mBatchSize; i++)
    {
        auto const offset = i * mMaxDecodingTokens * mVocabSize + lastLogitsOffsetHost[i] * mVocabSize;
        CUDA_CHECK(cudaMemcpy(hostLogits.data() + i * mVocabSize,
            static_cast<LogitsType*>(mBaseModel->getDeviceBuffer("logits")) + offset, mVocabSize * sizeof(LogitsType),
            cudaMemcpyDeviceToHost));
    }
    return;
}

void Eagle::initDecodingPhaseCudaGraph()
{
    mBaseModel->initDecodingPhaseCudaGraph();
    mDraftModel->initDecodingPhaseCudaGraph();
}

void Eagle::generate(std::vector<int32_t> const& inputIds, std::vector<int32_t> contextLengths,
    std::vector<std::vector<int32_t>>& outputIds, GenerationConfig generationConfig, int32_t endIds,
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
    CUDA_CHECK(cudaMemcpyAsync(inputIdsDevice, inputIds.data(), mBatchSize * contextLengths[0] * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream));
    setupExtraInputsForDraftModelContext(contextLengths);

    // Initialize decoding phase cuda graph
    initDecodingPhaseCudaGraph();

    mBaseModel->generateForContext(inputIdsDevice, contextLengths, lastTokenIds, {2, {mBatchSize, contextLengths[0]}});
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
            = static_cast<void*>(static_cast<int32_t*>(mBaseModel->getDeviceBuffer("input_ids")) + 1);
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

        invokeSamplingAndAccept(static_cast<int32_t*>(mEagleDeviceBuffer["draftIds"]), mMaxDecodingTokens, endIds);
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
                mBatchSize * mMaxDraftTokensPerStep * mHiddenDim * sizeof(HiddenStatesType), mStream));
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

            invokeSamplingAndAccept(static_cast<int32_t*>(mEagleDeviceBuffer["draftIds"]), mMaxDecodingTokens, endIds);
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
        // Copy output directly as int32_t
        CUDA_CHECK(cudaMemcpyAsync(outputIds[i].data(),
            static_cast<int32_t*>(mBaseModel->getDeviceBuffer("input_ids")) + initContextLengths[i],
            newLen * sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    return;
}

void Eagle::invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(int32_t layerIdx, HiddenStatesType* hs_draft)
{
    drivellm::kernel::UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<HiddenStatesType>
        params;
    params.outputIdsAllDraft = static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsAllDraft"]);
    params.selectedOutputIdsDraft = static_cast<int32_t*>(mEagleDeviceBuffer["selectedOutputIdsDraft"]);
    params.inputHiddenStatesDraft = hs_draft;
    params.outputHiddenStatesDraft
        = static_cast<HiddenStatesType*>(mEagleDeviceBuffer["hiddenStatesDraftDecodeFromDraft"]);
    params.treeMaskInit = static_cast<bool*>(mEagleDeviceBuffer["treeMaskInit"]);
    params.treeMaskInput = static_cast<bool*>(mEagleDeviceBuffer["treeMaskInput"]);
    params.treeMaskUpdate = static_cast<bool*>(mEagleDeviceBuffer["treeMaskUpdate"]);
    params.treeMaskUpdateforAttention = static_cast<bool*>(mEagleDeviceBuffer["treeMaskUpdateforAttention"]);
    params.packedTreeMaskUpdateforAttention
        = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskUpdateforAttention"]);
    params.packedTreeMaskUpdateforAttentionNoPadding
        = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskUpdateforAttentionNoPadding"]);
    params.treeIndices = static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.treePositionIds = static_cast<int32_t*>(mEagleDeviceBuffer["treePositionIds"]);
    params.intermediateScores = static_cast<float*>(mEagleDeviceBuffer["intermediateScores"]);
    params.cumScoresForThirdTopk = static_cast<float*>(mEagleDeviceBuffer["cumScores"]);
    params.outputIdsForThirdTopk = static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.allTokens = static_cast<int32_t*>(mEagleDeviceBuffer["allTokens"]);
    if (mIsEagle3)
    {
        params.draftVoc = static_cast<int32_t*>(mEagleDeviceBuffer["draftVoc"]);
    }
    else
    {
        params.draftVoc = nullptr;
    }
    params.layerIdx = layerIdx;
    params.maxLength = mMaxDraftTokensPerStep;
    params.curContextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));
    drivellm::kernel::dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores<
        HiddenStatesType>(params, mEagleCommonParams);
}

void Eagle::invokeUpdateCumScoresAndParentsIds(int32_t layerIdx)
{
    auto const bias1 = layerIdx > 1 ? mTopK : 0;
    auto const bias2 = std::max(0, layerIdx - 2);
    auto const bias = 1 + mTopK * mTopK * bias2 + bias1;

    drivellm::kernel::UpdateCumScoresAndParentsIdsParams params;
    params.outputLogProbsAllDraft = static_cast<float*>(mEagleDeviceBuffer["outputLogProbsAllDraftFloat"]);
    params.intermediateScores = static_cast<float*>(mEagleDeviceBuffer["intermediateScores"]);
    // cu_scores = topk_p + params.intermediateScores
    params.cumScores = static_cast<float*>(mEagleDeviceBuffer["cumScores"]);
    params.outputIdsCurrentDraft = static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]);
    params.parentsIds = static_cast<int64_t*>(mEagleDeviceBuffer["parentsIds"]);
    params.bias = bias;
    params.layerIdx = layerIdx;
    drivellm::kernel::dispatchUpdateCumScoresAndParentsIds(params, mEagleCommonParams);
}

void Eagle::draftDecodePostProcess(int32_t layerIdx)
{
    auto logits_last_token_draft = mDraftModel->getDeviceBuffer("logits");

    int32_t batchSize;
    void* workspace;
    size_t workspaceSize;

    if (layerIdx == 0)
    {
        batchSize = 1 * mBatchSize;
        workspace = mEagleDeviceBuffer["topk1Workspace"];
        workspaceSize = drivellm::getSelectAllTopKWorkspaceSize(batchSize, mDraftVocabSize, mTopK);
    }
    else
    {
        batchSize = mTopK * mBatchSize;
        workspace = mEagleDeviceBuffer["topk2Workspace"];
        workspaceSize = drivellm::getSelectAllTopKWorkspaceSize(batchSize, mDraftVocabSize, mTopK);
    }

    // Use selectAllTopK to get all top-K elements with log probabilities
    drivellm::selectAllTopKFromLogits(reinterpret_cast<LogitsType const*>(logits_last_token_draft),
        static_cast<float*>(mEagleDeviceBuffer["outputLogProbsAllDraftFloat"]), // top_k_values (float log probs)
        static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsAllDraft"]),         // top_k_indices
        batchSize, mDraftVocabSize, mTopK, workspace, workspaceSize, mStream,
        true,  // return_log_probs
        false, // normalize log probs
        true); // softmax is already computed

    auto const hiddenStatesDraft = mDraftModel->getDeviceBuffer("hidden_states");

    if (layerIdx == 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(mEagleDeviceBuffer["allScores"], mEagleDeviceBuffer["outputLogProbsAllDraftFloat"],
            mTopK * sizeof(float), cudaMemcpyDeviceToDevice, mStream));
        invokeUpdateCumScoresAndParentsIds(layerIdx);
        CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["parentsIds"], 0, sizeof(int64_t), mStream)); // first = 0,init 0
    }
    else
    {

        invokeUpdateCumScoresAndParentsIds(layerIdx);
        auto const scoresDevicePtr
            = static_cast<float*>(mEagleDeviceBuffer["allScores"]) + (layerIdx - 1) * mTopK * mTopK + mTopK;
        CUDA_CHECK(cudaMemcpyAsync(scoresDevicePtr, mEagleDeviceBuffer["cumScores"], mTopK * mTopK * sizeof(float),
            cudaMemcpyDeviceToDevice, mStream));
        // Use selectAllTopK with float since cumScores is float
        // 10 out of 100
        auto allocatedWorkspaceSize3 = drivellm::getSelectAllTopKWorkspaceSize(1, mTopK * mTopK, mTopK);
        drivellm::selectAllTopKFromLogits(static_cast<float const*>(mEagleDeviceBuffer["cumScores"]), // input as float
            nullptr, // don't need values output for this case
            static_cast<int32_t*>(mEagleDeviceBuffer["outputIdsCurrentDraft"]), // top_k_indices
            1,                                                                  // batch_size
            mTopK * mTopK,                                                      // vocab_size
            mTopK,                                                              // top_k
            mEagleDeviceBuffer["topk3Workspace"], allocatedWorkspaceSize3, mStream,
            false, // don't return log probs
            false, // don't normalize
            false  // compute softmax
        );
    }
    // prepare for next layer
    invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(
        layerIdx, static_cast<HiddenStatesType*>(hiddenStatesDraft));
}

void Eagle::invokeAssembleDraftIdsAndPathAndMaskAndPositionIds()
{
    // Use selectAllTopK with float since allScores is float
    auto allocatedWorkspaceSize4
        = drivellm::getSelectAllTopKWorkspaceSize(mBatchSize, mMaxPathLen * mTopK * mTopK, mMaxDraftTokens);
    drivellm::selectAllTopKFromLogits(static_cast<float const*>(mEagleDeviceBuffer["allScores"]), // input as float
        nullptr,                                                    // don't need values output for this case
        static_cast<int32_t*>(mEagleDeviceBuffer["fourthTopKIds"]), // top_k_indices
        mBatchSize,                                                 // batch_size
        mMaxPathLen * mTopK * mTopK,                                // vocab_size
        mMaxDraftTokens,                                            // top_k
        mEagleDeviceBuffer["topk4Workspace"], allocatedWorkspaceSize4, mStream,
        false, // don't return log probs
        false, // don't normalize
        false  // compute softmax
    );

    drivellm::kernel::AssembleDraftIdsAndPathAndMaskAndPositionIdsParams assembleParams;
    assembleParams.fourthTopKIds = static_cast<int32_t*>(mEagleDeviceBuffer["fourthTopKIds"]);
    assembleParams.allDraftIds = static_cast<int32_t*>(mEagleDeviceBuffer["allTokens"]);
    assembleParams.allDraftIdsAncestors = static_cast<int64_t*>(mEagleDeviceBuffer["parentsIds"]);
    assembleParams.modelInputIds = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("input_ids"));
    assembleParams.contextLengths = static_cast<int32_t*>(mBaseModel->getDeviceBuffer("context_lengths"));

    assembleParams.treeMask = static_cast<bool*>(mEagleDeviceBuffer["treeMaskVerification"]);
    assembleParams.positionIds = static_cast<int32_t*>(mEagleDeviceBuffer["positionIdsVerification"]);
    assembleParams.draftIds = static_cast<int32_t*>(mEagleDeviceBuffer["draftIds"]);
    assembleParams.draftIdsAncestors = static_cast<int64_t*>(mEagleDeviceBuffer["draftIdsAncestors"]);
    assembleParams.paths = static_cast<int32_t*>(mEagleDeviceBuffer["paths"]);
    assembleParams.validPathNum = static_cast<int32_t*>(mEagleDeviceBuffer["validPathNum"]);
    assembleParams.packedTreeMaskVerification = static_cast<int32_t*>(mEagleDeviceBuffer["packedTreeMaskVerification"]);
    drivellm::kernel::dispatchAssembleDraftIdsAndPathAndMaskAndPositionIds(assembleParams, mEagleCommonParams);
}

void Eagle::draftModelDecodeInfer(std::vector<int32_t> tempContextLengthForDraft)
{

    int layerIdx = 0;
    draftDecodePostProcess(0);
    std::vector<int64_t> lastTokenIds(mTopK);
    CUDA_CHECK(cudaMemsetAsync(mEagleDeviceBuffer["hiddenStatesDraftDecode"], 0,
        mBatchSize * mMaxDraftTokensPerStep * mTargetOutputHiddenDim * sizeof(HiddenStatesType), mStream));
    std::for_each(tempContextLengthForDraft.begin(), tempContextLengthForDraft.end(),
        [this](int& val) { val += mMaxDraftTokensPerStep; });
    for (int32_t treeIdx = 0; treeIdx < mMaxPathLen - 1; treeIdx++)
    {
        std::iota(lastTokenIds.begin(), lastTokenIds.end(), layerIdx * mTopK);
        ++layerIdx;
        mDraftModel->generateForDecode(tempContextLengthForDraft, lastTokenIds);
        draftDecodePostProcess(layerIdx);
    }

    invokeAssembleDraftIdsAndPathAndMaskAndPositionIds();
}

void Eagle::updateGenerationStatus(
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
    for (int64_t bi = 0; bi < mBatchSize; bi++)
    {
        if (finishedStates[bi])
        {
            unfinishedBatchNum--;
        }
    }
}

} // namespace rt
} // namespace drivellm
