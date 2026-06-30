/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/decoding/dflashDecoder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/speculative/dflashAcceptKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
int32_t dflashDraftProfileForRound(int32_t generationRound)
{
    return generationRound == 0 ? kPrefillProfile : kDecodeProfile;
}
} // namespace

DFlashDecoder::DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    SpecDecodeDraftingConfig const& /* draftingConfig */, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
{
    auto const& deployment = runtime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlashDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DFlashDecoder: draft config is required.");
    ELLM_CHECK(baseCfg.specDecodeType == SpecDecodeMode::kDFlash,
        "DFlashDecoder requires a base engine exported with spec_decode_type=dflash and engine_role=base.");
    ELLM_CHECK(baseCfg.reducedVocabSize == 0, "DFlash Phase 1 does not support reduced-vocabulary base engines.");

    mBlockSize = deployment.specConfig->verifySize;
    mMaskTokenId = baseCfg.dflashMaskTokenId > 0 ? baseCfg.dflashMaskTokenId : deployment.draft->dflashMaskTokenId;
    mDraftHiddenSize = deployment.specConfig->draftHiddenSize;
    mBaseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
    mDraftVocabSize = deployment.draft->outputVocabSize;

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();
    int32_t const maxSeqForDraft = baseCfg.maxKVCacheCapacity;

    // Load draft engine using the registry selected from the deployment's spec-decode mode.
    auto const draftEnginePath = engineDir / "spec_draft.engine";
    LOG_INFO("DFlashDecoder: loading draft engine from %s", draftEnginePath.string().c_str());
    mDraftExecutor = EngineExecutor::createForDraft(draftEnginePath, deployment);
    validateAgainstEngine(*deployment.draft, *mDraftExecutor, "dflash_draft");

    // Allocate draft engine I/O tensors
    mDraftInputsEmbeds = Tensor({maxBatch, mBlockSize, mDraftHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "DFlash::draftInputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, maxSeqForDraft, mBaseOutputHiddenDim}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlash::draftTargetHidden");
    mDraftOutputLogits = Tensor({maxBatch, mBlockSize, mDraftVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "DFlash::draftOutputLogits");

    // Proposal attention inputs
    int32_t const packedMaskLen = divUp(mBlockSize, 32);
    mDraftPackedAttentionMask = Tensor(
        {maxBatch, mBlockSize, packedMaskLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftPackedMask");
    mDraftAttentionPosId
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftAttentionPosId");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftContextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftDeltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftDeltaLens");

    // Set up the draft tensor map with new cached bindings
    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    mDraftTensorMap.set(binding_names::kContextLengths, mDraftContextLengths);
    mDraftTensorMap.set(binding_names::kDFlashDeltaLengths, mDraftDeltaLens);

    // KV cache bindings: bind to draft cache manager's combined KV cache (index 1)
    {
        auto& kvMgr = mDraftCacheManager.getKVCacheManager();
        LLMEngineConfig const& draftCfg = *deployment.draft;
        int32_t localAttnIdx = 0;
        for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(draftCfg.layerTypes.size()); ++absIdx)
        {
            if (draftCfg.layerTypes[absIdx] != HybridCacheManager::LayerType::kAttention)
            {
                continue;
            }
            auto& combinedKV = kvMgr.getCombinedKVCache(localAttnIdx);
            mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
            mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV);
            ++localAttnIdx;
        }
    }

    // kvcache_start_index: use draft cache lengths
    mDraftTensorMap.set(binding_names::kKVCacheStartIndex, mDraftCacheManager.getKVCacheLengths());

    // RoPE cos/sin: shared with base (from the shared RoPE pool)
    {
        LLMEngineConfig const& draftCfg = *deployment.draft;
        if (draftCfg.ropeConfig.type == RopeType::kMRope)
        {
            mDraftTensorMap.set(binding_names::kRopeCosSin, mRuntime.base.pipelineIO.mropeCosSin);
        }
        else
        {
            mDraftTensorMap.set(binding_names::kRopeCosSin,
                mRuntime.base.sharedResources.ropePool.getOrCreate(
                    draftCfg.ropeConfig, draftCfg.rotaryDim, baseCfg.maxKVCacheCapacity, nullptr));
        }
    }

    // Allocate accept/verify buffers
    mDraftTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::draftTokenIds");
    mVerifyTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::verifyTokenIds");
    mAcceptedTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::acceptedIds");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::acceptLength");
    mHostAcceptLengths = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostAcceptLengths");
    mHostAcceptedTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostAcceptedIds");
    mHostDraftInputIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostDraftInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostDeltaLens");

    // Pre-allocate argmax scratch for sequential accept
    mArgmaxScratch
        = Tensor({maxBatch * mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::argmaxScratch");

    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::lastAcceptedTokens");

    LOG_INFO(
        "DFlashDecoder initialized (cached KV path): blockSize=%d, maskTokenId=%d, maxBatch=%d, "
        "draftHiddenSize=%d, baseOutputHiddenDim=%d, draftVocabSize=%d",
        mBlockSize, mMaskTokenId, maxBatch, mDraftHiddenSize, mBaseOutputHiddenDim, mDraftVocabSize);
}

char const* DFlashDecoder::unsupportedReason(LLMGenerationRequest const& request) const noexcept
{
    return spec_decode_utils::isGreedyCompatible(request);
}

bool DFlashDecoder::decodeStep(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dflash_decode, "DFlashDecoder::decodeStep", nvtx_colors::GREEN);
    cudaGetLastError();

    if (!runDraftForward(context))
    {
        LOG_ERROR("DFlashDecoder: draft forward failed.");
        return false;
    }

    if (!runBaseVerification(context))
    {
        LOG_ERROR("DFlashDecoder: base verification failed.");
        return false;
    }

    return true;
}

bool DFlashDecoder::runDraftForward(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dflash_draft, "DFlashDecoder::runDraftForward", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("DFlashDecoder: draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockSize;

    // Step 1: Prepare draft input token IDs: [last_accepted_token, mask_id, ..., mask_id]
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    int32_t* hostLastAccepted = mHostLastAcceptedTokens.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostLastAccepted[b] = context.tokenIds[b].back();
        hostDraftInputIds[b * BS] = hostLastAccepted[b];
        for (int32_t j = 1; j < BS; ++j)
        {
            hostDraftInputIds[b * BS + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mLastAcceptedTokens.rawPointer(), mHostLastAcceptedTokens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Step 2: Embedding lookup -> draft inputs_embeds [B, BS, draftHiddenSize]
    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    // Step 3: Prepare target_hidden_delta from base hidden states.
    // On round 0: delta = prefill hidden states, per-batch lengths from effectivePrefillLengths.
    // On round > 0: delta = accepted base hidden states, per-batch from acceptLen.
    // The KV cache update plugin uses per-batch delta_lengths[b] to skip padded rows.
    int64_t maxDeltaLen;
    int64_t sourceSeqLen; // stride of baseHiddenStates along dim 1
    check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();

    if (context.generationRound == 0)
    {
        // Prompts may be padded to a batch max; update only each sequence's effective prefill length.
        sourceSeqLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
        maxDeltaLen = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int32_t const prefillLen = context.effectivePrefillLengths[b];
            hostDeltaLens[b] = prefillLen;
            maxDeltaLen = std::max(maxDeltaLen, static_cast<int64_t>(prefillLen));
        }
    }
    else
    {
        // After verify: baseHiddenStates is shaped [B, BS, dim]
        sourceSeqLen = BS;
        int32_t const* hostAccLens = mHostAcceptLengths.dataPointer<int32_t>();
        maxDeltaLen = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            hostDeltaLens[b] = hostAccLens[b];
            maxDeltaLen = std::max(maxDeltaLen, static_cast<int64_t>(hostAccLens[b]));
        }
    }

    // Upload per-batch delta_lengths for the KV cache update plugin guard
    check::check(mDraftDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), mHostDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Source: baseHiddenStates [B, sourceSeqLen, Hout] — stride = sourceSeqLen * Hout
    // Dest:   mDraftTargetHidden [B, maxDeltaLen, Hout] — stride = maxDeltaLen * Hout
    // Copy the first maxDeltaLen rows per batch. Rows beyond delta_lengths[b] are padding
    // (skipped by the KV cache update plugin).
    check::check(
        mDraftTargetHidden.reshape({activeBatchSize, maxDeltaLen, mBaseOutputHiddenDim}), "Tensor reshape failed");
    {
        size_t const elementBytes = utils::getTypeSize(mDraftTargetHidden.getDataType());
        size_t const rowBytes = static_cast<size_t>(mBaseOutputHiddenDim) * elementBytes;
        size_t const dstPitch = static_cast<size_t>(maxDeltaLen) * rowBytes;
        size_t const srcPitch = static_cast<size_t>(sourceSeqLen) * rowBytes;
        size_t const widthBytes = static_cast<size_t>(maxDeltaLen) * rowBytes;
        CUDA_CHECK(cudaMemcpy2DAsync(mDraftTargetHidden.rawPointer(), dstPitch,
            mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(), srcPitch, widthBytes, activeBatchSize,
            cudaMemcpyDeviceToDevice, context.stream));
    }

    // Step 4: Prepare proposal attention inputs.
    // KV length advancement is per sequence, so proposal positions and masks must consume delta_lengths[b].
    {
        int32_t const pmLen = divUp(BS, 32);
        check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, BS, pmLen}), "Tensor reshape failed");
        check::check(mDraftAttentionPosId.reshape({activeBatchSize, BS}), "Tensor reshape failed");
        check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

        Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
        kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
            mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
            mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), activeBatchSize,
            context.stream);
    }

    // Step 5: Execute draft engine
    check::check(mDraftOutputLogits.reshape({activeBatchSize, BS, mDraftVocabSize}), "Tensor reshape failed");

    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/BS,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/static_cast<int64_t>(maxDeltaLen),
        /*.attnMaskSeqLen=*/BS,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(BS, 32)),
        /*.startIndexLen=*/activeBatchSize,
    };

    cudaGetLastError();

    bool draftSuccess = mDraftExecutor->prepare(
        dflashDraftProfileForRound(context.generationRound), draftDims, mDraftTensorMap, context.stream);
    if (draftSuccess)
    {
        draftSuccess = mDraftExecutor->execute(context.stream);
    }
    if (!draftSuccess)
    {
        LOG_ERROR("DFlashDecoder: draft engine execution failed.");
        return false;
    }

    // Step 6: Commit per-batch delta lengths to the draft cache manager.
    // Round 0 uses effective prefill lengths; later rounds use accept lengths from verification.
    {
        if (context.generationRound == 0)
        {
            check::check(mDraftDeltaLenCommit.reshape({activeBatchSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLenCommit.rawPointer(), mHostDeltaLens.rawPointer(),
                activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
            mDraftCacheManager.commitSequenceLength(mDraftDeltaLenCommit, context.stream);
        }
        else
        {
            // mAcceptLength holds per-batch accept lengths from previous runBaseVerification.
            // After batch eviction, activeBatchSize may be smaller than mAcceptLength's shape.
            // Reshape to current activeBatchSize (evicted entries have been compacted).
            check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
            mDraftCacheManager.commitSequenceLength(mAcceptLength, context.stream);
        }
    }

    // Step 7: Argmax draft logits -> draft_tokens [B*BS, 1] -> [B, BS]
    check::check(mDraftOutputLogits.reshape({activeBatchSize * BS, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize * BS, 1}), "Tensor reshape failed");
    selectAllTopK(mDraftOutputLogits, std::nullopt, mDraftTokenIds, 1, mRuntime.sampling.workspace, context.stream);
    check::check(mDraftTokenIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");

    // Step 8: Build verify input on GPU: [last_accepted_token, draft_1, ..., draft_{BS-1}]
    check::check(mVerifyTokenIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    kernel::dflashBuildVerifyTokens(
        mLastAcceptedTokens, mDraftTokenIds, mVerifyTokenIds, activeBatchSize, BS, context.stream);

    return true;
}

bool DFlashDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dflash_verify, "DFlashDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockSize;

    // Step 1: Copy verify tokens to idsInput for base engine
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mVerifyTokenIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));

    // Step 2: Embedding lookup for base engine
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, BS, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

    // Step 3: Prepare base engine outputs
    int32_t const selectTokenSize = activeBatchSize * BS;
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, mBaseOutputHiddenDim}),
        "Tensor reshape failed");

    // Step 4: Prepare packed causal attention mask and position IDs for base verification.
    {
        int32_t const verifySize = BS;
        check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                         {activeBatchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifySize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifySize}),
            "Tensor reshape failed");

        Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
        kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifySize,
            mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
            mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
            mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
            mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), activeBatchSize, context.stream);
    }

    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(activeBatchSize, BS);

    // Step 5: Execute base engine verification
    cudaGetLastError();
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(activeBatchSize, BS);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.executor.execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("DFlashDecoder: base verification execution failed.");
        return false;
    }

    // Step 6: GPU-side sequential accept
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, BS, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");

    kernel::dflashSequentialAccept(mRuntime.base.pipelineIO.outputLogits, mVerifyTokenIds, mAcceptedTokenIds,
        mAcceptLength, mArgmaxScratch, activeBatchSize, BS, mRuntime.deployment.base.outputVocabSize, context.stream);

    // Step 7: KV cache commit (base)
    mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);

    // Reshape baseHiddenStates to [B, BS, dim]
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, BS, mBaseOutputHiddenDim}),
        "Tensor reshape failed");

    mRuntime.base.cacheManager.getMambaCacheManager().scatterMtpStates(mAcceptLength, context.stream);

    // Step 8: Append accepted tokens to context (includes D2H sync)
    spec_decode_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, BS, mRuntime.tokenizer, context.stream);

    return true;
}

bool DFlashDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};

    static constexpr int32_t kSimulateCacheLength{128};
    int32_t const BS = mBlockSize;
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;

    // ScopeGuard: reset cache state after capture
    struct ScopeGuard
    {
        std::function<void()> cleanup;
        ~ScopeGuard() noexcept
        {
            if (cleanup)
            {
                cleanup();
            }
        }
    } stateGuard{[&]() noexcept {
        std::vector<int32_t> zeroCacheLens(mRuntime.maxRuntimeBatchSize, 0);
        Tensor zeroCacheLensTensor(
            zeroCacheLens.data(), {mRuntime.maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        if (!mRuntime.base.executor.prepare(
                kDecodeProfile, mRuntime.deployment.base.resetDims(), mRuntime.base.tensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    }};

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        // Simulate a cache state with some tokens already committed
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        // --- Draft proposal CUDA graph capture ---
        // Capture for each possible deltaLen (1 to BS, matching possible accept lengths)
        {
            int32_t const pmLen = divUp(BS, 32);
            for (int32_t simDeltaLen = 1; simDeltaLen <= BS; ++simDeltaLen)
            {
                check::check(mDraftInputsEmbeds.reshape({batchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
                check::check(
                    mDraftTargetHidden.reshape({batchSize, static_cast<int64_t>(simDeltaLen), mBaseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(mDraftOutputLogits.reshape({batchSize, BS, mDraftVocabSize}), "Tensor reshape failed");
                check::check(mDraftPackedAttentionMask.reshape({batchSize, BS, pmLen}), "Tensor reshape failed");
                check::check(mDraftAttentionPosId.reshape({batchSize, BS}), "Tensor reshape failed");
                check::check(mDraftContextLengths.reshape({batchSize}), "Tensor reshape failed");

                // Upload simulated delta_lengths for graph capture
                std::vector<int32_t> simDeltaLens(batchSize, simDeltaLen);
                check::check(mDraftDeltaLens.reshape({batchSize}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), simDeltaLens.data(),
                    batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

                // Draft cache was reset to kSimulateCacheLength, prep kernel uses per-batch deltaLengths
                Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
                kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
                    mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
                    mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), batchSize,
                    stream);

                InferenceDims const draftDims{
                    /*.batch=*/batchSize,
                    /*.seqLen=*/BS,
                    /*.kvLen=*/draftKVCapacity,
                    /*.selectLen=*/static_cast<int64_t>(simDeltaLen),
                    /*.attnMaskSeqLen=*/BS,
                    /*.ropeBatch=*/1,
                    /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
                    /*.startIndexLen=*/batchSize,
                };

                if (mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, stream))
                {
                    draftProposalCaptureStatus &= mDraftExecutor->captureGraph(stream);
                }
                else
                {
                    LOG_WARNING("DFlash: failed to prepare draft for graph capture (batch=%d, delta=%d)", batchSize,
                        simDeltaLen);
                    draftProposalCaptureStatus = false;
                }
            }
        }

        // --- Base verification CUDA graph capture ---
        {
            int32_t const verifySize = BS;
            int32_t const selectTokenSize = batchSize * verifySize;
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                             {selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, mBaseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                             {batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}),
                "Tensor reshape failed");

            Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
            kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifySize,
                mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
                mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), batchSize, stream);

            if (mRuntime.preprocess.deepstack)
            {
                mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
            }

            mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(batchSize, verifySize);

            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifySize);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    LOG_INFO("DFlashDecoder: CUDA graph capture complete (draft=%s, baseVerify=%s)",
        draftProposalCaptureStatus ? "ok" : "FAILED", baseVerificationCaptureStatus ? "ok" : "FAILED");

    return draftProposalCaptureStatus && baseVerificationCaptureStatus;
}

int64_t DFlashDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void DFlashDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

bool DFlashDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptKVCacheDraft.find(key) != mSystemPromptKVCacheDraft.end();
}

void DFlashDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream)
{
    check::check(mSystemPromptKVCacheDraft.count(key) > 0, "DFlash system prompt cache missing for draft model");
    mDraftCacheManager.restoreKVCache(mSystemPromptKVCacheDraft[key].kvCacheLayers, batchIdx, stream);
}

bool DFlashDecoder::runSystemPromptPrefill(DecodingInferenceContext& context)
{
    // DFlash "draft prefill" updates target KV cache from base hidden states.
    // After base prefill, baseHiddenStates contains the multi-layer hidden features
    // for the full prompt. We run the draft engine to update these entries in the
    // draft KV cache. The proposal logits are ignored — only the target KV matters.
    //
    // This is equivalent to the first-round draft forward (generationRound==0),
    // but we don't do proposal/verify — just update the cache and commit.
    int32_t const activeBatchSize = context.activeBatchSize;
    int64_t const prefillLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
    int32_t const BS = mBlockSize;

    // Prepare inputs_embeds: [last_token, mask, ..., mask] for the draft engine
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostDraftInputIds[b * BS] = context.tokenIds[b].back();
        for (int32_t j = 1; j < BS; ++j)
        {
            hostDraftInputIds[b * BS + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    // Target hidden delta = full prefill hidden states
    check::check(
        mDraftTargetHidden.reshape({activeBatchSize, prefillLen, mBaseOutputHiddenDim}), "Tensor reshape failed");
    size_t const targetHiddenBytes = static_cast<size_t>(activeBatchSize) * prefillLen * mBaseOutputHiddenDim
        * utils::getTypeSize(mDraftTargetHidden.getDataType());
    CUDA_CHECK(cudaMemcpyAsync(mDraftTargetHidden.rawPointer(), mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(),
        targetHiddenBytes, cudaMemcpyDeviceToDevice, context.stream));

    // Upload delta_lengths for system prompt prefill (uniform prefillLen)
    {
        check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
        int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();
        std::fill_n(hostDeltaLens, activeBatchSize, static_cast<int32_t>(prefillLen));
        check::check(mDraftDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), mHostDeltaLens.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    }

    // Prepare proposal inputs (draft cache is empty at this point)
    int32_t const pmLen = divUp(BS, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, BS, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), activeBatchSize,
        context.stream);

    // Run draft engine (updates target KV cache; proposal logits are ignored)
    check::check(mDraftOutputLogits.reshape({activeBatchSize, BS, mDraftVocabSize}), "Tensor reshape failed");
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        activeBatchSize, BS, draftKVCapacity, prefillLen, BS, 1, static_cast<int64_t>(pmLen), activeBatchSize};

    cudaGetLastError();
    bool ok = mDraftExecutor->prepare(kPrefillProfile, draftDims, mDraftTensorMap, context.stream);
    if (ok)
        ok = mDraftExecutor->execute(context.stream);
    if (!ok)
    {
        LOG_ERROR("DFlash: system prompt draft prefill failed.");
        return false;
    }

    // Commit per-batch prefill lengths to draft cache manager.
    check::check(mDraftDeltaLenCommit.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLenCommit.rawPointer(), mDraftDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    mDraftCacheManager.commitSequenceLength(mDraftDeltaLenCommit, context.stream);

    return true;
}

void DFlashDecoder::saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
    std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream)
{
    constexpr int32_t CACHE_BATCH_IDX{0};
    SystemPromptKVCache savedCache;
    savedCache.systemPrompt = prompt;
    savedCache.tokenizedPrompt = tokenizedPrompt;
    savedCache.kvCacheLayers = mDraftCacheManager.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, stream);
    mSystemPromptKVCacheDraft.insert({key, std::move(savedCache)});
}

void DFlashDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
}

void DFlashDecoder::onBatchEvict(std::vector<int32_t> const& /* batchMapping */, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    mDraftCacheManager.compactBatch(deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    mDraftCacheManager.setActiveBatchSize(newActiveBatch);
}

} // namespace rt
} // namespace trt_edgellm
