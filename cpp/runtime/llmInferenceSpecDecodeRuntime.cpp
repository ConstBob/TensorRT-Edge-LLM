/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "llmInferenceSpecDecodeRuntime.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmRuntimeUtils.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace
{
//! Optimization-profile indices for the composable stack. Profile 0 is prefill, profile 1 is decode
//! (including EAGLE tree-verification / proposal / accept). These match the profile layout baked
//! into the engines by `llmBuilder`.
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

std::tuple<std::string, std::string> keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

} // namespace

namespace rt
{

void SpecDecodeInferenceContext::initialize(int32_t _activeBatchSize, int32_t _maxGenerateLength,
    rt::OptionalInputTensor const& _visualEmbeddings, rt::OptionalInputTensors const& _deepstackFeatures,
    std::string const& _loraWeightsName, cudaStream_t _stream)
{
    systemPrompts.resize(_activeBatchSize);
    rawBatchedInputIds.reserve(_activeBatchSize);
    tokenIds.resize(_activeBatchSize);
    currentGenerateLengths.resize(_activeBatchSize, 0);
    effectivePrefillLengths.resize(_activeBatchSize, 0);
    finishedStates.resize(_activeBatchSize, 0);
    slotStreams.clear();
    slotStreams.resize(_activeBatchSize);
    stopStringsPerSlot.clear();
    stopStringsPerSlot.resize(_activeBatchSize);

    // Initialize batch index mapping (identity mapping initially)
    batchIndexMapping.resize(_activeBatchSize);
    for (int32_t i = 0; i < _activeBatchSize; ++i)
    {
        batchIndexMapping[i] = i;
    }

    // Clear completed batch storage
    completedBatches.clear();

    visualEmbeddings = _visualEmbeddings;
    deepstackFeatures = _deepstackFeatures;
    generationRound = 0;
    maxGenerateLength = _maxGenerateLength;
    activeBatchSize = _activeBatchSize;
    loraWeightsName = _loraWeightsName;
    stream = _stream;
}

LLMInferenceSpecDecodeRuntime::LLMInferenceSpecDecodeRuntime(std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    EagleDraftingConfig const& draftingConfig, cudaStream_t stream)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream);
}

LLMInferenceSpecDecodeRuntime::LLMInferenceSpecDecodeRuntime(std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    cudaStream_t stream)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, std::nullopt, stream);
}

void LLMInferenceSpecDecodeRuntime::initializeCommon(std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<EagleDraftingConfig> const& draftingConfig, cudaStream_t stream)
{
    // -----------------------------------------------------------------------
    // 1. Load shared embedding table (shared between base and draft models).
    // -----------------------------------------------------------------------
    std::filesystem::path const embeddingPath = std::filesystem::path(engineDir) / "embedding.safetensors";
    mEmbedding = loadEmbeddingTable(embeddingPath, stream);

    // -----------------------------------------------------------------------
    // 2. Parse engine configurations and attach user drafting (bundle factory
    //    performs cross-engine consistency and drafting-vs-capacity checks).
    // -----------------------------------------------------------------------
    std::filesystem::path const baseEnginePath = draftingConfig.has_value()
        ? std::filesystem::path(engineDir) / "eagle_base.engine"
        : std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const baseConfigPath = draftingConfig.has_value()
        ? std::filesystem::path(engineDir) / "base_config.json"
        : std::filesystem::path(engineDir) / "config.json";
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{std::filesystem::path(engineDir) / "draft_config.json"}
        : std::nullopt;

    mDeployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);

    ELLM_CHECK(mDeployment.base.numDeepstackFeatures <= 0 || !multimodalEngineDir.empty(),
        "--multimodalEngineDir is required for VLM engine.");

    // -----------------------------------------------------------------------
    // 3. Construct Runners (registries built internally from the parsed configs).
    // -----------------------------------------------------------------------
    try
    {
        mBaseExecutor = EngineExecutor::createForLLM(baseEnginePath, mDeployment.base);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    if (draftingConfig.has_value())
    {
        std::filesystem::path const draftEnginePath = std::filesystem::path(engineDir) / "eagle_draft.engine";
        try
        {
            mDraftExecutor = EngineExecutor::createForEagleDraft(draftEnginePath, mDeployment);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize draft EngineExecutor: %s", e.what());
            throw std::runtime_error("Failed to initialize draft EngineExecutor: " + std::string(e.what()));
        }
        LOG_INFO("Draft EngineExecutor successfully loaded from %s.", draftEnginePath.c_str());
    }

    // -----------------------------------------------------------------------
    // 4. Validate engine binding dtypes against the parsed configs. Config is
    //    the source of truth; engine bindings are cross-checked here.
    // -----------------------------------------------------------------------
    validateAgainstEngine(mDeployment.base, *mBaseExecutor, "base");
    if (mDraftExecutor)
    {
        validateAgainstEngine(*mDeployment.draft, *mDraftExecutor, "draft");
    }

    // -----------------------------------------------------------------------
    // 5. Set runtime batch size.
    // -----------------------------------------------------------------------
    mMaxRuntimeBatchSize = mDeployment.maxRuntimeBatchSize();
    LOG_INFO("Runtime batch size set to: %d (from engine bundle)", mMaxRuntimeBatchSize);

    // -----------------------------------------------------------------------
    // 6. SharedResources + PipelineIO. PipelineIO is held via unique_ptr so
    //    its address is stable for the TensorMap pointers below (TensorMap
    //    stores non-owning Tensor* into PipelineIO members).
    // -----------------------------------------------------------------------
    bool const hasDraft = draftingConfig.has_value();
    if (hasDraft)
    {
        mSharedResources = SharedResources::createForEagle(mDeployment, mMaxRuntimeBatchSize, loraWeightsMap, stream);
        mPipelineIO
            = std::make_unique<PipelineIO>(PipelineIO::createForEagle(mDeployment, mMaxRuntimeBatchSize, stream));
    }
    else
    {
        mSharedResources = SharedResources::createForLLM(mDeployment.base, loraWeightsMap, stream);
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForLLM(mDeployment.base, stream));
    }

    // Externalized model weights: the SharedResources factory only allocates an
    // empty manager. Load external weights and validate against engine inputs.
    // External weights currently apply to the base engine only (no draft-engine load call).
    mSharedResources->externalWeightManager->load(std::filesystem::path(engineDir), baseConfigPath, stream);
    mSharedResources->externalWeightManager->validateAgainstEngine(*mBaseExecutor, "base");

    // -----------------------------------------------------------------------
    // 7. Build base TensorMap (kvCacheIndex=0) and publish static external
    //    weight bindings. EAGLE adds tree-mask / position IDs to this same
    //    map further down.
    // -----------------------------------------------------------------------
    buildTensorMap(mBaseTensorMap, *mPipelineIO, *mSharedResources, mDeployment.base, /*kvCacheIndex=*/0);
    mSharedResources->externalWeightManager->registerTensorMapEntries(mBaseTensorMap);

    // -----------------------------------------------------------------------
    // 8. EAGLE tree-building scratch + draft TensorMap. Engine-bound EAGLE
    //    tensors (packed mask, position IDs, dummy KV start index, tree-sized
    //    selectTokenIndices) live on PipelineIO — their bindings are wired
    //    into both maps by `buildTensorMap` / `buildTensorMapForEagleDraft`.
    //    Only tree-building scratch that never crosses the engine boundary
    //    stays on the runtime.
    // -----------------------------------------------------------------------
    if (hasDraft)
    {
        int32_t const effectiveMaxDraftTreeSize = mDeployment.effectiveMaxDraftTreeSize();
        mDraftTreeSize = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mDraftTreeSize");
        mDraftTreeMask = rt::Tensor({mMaxRuntimeBatchSize, effectiveMaxDraftTreeSize, effectiveMaxDraftTreeSize},
            rt::DeviceType::kGPU, DataType::kINT8, "LLMInferenceSpecDecodeRuntime::mDraftTreeMask");

        buildTensorMapForEagleDraft(mDraftTensorMap, *mPipelineIO, *mSharedResources, *mDeployment.draft);
    }

    // -----------------------------------------------------------------------
    // 9. LoRA: register engine bindings and seed the base tensor map with
    //    dummy / active adapter tensors. Only the base engine carries LoRA
    //    bindings — draft does not.
    // -----------------------------------------------------------------------
    if (mSharedResources->loraManager)
    {
        mSharedResources->loraManager->initializeEngineBindings(*mBaseExecutor);
        mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
    }

    // -----------------------------------------------------------------------
    // 10. Preprocessors.
    // -----------------------------------------------------------------------
    mStepPreparer = std::make_unique<StepPreparer>(mDeployment.base);
    mEmbeddingPre = std::make_unique<EmbeddingPreprocessor>(mEmbedding, mDeployment.base);
    if (mDeployment.base.numDeepstackFeatures > 0)
    {
        mDeepstack = std::make_unique<DeepstackBinding>(mPipelineIO->deepstackEmbeds, mSharedResources->zeroBuffer);
    }

    // -----------------------------------------------------------------------
    // 11. Allocate runtime-local tensors (sampling workspace, draft tree tables,
    //     host pinned scratch, batch-eviction mapping).
    // -----------------------------------------------------------------------
    int32_t const effectiveMaxTreeSize = hasDraft ? mDeployment.effectiveMaxDraftTreeSize() : 1;
    int32_t const effectiveDraftTopK = hasDraft ? draftingConfig->draftingTopK : 1;
    int32_t const effectiveMaxAcceptDepth = hasDraft ? draftingConfig->draftingStep + 1 : 1;
    int32_t const maxInputLength = hasDraft
        ? std::max(mDeployment.base.maxSupportedInputLength, mDeployment.draft->maxSupportedInputLength)
        : mDeployment.base.maxSupportedInputLength;
    int32_t const maxSamplingSize = hasDraft ? std::max(mMaxRuntimeBatchSize * effectiveMaxTreeSize,
                                                   mMaxRuntimeBatchSize * effectiveDraftTopK * effectiveDraftTopK)
                                             : mMaxRuntimeBatchSize;

    int32_t const draftFullTableLength = hasDraft
        ? 1 + effectiveDraftTopK + (draftingConfig->draftingStep - 1) * effectiveDraftTopK * effectiveDraftTopK
        : 0;

    // Reserve enough workspace for sampling, accounting for batch dimension in draft proposal stage.
    // Always include vanilla sampling workspace size because per-request disable_spec_decode
    // can fall back to topK/topP sampling even when draft is loaded.
    int32_t const vanillaSamplingWorkspaceSize
        = static_cast<int32_t>(getTopKtopPSamplingWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize,
            SamplingParams(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1.0f, 0, 0.9f)));
    int32_t const maxSamplingWorkspaceSize = hasDraft
        ? std::max({vanillaSamplingWorkspaceSize,
              static_cast<int32_t>(
                  getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1)),
              static_cast<int32_t>(getSelectAllTopKWorkspaceSize(
                  mMaxRuntimeBatchSize * effectiveDraftTopK, mDeployment.draft->outputVocabSize, effectiveDraftTopK))})
        : vanillaSamplingWorkspaceSize;

    try
    {
        mIdsInput = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mIdsInput");

        if (hasDraft)
        {
            mDraftTokenIdsFullTable = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU,
                DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mDraftTokenIdsFullTable");
            mDraftTokenScoreFullTable = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU,
                DataType::kFLOAT, "LLMInferenceSpecDecodeRuntime::mDraftTokenScoreFullTable");
            mDraftTokenPredecessorFullTable
                = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kINT32,
                    "LLMInferenceSpecDecodeRuntime::mDraftTokenPredecessorFullTable");
            // Draft vocab mapping table is 1D and shared across all batches (not batch-dependent)
            mDraftVocabMappingTable = rt::Tensor({mDeployment.draft->outputVocabSize}, rt::DeviceType::kGPU,
                DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mDraftVocabMappingTable");
            mDraftTreeRootTokenId = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
                "LLMInferenceSpecDecodeRuntime::mDraftTreeRootTokenId");
            mDraftTokenIdsTable = rt::Tensor({mMaxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK},
                rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mDraftTokenIdsTable");
            mDraftTokenScoresTable = rt::Tensor({mMaxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK},
                rt::DeviceType::kGPU, DataType::kFLOAT, "LLMInferenceSpecDecodeRuntime::mDraftTokenScoresTable");
            mDraftTokenIntermediateScores = rt::Tensor({mMaxRuntimeBatchSize, effectiveDraftTopK}, rt::DeviceType::kGPU,
                DataType::kFLOAT, "LLMInferenceSpecDecodeRuntime::mDraftTokenIntermediateScores");
            mDraftTokenIntermediateParents
                = rt::Tensor({mMaxRuntimeBatchSize, effectiveDraftTopK}, rt::DeviceType::kGPU, DataType::kINT32,
                    "LLMInferenceSpecDecodeRuntime::mDraftTokenIntermediateParents");
            mAcceptedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, effectiveMaxAcceptDepth}, rt::DeviceType::kGPU,
                DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mAcceptedTokenIds");
            mAcceptedTokenIndices = rt::Tensor({mMaxRuntimeBatchSize, effectiveMaxAcceptDepth}, rt::DeviceType::kGPU,
                DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mAcceptedTokenIndices");
            mAcceptLength = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
                "LLMInferenceSpecDecodeRuntime::mAcceptLength");
        }

        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceSpecDecodeRuntime::mSamplingWorkspace");
        mSamplingIndices = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mSamplingIndices");
        mSamplingScores = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT,
            "LLMInferenceSpecDecodeRuntime::mSamplingScores");

        // Batch mapping tensor for batch eviction.
        mDeviceBatchMapping = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mDeviceBatchMapping");

        mHostPackedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mHostPackedTokenIds");
        mHostSelectedTokenIds = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mHostSelectedTokenIds");
        mHostReuseKVCacheLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mHostReuseKVCacheLengths");

        // Pre-allocate multimodal indices tensor (used for audio/vision embedding lookup).
        mMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceSpecDecodeRuntime::mMultimodalIndices");

        if (hasDraft)
        {
            mHostAcceptLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
                "LLMInferenceSpecDecodeRuntime::mHostAcceptLengths");
            mHostAcceptedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, effectiveMaxAcceptDepth}, rt::DeviceType::kCPU,
                DataType::kINT32, "LLMInferenceSpecDecodeRuntime::mHostAcceptedTokenIds");
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate runtime tensors: %s", e.what());
        throw std::runtime_error("Failed to allocate runtime tensors: " + std::string(e.what()));
    }
    LOG_INFO("Runtime tensors successfully allocated.");

    // -----------------------------------------------------------------------
    // 12. Load draft-to-base vocab mapping table (EAGLE) and the optional base
    //     model reduced-vocab mapping table.
    //     MTP draft shares vocab with base (no d2t mapping needed); fill with
    //     zeros (identity) so the mapping is a no-op.
    // -----------------------------------------------------------------------
    if (hasDraft)
    {
        std::filesystem::path const d2tPath = std::filesystem::path(engineDir) / "d2t.safetensors";
        if (std::filesystem::exists(d2tPath))
        {
            std::vector<rt::Tensor> d2tTensors;
            ELLM_CHECK(safetensors::loadSafetensors(d2tPath, d2tTensors, stream),
                "Failed to load d2t.safetensors from model directory: " + engineDir);
            check::check(d2tTensors.size() == 1, "d2t.safetensors should contain exactly one tensor");
            check::check(d2tTensors[0].getShape().getNumDims() == 1, "d2t tensor should be 1D");
            check::check(d2tTensors[0].getShape()[0] == mDeployment.draft->outputVocabSize,
                "d2t tensor length should match draft vocab size");
            mDraftVocabMappingTable = std::move(d2tTensors[0]);
        }
        else
        {
            LOG_INFO("d2t.safetensors not found (MTP draft shares vocab with base), using identity mapping.");
            CUDA_CHECK(cudaMemsetAsync(
                mDraftVocabMappingTable.rawPointer(), 0, mDraftVocabMappingTable.getMemoryCapacity(), stream));
        }
    }

    if (mDeployment.base.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for base model reduced vocab size: %d -> %d",
            mDeployment.base.reducedVocabSize, mDeployment.base.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kVocabMapFileName) + " from model directory: " + engineDir);

        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mDeployment.base.reducedVocabSize,
            "vocab_map tensor length should match base model reduced vocab size");
        mBaseVocabMappingTable = std::move(vocabMapTensors[0]);
        LOG_INFO("Base model vocabulary mapping table successfully loaded.");
    }

    // -----------------------------------------------------------------------
    // 13. Tokenizer.
    // -----------------------------------------------------------------------
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Failed to load tokenizer from model directory: " + engineDir);
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // -----------------------------------------------------------------------
    // 14. Optional multimodal runners.
    // -----------------------------------------------------------------------
    if (!multimodalEngineDir.empty())
    {
        auto tryLoadRunner = [&](std::string const& dir, std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            try
            {
                LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
                auto runner = MultimodalRunner::create(
                    dir, mDeployment.base.maxSupportedBatchSize, mDeployment.base.maxKVCacheCapacity, stream);
                LOG_INFO("%s runner successfully initialized", name.c_str());
                return runner;
            }
            catch (std::exception const& e)
            {
                LOG_DEBUG("Failed to load %s runner from %s: %s", name.c_str(), dir.c_str(), e.what());
                return nullptr;
            }
        };

        mAudioRunner = tryLoadRunner(multimodalEngineDir + "/audio", "Audio");
        mVisionRunner = tryLoadRunner(multimodalEngineDir + "/visual", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = tryLoadRunner(multimodalEngineDir, "Vision");
        }

        // At least one multimodal runner must be available
        ELLM_CHECK(mAudioRunner || mVisionRunner, "No valid multimodal engine found in " + multimodalEngineDir);

        // Try to load action expert from multimodalEngineDir/action
        try
        {
            std::string actionDir = multimodalEngineDir + "/action";
            LOG_INFO("Attempting to load Action runner from %s", actionDir.c_str());
            mActionRunner = std::make_unique<Alpamayo1ActionRunner>(
                actionDir, stream, mSharedResources->cacheManagers[0]->getKVCacheManager().getConfig());
            LOG_INFO("Alpamayo 1 action expert loaded.");
        }
        catch (std::exception const& e)
        {
            LOG_INFO("Failed to load Action runner from %s: %s", (multimodalEngineDir + "/action").c_str(), e.what());
        }

        // Validate that the action engine's max KV cache capacity matches the LLM engine's.
        if (mActionRunner)
        {
            int32_t const actionMaxKVCacheCapacity = mActionRunner->getMaxKVCacheCapacity();
            int32_t const llmMaxKVCacheCapacity = mDeployment.base.maxKVCacheCapacity;
            ELLM_CHECK(actionMaxKVCacheCapacity == llmMaxKVCacheCapacity,
                format::fmtstr(
                    "Action engine max_kv_cache_capacity (%d) does not match LLM engine max_kv_cache_capacity (%d). "
                    "Re-export and rebuild the action engine with --max_kv_cache_capacity=%d to match the LLM engine.",
                    actionMaxKVCacheCapacity, llmMaxKVCacheCapacity, llmMaxKVCacheCapacity));
        }
    }

    // -----------------------------------------------------------------------
    // 15. Shared execution context memory for all engines (base, optional
    //     draft, and optional vision/audio). All engines execute serially so
    //     they can share a single buffer sized to the max requirement.
    // -----------------------------------------------------------------------
    int64_t const baseContextMemorySize = mBaseExecutor->getRequiredContextMemorySize();
    int64_t const draftContextMemorySize = mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
    int64_t const visionContextMemorySize = mVisionRunner ? mVisionRunner->getRequiredContextMemorySize() : 0;
    int64_t const audioContextMemorySize = mAudioRunner ? mAudioRunner->getRequiredContextMemorySize() : 0;
    int64_t const actionContextMemorySize = mActionRunner ? mActionRunner->getRequiredContextMemorySize() : 0;
    int64_t const sharedContextMemorySize = std::max({baseContextMemorySize, draftContextMemorySize,
        visionContextMemorySize, audioContextMemorySize, actionContextMemorySize});
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "LLMInferenceSpecDecodeRuntime::mSharedExecContextMemory");
    mBaseExecutor->setContextMemory(mSharedExecContextMemory);
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(mSharedExecContextMemory);
    }
    if (mVisionRunner)
    {
        mVisionRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mAudioRunner)
    {
        mAudioRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mActionRunner)
    {
        mActionRunner->setContextMemory(mSharedExecContextMemory);
    }
    LOG_INFO(
        "Setup shared execution context memory: %zu bytes (base requires: %zu, draft requires: %zu, vision requires: "
        "%zu, audio requires: %zu, action requires: %zu)",
        static_cast<size_t>(sharedContextMemorySize), static_cast<size_t>(baseContextMemorySize),
        static_cast<size_t>(draftContextMemorySize), static_cast<size_t>(visionContextMemorySize),
        static_cast<size_t>(audioContextMemorySize), static_cast<size_t>(actionContextMemorySize));
}

void LLMInferenceSpecDecodeRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

bool LLMInferenceSpecDecodeRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
{
    // Clear per-request portal state. Buffers themselves stay allocated and are
    // reshaped/overwritten when populated below — see getBaseModelHiddenStates() contract.
    mHiddenStatesRegistry.clear();
    mLastPrefillLength = 0;
    mLastInputTokenIds.clear();

    // Clear per-request response state. On failure (early return) the four vectors
    // stay empty; on success they are repopulated together below to matched sizes.
    response.outputIds.clear();
    response.outputTexts.clear();
    response.outputTrajectories.clear();
    response.finishReasons.clear();

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const enableSpecDecode = hasDraftModel() && !request.disableSpecDecode;
    std::string const& loraWeightsName = request.loraWeightsName;

    if (!validateRequestConfig(request))
    {
        return false;
    }

    if (!validateStreamingSubmission(request))
    {
        return false;
    }

    // Speculative decoding only supports greedy; override non-default sampling params.
    bool const hasNonDefaultSampling
        = (request.topK > 1 || request.topP < 1.0f || std::fabs(request.temperature - 1.0f) > 1e-3f);
    if (enableSpecDecode && hasNonDefaultSampling)
    {
        LOG_WARNING("Spec-decode active: overriding sampling params to greedy (ignoring temp/topK/topP).");
    }

    int32_t maxGenerateLength = request.maxGenerateLength;

    // Apply chat template for all requests (common for both multimodal and non-multimodal)
    request.formattedRequests.resize(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template to populate both formatted system prompt and full formatted prompt
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.enableThinking);
    }

    SpecDecodeInferenceContext context;
    context.initialize(
        activeBatchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    bool const supportsMultimodalInput
        = (mAudioRunner != nullptr) || (mVisionRunner != nullptr) || (mActionRunner != nullptr);

    if (supportsMultimodalInput)
    {
        if (!multiModalRuntimePreprocess(request, context, stream))
        {
            return false;
        }
    }
    else
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
            context.rawBatchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (context.rawBatchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
    }

    // Forward sampling params to context; spec-decode forces greedy.
    context.temperature = enableSpecDecode ? 1.0f : request.temperature;
    context.topP = enableSpecDecode ? 1.0f : request.topP;
    context.topK = enableSpecDecode ? 0 : request.topK;
    context.outputThinkerEmbeddings = outputThinkerEmbeddings;
    context.onTokenGenerated = request.onTokenGenerated;

    // Forward per-slot stop strings and cache the longest length to avoid
    // recomputing it on every emitChunks iteration.
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        context.stopStringsPerSlot[i] = request.requests[i].stopStrings;
        size_t maxLen = 0;
        for (auto const& s : request.requests[i].stopStrings)
        {
            if (s.size() > maxLen)
            {
                maxLen = s.size();
            }
        }
        context.slotStreams[i].maxStopLen = maxLen;
    }

    // The spec-decode path needs extra KV reserve for draft tokens during verification.
    constexpr int32_t kDRAFT_KVCACHE_RESERVE_LENGTH{100};
    int32_t const kvCacheCapacity = enableSpecDecode
        ? std::max(mDeployment.base.maxKVCacheCapacity, mDeployment.draft->maxKVCacheCapacity)
        : mDeployment.base.maxKVCacheCapacity;
    int32_t const kvcReserve = enableSpecDecode ? kDRAFT_KVCACHE_RESERVE_LENGTH : 0;

    // In production, the system-prompt KV cache is saved during warm-up.
    // We disable profiling here to make benchmarking closer to production inference result.
    bool profilingEnabled = getProfilingEnabled();
    if (profilingEnabled)
    {
        setProfilingEnabled(false);
    }

    // Generate system prompt KVCache for each sequence in the batch
    if (request.saveSystemPromptKVCache)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(context, i);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache for request %d in batch. "
                    "Continue to handle the request without saving the system prompt KVCache.",
                    i);
            }
        }
    }

    if (profilingEnabled)
    {
        setProfilingEnabled(true);
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(context))
    {
        LOG_ERROR("Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // ── Streaming setup ──────────────────────────────────────────────────────
    // Attach first, record in slotStreams only on success — a throw from attach
    // keeps foreign channels out of the finalizer's reach. Seed sentTokenCount
    // to the prompt length so streaming emits only generated tokens.
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (request.streamChannels.empty() || !request.streamChannels[i])
        {
            continue;
        }
        attachStreamChannel(request.streamChannels[i], context.batchIndexMapping[i]);
        auto& slot = context.slotStreams[i];
        slot.channel = request.streamChannels[i];
        slot.sentTokenCount = context.tokenIds[i].size();
        slot.lastEmittedTokenCount = slot.sentTokenCount;
    }
    StreamChannelFinalizer streamFinalizer(context, *mTokenizer);

    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(
        context.effectivePrefillLengths, request.maxGenerateLength, kvCacheCapacity, kvcReserve);
    if (clampedMaxGenerateLength != context.maxGenerateLength)
    {
        context.maxGenerateLength = clampedMaxGenerateLength;
        LOG_WARNING("Reduce max generation length to %d", context.maxGenerateLength);
    }
    if (context.maxGenerateLength <= 0)
    {
        LOG_ERROR("Insufficient KV cache capacity for generation for this request.");
        return false;
    }

    // Prefill from the base model and run spec-decode inference.
    bool const prefillStatus = runBaseModelPrefill(context);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    // Populate the base-model hidden-states portal so consumers (Qwen3-Omni Talker via
    // streaming callback or post-handleRequest sequential consumer) can fetch the buffers
    // by layer index. See getBaseModelHiddenStates() / getBaseModelInputTokenIds() for the
    // lifetime contract.
    int32_t prefillSequenceLength = 0;
    if (outputThinkerEmbeddings)
    {
        prefillSequenceLength
            = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());

        // Layer 0: back up post-multimodal input embeddings before the decode loop reshapes
        // mPipelineIO->inputsEmbeds to {BS,1,H} (scrambling the contiguous {BS,prefillLen,H} layout).
        // The backup buffer lives on PipelineIO and is lazy-allocated at maxISL on first
        // streaming request, then reshaped per request — see getBaseModelHiddenStates() lifetime contract.
        rt::Tensor& prefillEmbedsBackup = mPipelineIO->prefillEmbedsBackup;
        if (prefillEmbedsBackup.isEmpty())
        {
            prefillEmbedsBackup = rt::Tensor(
                {mMaxRuntimeBatchSize, mDeployment.base.maxSupportedInputLength, mDeployment.base.hiddenSize},
                rt::DeviceType::kGPU, DataType::kHALF, "PipelineIO::prefillEmbedsBackup");
        }
        check::check(prefillEmbedsBackup.reshape({activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize}),
            "Tensor reshape failed");
        size_t const prefillBytes = static_cast<size_t>(activeBatchSize) * prefillSequenceLength
            * mDeployment.base.hiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(prefillEmbedsBackup.rawPointer(), mPipelineIO->inputsEmbeds.rawPointer(),
            prefillBytes, cudaMemcpyDeviceToDevice, stream));

        mLastPrefillLength = prefillSequenceLength;
        mLastInputTokenIds = context.rawBatchedInputIds;
        mHiddenStatesRegistry[0] = &prefillEmbedsBackup;
        // Layer N (acceptHiddenLayer) is registered after the engine-output reshape below.
    }

    // Lambda to check if all batches are finished
    auto checkAllFinished = [&]() {
        // Check if all batches have been evicted
        if (context.activeBatchSize == 0)
        {
            return true;
        }
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (!context.finishedStates[i])
            {
                return false;
            }
        }
        return true;
    };

    // Used for Alpamayo 1
    int32_t trajFutureStartId = 0;
    if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        trajFutureStartId = static_cast<int32_t>(mTokenizer->getTokenId("<|traj_future_start|>"));
    }

    // Lambda to update finish states based on EOS and max_length. Latches
    // terminalReason atomically with the state flip — the !finishedStates guard
    // keeps first-writer-wins semantics relative to applyCancellationToFinishStates.
    auto updateFinishStates = [&]() {
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (context.finishedStates[i])
            {
                continue; // Respect first-writer-wins (cancel may have fired).
            }
            auto& s = context.slotStreams[i];
            // terminalReason is set for all slots; non-streaming slots surface it via
            // BatchResult.terminalReason → response.finishReasons.
            if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
            {
                if (context.tokenIds[i].size() > 1 && trajFutureStartId >= 0
                    && context.tokenIds[i][context.tokenIds[i].size() - 2] == trajFutureStartId)
                {
                    context.finishedStates[i] = 1;
                    s.terminalReason = FinishReason::kEndId;
                    LOG_DEBUG("Batch %d finished, reason: traj_future_start", i);
                    continue;
                }
            }
            else
            {
                // Check EOS
                if (!context.tokenIds[i].empty() && context.tokenIds[i].back() == mTokenizer->getEosId())
                {
                    context.finishedStates[i] = 1;
                    s.terminalReason = FinishReason::kEndId;
                    LOG_DEBUG("Batch %d finished, reason: EOS", i);
                    continue;
                }
            }
            // Check max length
            if (context.currentGenerateLengths[i] >= context.maxGenerateLength)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kLength;
                LOG_DEBUG(
                    "Batch %d finished, total tokens=%d, reason: max_length", i, context.currentGenerateLengths[i]);
                continue;
            }
        }

        // Stop-string override pass — runs after EOS/length so it can override
        // kEndId/kLength (user-relevant cause). Cancel/error still win because
        // decodePerSlot skipped the match when those reasons were latched.
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            auto& s = context.slotStreams[i];
            if (s.stopMatchedThisIter && s.terminalReason != FinishReason::kCancelled
                && s.terminalReason != FinishReason::kError)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kStopWords;
                LOG_DEBUG("Batch %d finished, reason: stop_words", i);
            }
        }
    };

    // Post-prefill per-iter pipeline:
    //   cancel → decode (emitDelta + stop match) → finalize (EOS/length/stop) → emit
    applyCancellationToFinishStates(context);
    decodePerSlot(context, *mTokenizer);
    updateFinishStates();
    emitChunks(context);

    // If everything finished during prefill, evict once so activeBatchSize reaches 0
    if (checkAllFinished() && context.activeBatchSize > 0)
    {
        bool const batchEvictStatus = performBatchEvict(context);
        if (!batchEvictStatus)
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    while (!checkAllFinished())
    {
        // Observe any consumer cancels at the top of the iteration so they land
        // first in the per-slot terminalReason latch.
        applyCancellationToFinishStates(context);

        if (enableSpecDecode)
        {
            if (context.generationRound == 0)
            {
                bool const draftPrefillStatus = runDraftModelPrefill(context);
                if (!draftPrefillStatus)
                {
                    LOG_ERROR("Failed to execute prefill step for draft model.");
                    return false;
                }
            }
            else
            {
                bool const draftAcceptTokenStatus = runDraftModelAcceptToken(context);
                if (!draftAcceptTokenStatus)
                {
                    LOG_ERROR("Failed to execute accept token step for draft model.");
                    return false;
                }
            }

            bool const draftTreeConstructionStatus = constructDraftTree(context);
            if (!draftTreeConstructionStatus)
            {
                LOG_ERROR("Failed to construct draft tree.");
                return false;
            }

            bool const baseModelVerificationStatus = runBaseModelVerification(context);
            if (!baseModelVerificationStatus)
            {
                LOG_ERROR("Failed to verify token draft tree with base model.");
                return false;
            }
        }
        else
        {
            bool const vanillaDecodingStatus = runVanillaDecoding(context);
            if (!vanillaDecodingStatus)
            {
                LOG_ERROR("Failed to decode tokens with vanilla decoding.");
                return false;
            }
        }

        // Per-iter pipeline: decode → finalize finish state → emit chunks.
        decodePerSlot(context, *mTokenizer);
        updateFinishStates();
        emitChunks(context);
        context.generationRound += 1;

        // Perform batch eviction if needed (after verification, before updating finish states)
        bool const batchEvictStatus = performBatchEvict(context);
        if (!batchEvictStatus)
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    if (context.activeBatchSize != 0)
    {
        LOG_ERROR("Eviction failure, there should be no active batch at the end of the inference. activeBatchSize: %d",
            context.activeBatchSize);
        return false;
    }

    // Record metrics - accumulate across all batches (active + evicted)
    int32_t totalReusedTokens = 0;
    int32_t totalComputedTokens = 0;
    int32_t totalGeneratedTokens = 0;
    int32_t totalIterations = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += (rawPromptLength - computedLength);
        totalComputedTokens += computedLength;
        totalGeneratedTokens += batchResult.generateLength;
        totalIterations += batchResult.actualIterations;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens);
    if (enableSpecDecode)
    {
        mEagleGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens);
    }
    else
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Save output ids and decoded texts to response.
    // Maintain original batch order using original batch indices.
    response.outputIds.resize(context.completedBatches.size());
    response.outputTexts.resize(context.completedBatches.size());
    response.outputTrajectories.resize(context.completedBatches.size());
    response.finishReasons.resize(context.completedBatches.size(), FinishReason::kNotFinished);

    // Add outputs from completed batches (using saved original indices)
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t genLength = batchResult.generateLength;

        // Log acceptance metrics for evicted batch
        if (enableSpecDecode)
        {
            int32_t const verificationTokens = genLength > 0 ? genLength - 1 : 0;
            float const acceptanceRate = batchResult.actualIterations > 0
                ? static_cast<float>(verificationTokens) / static_cast<float>(batchResult.actualIterations)
                : 0.0f;
            LOG_DEBUG(
                "Batch (completed with SpecDecode, original idx %d) - Acceptance rate: %.3f, Generated tokens: %d, "
                "Iterations: %d",
                originalIdx, acceptanceRate, genLength, batchResult.actualIterations);
        }

        // Extract generated tokens
        int32_t const totalLength = static_cast<int32_t>(batchResult.tokenIds.size());

        check::check(totalLength >= genLength, "Total length should be greater than or equal to generated length");
        response.outputIds[originalIdx] = std::vector<int32_t>(
            batchResult.tokenIds.begin() + (totalLength - genLength), batchResult.tokenIds.end());
        response.outputTexts[originalIdx] = mTokenizer->decode(response.outputIds[originalIdx], true);
        response.finishReasons[originalIdx] = batchResult.terminalReason;

        // Trim this slot's own stop strings from its output text by delegating
        // to applyStopStringMatch with isFinal=true — single source of truth
        // for earliest-position-wins semantics, shared with the streaming path.
        // outputIds is intentionally left intact (full token stream).
        if (originalIdx < static_cast<int32_t>(request.requests.size())
            && !request.requests[originalIdx].stopStrings.empty())
        {
            auto const& slotStops = request.requests[originalIdx].stopStrings;
            size_t maxLen = 0;
            for (auto const& s : slotStops)
            {
                maxLen = std::max(maxLen, s.size());
            }
            auto& text = response.outputTexts[originalIdx];
            auto outcome = applyStopStringMatch(text, slotStops, maxLen, /*isFinal=*/true);
            text = std::move(outcome.emitted);
            if (outcome.stopMatched)
            {
                // emitDelta (incremental) and one-shot Tokenizer::decode can differ at BPE
                // piece boundaries — upgrade the reason if one-shot surfaced a stop the
                // streaming-path matcher missed.
                response.finishReasons[originalIdx] = FinishReason::kStopWords;
            }
        }
    }

    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });
    // If action engine is loaded, run one batched trajectory sample and fill output for all batch items.
    if (hasTrajectoryHistory && mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        if (!mVisionRunner)
        {
            LOG_ERROR("Alpamayo1ActionRunner requires a vision runner (e.g. QwenViTRunner) for MRoPE rope deltas.");
            return false;
        }

        multimodal::ModelType const visionType = mVisionRunner->getModelType();
        bool const isQwen3ViT = visionType == multimodal::ModelType::QWEN3_VL;
        if (!isQwen3ViT)
        {
            LOG_ERROR(
                "Alpamayo1ActionRunner requires a Qwen3-VL vision runner but a different vision runner is loaded.");
            return false;
        }
        // MultimodalRunner::create() uses QwenViTRunner only for Qwen3-VL.
        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltas = qwenVision->getMropeRopeDeltasPerBatch();
        rt::HybridCacheManager& kvcache = *mSharedResources->cacheManagers[0];
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories
            = mActionRunner->sampleTrajectory(stream, activeBatchSize, kvcache, ropeDeltas);
        if (trajectories.size() != static_cast<size_t>(activeBatchSize))
        {
            LOG_ERROR("Alpamayo1ActionRunner trajectory sampling failed.");
            return false;
        }
        for (size_t i = 0; i < trajectories.size() && i < static_cast<size_t>(activeBatchSize); ++i)
        {
            if (!trajectories[i].empty())
            {
                response.outputTrajectories[i] = std::move(trajectories[i]);
            }
        }
    }

    // Reshape engine-output hidden states to the actual prefill size and register layer N
    // (acceptHiddenLayer) in the portal. The buffers live on PipelineIO; the registry
    // here just records non-owning pointers consumers fetch via getBaseModelHiddenStates().
    if (outputThinkerEmbeddings)
    {
        rt::Tensor& outputHiddenStates = mPipelineIO->outputHiddenStates;
        check::check(outputHiddenStates.reshape({activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize}),
            "Tensor reshape failed");
        mHiddenStatesRegistry[request.acceptHiddenLayer] = &outputHiddenStates;
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::validateRequestConfig(LLMGenerationRequest const& request)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    if (activeBatchSize == 0)
    {
        LOG_ERROR("Empty request with no requests");
        return false;
    }

    if (activeBatchSize > mMaxRuntimeBatchSize)
    {
        LOG_ERROR(
            "Requested batch size %d exceeds maximum supported batch size %d", activeBatchSize, mMaxRuntimeBatchSize);
        return false;
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (request.requests[i].messages.empty())
        {
            LOG_ERROR("Request %d in batch is empty: no messages provided", i);
            return false;
        }
    }
    if (hasAudio && !mAudioRunner)
    {
        LOG_ERROR("Request contains audio input, but this runtime does not have an audio runner.");
        return false;
    }
    if (hasVision && !mVisionRunner)
    {
        LOG_ERROR("Request contains vision input, but this runtime does not have a vision runner.");
        return false;
    }
    if (hasTrajectoryHistory && !mActionRunner)
    {
        LOG_ERROR("Request contains trajectory history input, but this runtime does not have an action runner.");
        return false;
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::multiModalRuntimePreprocess(
    LLMGenerationRequest const& request, SpecDecodeInferenceContext& context, cudaStream_t stream)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    // Clear request-scoped multimodal state up front so previous requests cannot leak through reused runtime members.
    context.visualEmbeddings = std::nullopt;
    context.audioEmbeddings = std::nullopt;
    context.deepstackFeatures.clear();
    // Treat multimodal indices as request-scoped state. Only request paths that explicitly rebuild
    // mMultimodalIndices for the current request should observe a non-empty tensor downstream.
    check::check(mMultimodalIndices.reshape({0}), "Tensor reshape failed");

    // Mark multimodal preprocessing and inference for NVTX profiling
    NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);

    std::vector<std::vector<int32_t>> batchedInputIds;

    // MRope cos/sin output cache is supplied only for MRope-based runners (QwenViT, Qwen3OmniAudio).
    // Runners with standard RoPE (InternViT, Phi4MMViT) ignore it; see MultimodalRunner::preprocess.
    rt::OptionalOutputTensor mropeCosSinOut = (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        ? rt::OptionalOutputTensor{std::ref(mPipelineIO->mropeCosSin)}
        : std::nullopt;

    // Process audio inputs (if present)
    if (hasAudio && mAudioRunner)
    {
        LOG_INFO("Processing audio inputs");
        if (!mAudioRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Audio preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mAudioRunner->infer(stream))
        {
            LOG_ERROR("Audio inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process vision inputs (if present)
    if (hasVision && mVisionRunner)
    {
        LOG_INFO("Processing vision inputs");
        if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Vision preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mVisionRunner->infer(stream))
        {
            LOG_ERROR("Vision inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process action inputs (if present)
    if (hasTrajectoryHistory && mActionRunner)
    {
        LOG_INFO("Processing trajectory history inputs");
        if (!mActionRunner->preprocess(request, batchedInputIds, mTokenizer.get()))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Trajectory history preprocessing failed. This request cannot be handled.");
            return false;
        }
    }

    if (!hasAudio && !hasVision)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.push_back(mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (batchedInputIds.back().empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
        if (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        {
            rt::Tensor& ropeCosSinCache = mPipelineIO->mropeCosSin;
            check::check(ropeCosSinCache.reshape({mDeployment.base.maxSupportedBatchSize,
                             mDeployment.base.maxKVCacheCapacity, mDeployment.base.rotaryDim}),
                "Tensor reshape failed");
            kernel::initializeTextOnlyMRopeCosSin(ropeCosSinCache.dataPointer<float>(),
                mDeployment.base.ropeConfig.rotaryTheta, mDeployment.base.rotaryDim,
                mDeployment.base.maxKVCacheCapacity, mDeployment.base.maxSupportedBatchSize, stream);
        }
    }

    // Get embeddings from independent runners — gate on request having multimodal data,
    // not just runner existence, to avoid leaking stale embeddings from previous requests.
    rt::OptionalInputTensor visionEmbeddings
        = (hasVision && mVisionRunner) ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = (hasAudio && mAudioRunner) ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors deepstackFeatures
        = (hasVision && mVisionRunner) ? mVisionRunner->getDeepstackFeatures() : rt::OptionalInputTensors{};

    context.visualEmbeddings = visionEmbeddings;
    context.deepstackFeatures = deepstackFeatures;
    context.audioEmbeddings = audioEmbeddings;

    // Populate system prompts and raw input IDs from batchedInputIds
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
        context.rawBatchedInputIds.push_back(batchedInputIds[i]);
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelPrefill(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_base_prefill,
        ("EAGLE_BASE_PREFILL[" + std::to_string(context.activeBatchSize) + "]").c_str(), nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const inputIdsLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    int32_t const baseOutputHiddenDim = mDeployment.eagle.has_value() ? mDeployment.eagle->baseOutputHiddenDim : 0;

    // Reshape IO tensors for this step.
    check::check(mIdsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(mPipelineIO->hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mPipelineIO->inputsEmbeds.reshape({activeBatchSize, inputIdsLength, mDeployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
        "Tensor reshape failed");
    if (hasDraftModel())
    {
        // EAGLE: base engine emits hidden states that feed the draft engine.
        check::check(mPipelineIO->baseHiddenStates.reshape({activeBatchSize, inputIdsLength, baseOutputHiddenDim}),
            "Tensor reshape failed");
    }

    // Populate host-side context lengths with effective (unpadded) prefill lengths and pack tokens.
    int32_t* hostCtxLenData = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    check::check(mHostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mHostPackedTokenIds.dataPointer<int32_t>();

    // Clear the entire pinned buffer first so trailing pad slots from prior batches don't leak into the
    // multimodal-indices walk, which scans all inputIdsLength positions per row, not just up to context_length.
    std::fill(hostPackedTokenIdsData, hostPackedTokenIdsData + activeBatchSize * inputIdsLength, 0);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        hostCtxLenData[i] = context.effectivePrefillLengths[i];
        std::copy(context.tokenIds[i].begin(), context.tokenIds[i].end(), hostPackedTokenIdsData + i * inputIdsLength);
    }

    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Embedding lookup (text / vision / audio-multimodal) into mPipelineIO->inputsEmbeds;
    // deepstack slots are populated from features or zero-filled depending on the request.
    mEmbeddingPre->embed(mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream);
    mEmbeddingPre->prepareDeepstack(mIdsInput, context.deepstackFeatures, *mPipelineIO, context.stream);

    // Dispatch per-step sequence prep (context lengths H2D, selectTokenIndices).
    mStepPreparer->prepare(
        InferencePhase::kPrefill, activeBatchSize, (*mSharedResources->cacheManagers[0]), *mPipelineIO, context.stream);
    // Bind real deepstack features for this prefill (no-op when feature absent).
    if (mDeepstack)
    {
        mDeepstack->useRealFeatures(mBaseTensorMap);
    }

    // Execute base prefill through the EngineExecutor. Empty-cache is
    // runtime-dynamic; prefillDims uses it to set InferenceDims::startIndexLen
    // (0 for the "initial prefill" sentinel, else batch).
    bool const baseKVAllEmpty = (*mSharedResources->cacheManagers[0]).getKVCacheAllEmpty();
    auto const prefillDims = mDeployment.base.prefillDims(activeBatchSize, inputIdsLength, baseKVAllEmpty);
    check::check(mBaseExecutor->prepare(kPrefillProfile, prefillDims, mBaseTensorMap, context.stream),
        "Failed to prepare base model for prefill step.");
    check::check(mBaseExecutor->execute(context.stream), "Failed to execute base model for prefill step.");
    (*mSharedResources->cacheManagers[0]).commitSequenceLength(mPipelineIO->contextLengths, context.stream);

    // Sampling from the prefill stage logits follows the same policy as vanilla decoding.
    // Spec-decode forces greedy sampling upstream (handleRequest sets topK/topP/temperature to greedy
    // defaults when enableSpecDecode), so this branch is a no-op in the EAGLE path.
    check::check(mSamplingIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mDeployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(
            mPipelineIO->outputLogits, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mPipelineIO->outputLogits, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace,
            context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary.
    if (mDeployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mSamplingIndices, mBaseVocabMappingTable, context.stream);
    }

    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (!context.finishedStates[i])
        {
            context.tokenIds[i].push_back(hostSelectedTokenIdsData[i]);
            context.currentGenerateLengths[i] += 1;

            // Fire the per-token callback for the prefill-sampled token. runVanillaDecoding
            // dispatches the callback for every decode token, so emitting here keeps the sequence
            // complete for streaming consumers (e.g. the Qwen3-Omni Thinker-Talker pipeline).
            if (context.onTokenGenerated.has_value())
            {
                bool const isFinished = context.finishedStates[i] != 0;
                TokenCallbackInfo info{hostSelectedTokenIdsData[i], i, context.generationRound, isFinished};
                context.onTokenGenerated.value()(info);
            }
        }
    }
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelPrefill(SpecDecodeInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mDeployment.eagle.has_value());
    assert(mDeployment.draft.has_value());
    TIME_STAGE(metrics::StageNames::kEAGLE_DRAFT_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_draft_prefill,
        ("EAGLE_DRAFT_PREFILL[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::DARK_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mDeployment.eagle->draftHiddenSize;
    int32_t const draftVocabSize = mDeployment.draft->outputVocabSize;

    // F_n = F(H_n, Token_{n+1}): draft consumes base hidden states + tokens shifted by one.
    int32_t const inputIdsLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());

    check::check(mPipelineIO->baseHiddenStates.getShape()[0] == activeBatchSize
            && mPipelineIO->baseHiddenStates.getShape()[1] == inputIdsLength,
        "BaseHiddenStates shape [batch, seq_len, hidden_dim] shall match with [activeBatchSize, inputIdsLength, "
        "hidden_dim]");

    // Reshape input/output tensors.
    check::check(mIdsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(mPipelineIO->draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(
        mPipelineIO->draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}), "Tensor reshape failed");
    check::check(mPipelineIO->hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    // Clear garbage data in the draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(mPipelineIO->draftHiddenStatesIn.rawPointer(), 0,
        mPipelineIO->draftHiddenStatesIn.getMemoryCapacity(), context.stream));

    // Pack token IDs (skip first token) into host pinned memory.
    check::check(mHostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mHostPackedTokenIds.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(
            context.tokenIds[i].begin() + 1, context.tokenIds[i].end(), hostPackedTokenIdsData + i * inputIdsLength);
    }
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Draft embeddings (text-only or image-insertion variant for vision-capable models).
    check::check(
        mPipelineIO->inputsEmbeds.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}), "Tensor reshape failed");
    if (context.visualEmbeddings.has_value())
    {
        rt::Tensor const& imageEmbedsTensor = context.visualEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(),
            imageEmbedsTensor, mPipelineIO->inputsEmbeds, context.stream);
    }
    else
    {
        kernel::embeddingLookup(
            mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(), mPipelineIO->inputsEmbeds, context.stream);
    }

    // Populate GPU context_lengths + selectTokenIndices (last real token per sequence) for the draft prefill.
    int32_t* ctxLenData = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        ctxLenData[i] = context.effectivePrefillLengths[i];
    }
    CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->contextLengths.rawPointer(), ctxLenData, activeBatchSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    check::check(mPipelineIO->selectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(
        mPipelineIO->hostSelectTokenIndices.reshape({activeBatchSize, 1}), "hostSelectTokenIndices reshape failed");
    int64_t* selectData = mPipelineIO->hostSelectTokenIndices.dataPointer<int64_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectData[i] = context.effectivePrefillLengths[i] - 1;
    }
    CUDA_CHECK(
        cudaMemcpyAsync(mPipelineIO->selectTokenIndices.rawPointer(), mPipelineIO->hostSelectTokenIndices.rawPointer(),
            activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, context.stream));

    // Draft prefill always runs with an empty draft KV cache on round 0, so
    // `startIndexLen` resolves to 0 (the engine's "initial prefill" sentinel);
    // subsequent rounds pass a non-empty cache and use [batch]. InferenceDims does
    // all the shape work — no per-call rebind of kvcache_start_index.
    bool const draftKVAllEmpty = (*mSharedResources->cacheManagers[1]).getKVCacheAllEmpty();
    auto const prefillDims = mDeployment.draft->prefillDims(activeBatchSize, inputIdsLength, draftKVAllEmpty);
    bool prefillSuccess = mDraftExecutor->prepare(kPrefillProfile, prefillDims, mDraftTensorMap, context.stream);
    if (prefillSuccess)
    {
        prefillSuccess = mDraftExecutor->execute(context.stream);
    }
    if (prefillSuccess)
    {
        (*mSharedResources->cacheManagers[1]).commitSequenceLength(mPipelineIO->contextLengths, context.stream);
    }
    if (!prefillSuccess)
    {
        LOG_ERROR("Failed to execute prefill step for draft model.");
        return false;
    }
    return true;
}

bool LLMInferenceSpecDecodeRuntime::constructDraftTree(SpecDecodeInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mDeployment.eagle.has_value());
    assert(mDeployment.draft.has_value());
    TIME_STAGE(metrics::StageNames::kEAGLE_CONSTRUCT_DRAFT_TREE, context.stream);
    NVTX_SCOPED_RANGE(nvtx_construct_tree,
        ("EAGLE_CONSTRUCT_TREE[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mDeployment.eagle->draftHiddenSize;
    int32_t const baseOutputHiddenDim = mDeployment.eagle->baseOutputHiddenDim;
    int32_t const draftVocabSize = mDeployment.draft->outputVocabSize;

    // Reshape draft tree tensors to match activeBatchSize for dynamic batching
    int32_t const draftTopK = mDeployment.eagle->draftingTopK;
    int32_t const draftFullTableLength = static_cast<int32_t>(mDraftTokenIdsFullTable.getShape()[1]);
    check::check(mDraftTokenIdsFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(mDraftTokenScoreFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(
        mDraftTokenPredecessorFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(mDraftTreeRootTokenId.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftTokenIdsTable.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenScoresTable.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenIntermediateScores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenIntermediateParents.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");

    // Record root token (last committed token selected by base model) id for the draft tree for each batch.
    std::vector<int32_t> rootTokenIds(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        rootTokenIds[i] = context.tokenIds[i].back();
    }
    CUDA_CHECK(cudaMemcpyAsync(mDraftTreeRootTokenId.rawPointer(), rootTokenIds.data(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Sampling from the logits output, collect draftTopK tokens as first level under "root".
    check::check(mSamplingIndices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    check::check(mSamplingScores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    selectAllTopK(mPipelineIO->outputLogits, std::ref(mSamplingScores), mSamplingIndices, draftTopK, mSamplingWorkspace,
        context.stream);

    // Initialize data structures to describe the whole draft tree.
    kernel::initializeDraftTreeTables(mSamplingIndices, mSamplingScores, mDraftTreeRootTokenId, mDraftVocabMappingTable,
        mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, context.stream);

    // Reset hidden states output of base model and input hidden states of draft model to clear garbage data.
    CUDA_CHECK(cudaMemsetAsync(mPipelineIO->baseHiddenStates.rawPointer(), 0,
        mPipelineIO->baseHiddenStates.getMemoryCapacity(), context.stream));
    CUDA_CHECK(cudaMemsetAsync(mPipelineIO->draftHiddenStatesIn.rawPointer(), 0,
        mPipelineIO->draftHiddenStatesIn.getMemoryCapacity(), context.stream));

    // Construct padded input tensors for the draft engine. Results are only collected from relevant indices.
    int32_t const paddedDraftTreeSize = mDeployment.eagle->draftingStep * draftTopK;
    check::check(mIdsInput.reshape({activeBatchSize, paddedDraftTreeSize}), "Tensor reshape failed");
    check::check(mPipelineIO->baseHiddenStates.reshape({activeBatchSize, paddedDraftTreeSize, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(mPipelineIO->draftHiddenStatesIn.reshape({activeBatchSize, paddedDraftTreeSize, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mDraftTreeSize.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mDraftTreeMask.reshape({activeBatchSize, paddedDraftTreeSize, paddedDraftTreeSize}), "Tensor reshape failed");
    check::check(mPipelineIO->packedAttentionMask.reshape(
                     {activeBatchSize, paddedDraftTreeSize, static_cast<int64_t>(divUp(paddedDraftTreeSize, 32))}),
        "Tensor reshape failed");

    // Assemble the initial draft tree input (must happen before reshaping the draft hidden states output).
    kernel::assembleInitialDraftTreeInput(mDraftTokenIdsFullTable, mPipelineIO->draftHiddenStatesOut, mIdsInput,
        mPipelineIO->draftHiddenStatesIn, mDraftTreeSize, mDraftTreeMask, draftTopK, context.stream);

    // Output tensors must be 3D: [batch_size, num_tokens, vocab_size/hidden_dim] for draft proposal.
    check::check(
        mPipelineIO->outputLogits.reshape({activeBatchSize, draftTopK, draftVocabSize}), "Tensor reshape failed");
    check::check(mPipelineIO->draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
        "Tensor reshape failed");

    for (int32_t round = 0; round < mDeployment.eagle->draftingStep - 1; round++)
    {
        if (round == 0)
        {
            kernel::assembleInitialIntermediateData(mSamplingScores, mDraftTokenIntermediateParents,
                mDraftTokenIntermediateScores, draftTopK, context.stream);
        }
        else
        {
            check::check(mSamplingIndices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
            check::check(mSamplingScores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
            selectAllTopK(mDraftTokenScoresTable, std::ref(mSamplingScores), mSamplingIndices, draftTopK,
                mSamplingWorkspace, context.stream);
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mPipelineIO->draftHiddenStatesOut, mSamplingIndices,
                mIdsInput, mPipelineIO->draftHiddenStatesIn, mDraftTreeSize, mDraftTreeMask, draftTopK, round,
                context.stream);
            kernel::assembleIntermediateData(mSamplingScores, mSamplingIndices, mDraftTokenIntermediateScores,
                mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

        check::check(
            mPipelineIO->outputLogits.reshape({activeBatchSize, draftTopK, draftVocabSize}), "Tensor reshape failed");
        check::check(mPipelineIO->draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
            "Tensor reshape failed");

        // Perform embedding lookup for draft proposal input (draft proposal only has text, no images).
        check::check(mPipelineIO->inputsEmbeds.reshape({activeBatchSize, paddedDraftTreeSize, draftHiddenSize}),
            "Tensor reshape failed");
        kernel::embeddingLookup(
            mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(), mPipelineIO->inputsEmbeds, context.stream);

        // Prepare EAGLE inputs: packed mask, position IDs, select token indices, context lengths.
        // selectTokenIndices is reshaped to [batch, draftTopK] because the draft
        // engine's last_token_ids selects one token per tree branch (the top-K
        // candidates at this proposal level), matching the 3D logits output
        // shape [batch, draftTopK, draftVocabSize].
        {
            rt::Tensor const& draftKVCacheLengths = (*mSharedResources->cacheManagers[1]).getKVCacheLengths();
            check::check(
                mPipelineIO->selectTokenIndices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
            check::check(mPipelineIO->contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
            check::check(
                mPipelineIO->eaglePositionIds.reshape({activeBatchSize, paddedDraftTreeSize}), "Tensor reshape failed");
            kernel::prepareEagleDraftProposalInputs(mDraftTreeMask, mDraftTreeSize, draftKVCacheLengths,
                mPipelineIO->packedAttentionMask, mPipelineIO->eaglePositionIds, mPipelineIO->selectTokenIndices,
                mPipelineIO->contextLengths, context.stream);
        }

        // Execute draft proposal via EngineExecutor (uses kDecodeProfile for tree decoding).
        // kvcache_start_index is a static binding: proposalDims sets
        // InferenceDims::startIndexLen = batch (the draft KV cache is non-empty
        // after prefill committed its sequence length), so no rebind needed.
        auto const proposalDims = mDeployment.draft->proposalDims(activeBatchSize, paddedDraftTreeSize, draftTopK);
        check::check(mDraftExecutor->prepare(kDecodeProfile, proposalDims, mDraftTensorMap, context.stream),
            "Failed to prepare draft model for draft proposal step.");
        check::check(mDraftExecutor->execute(context.stream), "Failed to execute draft model for draft proposal step.");

        // Collect TopK results from each lane. Reshape to 2D for selectAllTopK.
        check::check(
            mPipelineIO->outputLogits.reshape({activeBatchSize * draftTopK, draftVocabSize}), "Tensor reshape failed");
        check::check(mPipelineIO->draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, draftHiddenSize}),
            "Tensor reshape failed");
        check::check(mSamplingIndices.reshape({activeBatchSize * draftTopK, draftTopK}), "Tensor reshape failed");
        check::check(mSamplingScores.reshape({activeBatchSize * draftTopK, draftTopK}), "Tensor reshape failed");
        selectAllTopK(mPipelineIO->outputLogits, std::ref(mSamplingScores), mSamplingIndices, draftTopK,
            mSamplingWorkspace, context.stream);

        // Reshape sampling indices/scores back for subsequent kernels.
        // Note: draftHiddenStatesOut stays in 2D for assembleDraftTreeInput in the next round.
        check::check(mSamplingIndices.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
        check::check(mSamplingScores.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");

        kernel::computeCuScoresAndTranslateToken(mSamplingIndices, mSamplingScores, mDraftTokenIntermediateScores,
            mDraftVocabMappingTable, mDraftTokenIdsTable, mDraftTokenScoresTable, draftTopK, context.stream);
        kernel::updateDraftTreeFullTables(mDraftTokenIdsTable, mDraftTokenScoresTable, mDraftTokenIntermediateParents,
            mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, round,
            context.stream);
    }

    // Pick top candidates and produce the verification tree.
    check::check(
        mSamplingIndices.reshape({activeBatchSize, mDeployment.eagle->verifyTreeSize}), "Tensor reshape failed");
    selectAllTopK(mDraftTokenScoreFullTable, std::nullopt, mSamplingIndices, mDeployment.eagle->verifyTreeSize,
        mSamplingWorkspace, context.stream);

    check::check(mIdsInput.reshape({activeBatchSize, mDeployment.eagle->verifyTreeSize}), "Tensor reshape failed");
    check::check(
        mDraftTreeMask.reshape({activeBatchSize, mDeployment.eagle->verifyTreeSize, mDeployment.eagle->verifyTreeSize}),
        "Tensor reshape failed");
    check::check(mPipelineIO->packedAttentionMask.reshape({activeBatchSize, mDeployment.eagle->verifyTreeSize,
                     static_cast<int64_t>(divUp(mDeployment.eagle->verifyTreeSize, 32))}),
        "Tensor reshape failed");
    kernel::constructVerificationDraftTree(mDraftTokenIdsFullTable, mDraftTokenPredecessorFullTable, mSamplingIndices,
        mIdsInput, mDraftTreeMask, context.stream);

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelVerification(SpecDecodeInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mDeployment.eagle.has_value());
    TIME_STAGE(metrics::StageNames::kEAGLE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_verify,
        ("EAGLE_VERIFY[R" + std::to_string(context.generationRound) + "," + std::to_string(context.activeBatchSize)
            + "]")
            .c_str(),
        nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const baseOutputHiddenDim = mDeployment.eagle.has_value() ? mDeployment.eagle->baseOutputHiddenDim : 0;

    // Cache sub-manager references used throughout this function.
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();

    check::check(
        mIdsInput.getShape()[0] == activeBatchSize && mIdsInput.getShape()[1] == mDeployment.eagle->verifyTreeSize,
        "IdsInput shall have shape [batch_size, verify_tree_size]");
    check::check(mDraftTreeMask.getShape()[0] == activeBatchSize
            && mDraftTreeMask.getShape()[1] == mDeployment.eagle->verifyTreeSize
            && mDraftTreeMask.getShape()[2] == mDeployment.eagle->verifyTreeSize,
        "DraftTreeMask shall have shape [batch_size, verify_tree_size, verify_tree_size]");
    check::check(mPipelineIO->packedAttentionMask.getShape()[0] == activeBatchSize
            && mPipelineIO->packedAttentionMask.getShape()[1] == mDeployment.eagle->verifyTreeSize
            && mPipelineIO->packedAttentionMask.getShape()[2]
                == static_cast<int64_t>(divUp(mDeployment.eagle->verifyTreeSize, 32)),
        "PackedAttentionMask shall have shape [batch_size, verify_tree_size, packed_mask_len]");

    // Perform embedding lookup for base model verification (Eagle base tree decoding only has text, no images).
    check::check(mPipelineIO->inputsEmbeds.reshape(
                     {activeBatchSize, mDeployment.eagle->verifyTreeSize, mDeployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(
        mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(), mPipelineIO->inputsEmbeds, context.stream);

    // Engine expects 2D tensors: [batch_size * verify_tree_size, vocab_size/hidden_dim].
    int32_t const selectTokenSize = activeBatchSize * mDeployment.eagle->verifyTreeSize;
    check::check(mPipelineIO->outputLogits.reshape({selectTokenSize, mDeployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(
        mPipelineIO->baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}), "Tensor reshape failed");

    // Prepare EAGLE inputs: pack INT8 mask -> INT32, compute position IDs, select token indices, context lengths.
    {
        int32_t const verifyTreeSize = mDeployment.eagle->verifyTreeSize;
        rt::Tensor const& baseKVCacheLengths = (*mSharedResources->cacheManagers[0]).getKVCacheLengths();
        check::check(
            mPipelineIO->selectTokenIndices.reshape({activeBatchSize, verifyTreeSize}), "Tensor reshape failed");
        check::check(mPipelineIO->contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mPipelineIO->eaglePositionIds.reshape({activeBatchSize, verifyTreeSize}), "Tensor reshape failed");
        kernel::prepareEagleBaseTreeDecodingInputs(mDraftTreeMask, baseKVCacheLengths, mPipelineIO->packedAttentionMask,
            mPipelineIO->eaglePositionIds, mPipelineIO->selectTokenIndices, mPipelineIO->contextLengths,
            context.stream);
    }

    // Tree verify has no real deepstack features — the vision contribution was
    // already absorbed into base hidden states during prefill. Rebind to the
    // shared {1, 1, H} zero broadcast so the engine's elementwise add is a no-op.
    if (mDeepstack)
    {
        mDeepstack->useZeroTarget(mBaseTensorMap);
    }

    // MTP: reshape recurrent/conv intermediate state outputs to match runtime dims before execute.
    // TRT writes these contiguously as [activeBatchSize, verifyTreeSize, ...]. No-op when MTP disabled.
    mambaMgr.reshapeIntermediateStates(activeBatchSize, mDeployment.eagle->verifyTreeSize);

    // Execute base tree decoding via EngineExecutor (uses kDecodeProfile, NOT kPrefillProfile).
    // kvcache_start_index is a static binding; treeVerifyDims sets
    // InferenceDims::startIndexLen = batch (base KV cache non-empty after prefill).
    auto const treeDims = mDeployment.base.treeVerifyDims(activeBatchSize, mDeployment.eagle->verifyTreeSize);
    bool verifySuccess = mBaseExecutor->prepare(kDecodeProfile, treeDims, mBaseTensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mBaseExecutor->execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("Failed to execute base tree verification step for base model.");
        return false;
    }

    // Reshape accepted token tensors to match activeBatchSize for dynamic batching.
    int32_t const maxAcceptDepth = mDeployment.eagle->draftingStep + 1;
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    // Eagle accept: collects accepted token ids and indices; uses sampling workspace as scratch.
    rt::OptionalInputTensor vocabMappingTable
        = (mDeployment.base.reducedVocabSize > 0) ? std::optional{std::ref(mBaseVocabMappingTable)} : std::nullopt;
    kernel::eagleAccept(mPipelineIO->outputLogits, mIdsInput, mDraftTreeMask, mAcceptedTokenIds, mAcceptedTokenIndices,
        mAcceptLength, vocabMappingTable, mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(),
        context.stream);

    // Inplace update KVCache and base hidden states from accepted indices, commit new KV lengths.
    rt::Tensor const& kvCacheLengths = cacheMgrBase.getKVCacheLengths();
    auto& kvMgrBase = cacheMgrBase.getKVCacheManager();

    // The EAGLE base verify is per-head-dim-group batched: one launch per group covers
    // every layer in that group, addressing per-layer storage through a device-resident
    // KVLayerInfo array owned by HybridCacheManager. Uniform models (every model that goes
    // through EAGLE today) yield a single group; hybrid Gemma4-style layouts would yield
    // one group per distinct headDim with no further code change.
    auto const kvHeadDimGroups = cacheMgrBase.getKVHeadDimGroups();
    auto const kvCacheType = kvMgrBase.getConfig().kvCacheType;

    // Reshape input hidden states from 2D [batch*verify_tree_size, hidden_dim] to 3D
    // [batch, verify_tree_size, hidden_dim]
    check::check(mPipelineIO->baseHiddenStates.reshape(
                     {activeBatchSize, mDeployment.eagle->verifyTreeSize, baseOutputHiddenDim}),
        "Tensor reshape failed");

    // INPLACE updates:
    //   - eagleBaseCommitKVCache rewrites each layer's KV cache so accepted tokens occupy
    //     contiguous slots starting at pastKvCacheLength.
    //   - eagleBaseAssembleHiddenState compacts the hidden-state buffer:
    //       Before: [Batch0: Token0...Token59][Batch1: Token0...Token59]...   (stride=60 per batch)
    //       After:  [Batch0: Sel0...Sel6][Batch1: Sel0...Sel6]...              (stride=maxAcceptDepth)
    for (auto const& group : kvHeadDimGroups)
    {
        kernel::eagleBaseCommitKVCache(mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, group.deviceLayerInfos,
            group.numLayers, group.headDim, group.maxKVHeads, activeBatchSize, maxAcceptDepth, kvCacheType,
            context.stream);
    }

    // Hidden-state assembly is not idempotent across repeated calls (it compacts in place), so
    // run it exactly once — separately from the per-head-dim-group KV commit loop above.
    kernel::eagleBaseAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, mPipelineIO->baseHiddenStates, context.stream);

    cacheMgrBase.commitSequenceLength(mAcceptLength, context.stream);

    // MTP: roll back recurrent/conv states to last accepted step. No-op when MTP is disabled.
    mambaMgr.scatterMtpStates(mAcceptLength, context.stream);

    // Reshape to the compacted layout [batch, maxAcceptDepth, hiddenDim].
    check::check(mPipelineIO->baseHiddenStates.reshape({activeBatchSize, maxAcceptDepth, baseOutputHiddenDim}),
        "Tensor reshape failed");

    // Pull collected results from device to host pinned memory for all batches.
    check::check(mHostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mHostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* hostAcceptLengthsData = mHostAcceptLengths.dataPointer<int32_t>();
    int32_t* hostAcceptedTokenIdsData = mHostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengthsData, mAcceptLength.rawPointer(), activeBatchSize * sizeof(int32_t),
        cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(hostAcceptedTokenIdsData, mAcceptedTokenIds.rawPointer(),
        activeBatchSize * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Update tokenIds and check for EOS for each batch.
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int32_t const acceptLength = hostAcceptLengthsData[batchIdx];
        for (int32_t i = 0; i < acceptLength; i++)
        {
            int32_t const token = hostAcceptedTokenIdsData[batchIdx * maxAcceptDepth + i];
            context.tokenIds[batchIdx].push_back(token);
            context.currentGenerateLengths[batchIdx]++;

            // Abandon tokens after EOS.
            if (token == mTokenizer->getEosId())
            {
                break;
            }
        }
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runVanillaDecoding(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_GENERATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_vanilla_decoding,
        ("VANILLA_DECODING[R" + std::to_string(context.generationRound) + "," + std::to_string(context.activeBatchSize)
            + "]")
            .c_str(),
        nvtx_colors::BLUE);

    // runVanillaDecoding is only invoked in the non-spec-decode branch of handleRequest; no hasDraftModel()
    // gate is needed. Fully routed through the new composable stack.
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(mHostPackedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mHostPackedTokenIds.dataPointer<int32_t>();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const lastTokenId = context.tokenIds[i].back();
        hostPackedTokenIdsData[i] = lastTokenId;
    }

    check::check(mIdsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), mHostPackedTokenIds.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(
        mPipelineIO->inputsEmbeds.reshape({activeBatchSize, 1, mDeployment.base.hiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(), mPipelineIO->inputsEmbeds, context.stream);

    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
        "Tensor reshape failed");

    // Dispatch per-step sequence prep (contextLengths, selectTokenIndices).
    mStepPreparer->prepare(
        InferencePhase::kDecode, activeBatchSize, (*mSharedResources->cacheManagers[0]), *mPipelineIO, context.stream);
    // Bind zero-broadcast deepstack for non-prefill phases.
    if (mDeepstack)
    {
        mDeepstack->useZeroTarget(mBaseTensorMap);
    }

    // Execute vanilla decoding via EngineExecutor + KV commit (+1 per decode step).
    auto const decodeDims = mDeployment.base.decodeDims(activeBatchSize);
    bool decodingStatus = mBaseExecutor->prepare(kDecodeProfile, decodeDims, mBaseTensorMap, context.stream);
    if (decodingStatus)
    {
        decodingStatus = mBaseExecutor->execute(context.stream);
    }
    if (decodingStatus)
    {
        (*mSharedResources->cacheManagers[0]).commitSequenceLength(/*increment=*/1, context.stream);
    }
    if (!decodingStatus)
    {
        LOG_ERROR("Failed to execute vanilla decoding step for base model.");
        return false;
    }

    // Use topKtopPSampling when sampling params differ from greedy defaults (temperature=1.0, topK<=1, topP=1.0).
    // Temperature <= 1e-3 is treated as greedy to avoid softmax numerical instability.
    check::check(mSamplingIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mDeployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(
            mPipelineIO->outputLogits, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        // Greedy decoding (temperature ~= 0 or default)
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mPipelineIO->outputLogits, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace,
            context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary
    if (mDeployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mSamplingIndices, mBaseVocabMappingTable, context.stream);
    }

    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Update tokenIds and generation length for each sequence
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.tokenIds[i].push_back(hostSelectedTokenIdsData[i]);
        context.currentGenerateLengths[i] += 1;

        if (context.onTokenGenerated.has_value())
        {
            bool const isFinished = context.finishedStates[i] != 0;
            TokenCallbackInfo info{hostSelectedTokenIdsData[i], i, context.generationRound, isFinished};
            context.onTokenGenerated.value()(info);
        }
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelAcceptToken(SpecDecodeInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mDeployment.eagle.has_value());
    assert(mDeployment.draft.has_value());
    NVTX_SCOPED_RANGE(nvtx_draft_accept,
        ("EAGLE_DRAFT_ACCEPT[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::YELLOW);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mDeployment.eagle->draftHiddenSize;
    int32_t const draftVocabSize = mDeployment.draft->outputVocabSize;

    // Base verification produces [activeBatchSize, accepted_length, hidden_dim] in baseHiddenStates.
    int64_t const inputIdsLength = mPipelineIO->baseHiddenStates.getShape()[1];

    check::check(mIdsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(mPipelineIO->draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(
        mPipelineIO->draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}), "Tensor reshape failed");

    // Clear garbage in draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(mPipelineIO->draftHiddenStatesIn.rawPointer(), 0,
        mPipelineIO->draftHiddenStatesIn.getMemoryCapacity(), context.stream));

    // Copy accepted token ids into mIdsInput (first inputIdsLength per batch).
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int32_t*>(mIdsInput.rawPointer()) + i * inputIdsLength,
            static_cast<int32_t*>(mAcceptedTokenIds.rawPointer()) + i * mAcceptedTokenIds.getShape()[1],
            inputIdsLength * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    }

    // Embedding lookup (text-only).
    check::check(
        mPipelineIO->inputsEmbeds.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        mIdsInput, mEmbedding.table, mEmbedding.scalesAsOptional(), mPipelineIO->inputsEmbeds, context.stream);

    // Prepare EAGLE inputs: packed causal mask, position IDs, select token indices, context lengths.
    {
        int32_t const acceptedTokenNum = static_cast<int32_t>(inputIdsLength);
        rt::Tensor const& draftKVCacheLengths = (*mSharedResources->cacheManagers[1]).getKVCacheLengths();
        check::check(mPipelineIO->selectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        check::check(mPipelineIO->contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(
            mPipelineIO->eaglePositionIds.reshape({activeBatchSize, acceptedTokenNum}), "Tensor reshape failed");
        check::check(mPipelineIO->packedAttentionMask.reshape(
                         {activeBatchSize, acceptedTokenNum, static_cast<int64_t>(divUp(acceptedTokenNum, 32))}),
            "Tensor reshape failed");
        kernel::prepareEagleAcceptDecodeTokenInputs(draftKVCacheLengths, mAcceptLength,
            mPipelineIO->packedAttentionMask, mPipelineIO->eaglePositionIds, mPipelineIO->selectTokenIndices,
            mPipelineIO->contextLengths, context.stream);
    }

    // After round 0, draft KV cache is non-empty; acceptDims sets
    // InferenceDims::startIndexLen = batch, so the static binding carries the
    // right shape — no rebind.
    auto const acceptDims = mDeployment.draft->acceptDims(activeBatchSize, inputIdsLength);
    check::check(mDraftExecutor->prepare(kDecodeProfile, acceptDims, mDraftTensorMap, context.stream),
        "Failed to prepare draft model for accept token step.");
    check::check(mDraftExecutor->execute(context.stream), "Failed to execute draft model for accept token step.");
    (*mSharedResources->cacheManagers[1]).commitSequenceLength(mAcceptLength, context.stream);

    return true;
}

bool LLMInferenceSpecDecodeRuntime::captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream)
{
    auto captureOnce = [&](std::string const& loraName) -> bool {
        if (mSharedResources->loraManager)
        {
            if (loraName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(loraName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        if (!mBaseExecutor->prepare(kDecodeProfile, dims, mBaseTensorMap, stream))
        {
            return false;
        }
        return mBaseExecutor->captureGraph(stream);
    };

    bool ok = captureOnce(mEmptyLoraWeightsName);
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        for (auto const& loraWeightsName : mSharedResources->loraManager->getAdapterNames())
        {
            ok &= captureOnce(loraWeightsName);
        }
    }
    return ok;
}

bool LLMInferenceSpecDecodeRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool draftAcceptCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};
    bool baseVanillaDecodingCaptureStatus{true};

    bool const hasDraft = hasDraftModel();

    // EAGLE: simulate KV cache state so warmup enqueueV3 doesn't write out-of-bounds. Without valid
    // KV cache lengths, applyRopeWriteKV computes negative indices.
    static constexpr int32_t kSimulateCacheLength{128};

    // RAII scope guard to restore KV cache state + draft tensor map on scope exit (normal or exception).
    auto const restoreState = [&]() noexcept {
        if (!hasDraft)
        {
            return;
        }
        std::vector<int32_t> zeroCacheLens(mMaxRuntimeBatchSize, 0);
        rt::Tensor zeroCacheLensTensor(
            zeroCacheLens.data(), {mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        (*mSharedResources->cacheManagers[0]).resetForNewSequences(zeroCacheLensTensor, stream);
        (*mSharedResources->cacheManagers[1]).resetForNewSequences(zeroCacheLensTensor, stream);

        // Reset base runner context to clean state so no EAGLE tree state lingers for vanilla runs.
        // Lambda is noexcept so we can't throw on failure — just log. Downstream graph-capture
        // teardown continues regardless; a failure here most likely means TRT is already in a bad
        // state (e.g. cuBLAS handle freed) and the higher-level error path will surface it.
        if (!mBaseExecutor->prepare(kDecodeProfile, mDeployment.base.resetDims(), mBaseTensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    };

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
    } stateGuard{restoreState};

    int32_t const draftTopK = hasDraft ? mDeployment.eagle->draftingTopK : 1;
    int32_t const paddedDraftTreeSize = hasDraft ? mDeployment.eagle->draftingStep * draftTopK : 1;
    int32_t const draftingStep = hasDraft ? mDeployment.eagle->draftingStep : 1;
    int32_t const draftHiddenSize = hasDraft ? mDeployment.eagle->draftHiddenSize : 0;
    int32_t const baseOutputHiddenDim = mDeployment.eagle.has_value() ? mDeployment.eagle->baseOutputHiddenDim : 0;
    int32_t const draftVocabSize = hasDraft ? mDeployment.draft->outputVocabSize : 0;

    for (int32_t batchSize = 1; batchSize <= mMaxRuntimeBatchSize; ++batchSize)
    {
        if (hasDraft)
        {
            // Simulate KV cache state (both base and draft caches have kSimulateCacheLength entries).
            {
                std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
                rt::Tensor simCacheLensTensor(
                    simCacheLens.data(), {batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

                (*mSharedResources->cacheManagers[0]).resetForNewSequences(simCacheLensTensor, stream);
                (*mSharedResources->cacheManagers[1]).resetForNewSequences(simCacheLensTensor, stream);

                // Set context_lengths to simulated values so RoPE indices are valid.
                std::vector<int32_t> simCtxLens(batchSize, kSimulateCacheLength + paddedDraftTreeSize);
                CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->contextLengths.rawPointer(), simCtxLens.data(),
                    batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            }

            // --- Draft proposal capture ---
            {
                check::check(
                    mPipelineIO->baseHiddenStates.reshape({batchSize, paddedDraftTreeSize, baseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(
                    mPipelineIO->draftHiddenStatesIn.reshape({batchSize, paddedDraftTreeSize, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(mDraftTreeSize.reshape({batchSize}), "Tensor reshape failed");
                check::check(mDraftTreeMask.reshape({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}),
                    "Tensor reshape failed");
                check::check(mPipelineIO->packedAttentionMask.reshape({batchSize, paddedDraftTreeSize,
                                 static_cast<int64_t>(divUp(paddedDraftTreeSize, 32))}),
                    "Tensor reshape failed");
                check::check(
                    mPipelineIO->outputLogits.reshape({batchSize, draftTopK, draftVocabSize}), "Tensor reshape failed");
                check::check(mPipelineIO->draftHiddenStatesOut.reshape({batchSize, draftTopK, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(mPipelineIO->inputsEmbeds.reshape({batchSize, paddedDraftTreeSize, draftHiddenSize}),
                    "Tensor reshape failed");

                // Warmup EAGLE kernel outputs so bound tensors have valid values during capture.
                {
                    rt::Tensor const& draftKVCacheLengths = (*mSharedResources->cacheManagers[1]).getKVCacheLengths();
                    check::check(
                        mPipelineIO->selectTokenIndices.reshape({batchSize, draftTopK}), "Tensor reshape failed");
                    check::check(mPipelineIO->eaglePositionIds.reshape({batchSize, paddedDraftTreeSize}),
                        "Tensor reshape failed");
                    kernel::prepareEagleDraftProposalInputs(mDraftTreeMask, mDraftTreeSize, draftKVCacheLengths,
                        mPipelineIO->packedAttentionMask, mPipelineIO->eaglePositionIds,
                        mPipelineIO->selectTokenIndices, mPipelineIO->contextLengths, stream);
                }

                auto const proposalDims = mDeployment.draft->proposalDims(batchSize, paddedDraftTreeSize, draftTopK);
                if (mDraftExecutor->prepare(kDecodeProfile, proposalDims, mDraftTensorMap, stream))
                {
                    draftProposalCaptureStatus &= mDraftExecutor->captureGraph(stream);
                }
                else
                {
                    draftProposalCaptureStatus = false;
                }
            }

            // --- Draft accept decode token capture ---
            {
                check::check(mPipelineIO->outputLogits.reshape({batchSize, draftVocabSize}), "Tensor reshape failed");
                check::check(
                    mPipelineIO->draftHiddenStatesOut.reshape({batchSize, draftHiddenSize}), "Tensor reshape failed");

                for (int32_t acceptLength = 1; acceptLength <= draftingStep + 1; acceptLength++)
                {
                    check::check(mPipelineIO->baseHiddenStates.reshape({batchSize, acceptLength, baseOutputHiddenDim}),
                        "Tensor reshape failed");
                    check::check(mPipelineIO->draftHiddenStatesIn.reshape({batchSize, acceptLength, draftHiddenSize}),
                        "Tensor reshape failed");
                    check::check(mAcceptLength.reshape({batchSize}), "Tensor reshape failed");
                    std::vector<int32_t> acceptLengthsVec(batchSize, acceptLength);
                    CUDA_CHECK(cudaMemcpyAsync(mAcceptLength.rawPointer(), acceptLengthsVec.data(),
                        batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    check::check(mPipelineIO->inputsEmbeds.reshape({batchSize, acceptLength, draftHiddenSize}),
                        "Tensor reshape failed");
                    check::check(mPipelineIO->packedAttentionMask.reshape(
                                     {batchSize, acceptLength, static_cast<int64_t>(divUp(acceptLength, 32))}),
                        "Tensor reshape failed");

                    {
                        rt::Tensor const& draftKVCacheLengths
                            = (*mSharedResources->cacheManagers[1]).getKVCacheLengths();
                        check::check(mPipelineIO->selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");
                        check::check(mPipelineIO->contextLengths.reshape({batchSize}), "Tensor reshape failed");
                        check::check(
                            mPipelineIO->eaglePositionIds.reshape({batchSize, acceptLength}), "Tensor reshape failed");
                        kernel::prepareEagleAcceptDecodeTokenInputs(draftKVCacheLengths, mAcceptLength,
                            mPipelineIO->packedAttentionMask, mPipelineIO->eaglePositionIds,
                            mPipelineIO->selectTokenIndices, mPipelineIO->contextLengths, stream);
                    }

                    auto const acceptDims = mDeployment.draft->acceptDims(batchSize, acceptLength);
                    if (mDraftExecutor->prepare(kDecodeProfile, acceptDims, mDraftTensorMap, stream))
                    {
                        draftAcceptCaptureStatus &= mDraftExecutor->captureGraph(stream);
                    }
                    else
                    {
                        draftAcceptCaptureStatus = false;
                    }
                }
            }

            // --- Base verification capture ---
            {
                int32_t const verifyTreeSize = mDeployment.eagle->verifyTreeSize;
                int32_t const selectTokenSize = batchSize * verifyTreeSize;
                check::check(mPipelineIO->outputLogits.reshape({selectTokenSize, mDeployment.base.outputVocabSize}),
                    "Tensor reshape failed");
                check::check(mPipelineIO->baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(
                    mDraftTreeMask.reshape({batchSize, verifyTreeSize, verifyTreeSize}), "Tensor reshape failed");
                check::check(mPipelineIO->packedAttentionMask.reshape(
                                 {batchSize, verifyTreeSize, static_cast<int64_t>(divUp(verifyTreeSize, 32))}),
                    "Tensor reshape failed");
                check::check(
                    mPipelineIO->inputsEmbeds.reshape({batchSize, verifyTreeSize, mDeployment.base.hiddenSize}),
                    "Tensor reshape failed");

                {
                    rt::Tensor const& baseKVCacheLengths = (*mSharedResources->cacheManagers[0]).getKVCacheLengths();
                    check::check(
                        mPipelineIO->selectTokenIndices.reshape({batchSize, verifyTreeSize}), "Tensor reshape failed");
                    check::check(mPipelineIO->contextLengths.reshape({batchSize}), "Tensor reshape failed");
                    check::check(
                        mPipelineIO->eaglePositionIds.reshape({batchSize, verifyTreeSize}), "Tensor reshape failed");
                    kernel::prepareEagleBaseTreeDecodingInputs(mDraftTreeMask, baseKVCacheLengths,
                        mPipelineIO->packedAttentionMask, mPipelineIO->eaglePositionIds,
                        mPipelineIO->selectTokenIndices, mPipelineIO->contextLengths, stream);
                }

                // Tree-verify bindings are already populated by
                // `prepareEagleBaseTreeDecodingInputs` above — do NOT call
                // StepPreparer here (its Decode branch would clobber
                // selectTokenIndices and contextLengths). Just swap deepstack
                // to the zero-broadcast binding.
                if (mDeepstack)
                {
                    mDeepstack->useZeroTarget(mBaseTensorMap);
                }

                auto const treeDims = mDeployment.base.treeVerifyDims(batchSize, verifyTreeSize);
                baseVerificationCaptureStatus &= captureBaseGraphWithLoraFanout(treeDims, stream);
            }
        }

        // --- Base vanilla decoding capture (always needed; same path for vanilla and EAGLE runtimes) ---
        {
            check::check(mPipelineIO->inputsEmbeds.reshape({batchSize, 1, mDeployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mPipelineIO->outputLogits.reshape({batchSize, mDeployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mPipelineIO->selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");

            mStepPreparer->prepare(
                InferencePhase::kDecode, batchSize, (*mSharedResources->cacheManagers[0]), *mPipelineIO, stream);
            if (mDeepstack)
            {
                mDeepstack->useZeroTarget(mBaseTensorMap);
            }

            auto const decodeDims = mDeployment.base.decodeDims(batchSize);
            baseVanillaDecodingCaptureStatus &= captureBaseGraphWithLoraFanout(decodeDims, stream);
        }
    }

    bool const captureStatus = draftProposalCaptureStatus && draftAcceptCaptureStatus && baseVerificationCaptureStatus
        && baseVanillaDecodingCaptureStatus;
    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for all stages.");
    }
    else
    {
        LOG_WARNING(
            "Failed to capture decoding CUDA graphs for some stages. The inference can proceed without "
            "CUDA graph capture, but at cost of performance degradation.");
    }

    return captureStatus;
}

void LLMInferenceSpecDecodeRuntime::restoreRecurrentStates(
    int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;

        if (layer < static_cast<int32_t>(cachedStates.recurrentStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(recurrentDst, cachedStates.recurrentStateContents[layer].rawPointer(),
                recurrentBatchBytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        }

        if (layer < static_cast<int32_t>(cachedStates.convStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(convDst, cachedStates.convStateContents[layer].rawPointer(), convBatchBytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
        }
    }
}

void LLMInferenceSpecDecodeRuntime::zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;
        CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
    }
}

bool LLMInferenceSpecDecodeRuntime::setUpForPrefillExecution(SpecDecodeInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    // LoRA switching goes through the LoRAManager on SharedResources.
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        try
        {
            if (context.loraWeightsName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(context.loraWeightsName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to switch LoRA weights to %s: %s", context.loraWeightsName.c_str(), e.what());
            return false;
        }
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    std::vector<std::vector<int32_t>> const& batchedInputIds = context.rawBatchedInputIds;
    bool const hasDraft = hasDraftModel();
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];

    // Record the length of the reused KVCache for each sequence.
    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);

    context.tokenIds.clear();
    context.tokenIds.resize(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const& prompt = context.systemPrompts[i];
        auto const promptKey = keySystemPromptWithLoraWeights(prompt, context.loraWeightsName);
        if (mSystemPromptKVCacheBase.count(promptKey) > 0)
        {
            auto& precachedKVCacheBase = mSystemPromptKVCacheBase[promptKey];
            auto const& kvCacheLayersBase = precachedKVCacheBase.kvCacheLayers;
            cacheMgrBase.restoreKVCache(kvCacheLayersBase, i, context.stream);

            if (hasDraft)
            {
                check::check(mSystemPromptKVCacheDraft.count(promptKey) > 0,
                    "System prompt cache inconsistency between base and draft model");
                auto& precachedKVCacheDraft = mSystemPromptKVCacheDraft[promptKey];
                auto const& kvCacheLayersDraft = precachedKVCacheDraft.kvCacheLayers;
                auto& cacheMgrDraft = *mSharedResources->cacheManagers[1];
                cacheMgrDraft.restoreKVCache(kvCacheLayersDraft, i, context.stream);
            }

            // Restore recurrent/conv states for hybrid models (vanilla only — EAGLE bundles never hybrid).
            if (mDeployment.base.numLinearAttnLayers > 0)
            {
                restoreRecurrentStates(i, precachedKVCacheBase, context.stream);
            }

            // Per-layer saved KV tensor shape is [2, numKVHeads, sequenceLength, headDim]; shape[2] == seqLen.
            check::check(!kvCacheLayersBase.empty(), "System prompt KV cache must have at least one layer.");
            auto reuseLength = math::cast<size_t>(kvCacheLayersBase[0].getShape()[2]);
            check::check(reuseLength > 0 && reuseLength < batchedInputIds[i].size(),
                "The reuse length shall be larger than 0 and not exceed the input length.");
            // Reuse N-1 tokens from the cached prefix so the Nth token is treated as real input in prefill;
            // this keeps the draft prefill boundary aligned with the true next-token position.
            auto const effectiveReuseLength = reuseLength - 1;
            reuseKVCacheLengthsData[i] = math::cast<int32_t>(effectiveReuseLength);

            context.tokenIds[i].assign(batchedInputIds[i].begin() + effectiveReuseLength, batchedInputIds[i].end());
            context.effectivePrefillLengths[i] = math::cast<int32_t>(batchedInputIds[i].size() - effectiveReuseLength);

            bool const matchIds = std::equal(precachedKVCacheBase.tokenizedPrompt.begin(),
                precachedKVCacheBase.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "Though system prompt strings are matched, token_ids are not perfectly aligned."
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            context.tokenIds[i] = batchedInputIds[i];
            context.effectivePrefillLengths[i] = static_cast<int32_t>(batchedInputIds[i].size());
            reuseKVCacheLengthsData[i] = 0;

            if (mDeployment.base.numLinearAttnLayers > 0)
            {
                zeroRecurrentStates(i, context.stream);
            }
        }
    }

    int32_t const maxInputLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    if (maxInputLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("The max input length (%d) exceeds the max supported input length (%d) of the LLM Engine.",
            maxInputLength, mDeployment.base.maxSupportedInputLength);
        return false;
    }

    cacheMgrBase.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    if (hasDraft)
    {
        auto& cacheMgrDraft = *mSharedResources->cacheManagers[1];
        cacheMgrDraft.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    }
    return true;
}

bool LLMInferenceSpecDecodeRuntime::genAndSaveSystemPromptKVCache(
    SpecDecodeInferenceContext& context, int32_t genAndSaveBatchIdx)
{
    std::string const& loraWeightsName = context.loraWeightsName;
    std::string const prompt = context.systemPrompts[genAndSaveBatchIdx];
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);

    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    bool const hasDraft = hasDraftModel();
    if (mSystemPromptKVCacheBase.find(promptKey) != mSystemPromptKVCacheBase.end()
        && (!hasDraft || mSystemPromptKVCacheDraft.find(promptKey) != mSystemPromptKVCacheDraft.end()))
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());

    if (promptIdsLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (base=%d)", promptIdsLength,
            mDeployment.base.maxSupportedInputLength);
        return false;
    }

    if (hasDraft && promptIdsLength > mDeployment.draft->maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (draft=%d)", promptIdsLength,
            mDeployment.draft->maxSupportedInputLength);
        return false;
    }

    // Temporary single-batch context to reuse the existing prefill functions.
    SpecDecodeInferenceContext tempContext;
    tempContext.initialize(1, 1, context.visualEmbeddings, context.deepstackFeatures, loraWeightsName, context.stream);
    tempContext.systemPrompts[0] = prompt;
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;

    if (!setUpForPrefillExecution(tempContext))
    {
        LOG_ERROR("Prefill execution setup failed for system prompt KVCache generation.");
        return false;
    }

    bool prefillStatus = runBaseModelPrefill(tempContext);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute base model prefill for system prompt KVCache generation.");
        return false;
    }

    // Tokens produced during system KV-cache reuse prefill do not count as generated tokens.
    tempContext.currentGenerateLengths[0] -= 1;

    if (hasDraft)
    {
        bool draftPrefillStatus = runDraftModelPrefill(tempContext);
        if (!draftPrefillStatus)
        {
            LOG_ERROR("Failed to execute draft model prefill for system prompt KVCache generation.");
            return false;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Capture base KV cache content from the new-stack shared KV cache.
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    constexpr int32_t CACHE_BATCH_IDX{0};

    SystemPromptKVCache savedKVCacheBase;
    savedKVCacheBase.systemPrompt = prompt;
    savedKVCacheBase.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheBase.kvCacheLayers = cacheMgrBase.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, context.stream);

    // Save recurrent / conv states for hybrid layers.
    if (mDeployment.base.numLinearAttnLayers > 0)
    {
        savedKVCacheBase.recurrentStateContents = cacheMgrBase.captureRecurrentStates(CACHE_BATCH_IDX, context.stream);
        savedKVCacheBase.convStateContents = cacheMgrBase.captureConvStates(CACHE_BATCH_IDX, context.stream);
    }

    mSystemPromptKVCacheBase.insert({promptKey, std::move(savedKVCacheBase)});

    if (hasDraft)
    {
        auto& cacheMgrDraft = *mSharedResources->cacheManagers[1];

        SystemPromptKVCache savedKVCacheDraft;
        savedKVCacheDraft.systemPrompt = prompt;
        savedKVCacheDraft.tokenizedPrompt = tokenizedPrompt;
        savedKVCacheDraft.kvCacheLayers
            = cacheMgrDraft.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, context.stream);
        mSystemPromptKVCacheDraft.insert({promptKey, std::move(savedKVCacheDraft)});
    }

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    LOG_DEBUG("System prompt KVCache saved for batch %d: {%s}", genAndSaveBatchIdx, prompt.c_str());

    return true;
}

bool LLMInferenceSpecDecodeRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCacheBase.find(promptKey) != mSystemPromptKVCacheBase.end())
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }
    SpecDecodeInferenceContext tempContext;
    tempContext.initialize(1, 1, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    tempContext.systemPrompts[0] = prompt;
    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;
    return genAndSaveSystemPromptKVCache(tempContext, 0);
}

bool LLMInferenceSpecDecodeRuntime::performBatchEvict(SpecDecodeInferenceContext& context)
{
    // Check if any batch has finished
    bool hasFinishedBatch = false;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            hasFinishedBatch = true;
            break;
        }
    }

    if (!hasFinishedBatch)
    {
        return true;
    }

    int32_t const oldActiveBatch = context.activeBatchSize;

    // Build batch mapping
    std::vector<int32_t> batchMapping = buildBatchMapping(context.finishedStates);

    // Calculate new active batch size
    int32_t newActiveBatch = 0;
    for (auto newIdx : batchMapping)
    {
        if (newIdx >= 0)
        {
            newActiveBatch = std::max(newActiveBatch, newIdx + 1);
        }
    }

    // Log eviction details
    std::vector<int32_t> evictedIndices;
    for (int32_t i = 0; i < oldActiveBatch; ++i)
    {
        if (batchMapping[i] < 0)
        {
            evictedIndices.push_back(i);
        }
    }
    LOG_DEBUG("Batch eviction: %d active batches to %d remaining (evicted %d batch(es): indices [%s])", oldActiveBatch,
        newActiveBatch, static_cast<int32_t>(evictedIndices.size()),
        [&evictedIndices]() {
            std::string result;
            for (size_t i = 0; i < evictedIndices.size(); ++i)
            {
                if (i > 0)
                {
                    result += ", ";
                }
                result += std::to_string(evictedIndices[i]);
            }
            return result;
        }()
            .c_str());

    // Upload batch mapping to GPU
    check::check(mDeviceBatchMapping.reshape({oldActiveBatch}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(), oldActiveBatch * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    // Compact base model caches (KV + Mamba) via the HybridCacheManager single-call API.
    auto& baseCacheMgr = *mSharedResources->cacheManagers[0];
    baseCacheMgr.compactBatch(mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
    baseCacheMgr.setActiveBatchSize(newActiveBatch);

    bool const hasDraft = hasDraftModel();
    if (hasDraft)
    {
        auto& draftCacheMgr = *mSharedResources->cacheManagers[1];
        draftCacheMgr.compactBatch(mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
        draftCacheMgr.setActiveBatchSize(newActiveBatch);

        // Compact draft model's RoPE CosSin cache if it's per-batch (MRope). Draft model shares the
        // shared RoPE pool; if MRope is enabled, the draft config owns a distinct entry there.
        if (mDeployment.draft->ropeConfig.type == RopeType::kMRope)
        {
            rt::Tensor& draftRopeCache = mSharedResources->ropePool.getOrCreate(mDeployment.draft->ropeConfig,
                mDeployment.draft->rotaryDim, mDeployment.draft->maxKVCacheCapacity, nullptr);
            if (draftRopeCache.getShape().getNumDims() == 3 && draftRopeCache.getShape()[0] == oldActiveBatch
                && newActiveBatch > 0)
            {
                kernel::compactTensorBatch(draftRopeCache, mDeviceBatchMapping, draftRopeCache, oldActiveBatch,
                    newActiveBatch, context.stream);
                auto const seqLen = static_cast<int32_t>(draftRopeCache.getShape()[1]);
                auto const rotaryDim = static_cast<int32_t>(draftRopeCache.getShape()[2]);
                check::check(draftRopeCache.reshape({newActiveBatch, seqLen, rotaryDim}), "Tensor reshape failed");
            }
        }
    }

    // Compact base model's RoPE cache (stored per-batch for MRope on mPipelineIO->mropeCosSin).
    if (mDeployment.base.ropeConfig.type == RopeType::kMRope && newActiveBatch > 0)
    {
        rt::Tensor& baseRopeCache = mPipelineIO->mropeCosSin;
        if (baseRopeCache.getShape().getNumDims() == 3 && baseRopeCache.getShape()[0] == oldActiveBatch)
        {
            kernel::compactTensorBatch(
                baseRopeCache, mDeviceBatchMapping, baseRopeCache, oldActiveBatch, newActiveBatch, context.stream);
            auto const seqLen = static_cast<int32_t>(baseRopeCache.getShape()[1]);
            auto const rotaryDim = static_cast<int32_t>(baseRopeCache.getShape()[2]);
            check::check(baseRopeCache.reshape({newActiveBatch, seqLen, rotaryDim}), "Tensor reshape failed");
        }
    }

    // Compact cross-round GPU tensors that are read (not just written) in the next round.

    // 1. baseHiddenStates: read by runDraftModelAcceptToken in next round. [batch, maxAcceptDepth, baseHiddenDim]
    if (mPipelineIO->baseHiddenStates.getShape().getNumDims() == 3
        && mPipelineIO->baseHiddenStates.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(mPipelineIO->baseHiddenStates, mDeviceBatchMapping, mPipelineIO->baseHiddenStates,
            oldActiveBatch, newActiveBatch, context.stream);
        auto const dim1 = static_cast<int32_t>(mPipelineIO->baseHiddenStates.getShape()[1]);
        auto const dim2 = static_cast<int32_t>(mPipelineIO->baseHiddenStates.getShape()[2]);
        check::check(mPipelineIO->baseHiddenStates.reshape({newActiveBatch, dim1, dim2}), "Tensor reshape failed");
    }

    // 2/3. Accepted token ids + accept length (draft-only, read in the next round).
    if (hasDraft)
    {
        if (mAcceptedTokenIds.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
        {
            kernel::compactTensorBatch(mAcceptedTokenIds, mDeviceBatchMapping, mAcceptedTokenIds, oldActiveBatch,
                newActiveBatch, context.stream);
            auto const maxAcceptDepth = static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]);
            check::check(mAcceptedTokenIds.reshape({newActiveBatch, maxAcceptDepth}), "Tensor reshape failed");
        }

        if (mAcceptLength.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
        {
            kernel::compactTensorBatch(
                mAcceptLength, mDeviceBatchMapping, mAcceptLength, oldActiveBatch, newActiveBatch, context.stream);
            check::check(mAcceptLength.reshape({newActiveBatch}), "Tensor reshape failed");
        }
    }

    // Compact CPU context
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Save evicted batches' results before compacting (using original batch index)
    for (size_t i = 0; i < batchMapping.size(); ++i)
    {
        if (batchMapping[i] < 0 && context.finishedStates[i])
        {
            // This batch is evicted and finished, save its results with original index
            int32_t originalIdx = context.batchIndexMapping[i];

            // Create and populate BatchResult with all related data
            BatchResult result;
            result.tokenIds = std::move(context.tokenIds[i]);
            result.generateLength = context.currentGenerateLengths[i];
            result.actualIterations = context.generationRound;
            result.rawBatchedInputIds = std::move(context.rawBatchedInputIds[i]);
            result.effectivePrefillLength = context.effectivePrefillLengths[i];
            result.terminalReason = context.slotStreams[i].terminalReason;

            context.completedBatches[originalIdx] = std::move(result);
        }
    }

    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.effectivePrefillLengths);
    rt::compactVector(batchMapping, context.batchIndexMapping);
    rt::compactVector(batchMapping, context.slotStreams);
    rt::compactVector(batchMapping, context.stopStringsPerSlot);

    // Update active batch size
    context.activeBatchSize = newActiveBatch;

    return true;
}

} // namespace rt
} // namespace trt_edgellm
