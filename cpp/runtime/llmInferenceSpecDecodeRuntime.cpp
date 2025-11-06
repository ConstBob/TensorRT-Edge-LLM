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

#include "llmInferenceSpecDecodeRuntime.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{
// Left a utility function here in case we want to move to a better hashing method.
size_t hashSystemPrompt(std::string const& systemPrompt)
{
    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, systemPrompt);
    return hashValue;
}
} // namespace

namespace rt
{

void SpecDecodeInferenceContext::initialize(int32_t _activeBatchSize, int32_t _maxGenerateLength,
    rt::OptionalInputTensor const& _mutimodalEmbeddings, cudaStream_t _stream)
{
    systemPrompts.resize(_activeBatchSize);
    rawBatchedInputIds.reserve(_activeBatchSize);
    tokenIds.resize(_activeBatchSize);
    currentGenerateLengths.resize(_activeBatchSize, 0);
    promptLengths.resize(_activeBatchSize, 0);
    finishedStates.resize(_activeBatchSize, false);
    actualIterations.resize(_activeBatchSize, 0);
    multimodalEmbeddings = _mutimodalEmbeddings;
    generationRound = 0;
    maxGenerateLength = _maxGenerateLength;
    activeBatchSize = _activeBatchSize;
    stream = _stream;
}

LLMInferenceSpecDecodeRuntime::LLMInferenceSpecDecodeRuntime(std::string const& engineDir,
    std::string const& multimodalEngineDir, EagleDraftingConfig const& draftingConfig, cudaStream_t stream)
{
    mDraftingConfig = draftingConfig;

    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "eagle_base.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "base_config.json";
    // Currently, we don't support LoRA weights along with Eagle SpecDecode.
    std::unordered_map<std::string, std::string> loraWeightsMap{};
    try
    {
        mBaseEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize LLMEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("LLMEngineRunner successfully loaded and initialized eagle base engine.");
    mBaseEngineConfig = mBaseEngineRunner->getEngineConfig();

    std::filesystem::path const draftEnginePath = std::filesystem::path(engineDir) / "eagle_draft.engine";
    std::filesystem::path const draftConfigPath = std::filesystem::path(engineDir) / "draft_config.json";
    try
    {
        mDraftEngineRunner = std::make_unique<EagleDraftEngineRunner>(draftEnginePath, draftConfigPath, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize EagleDraftEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize EagleDraftEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("EagleDraftEngineRunner successfully initialized.");
    mDraftEngineConfig = mDraftEngineRunner->getDraftEngineConfig();

    // Set runtime batch size to the minimum of base and draft engine's maxSupportedBatchSize
    // This ensures CUDA graph capture and tensor allocation work for both engines
    mMaxRuntimeBatchSize = std::min(mBaseEngineConfig.maxSupportedBatchSize, mDraftEngineConfig.maxSupportedBatchSize);
    LOG_INFO("Runtime batch size set to: %d (base engine max: %d, draft engine max: %d)", mMaxRuntimeBatchSize,
        mBaseEngineConfig.maxSupportedBatchSize, mDraftEngineConfig.maxSupportedBatchSize);

    // Validate drafting configuration against engine capabilities
    // maxDraftTreeSize controls the maximum input length to draft proposal step
    int32_t const requiredDraftInputSize = mDraftingConfig.draftingStep * mDraftingConfig.draftingTopK;
    if (requiredDraftInputSize > mDraftEngineConfig.maxDraftTreeSize)
    {
        LOG_ERROR(
            "Drafting config requires %d draft input tokens (draftingStep=%d * draftingTopK=%d) but engine supports "
            "max %d draft tree size",
            requiredDraftInputSize, mDraftingConfig.draftingStep, mDraftingConfig.draftingTopK,
            mDraftEngineConfig.maxDraftTreeSize);
        throw std::runtime_error("Drafting configuration exceeds engine draft tree size capability");
    }

    // Validate that verifyTreeSize doesn't exceed the maximum draft tree size
    if (mDraftingConfig.verifyTreeSize > mDraftEngineConfig.maxDraftTreeSize)
    {
        LOG_ERROR("Drafting config verifyTreeSize (%d) exceeds engine maxDraftTreeSize (%d)",
            mDraftingConfig.verifyTreeSize, mDraftEngineConfig.maxDraftTreeSize);
        throw std::runtime_error("Verify tree size exceeds engine maximum draft tree size");
    }

    // Allocate runtime tensors till max supported size.
    int32_t const maxDraftTreeSize = std::max(mDraftEngineConfig.maxDraftTreeSize, mDraftingConfig.verifyTreeSize);
    int32_t const draftTopK = mDraftingConfig.draftingTopK;
    // maxSamplingSize needs to account for batch dimension: max of (batchSize * verifyTreeSize) or (batchSize *
    // draftTopK * draftTopK)
    int32_t const maxSamplingSize
        = std::max(mMaxRuntimeBatchSize * maxDraftTreeSize, mMaxRuntimeBatchSize * draftTopK * draftTopK);
    int32_t const draftFullTableLength = 1 + draftTopK + (mDraftingConfig.draftingStep - 1) * draftTopK * draftTopK;

    LOG_DEBUG(
        "maxDraftTreeSize: %d, maxSamplingSize: %d, draftFullTableLength: %d to set up the SpecDecode inference "
        "runtime",
        maxDraftTreeSize, maxSamplingSize, draftFullTableLength);
    // Reserve enough workspace for sampling, accounting for batch dimension in draft proposal stage
    // In draft proposal loop, we process (batchSize * draftTopK) rows, each doing topK selection
    int32_t const maxSamplingWorkspaceSize
        = std::max(getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize, mBaseEngineConfig.vocabSize, 1),
            getSelectAllTopKWorkspaceSize(
                mMaxRuntimeBatchSize * draftTopK, mDraftEngineConfig.draftModelVocabSize, draftTopK));

    try
    {
        mIdsInput = rt::Tensor(
            {mMaxRuntimeBatchSize, mBaseEngineConfig.maxSupportedInputLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mContextLengthsInput = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
        // Allocate mLogitsOutput with max capacity to support both draft (smaller vocab) and base (larger vocab)
        // operations Max size needed: batch_size * verify_tree_size * base_vocab_size for base verification
        int32_t const maxLogitsSize = mMaxRuntimeBatchSize * maxDraftTreeSize;
        int32_t const maxVocabSize = std::max(mBaseEngineConfig.vocabSize, mDraftEngineConfig.draftModelVocabSize);
        mLogitsOutput = rt::Tensor({maxLogitsSize, maxVocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTreeSize = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTreeMask = rt::Tensor(
            {mMaxRuntimeBatchSize, maxDraftTreeSize, maxDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        mBaseHiddenStatesOutput = rt::Tensor(
            {mMaxRuntimeBatchSize, mBaseEngineConfig.maxSupportedInputLength, mBaseEngineConfig.outputHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftHiddenStatesInput = rt::Tensor(
            {mMaxRuntimeBatchSize, mBaseEngineConfig.maxSupportedInputLength, mDraftEngineConfig.draftModelHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftHiddenStatesOutput = rt::Tensor({mMaxRuntimeBatchSize, draftTopK, mDraftEngineConfig.draftModelHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftTokenIdsFullTable
            = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenScoreFullTable
            = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenPredecessorFullTable
            = rt::Tensor({mMaxRuntimeBatchSize, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftVocabMappingTable = rt::Tensor(
            {mMaxRuntimeBatchSize, mDraftEngineConfig.draftModelVocabSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTreeRootTokenId = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenIdsTable
            = rt::Tensor({mMaxRuntimeBatchSize, draftTopK * draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenScoresTable
            = rt::Tensor({mMaxRuntimeBatchSize, draftTopK * draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenIntermediateScores
            = rt::Tensor({mMaxRuntimeBatchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenIntermediateParents
            = rt::Tensor({mMaxRuntimeBatchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8);
        mSamplingIndices = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mSamplingScores = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

        // DraftModel prefill/accept-decode-token will also produce one layer of draft tree, so the max accepted
        // depth should be drafting step + 1.
        int32_t const maxAcceptDepth = mDraftingConfig.draftingStep + 1;
        mAcceptedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxAcceptDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        mAcceptedTokenIndices
            = rt::Tensor({mMaxRuntimeBatchSize, maxAcceptDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        mAcceptLength = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate runtime tensors: %s", e.what());
        throw std::runtime_error("Failed to allocate runtime tensors: " + std::string(e.what()));
    }
    LOG_INFO("Runtime tensors successfully allocated.");

    // Load conversion table from draft model vocab to base model vocab.
    std::vector<rt::Tensor> d2tTensors;
    if (!safetensors::loadSafetensors(std::filesystem::path(engineDir) / "d2t.safetensors", d2tTensors, stream))
    {
        LOG_ERROR("Failed to load d2t.safetensors from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load d2t.safetensors from model directory: " + engineDir);
    }

    // Check we have exactly one tensor and use it
    check::check(d2tTensors.size() == 1, "d2t.safetensors should contain exactly one tensor");
    check::check(d2tTensors[0].getShape().getNumDims() == 1, "d2t tensor should be 1D");
    check::check(d2tTensors[0].getShape()[0] == mDraftEngineConfig.draftModelVocabSize,
        "d2t tensor length should match draft vocab size");
    mDraftVocabMappingTable = std::move(d2tTensors[0]);

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // Check we have exactly one mImStartTokenId and use it
    // TODO: Remove this implicit usage of <|im_start|> to allow generalization.
    std::vector<int32_t> imStartTokenIdVec = mTokenizer->encode("<|im_start|>", false);
    check::check(imStartTokenIdVec.size() == 1, "imStartTokenIdVec should contain exactly one token");
    mImStartTokenId = imStartTokenIdVec[0];

    // Optional: Setup multimodal engine runner
    if (!multimodalEngineDir.empty())
    {
        try
        {
            mMultimodalRunner = MultimodalRunner::create(multimodalEngineDir, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize MultimodalRunner: %s", e.what());
            throw std::runtime_error("Failed to initialize MultimodalRunner: " + std::string(e.what()));
        }
        LOG_INFO("MultimodalRunner successfully loaded and initialized multimodal engine.");
    }
}

bool LLMInferenceSpecDecodeRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.prompts.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("Empty request with no prompts");
        return false;
    }

    if (activeBatchSize > mMaxRuntimeBatchSize)
    {
        LOG_ERROR(
            "Requested batch size %d exceeds maximum supported batch size %d", activeBatchSize, mMaxRuntimeBatchSize);
        return false;
    }

    // Use empty tensor for when no multimodal runner is available.
    // All other data input used by prefill step is already set up in setUpForPrefillExecution().
    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    int32_t maxGenerateLength = request.maxGenerateLength;

    // Initialize context for multi-batch
    SpecDecodeInferenceContext context;
    context.initialize(activeBatchSize, maxGenerateLength, multimodalEmbeddings, stream);

    // Preprocess user prompts and encode them.
    std::vector<std::vector<int32_t>> batchedInputIds;
    if (!mMultimodalRunner)
    {
        // Process each prompt in the batch
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            auto const& prompt = request.prompts[i];
            context.systemPrompts[i] = prompt.systemPrompt;

            std::string const inputText = context.systemPrompts[i] + prompt.userPrompt;
            context.rawBatchedInputIds.emplace_back(mTokenizer->encode(inputText, false));
            if (context.rawBatchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to tokenize input text for batch %d", i);
                return false;
            }
        }
    }
    else
    {
        // Process system prompts for all batches
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            // TODO: apply chat template for system prompt
            context.systemPrompts[i] = mMultimodalRunner->preprocessSystemPrompt(request.prompts[i].systemPrompt,
                mTokenizer.get(), mBaseEngineRunner->getRopeCosSinCacheTensor(), stream);
        }

        if (!mMultimodalRunner->preprocess(request, context.rawBatchedInputIds, mTokenizer.get(),
                mBaseEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR("Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("Multimodal inference failed. This request cannot be handled.");
            return false;
        }
    }

    // The boundary case for KVCache is not handled, during execution we need to write drafting KVCache.
    // Workaround the issue by enforce max input sequence length smaller than (KVCacheCapacity - 100)
    constexpr int32_t kDRAFT_KVCACHE_RESERVE_LENGTH{100};
    int32_t const perfillTokenLength = context.rawBatchedInputIds[0].size();
    int32_t const kvCacheCapacity
        = std::max(mBaseEngineConfig.maxSequenceLength, mDraftEngineConfig.kvCacheCapacityLength);
    if (perfillTokenLength + request.maxGenerateLength > (kvCacheCapacity - kDRAFT_KVCACHE_RESERVE_LENGTH))
    {
        maxGenerateLength = kvCacheCapacity - perfillTokenLength - kDRAFT_KVCACHE_RESERVE_LENGTH;
        LOG_WARNING(
            "With Eagle3, we need to write drafting KVCache which constrain us on sequence generation."
            "Reduce max Generation length to %d",
            maxGenerateLength);
    }

    // In production, the system-prompt KV cache is saved during warm-up.
    // We disable profiling here to make benchmarking closer to production inference result.
    bool profilingEnabled = getProfilingEnabled();
    if (profilingEnabled)
    {
        setProfilingEnabled(false);
    }

    // Generate system prompt KVCache for each sequence in the batch
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.currentBatchIndex = i;
        bool const saveCacheStatus = genAndSaveSystemPromptKVCache(context);
        if (!saveCacheStatus)
        {
            LOG_WARNING(
                "Failed to save system prompt KVCache for batch %d. "
                "May be KVCache reuse feature is not enabled in the engine.",
                i);
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

    // Prefill from the base model and run spec-decode inference.
    bool const prefillStatus = runBaseModelPrefill(context);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    // Lambda to check if all batches are finished
    auto checkAllFinished = [&]() {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!context.finishedStates[i])
                return false;
        }
        return true;
    };

    // Lambda to update finish states based on EOS and max_length
    auto updateFinishStates = [&]() {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (context.finishedStates[i])
                continue;

            // Check EOS
            if (!context.tokenIds[i].empty() && context.tokenIds[i].back() == mTokenizer->getEosId())
            {
                context.finishedStates[i] = true;
                LOG_DEBUG("Batch %d finished, reason: EOS", i);
                continue;
            }
            // Check max length
            if (context.currentGenerateLengths[i] >= context.maxGenerateLength)
            {
                context.finishedStates[i] = true;
                LOG_DEBUG(
                    "Batch %d finished, total tokens=%d, reason: max_length", i, context.currentGenerateLengths[i]);
                continue;
            }
        }
    };

    // Check if any batch finished immediately after prefill
    updateFinishStates();

    while (!checkAllFinished())
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

        // Update iterations and check finish conditions
        updateFinishStates();

        context.generationRound += 1;
    }

    // Record Eagle metrics - accumulate across all batches
    int32_t totalReusedTokens = 0;
    int32_t totalComputedTokens = 0;
    int32_t totalGeneratedTokens = 0;
    int32_t totalIterations = 0;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t rawPromptLength = static_cast<int32_t>(context.rawBatchedInputIds[i].size());
        int32_t computedLength = context.promptLengths[i];
        totalReusedTokens += (rawPromptLength - computedLength);
        totalComputedTokens += computedLength;
        totalGeneratedTokens += context.currentGenerateLengths[i];
        totalIterations += context.actualIterations[i];
    }
    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens);
    mEagleGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens);

    // Save output ids and decoded texts to response.
    response.outputIds.clear();
    response.outputTexts.clear();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Log acceptance metrics for this batch using its actual iterations
        // actualIterations[i] tracks how many rounds this specific sequence participated in before finishing
        // Note: Prefill generates 1 token but is not counted as an iteration.
        // To calculate true acceptance rate (tokens accepted per verification round), we need to:
        // 1. Subtract the 1 prefill token from currentGenerateLengths
        // 2. Divide by actualIterations (which counts verification rounds only)
        int32_t const verificationTokens = context.currentGenerateLengths[i] > 0
            ? context.currentGenerateLengths[i] - 1 // Subtract the 1 prefill token
            : 0;
        float const acceptanceRate = context.actualIterations[i] > 0
            ? static_cast<float>(verificationTokens) / static_cast<float>(context.actualIterations[i])
            : 0.0f;
        LOG_INFO("Batch %d - Acceptance rate: %.3f, Generated tokens: %d, Iterations: %d", i, acceptanceRate,
            context.currentGenerateLengths[i], context.actualIterations[i]);

        // Extract only the generated tokens (skip prompt and padding)
        // Generated tokens are the last currentGenerateLengths[i] tokens in the vector
        int32_t const genLength = context.currentGenerateLengths[i];
        int32_t const totalLength = static_cast<int32_t>(context.tokenIds[i].size());

        response.outputIds.emplace_back(
            context.tokenIds[i].begin() + (totalLength - genLength), context.tokenIds[i].end());

        response.outputTexts.emplace_back(mTokenizer->decode(response.outputIds[i], true));
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelPrefill(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);

    int32_t const activeBatchSize = context.activeBatchSize;

    // Prepare the inputs for prefill stage execution.
    // All sequences in the batch should have the same padded length
    int32_t const inputIdsLength = static_cast<int32_t>(context.tokenIds[0].size());
    if (inputIdsLength > mBaseEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("Input ids length %d is greater than the max supported input length %d", inputIdsLength,
            mBaseEngineConfig.maxSupportedInputLength);
        return false;
    }

    mIdsInput.reshape({activeBatchSize, inputIdsLength});
    mContextLengthsInput.reshape({activeBatchSize});
    mBaseHiddenStatesOutput.reshape({activeBatchSize, inputIdsLength, mBaseEngineConfig.outputHiddenDim});
    mLogitsOutput.reshape({activeBatchSize, mBaseEngineConfig.vocabSize});

    // Setup the input tensors. ContextLen input is on CPU.
    int32_t* ctxLenData = mContextLengthsInput.dataPointer<int32_t>();
    int32_t* idsInputData = mIdsInput.dataPointer<int32_t>();

    // Pack all sequences into the input tensor
    // Use actual prompt length (not padded length) for context_lengths to ensure we select the last real token
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        ctxLenData[i] = context.promptLengths[i]; // Use actual prompt length instead of padded length
        CUDA_CHECK(cudaMemcpyAsync(idsInputData + i * inputIdsLength, context.tokenIds[i].data(),
            inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    }

    bool const prefillSuccess = mBaseEngineRunner->executePrefillStep(mIdsInput, mContextLengthsInput,
        context.multimodalEmbeddings, mLogitsOutput, std::ref(mBaseHiddenStatesOutput), context.stream);
    if (!prefillSuccess)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    // Sampling from the Prefill stage logits using greedy Top1 sampling for each sequence，only collect the top1 index.
    mSamplingIndices.reshape({activeBatchSize, 1});
    constexpr int32_t kSAMPLING_TOP_K = 1;
    selectAllTopK(mLogitsOutput, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);

    // Pull the sampling indices from device to host for all sequences
    std::vector<int32_t> selectedTokenIds(activeBatchSize);
    CUDA_CHECK(cudaMemcpyAsync(selectedTokenIds.data(), mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Update tokenIds and generation length for each sequence
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (!context.finishedStates[i])
        {
            context.tokenIds[i].push_back(selectedTokenIds[i]);
            context.currentGenerateLengths[i] += 1;
        }
    }

    // The base prefill function produce output logits and (concatenated) hiddenStates for next step to use.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelPrefill(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kEAGLE_DRAFT_PREFILL, context.stream);

    int32_t const activeBatchSize = context.activeBatchSize;

    // Implement the draft prefill execution logic, prepare the input ids and hidden states inputs for the
    // eagle draft engine. The formulation of the feature "vector" is F_n = F(H_n, Token_{n+1}), therefore we
    // need to trim out the first token of the sequence from the token_ids input.
    // All sequences should have the same padded length
    // After base prefill: context.tokenIds has N+1 tokens, mBaseHiddenStatesOutput has N hidden states
    int32_t const inputIdsLength = static_cast<int32_t>(context.tokenIds[0].size()) - 1;
    check::check(mBaseHiddenStatesOutput.getShape()[0] == activeBatchSize
            && mBaseHiddenStatesOutput.getShape()[1] == inputIdsLength,
        "BaseHiddenStatesOutput shape [batch, seq_len, hidden_dim] shall match with [activeBatchSize, inputIdsLength, "
        "hidden_dim]");

    // Prepare input and output tensors.
    mIdsInput.reshape({activeBatchSize, inputIdsLength});
    mDraftHiddenStatesInput.reshape({activeBatchSize, inputIdsLength, mDraftEngineConfig.draftModelHiddenDim});
    mLogitsOutput.reshape({activeBatchSize, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({activeBatchSize, mDraftEngineConfig.draftModelHiddenDim});

    // Clear garbage data in the draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));

    // Copy input IDs for each batch
    int32_t* idsInputData = mIdsInput.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Shift the data pointer by 1 to start from the second token for each batch
        int32_t const* inputIdsData = context.tokenIds[i].data() + 1;
        CUDA_CHECK(cudaMemcpyAsync(idsInputData + i * inputIdsLength, inputIdsData, inputIdsLength * sizeof(int32_t),
            cudaMemcpyHostToDevice, context.stream));
    }

    bool const prefillSuccess = mDraftEngineRunner->executeEaglePrefillStep(mIdsInput, mBaseHiddenStatesOutput,
        mDraftHiddenStatesInput, mContextLengthsInput, context.multimodalEmbeddings, mLogitsOutput,
        mDraftHiddenStatesOutput, context.stream);
    if (!prefillSuccess)
    {
        LOG_ERROR("Failed to execute prefill step for draft model.");
        return false;
    }
    // No need to do sampling, directly produce logits (mLogitsOutput) and hidden states (mDraftHiddenStatesOutput) for
    // the next step.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::constructDraftTree(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kEAGLE_CONSTRUCT_DRAFT_TREE, context.stream);

    int32_t const activeBatchSize = context.activeBatchSize;

    // Core logic for eagle speculative decoding, construct the draft tree in an auto-regressive manner./
    // Inputs: Logits (mLogitsOutput) and draft hidden states (mDraftHiddenStatesOutput) from draft prefill
    // or draft model accept decoding operation.
    // Construct the draft tree table with multiple round of drafting. The descriptions of the draft tree are
    // stored in mDraftTokenIdsTable, mDraftTokenScoreTable, mDraftTokenPredecessorTable.

    // Reshape draft tree tensors to match activeBatchSize for dynamic batching
    int32_t const draftTopK = mDraftingConfig.draftingTopK;
    int32_t const draftFullTableLength = static_cast<int32_t>(mDraftTokenIdsFullTable.getShape()[1]);
    mDraftTokenIdsFullTable.reshape({activeBatchSize, draftFullTableLength});
    mDraftTokenScoreFullTable.reshape({activeBatchSize, draftFullTableLength});
    mDraftTokenPredecessorFullTable.reshape({activeBatchSize, draftFullTableLength});
    mDraftTreeRootTokenId.reshape({activeBatchSize});
    mDraftVocabMappingTable.reshape({activeBatchSize, mDraftEngineConfig.draftModelVocabSize});
    mDraftTokenIdsTable.reshape({activeBatchSize, draftTopK * draftTopK});
    mDraftTokenScoresTable.reshape({activeBatchSize, draftTopK * draftTopK});
    mDraftTokenIntermediateScores.reshape({activeBatchSize, draftTopK});
    mDraftTokenIntermediateParents.reshape({activeBatchSize, draftTopK});

    // Record root token (last committed token selected by base model) id for the draft tree for each batch.
    std::vector<int32_t> rootTokenIds(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        rootTokenIds[i] = context.tokenIds[i].back();
    }
    CUDA_CHECK(cudaMemcpyAsync(mDraftTreeRootTokenId.rawPointer(), rootTokenIds.data(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Sampling from the logits output, collect draftTopK tokens as first level under "root".
    mSamplingIndices.reshape({activeBatchSize, draftTopK});
    mSamplingScores.reshape({activeBatchSize, draftTopK});
    selectAllTopK(
        mLogitsOutput, std::ref(mSamplingScores), mSamplingIndices, draftTopK, mSamplingWorkspace, context.stream);

    // Initialize data structures to describe the whole draft tree.
    kernel::initializeDraftTreeTables(mSamplingIndices, mSamplingScores, mDraftTreeRootTokenId, mDraftVocabMappingTable,
        mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, context.stream);
    // Reset hidden states output of base model and input hidden states of draft model to clear the garbage data.
    CUDA_CHECK(cudaMemsetAsync(
        mBaseHiddenStatesOutput.rawPointer(), 0, mBaseHiddenStatesOutput.getMemoryCapacity(), context.stream));
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));

    // Construct input tensors to feed into the eagle draft engine. With current implementation, for simplicity, we
    // will use padded input and only collect results from indices we need.
    int32_t const paddedDraftTreeSize = mDraftingConfig.draftingStep * draftTopK;
    mIdsInput.reshape({activeBatchSize, paddedDraftTreeSize});
    mBaseHiddenStatesOutput.reshape({activeBatchSize, paddedDraftTreeSize, mBaseEngineConfig.outputHiddenDim});
    mDraftHiddenStatesInput.reshape({activeBatchSize, paddedDraftTreeSize, mDraftEngineConfig.draftModelHiddenDim});
    mDraftTreeSize.reshape({activeBatchSize});
    mDraftTreeMask.reshape({activeBatchSize, paddedDraftTreeSize, paddedDraftTreeSize});
    // Assemble the initial draft tree input here since we need to copy out the data in draftHiddenStatesOutput prior to
    // reshaping it.
    kernel::assembleInitialDraftTreeInput(mDraftTokenIdsFullTable, mDraftHiddenStatesOutput, mIdsInput,
        mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, draftTopK, context.stream);
    // Output tensors must be 3D: [batch_size, num_tokens, vocab_size/hidden_dim] for draft proposal
    mLogitsOutput.reshape({activeBatchSize, draftTopK, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({activeBatchSize, draftTopK, mDraftEngineConfig.draftModelHiddenDim});

    for (int32_t round = 0; round < mDraftingConfig.draftingStep - 1; round++)
    {
        if (round == 0)
        {
            // With first round of drafting, the input tensors have been assembled, we only need to save intermediate
            // information for the next step.
            kernel::assembleInitialIntermediateData(mSamplingScores, mDraftTokenIntermediateParents,
                mDraftTokenIntermediateScores, draftTopK, context.stream);
        }
        else
        {
            // Last round of drafting produce draftTopK x draftTopK candidate token for the layer, we need to pick the
            // top draftTopK, assemble input tensors, and save intermediate information.
            mSamplingIndices.reshape({activeBatchSize, draftTopK});
            mSamplingScores.reshape({activeBatchSize, draftTopK});
            selectAllTopK(mDraftTokenScoresTable, std::ref(mSamplingScores), mSamplingIndices, draftTopK,
                mSamplingWorkspace, context.stream);
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mDraftHiddenStatesOutput, mSamplingIndices, mIdsInput,
                mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, draftTopK, round, context.stream);
            kernel::assembleIntermediateData(mSamplingScores, mSamplingIndices, mDraftTokenIntermediateScores,
                mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

        // Ensure output tensors are 3D before calling draft proposal
        mLogitsOutput.reshape({activeBatchSize, draftTopK, mDraftEngineConfig.draftModelVocabSize});
        mDraftHiddenStatesOutput.reshape({activeBatchSize, draftTopK, mDraftEngineConfig.draftModelHiddenDim});

        // Invoke the eagle draft engine to produce the new round of logits and hidden states.
        bool const draftProposalStatus = mDraftEngineRunner->executeEagleDraftProposalStep(mIdsInput,
            mBaseHiddenStatesOutput, mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, mLogitsOutput,
            mDraftHiddenStatesOutput, context.stream);
        if (!draftProposalStatus)
        {
            LOG_ERROR("Failed to execute draft proposal step for draft model.");
            return false;
        }
        // Collect TopK results from each lane of output logits.
        // mLogitsOutput is now 3D: {activeBatchSize, draftTopK, vocabSize}
        // Reshape to 2D for selectAllTopK: {activeBatchSize * draftTopK, vocabSize}
        mLogitsOutput.reshape({activeBatchSize * draftTopK, mDraftEngineConfig.draftModelVocabSize});
        mDraftHiddenStatesOutput.reshape({activeBatchSize * draftTopK, mDraftEngineConfig.draftModelHiddenDim});
        mSamplingIndices.reshape({activeBatchSize * draftTopK, draftTopK});
        mSamplingScores.reshape({activeBatchSize * draftTopK, draftTopK});
        selectAllTopK(
            mLogitsOutput, std::ref(mSamplingScores), mSamplingIndices, draftTopK, mSamplingWorkspace, context.stream);

        // Reshape sampling indices/scores back to the expected format for subsequent kernel calls
        // Note: mDraftHiddenStatesOutput stays in 2D format for assembleDraftTreeInput in next round
        mSamplingIndices.reshape({activeBatchSize, draftTopK * draftTopK});
        mSamplingScores.reshape({activeBatchSize, draftTopK * draftTopK});

        // Update the draft tree tables with the new topK results. translate draft vocab token towards full vocab size.
        kernel::computeCuScoresAndTranslateToken(mSamplingIndices, mSamplingScores, mDraftTokenIntermediateScores,
            mDraftVocabMappingTable, mDraftTokenIdsTable, mDraftTokenScoresTable, draftTopK, context.stream);
        // Update results in the full draft table
        kernel::updateDraftTreeFullTables(mDraftTokenIdsTable, mDraftTokenScoresTable, mDraftTokenIntermediateParents,
            mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, round,
            context.stream);
    }

    // We have constructed the data structure for the draft table, now we need to pick the top candidates and produce
    // the verify tree and pass into the base model for verification.
    mSamplingIndices.reshape({activeBatchSize, mDraftingConfig.verifyTreeSize});
    selectAllTopK(mDraftTokenScoreFullTable, std::nullopt, mSamplingIndices, mDraftingConfig.verifyTreeSize,
        mSamplingWorkspace, context.stream);

    mIdsInput.reshape({activeBatchSize, mDraftingConfig.verifyTreeSize});
    mDraftTreeMask.reshape({activeBatchSize, mDraftingConfig.verifyTreeSize, mDraftingConfig.verifyTreeSize});
    kernel::constructVerificationDraftTree(mDraftTokenIdsFullTable, mDraftTokenPredecessorFullTable, mSamplingIndices,
        mIdsInput, mDraftTreeMask, context.stream);

    // This function will produce mIdsInput and mDraftTreeMask to describe established draft tree.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelVerification(SpecDecodeInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kEAGLE_BASE_VERIFICATION, context.stream);

    int32_t const activeBatchSize = context.activeBatchSize;

    // This function will consume idsInput and draftTreeMask. Use base model to verify the draft tree.
    // We need to collect the logits and hidden states (for further drafting step).
    check::check(
        mIdsInput.getShape()[0] == activeBatchSize && mIdsInput.getShape()[1] == mDraftingConfig.verifyTreeSize,
        "IdsInput shall have shape [batch_size, verify_tree_size]");
    check::check(mDraftTreeMask.getShape()[0] == activeBatchSize
            && mDraftTreeMask.getShape()[1] == mDraftingConfig.verifyTreeSize
            && mDraftTreeMask.getShape()[2] == mDraftingConfig.verifyTreeSize,
        "DraftTreeMask shall have shape [batch_size, verify_tree_size, verify_tree_size]");

    // Engine expects 2D tensors: [batch_size * verify_tree_size, vocab_size/hidden_dim]
    int32_t const selectTokenSize = activeBatchSize * mDraftingConfig.verifyTreeSize;
    mLogitsOutput.reshape({selectTokenSize, mBaseEngineConfig.vocabSize});
    mBaseHiddenStatesOutput.reshape({selectTokenSize, mBaseEngineConfig.outputHiddenDim});

    bool const verifySuccess = mBaseEngineRunner->executeEagleBaseTreeDecodingStep(
        mIdsInput, mDraftTreeMask, mLogitsOutput, mBaseHiddenStatesOutput, context.stream);
    if (!verifySuccess)
    {
        LOG_ERROR("Failed to execute base tree verification step for base model.");
        return false;
    }

    // Reshape accepted token tensors to match activeBatchSize for dynamic batching
    int32_t const maxAcceptDepth = mDraftingConfig.draftingStep + 1;
    mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth});
    mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth});
    mAcceptLength.reshape({activeBatchSize});

    // Collected accepted token ids and indices. Use sampling workspace for eagle accept process.
    kernel::eagleAccept(mLogitsOutput, mIdsInput, mDraftTreeMask, mAcceptedTokenIds, mAcceptedTokenIndices,
        mAcceptLength, mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(), context.stream);

    // Inplace update the KVCache and input hidden states from the accepted token indices.
    // Also commit KVCache to reflect the latest KVCache length (We can only do this after knowing how many tokens are
    // accepted).
    rt::Tensor const& kvCacheLengths = mBaseEngineRunner->getLinearKVCache().getKVCacheLengths();
    rt::Tensor kvCacheTensor = mBaseEngineRunner->getLinearKVCache().getKVCacheBuffer();

    // Reshape input hidden states from 2D [batch*verify_tree_size, hidden_dim] to 3D [batch, verify_tree_size,
    // hidden_dim]
    mBaseHiddenStatesOutput.reshape(
        {activeBatchSize, mDraftingConfig.verifyTreeSize, mBaseEngineConfig.outputHiddenDim});

    // INPLACE update: The kernel will update accepted tokens directly within the same buffer.
    //
    // Memory layout transformation (inplace):
    //   Before: [Batch0: Token0...Token59][Batch1: Token0...Token59]...  =(stride=60 per batch)
    //   After:  [Batch0: SelectedToken0...SelectedToken6][Batch1: SelectedToken0...SelectedToken6]... (Select and pad
    //   to stride=maxAcceptDepth 7)
    kernel::eagleBaseCommitKVCacheAndAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, kvCacheTensor, mBaseHiddenStatesOutput, context.stream);

    mBaseEngineRunner->getLinearKVCache().commitSequenceLength(mAcceptLength, context.stream);

    // Reshape to reflect the compacted layout [batch, maxAcceptDepth, hiddenDim]
    mBaseHiddenStatesOutput.reshape({activeBatchSize, maxAcceptDepth, mBaseEngineConfig.outputHiddenDim});

    // Pull collected results from device to host for all batches
    std::vector<int32_t> acceptLengths(activeBatchSize);
    std::vector<int32_t> acceptedTokenIds(activeBatchSize * maxAcceptDepth);

    CUDA_CHECK(cudaMemcpyAsync(acceptLengths.data(), mAcceptLength.rawPointer(), activeBatchSize * sizeof(int32_t),
        cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(acceptedTokenIds.data(), mAcceptedTokenIds.rawPointer(),
        activeBatchSize * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Update tokenIds and check for EOS for each batch
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        if (context.finishedStates[batchIdx])
            continue;

        int32_t const acceptLength = acceptLengths[batchIdx];
        // update iterations for each batch
        if (acceptLength > 0)
        {
            context.actualIterations[batchIdx] += 1;
        }
        for (int32_t i = 0; i < acceptLength; i++)
        {
            int32_t const token = acceptedTokenIds[batchIdx * maxAcceptDepth + i];
            context.tokenIds[batchIdx].push_back(token);
            context.currentGenerateLengths[batchIdx]++;

            if (token == mTokenizer->getEosId())
            {
                context.finishedStates[batchIdx] = true;
                LOG_DEBUG("Batch %d encountered EOS (token %d) at generation round %d", batchIdx, token,
                    context.generationRound);
                break;
            }
        }
    }

    // Produce base model hidden states for the next step
    // Note: Hidden states shape is already set by the kernel
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelAcceptToken(SpecDecodeInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;

    // Base model verifiction function is responsible for producing the output with correct shape.
    // Shape is [activeBatchSize, accepted_length, hidden_dim]
    int64_t const inputIdsLength = mBaseHiddenStatesOutput.getShape()[1];

    // Prepare input and output tensors.
    mIdsInput.reshape({activeBatchSize, inputIdsLength});
    mDraftHiddenStatesInput.reshape({activeBatchSize, inputIdsLength, mDraftEngineConfig.draftModelHiddenDim});
    mLogitsOutput.reshape({activeBatchSize, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({activeBatchSize, mDraftEngineConfig.draftModelHiddenDim});

    // Clear garbage data in the draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));

    // Prepare the IdsInput, we need to copy from mAcceptedTokenIds for each batch.
    // mAcceptedTokenIds is [activeBatchSize, maxAcceptDepth], we copy the first inputIdsLength tokens
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int32_t*>(mIdsInput.rawPointer()) + i * inputIdsLength,
            static_cast<int32_t*>(mAcceptedTokenIds.rawPointer()) + i * mAcceptedTokenIds.getShape()[1],
            inputIdsLength * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    }

    bool const acceptTokenSuccess
        = mDraftEngineRunner->executeEagleAcceptDecodeTokenStep(mIdsInput, mBaseHiddenStatesOutput,
            mDraftHiddenStatesInput, mAcceptLength, mLogitsOutput, mDraftHiddenStatesOutput, context.stream);
    if (!acceptTokenSuccess)
    {
        LOG_ERROR("Failed to execute accept token step for draft model.");
        return false;
    }
    // No need to do sampling, directly produce logits (mLogitsOutput) and hidden states (mDraftHiddenStatesOutput) for
    // the next step.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::captureDraftProposalCudaGraph(cudaStream_t stream)
{
    bool captureStatus{true};
    int32_t const draftTopK = mDraftingConfig.draftingTopK;
    int32_t const paddedDraftTreeSize = mDraftingConfig.draftingStep * draftTopK;

    // Capture CUDA graph for all supported batch sizes
    for (int32_t batchSize = 1; batchSize <= mMaxRuntimeBatchSize; ++batchSize)
    {
        mSamplingIndices.reshape({batchSize, draftTopK});
        mSamplingScores.reshape({batchSize, draftTopK});
        mIdsInput.reshape({batchSize, paddedDraftTreeSize});
        mBaseHiddenStatesOutput.reshape({batchSize, paddedDraftTreeSize, mBaseEngineConfig.outputHiddenDim});
        mDraftHiddenStatesInput.reshape({batchSize, paddedDraftTreeSize, mDraftEngineConfig.draftModelHiddenDim});
        mDraftTreeSize.reshape({batchSize});
        mDraftTreeMask.reshape({batchSize, paddedDraftTreeSize, paddedDraftTreeSize});
        // Output tensors must be 3D: [batch_size, num_tokens, vocab_size/hidden_dim] not 2D
        mLogitsOutput.reshape({batchSize, draftTopK, mDraftEngineConfig.draftModelVocabSize});
        mDraftHiddenStatesOutput.reshape({batchSize, draftTopK, mDraftEngineConfig.draftModelHiddenDim});

        // Don't pass multimodal embeddings during CUDA graph capture as they are invalid
        captureStatus &= mDraftEngineRunner->captureEagleDraftProposalCudaGraph(mIdsInput, mBaseHiddenStatesOutput,
            mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, mLogitsOutput, mDraftHiddenStatesOutput, stream);
    }

    if (captureStatus)
    {
        LOG_INFO(
            "Successfully captured the draft proposal CUDA graph for all batch sizes (1-%d).", mMaxRuntimeBatchSize);
    }
    else
    {
        LOG_WARNING("Failed to capture the draft proposal CUDA graph for some batch sizes.");
    }

    return captureStatus;
}

bool LLMInferenceSpecDecodeRuntime::captureDraftAcceptDecodeTokenCudaGraph(cudaStream_t stream)
{
    bool captureStatus{true};
    int32_t const draftingStep = mDraftingConfig.draftingStep;

    // Capture CUDA graph for all supported batch sizes and accept lengths
    for (int32_t batchSize = 1; batchSize <= mMaxRuntimeBatchSize; ++batchSize)
    {
        mLogitsOutput.reshape({batchSize, mDraftEngineConfig.draftModelVocabSize});
        mDraftHiddenStatesOutput.reshape({batchSize, mDraftEngineConfig.draftModelHiddenDim});

        // Don't pass multimodal embeddings during CUDA graph capture as they are invalid
        // TODO: consider using a single capture for max accept lengths.
        for (int32_t acceptLength = 1; acceptLength <= draftingStep + 1; acceptLength++)
        {
            mBaseHiddenStatesOutput.reshape({batchSize, acceptLength, mBaseEngineConfig.outputHiddenDim});
            mIdsInput.reshape({batchSize, acceptLength});
            mDraftHiddenStatesInput.reshape({batchSize, acceptLength, mDraftEngineConfig.draftModelHiddenDim});

            // Create a temporary acceptLength tensor for CUDA graph capture
            // All batches use the same acceptLength during graph capture
            mAcceptLength.reshape({batchSize});
            std::vector<int32_t> acceptLengthsVec(batchSize, acceptLength);
            CUDA_CHECK(cudaMemcpyAsync(mAcceptLength.rawPointer(), acceptLengthsVec.data(), batchSize * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            captureStatus
                &= mDraftEngineRunner->captureEagleAcceptDecodeTokenCudaGraph(mIdsInput, mBaseHiddenStatesOutput,
                    mDraftHiddenStatesInput, mAcceptLength, mLogitsOutput, mDraftHiddenStatesOutput, stream);
        }
    }

    if (captureStatus)
    {
        LOG_INFO(
            "Successfully captured the draft accept decode token CUDA graph for all batch sizes (1-%d) and accept "
            "lengths (1-%d).",
            mMaxRuntimeBatchSize, draftingStep + 1);
    }
    else
    {
        LOG_WARNING("Failed to capture the draft accept decode token CUDA graph for some combinations.");
    }

    return captureStatus;
}

bool LLMInferenceSpecDecodeRuntime::captureBaseVerificationCudaGraph(cudaStream_t stream)
{
    bool captureStatus{true};

    // Capture CUDA graph for all supported batch sizes
    for (int32_t batchSize = 1; batchSize <= mMaxRuntimeBatchSize; ++batchSize)
    {
        // Engine expects 2D tensors: [batch_size * verify_tree_size, vocab_size/hidden_dim]
        int32_t const selectTokenSize = batchSize * mDraftingConfig.verifyTreeSize;
        mLogitsOutput.reshape({selectTokenSize, mBaseEngineConfig.vocabSize});
        mBaseHiddenStatesOutput.reshape({selectTokenSize, mBaseEngineConfig.outputHiddenDim});

        mIdsInput.reshape({batchSize, mDraftingConfig.verifyTreeSize});
        mDraftTreeMask.reshape({batchSize, mDraftingConfig.verifyTreeSize, mDraftingConfig.verifyTreeSize});

        // Don't pass multimodal embeddings during CUDA graph capture as they are invalid
        captureStatus &= mBaseEngineRunner->captureEagleBaseTreeDecodingCudaGraph(
            mIdsInput, mDraftTreeMask, mLogitsOutput, mBaseHiddenStatesOutput, stream);
    }

    if (captureStatus)
    {
        LOG_INFO("Successfully captured the base model verification CUDA graph for all batch sizes (1-%d).",
            mMaxRuntimeBatchSize);
    }
    else
    {
        LOG_WARNING("Failed to capture the base model verification CUDA graph for some batch sizes.");
    }

    return captureStatus;
}

bool LLMInferenceSpecDecodeRuntime::setUpForPrefillExecution(SpecDecodeInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    std::vector<std::vector<int32_t>> const& batchedInputIds = context.rawBatchedInputIds;
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;

    rt::LinearKVCache& linearKVCacheBase = mBaseEngineRunner->getLinearKVCache();
    rt::LinearKVCache& linearKVCacheDraft = mDraftEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBufferBase = linearKVCacheBase.getKVCacheBuffer();
    rt::Tensor kvCacheBufferDraft = linearKVCacheDraft.getKVCacheBuffer();

    // Record the length of the reused KVCache for each sequence.
    // Use activeBatchSize (actual request size)
    rt::Tensor reuseKVCacheLengths = rt::Tensor({activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
    int32_t* reuseKVCacheLengthsData = reuseKVCacheLengths.dataPointer<int32_t>();

    // Initialize reuse lengths to 0 for all active sequences
    std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);

    // Search if the system prompt has been cached. If there are cached system prompts, insert
    // the pre-computed KVCache and remove the contents from inputIds.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPrompt(context.systemPrompts[i]);
        if (mSystemPromptKVCacheBase.find(promptHash) != mSystemPromptKVCacheBase.end())
        {
            auto& precachedKVCacheBase = mSystemPromptKVCacheBase[promptHash];
            auto& precachedKVCacheDraft = mSystemPromptKVCacheDraft[promptHash];
            auto const& kvCacheContentBase = precachedKVCacheBase.kvCacheContent;
            auto const& kvCacheContentDraft = precachedKVCacheDraft.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBufferBase, kvCacheContentBase, i, context.stream);
            kernel::instantiateKVCacheFromTensor(kvCacheBufferDraft, kvCacheContentDraft, i, context.stream);

            int32_t reuseLength = static_cast<int32_t>(kvCacheContentBase.getShape()[3]);
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
            reuseKVCacheLengthsData[i] = reuseLength;
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));

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
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    // Pack inputIds, instantiate input data for prefill step, and reset the KVCache state.
    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mBaseEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("The max input length (%d) exceeds the max supported input length (%d) of the LLM Engine.",
            maxInputLength, mBaseEngineConfig.maxSupportedInputLength);
        return false;
    }

    // The LLM Engine could also have minSupportedInputLength constraint.
    int32_t const packedInputLength = std::max(maxInputLength, mBaseEngineConfig.minSupportedInputLength);

    // Initialize tokenIds for each sequence in the batch
    context.tokenIds.clear();
    context.tokenIds.resize(activeBatchSize);

    // Save the actual prompt lengths (before padding) for each batch
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.promptLengths[i] = processedIdsLengths[i];
    }

    // Pad each sequence to the max length of this batch
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.tokenIds[i].resize(packedInputLength, mTokenizer->getPadId());
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), context.tokenIds[i].begin());
    }

    linearKVCacheBase.resetForNewSequences(reuseKVCacheLengths, context.stream);
    linearKVCacheDraft.resetForNewSequences(reuseKVCacheLengths, context.stream);

    return true;
}

bool LLMInferenceSpecDecodeRuntime::genAndSaveSystemPromptKVCache(SpecDecodeInferenceContext& context)
{
    check::check(mBaseEngineConfig.enableReuseKVCache == mDraftEngineConfig.enableReuseKVCache,
        "The system prompt KVCache feature is not same for base and draft model.");

    if (!mBaseEngineConfig.enableReuseKVCache)
    {
        LOG_DEBUG("System prompt KVCache feature is not enabled in the engine.");
        return false;
    }

    // Check if cache already exists
    int32_t const batchIdx = context.currentBatchIndex;
    std::string const prompt = context.systemPrompts[batchIdx];
    size_t const promptHash = hashSystemPrompt(prompt);
    if (mSystemPromptKVCacheBase.find(promptHash) != mSystemPromptKVCacheBase.end()
        && mSystemPromptKVCacheDraft.find(promptHash) != mSystemPromptKVCacheDraft.end())
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());

    if (promptIdsLength > mBaseEngineConfig.maxSupportedInputLength
        || promptIdsLength > mDraftEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (base=%d, draft=%d)", promptIdsLength,
            mBaseEngineConfig.maxSupportedInputLength, mDraftEngineConfig.maxSupportedInputLength);
        return false;
    }

    // Create a temporary single-batch context for system prompt KVCache generation
    // Reuse the existing prefill functions which will use runtime member tensors (mIdsInput, mLogitsOutput, etc.)
    SpecDecodeInferenceContext tempContext;
    tempContext.systemPrompts.resize(1);
    tempContext.systemPrompts[0] = prompt;
    tempContext.rawBatchedInputIds.emplace_back(tokenizedPrompt);
    tempContext.tokenIds.resize(1);
    tempContext.tokenIds[0] = tokenizedPrompt;
    tempContext.currentGenerateLengths.resize(1, 0);
    tempContext.promptLengths.resize(1, 0);
    tempContext.finishedStates.resize(1, false);
    tempContext.multimodalEmbeddings = context.multimodalEmbeddings;
    tempContext.generationRound = 0;
    tempContext.maxGenerateLength = 0; // Not generating, just caching
    tempContext.activeBatchSize = 1;
    tempContext.currentBatchIndex = 0;
    tempContext.stream = context.stream;

    // Setup for prefill execution: handles KV cache reset and applies any reused system prompt cache
    if (!setUpForPrefillExecution(tempContext))
    {
        LOG_ERROR("Prefill execution setup failed for system prompt KVCache generation.");
        return false;
    }

    // Run base model prefill (reuses mIdsInput, mLogitsOutput, mBaseHiddenStatesOutput)
    bool prefillStatus = runBaseModelPrefill(tempContext);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute base model prefill for system prompt KVCache generation.");
        return false;
    }

    // During system prompt KVCache generation, we know the next token is <|im_start|> (first token of user prompt)
    // Set it explicitly to avoid sampling randomness
    check::check(!tempContext.tokenIds[0].empty(), "Token IDs should not be empty at this point.");
    tempContext.tokenIds[0].back() = mImStartTokenId;

    // Tokens produced during system KV-cache reuse prefill do not count as generated tokens
    tempContext.currentGenerateLengths[0] -= 1;

    // Run draft model prefill (reuses mIdsInput, mLogitsOutput, mDraftHiddenStatesInput, mDraftHiddenStatesOutput)
    bool draftPrefillStatus = runDraftModelPrefill(tempContext);
    if (!draftPrefillStatus)
    {
        LOG_ERROR("Failed to execute draft model prefill for system prompt KVCache generation.");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Copy out the KVCache content from the prefill step
    auto& linearKVCacheBase = mBaseEngineRunner->getLinearKVCache();
    auto& linearKVCacheDraft = mDraftEngineRunner->getLinearKVCache();
    auto cacheConfigBase = linearKVCacheBase.getConfig();
    auto cacheConfigDraft = linearKVCacheDraft.getConfig();
    auto kvCacheBufferBase = linearKVCacheBase.getKVCacheBuffer();
    auto kvCacheBufferDraft = linearKVCacheDraft.getKVCacheBuffer();
    rt::Coords savedKVCacheShapeBase{
        cacheConfigBase.numDecoderLayers, 2, cacheConfigBase.numKVHeads, promptIdsLength, cacheConfigBase.headDim};
    rt::Coords savedKVCacheShapeDraft{
        cacheConfigDraft.numDecoderLayers, 2, cacheConfigDraft.numKVHeads, promptIdsLength, cacheConfigDraft.headDim};

    SystemPromptKVCache savedKVCacheBase;
    SystemPromptKVCache savedKVCacheDraft;
    savedKVCacheBase.systemPrompt = prompt;
    savedKVCacheBase.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheBase.kvCacheContent
        = rt::Tensor(savedKVCacheShapeBase, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT);
    savedKVCacheDraft.systemPrompt = prompt;
    savedKVCacheDraft.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheDraft.kvCacheContent
        = rt::Tensor(savedKVCacheShapeDraft, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT);

    // We only process one sequence at a time
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCacheBase.kvCacheContent, kvCacheBufferBase, CACHE_BATCH_IDX, context.stream);
    kernel::saveKVCacheIntoTensor(
        savedKVCacheDraft.kvCacheContent, kvCacheBufferDraft, CACHE_BATCH_IDX, context.stream);

    mSystemPromptKVCacheBase.insert({promptHash, std::move(savedKVCacheBase)});
    mSystemPromptKVCacheDraft.insert({promptHash, std::move(savedKVCacheDraft)});

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    LOG_DEBUG("System prompt KVCache saved for batch %d: {%s}", batchIdx, prompt.c_str());

    return true;
}

} // namespace rt
} // namespace trt_edgellm