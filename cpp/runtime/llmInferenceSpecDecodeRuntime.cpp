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

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/newEagleUtilKernels.h"
#include "sampler/sampling.h"
#include <fstream>
#include <functional>
#include <vector>

using namespace nvinfer1;

namespace drivellm
{
namespace rt
{

LLMInferenceSpecDecodeRuntime::LLMInferenceSpecDecodeRuntime(
    std::string const& engineDir, EagleDraftingConfig const& draftingConfig, cudaStream_t stream)
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

    // Allocate runtime tensors till max supported size.
    int32_t const maxDraftTreeSize = std::max(mDraftEngineConfig.maxDraftTreeSize, mDraftingConfig.verifyTreeSize);
    int32_t const draftTopK = mDraftingConfig.draftingTopK;
    int32_t const maxSamplingSize = std::max(maxDraftTreeSize, draftTopK * draftTopK);
    int32_t const draftFullTableLength = 1 + draftTopK + (mDraftingConfig.draftingStep - 1) * draftTopK * draftTopK;

    LOG_DEBUG(
        "maxDraftTreeSize: %d, maxSamplingSize: %d, draftFullTableLength: %d to set up the SpecDecode inference "
        "runtime",
        maxDraftTreeSize, maxSamplingSize, draftFullTableLength);
    // Reserve enough workspace for sampling, the size can be further optimized.
    int32_t const maxSamplingWorkspaceSize
        = std::max(getSelectAllTopKWorkspaceSize(kRUNTIME_BATCH_SIZE, mBaseEngineConfig.vocabSize, 1),
            getSelectAllTopKWorkspaceSize(draftTopK, mDraftEngineConfig.draftModelVocabSize, draftTopK));

    try
    {
        mIdsInput = rt::Tensor(
            {kRUNTIME_BATCH_SIZE, mBaseEngineConfig.maxSupportedInputLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mContextLengthsInput = rt::Tensor({kRUNTIME_BATCH_SIZE}, rt::DeviceType::kCPU, DataType::kINT32);
        mLogitsOutput
            = rt::Tensor({maxDraftTreeSize, mBaseEngineConfig.vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTreeSize = rt::Tensor({kRUNTIME_BATCH_SIZE}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTreeMask = rt::Tensor(
            {kRUNTIME_BATCH_SIZE, maxDraftTreeSize, maxDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        mBaseHiddenStatesOutput = rt::Tensor(
            {kRUNTIME_BATCH_SIZE, mBaseEngineConfig.maxSupportedInputLength, mBaseEngineConfig.outputHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftHiddenStatesInput = rt::Tensor(
            {kRUNTIME_BATCH_SIZE, mBaseEngineConfig.maxSupportedInputLength, mDraftEngineConfig.draftModelHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftHiddenStatesOutput = rt::Tensor({kRUNTIME_BATCH_SIZE, draftTopK, mDraftEngineConfig.draftModelHiddenDim},
            rt::DeviceType::kGPU, DataType::kHALF);
        mDraftTokenIdsFullTable
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenScoreFullTable
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenPredecessorFullTable
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftFullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftVocabMappingTable = rt::Tensor(
            {kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelVocabSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTreeRootTokenId = rt::Tensor({kRUNTIME_BATCH_SIZE}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenIdsTable
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftTopK * draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        mDraftTokenScoresTable
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftTopK * draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenIntermediateScores
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mDraftTokenIntermediateParents
            = rt::Tensor({kRUNTIME_BATCH_SIZE, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8);
        mSamplingIndices = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mSamplingScores = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

        // DraftModel prefill/accept-decode-token will also produce one layer of draft tree, so the max accepted
        // depth should be drafting step + 1.
        int32_t const maxAcceptDepth = mDraftingConfig.draftingStep + 1;
        mAcceptedTokenIds = rt::Tensor({kRUNTIME_BATCH_SIZE, maxAcceptDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        mAcceptedTokenIndices
            = rt::Tensor({kRUNTIME_BATCH_SIZE, maxAcceptDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        mAcceptLength = rt::Tensor({kRUNTIME_BATCH_SIZE}, rt::DeviceType::kGPU, DataType::kINT32);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate runtime tensors: %s", e.what());
        throw std::runtime_error("Failed to allocate runtime tensors: " + std::string(e.what()));
    }
    LOG_INFO("Runtime tensors successfully allocated.");

    // Load conversion table from draft model vocab to base model vocab.
    bool const draftVocabMappingTableLoaded
        = loadDraftVocabMappingTable(std::filesystem::path(engineDir) / "d2t.bin", stream);
    if (!draftVocabMappingTableLoaded)
    {
        LOG_ERROR("Failed to load draft vocab mapping table from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load draft vocab mapping table from model directory: " + engineDir);
    }

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());
}

// TODO: Remove the loading function from d2t.bin and unify it to use SafeTensor loader.
bool LLMInferenceSpecDecodeRuntime::loadDraftVocabMappingTable(
    std::filesystem::path const& draftVocPath, cudaStream_t stream)
{
    std::ifstream fin(draftVocPath, std::ios::binary);
    if (!fin)
    {
        LOG_ERROR("Failed to open d2t.bin in path %s, it must be provided for Eagle3.", draftVocPath.c_str());
        return false;
    }
    fin.seekg(0, std::ios::end);
    std::streamsize fileSize = fin.tellg();
    int32_t const draftVocabByteSize = mDraftEngineConfig.draftModelVocabSize * sizeof(int32_t);
    if (static_cast<int32_t>(fileSize) != draftVocabByteSize)
    {
        LOG_ERROR(
            "d2t.bin file size mismatch. Got: %d, Expected: %d", static_cast<int32_t>(fileSize), draftVocabByteSize);
        return false;
    }
    std::vector<int32_t> draftVocHost(mDraftEngineConfig.draftModelVocabSize);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(draftVocHost.data()), draftVocabByteSize);
    if (!fin)
    {
        LOG_ERROR("Failed to read d2t.bin in path %s", draftVocPath.c_str());
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mDraftVocabMappingTable.rawPointer(), draftVocHost.data(), draftVocabByteSize, cudaMemcpyHostToDevice, stream));
    LOG_INFO("Draft vocab mapping table successfully loaded from model directory: %s", draftVocPath.c_str());
    return true;
}

bool LLMInferenceSpecDecodeRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    if (request.prompts.size() != kRUNTIME_BATCH_SIZE)
    {
        LOG_ERROR("Only %d batch size is supported by current implementation. Supplied batch size: %zu",
            kRUNTIME_BATCH_SIZE, request.prompts.size());
        return false;
    }

    LLMGenerationRequest::Prompt const& prompt = request.prompts[0];
    std::string const inputText = prompt.systemPrompt + prompt.userPrompt;
    // set tokenizer to ignore special tokens for debugging with python
    std::vector<int32_t> const tokenizedInputIds = mTokenizer->encode(inputText, false);
    if (tokenizedInputIds.empty())
    {
        LOG_ERROR("Failed to tokenize input text: %s", inputText.c_str());
        return false;
    }
    // The boundary case for KVCache is not handled, during execution we need to write drafting KVCache.
    // Workaround the issue by enforce max input sequence length smaller than (KVCacheCapacity - 100)
    constexpr int32_t kDRAFT_KVCACHE_RESERVE_LENGTH{100};
    int32_t const prefillContextLength = tokenizedInputIds.size();
    int32_t const kvCacheCapacity
        = std::max(mBaseEngineConfig.maxSequenceLength, mDraftEngineConfig.kvCacheCapacityLength);
    int32_t maxGenerateLength = request.maxGenerateLength;
    if (prefillContextLength + request.maxGenerateLength > (kvCacheCapacity - kDRAFT_KVCACHE_RESERVE_LENGTH))
    {
        maxGenerateLength = kvCacheCapacity - prefillContextLength - kDRAFT_KVCACHE_RESERVE_LENGTH;
        LOG_WARNING(
            "With Eagle3, we need to write drafting KVCache which constrain us on sequence generation."
            "Reduce max Generation length to %d",
            maxGenerateLength);
    }

    SpecDecodeInferenceContext context{tokenizedInputIds, 0, maxGenerateLength, 0, stream};
    auto checkGenerateEndStatus = [this](SpecDecodeInferenceContext& context) {
        bool flag = (context.currentGenerateLength >= context.maxGenerateLength)
            || (context.tokenIds.back() == mTokenizer->getEosId());
        return flag;
    };

    // Prefill from the base model and run spec-decode inference.
    bool const prefillStatus = runBaseModelPrefill(context);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }
    while (!checkGenerateEndStatus(context))
    {
        if (context.generationRound == 0)
        {
            // Run draft model prefill.
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

        // Produce draft tree for Eagle decoding.
        bool const draftTreeConstructionStatus = constructDraftTree(context);
        if (!draftTreeConstructionStatus)
        {
            LOG_ERROR("Failed to construct draft tree.");
            return false;
        }

        // Run base model verification.
        bool const baseModelVerificationStatus = runBaseModelVerification(context);
        if (!baseModelVerificationStatus)
        {
            LOG_ERROR("Failed to verify token draft tree with base model.");
            return false;
        }
        context.generationRound += 1;
    }

    // Save output ids and decoded texts to response.
    response.outputIds.clear();
    response.outputTexts.clear();
    for (int32_t i = 0; i < kRUNTIME_BATCH_SIZE; ++i)
    {
        auto const acception_rate = (float) context.currentGenerateLength / context.generationRound;
        LOG_INFO("Acception_rate: %f newLen: %d iterNum: %d\n", acception_rate, context.currentGenerateLength,
            context.generationRound);

        response.outputIds.emplace_back(context.tokenIds.begin() + prefillContextLength, context.tokenIds.end());
        response.outputTexts.emplace_back(mTokenizer->decode(response.outputIds[i], true));
    }

    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelPrefill(SpecDecodeInferenceContext& context)
{
    // Prepare the inputs for prefill stage execution.
    int32_t const inputIdsLength = static_cast<int32_t>(context.tokenIds.size());
    if (inputIdsLength > mBaseEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("Input ids length %d is greater than the max supported input length %d", inputIdsLength,
            mBaseEngineConfig.maxSupportedInputLength);
        return false;
    }
    mIdsInput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength});
    mContextLengthsInput.reshape({kRUNTIME_BATCH_SIZE});
    mBaseHiddenStatesOutput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength, mBaseEngineConfig.outputHiddenDim});
    mLogitsOutput.reshape({kRUNTIME_BATCH_SIZE, mBaseEngineConfig.vocabSize});

    // Currently leave the multimodal embeddings empty.
    rt::Tensor multimodalEmbeddings{};
    // Setup the input tensors. ContextLen input is on CPU.
    int32_t* ctxLenData = mContextLengthsInput.dataPointer<int32_t>();
    ctxLenData[0] = inputIdsLength;
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), context.tokenIds.data(), inputIdsLength * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    // Reset status of the KVCache for the new sequence.
    std::vector<int32_t> reuseKVCacheLengths{0};
    rt::Tensor const reuseKVCacheLengthsTensor{reuseKVCacheLengths.data(), {1}, DeviceType::kCPU, DataType::kINT32};
    mBaseEngineRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengthsTensor, context.stream);

    bool const prefillSuccess = mBaseEngineRunner->executePrefillStep(mIdsInput, mContextLengthsInput,
        multimodalEmbeddings, mLogitsOutput, std::ref(mBaseHiddenStatesOutput), context.stream);
    if (!prefillSuccess)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    // Sampling from the Prefill stage logits using greedy Top1 sampling, only collect the top1 index.
    mSamplingIndices.reshape({kRUNTIME_BATCH_SIZE});
    constexpr int32_t kSAMPLING_TOP_K = 1;
    selectAllTopKFromLogits(mLogitsOutput.dataPointer<float>(), nullptr, mSamplingIndices.dataPointer<int32_t>(),
        kRUNTIME_BATCH_SIZE, mBaseEngineConfig.vocabSize, kSAMPLING_TOP_K, mSamplingWorkspace.rawPointer(),
        mSamplingWorkspace.getMemoryCapacity(), context.stream, false, false, false);

    // Pull the sampling indices from device to host.
    int32_t selectedTokenId;
    CUDA_CHECK(cudaMemcpyAsync(
        &selectedTokenId, mSamplingIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    context.tokenIds.push_back(selectedTokenId);
    context.currentGenerateLength += 1;
    // The base prefill function produce output logits and (concatenated) hiddenStates for next step to use.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelPrefill(SpecDecodeInferenceContext& context)
{
    // Implement the draft prefill execution logic, prepare the input ids and hidden states inputs for the
    // eagle draft engine. The formulation of the feature "vector" is F_n = F(H_n, Token_{n+1}), therefore we
    // need to trim out the first token of the sequence from the token_ids input.
    int32_t const inputIdsLength = static_cast<int32_t>(context.tokenIds.size()) - 1;
    check::check(mBaseHiddenStatesOutput.getShape()[1] == inputIdsLength,
        "BaseHiddenStatesOutput shall match with inputIdsLength");

    // Reset status of the draft engine KVCache for the new sequence.
    std::vector<int32_t> reuseKVCacheLengths{0};
    rt::Tensor const reuseKVCacheLengthsTensor{reuseKVCacheLengths.data(), {1}, DeviceType::kCPU, DataType::kINT32};
    mDraftEngineRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengthsTensor, context.stream);

    // Prepare input and output tensors.
    mIdsInput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength});
    mDraftHiddenStatesInput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength, mDraftEngineConfig.draftModelHiddenDim});
    mLogitsOutput.reshape({kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelHiddenDim});

    // Clear garbage data in the draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));
    // Shift the data pointer by 1 to start from the second token.
    int32_t const* inputIdsData = context.tokenIds.data() + 1;
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), inputIdsData, inputIdsLength * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    bool const prefillSuccess = mDraftEngineRunner->executeEaglePrefillStep(mIdsInput, mBaseHiddenStatesOutput,
        mDraftHiddenStatesInput, mLogitsOutput, mDraftHiddenStatesOutput, context.stream);
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
    // Core logic for eagle speculative decoding, construct the draft tree in an auto-regressive manner./
    // Inputs: Logits (mLogitsOutput) and draft hidden states (mDraftHiddenStatesOutput) from draft prefill
    // or draft model accept decoding operation.
    // Construct the draft tree table with multiple round of drafting. The descriptions of the draft tree are
    // stored in mDraftTokenIdsTable, mDraftTokenScoreTable, mDraftTokenPredecessorTable.

    // Record root token (last committed token selected by base model) id for the draft tree.
    int32_t const rootTokenId = context.tokenIds.back();
    CUDA_CHECK(cudaMemcpyAsync(
        mDraftTreeRootTokenId.rawPointer(), &rootTokenId, sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Sampling from the logits output, collect draftTopK tokens as first level under "root".
    int32_t const draftTopK = mDraftingConfig.draftingTopK;
    mSamplingIndices.reshape({kRUNTIME_BATCH_SIZE, draftTopK});
    mSamplingScores.reshape({kRUNTIME_BATCH_SIZE, draftTopK});
    selectAllTopKFromLogits(mLogitsOutput.dataPointer<float>(), mSamplingScores.dataPointer<float>(),
        mSamplingIndices.dataPointer<int32_t>(), kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelVocabSize, draftTopK,
        mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(), context.stream);

    // Initialize data structures to describe the whole draft tree.
    kernel::initializeDraftTreeTables(mSamplingIndices, mSamplingScores, mDraftTreeRootTokenId, mDraftVocabMappingTable,
        mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, context.stream);
    // Reset hidden states output of base model and input hidden states of draft model to clear the garbage data.
    CUDA_CHECK(cudaMemsetAsync(
        mBaseHiddenStatesOutput.rawPointer(), 0, mBaseHiddenStatesOutput.getMemoryCapacity(), context.stream));
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));

    // Construct input tensors to feed into the eagle draft engine. With current implementation, for simplicity, we
    // will use padded input amd only collect results from indices we need.
    int32_t const paddedDraftTreeSize = mDraftingConfig.draftingStep * draftTopK;
    mIdsInput.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize});
    mBaseHiddenStatesOutput.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize, mBaseEngineConfig.outputHiddenDim});
    mDraftHiddenStatesInput.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize, mDraftEngineConfig.draftModelHiddenDim});
    mDraftTreeSize.reshape({kRUNTIME_BATCH_SIZE});
    mDraftTreeMask.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize, paddedDraftTreeSize});
    // Assemble the initial draft tree input here since we need to copy out the data in draftHiddenStatesOutput prior to
    // reshaping it.
    kernel::assembleInitialDraftTreeInput(mDraftTokenIdsFullTable, mDraftHiddenStatesOutput, mIdsInput,
        mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, draftTopK, context.stream);
    mLogitsOutput.reshape({draftTopK, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({draftTopK, mDraftEngineConfig.draftModelHiddenDim});

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
            mSamplingIndices.reshape({kRUNTIME_BATCH_SIZE, draftTopK});
            mSamplingScores.reshape({kRUNTIME_BATCH_SIZE, draftTopK});
            selectAllTopKFromLogits(mDraftTokenScoresTable.dataPointer<float>(), mSamplingScores.dataPointer<float>(),
                mSamplingIndices.dataPointer<int32_t>(), kRUNTIME_BATCH_SIZE, draftTopK * draftTopK, draftTopK,
                mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(), context.stream);
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mDraftHiddenStatesOutput, mSamplingIndices, mIdsInput,
                mDraftHiddenStatesInput, mDraftTreeSize, mDraftTreeMask, draftTopK, round, context.stream);
            kernel::assembleIntermediateData(mSamplingScores, mSamplingIndices, mDraftTokenIntermediateScores,
                mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

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
        mSamplingIndices.reshape({kRUNTIME_BATCH_SIZE, draftTopK * draftTopK});
        mSamplingScores.reshape({kRUNTIME_BATCH_SIZE, draftTopK * draftTopK});
        selectAllTopKFromLogits(mLogitsOutput.dataPointer<float>(), mSamplingScores.dataPointer<float>(),
            mSamplingIndices.dataPointer<int32_t>(), kRUNTIME_BATCH_SIZE * draftTopK,
            mDraftEngineConfig.draftModelVocabSize, draftTopK, mSamplingWorkspace.rawPointer(),
            mSamplingWorkspace.getMemoryCapacity(), context.stream);

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
    mSamplingIndices.reshape({kRUNTIME_BATCH_SIZE, mDraftingConfig.verifyTreeSize});
    int64_t const fullDraftTableSize = mDraftTokenScoreFullTable.getShape()[1];
    selectAllTopKFromLogits(mDraftTokenScoreFullTable.dataPointer<float>(), nullptr,
        mSamplingIndices.dataPointer<int32_t>(), kRUNTIME_BATCH_SIZE, fullDraftTableSize,
        mDraftingConfig.verifyTreeSize, mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(),
        context.stream);

    mIdsInput.reshape({kRUNTIME_BATCH_SIZE, mDraftingConfig.verifyTreeSize});
    mDraftTreeMask.reshape({kRUNTIME_BATCH_SIZE, mDraftingConfig.verifyTreeSize, mDraftingConfig.verifyTreeSize});
    kernel::constructVerificationDraftTree(mDraftTokenIdsFullTable, mDraftTokenPredecessorFullTable, mSamplingIndices,
        mIdsInput, mDraftTreeMask, context.stream);

    // This function will produce mIdsInput and mDraftTreeMask to describe established draft tree.
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runBaseModelVerification(SpecDecodeInferenceContext& context)
{
    // This function will consume idsInput and draftTreeMask. Use base model to verify the draft tree.
    // We need to collect the logits and hidden states (for further drafting step).
    check::check(
        mIdsInput.getShape()[0] == kRUNTIME_BATCH_SIZE && mIdsInput.getShape()[1] == mDraftingConfig.verifyTreeSize,
        "IdsInput shall have shape [batch_size, verify_tree_size]");
    check::check(mDraftTreeMask.getShape()[0] == kRUNTIME_BATCH_SIZE
            && mDraftTreeMask.getShape()[1] == mDraftingConfig.verifyTreeSize
            && mDraftTreeMask.getShape()[2] == mDraftingConfig.verifyTreeSize,
        "DraftTreeMask shall have shape [batch_size, verify_tree_size, verify_tree_size]");
    mLogitsOutput.reshape({mDraftingConfig.verifyTreeSize, mBaseEngineConfig.vocabSize});
    mBaseHiddenStatesOutput.reshape({mDraftingConfig.verifyTreeSize, mBaseEngineConfig.outputHiddenDim});

    // No need to pass in multimodal embeddings.
    rt::Tensor const multimodalEmbeddings{};
    bool const verifySuccess = mBaseEngineRunner->executeEagleBaseTreeDecodingStep(
        mIdsInput, mDraftTreeMask, multimodalEmbeddings, mLogitsOutput, mBaseHiddenStatesOutput, context.stream);
    if (!verifySuccess)
    {
        LOG_ERROR("Failed to execute base tree verification step for base model.");
        return false;
    }
    // Collected accepted token ids and indices. Use sampling workspace for eagle accept process.
    kernel::eagleAccept(mLogitsOutput, mIdsInput, mDraftTreeMask, mAcceptedTokenIds, mAcceptedTokenIndices,
        mAcceptLength, mSamplingWorkspace.rawPointer(), mSamplingWorkspace.getMemoryCapacity(), context.stream);

    // Inplace update the KVCache and input hidden states from the accepted token indices.
    // Also commit KVCache to reflect the latest KVCache length (We can only do this after knowing how many tokens are
    // accepted).
    rt::Tensor const& kvCacheLengths = mBaseEngineRunner->getLinearKVCache().getKVCacheLengths();
    rt::Tensor kvCacheTensor = mBaseEngineRunner->getLinearKVCache().getKVCacheBuffer();
    kernel::eagleBaseCommitKVCacheAndAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, kvCacheTensor, mBaseHiddenStatesOutput, context.stream);
    mBaseEngineRunner->getLinearKVCache().commitSequenceLength(mAcceptLength, context.stream);

    int32_t acceptLength;
    std::vector<int32_t> acceptedTokenIds(mAcceptedTokenIds.getShape().volume());
    // Pull collected results from device to host record the selected tokens.
    CUDA_CHECK(cudaMemcpyAsync(
        &acceptLength, mAcceptLength.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(acceptedTokenIds.data(), mAcceptedTokenIds.rawPointer(),
        mAcceptedTokenIds.getShape().volume() * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    for (int32_t i = 0; i < acceptLength; i++)
    {
        context.tokenIds.push_back(acceptedTokenIds[i]);
    }
    context.currentGenerateLength += acceptLength;

    // Produce base model hidden states for the next step. Reshape to accepted length.
    // FIXME: Perform the reshape but model input and output for hidden states have different semantics.
    // The function produce hidden states (mBaseHiddenStatesOutput) and input ids (attached in inference context).
    mBaseHiddenStatesOutput.reshape({kRUNTIME_BATCH_SIZE, acceptLength, mBaseEngineConfig.outputHiddenDim});
    return true;
}

bool LLMInferenceSpecDecodeRuntime::runDraftModelAcceptToken(SpecDecodeInferenceContext& context)
{
    // Base model verifiction function is responsible for producing the output with correct shape.
    int64_t const inputIdsLength = mBaseHiddenStatesOutput.getShape()[1];

    // Prepare input and output tensors.
    mIdsInput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength});
    mDraftHiddenStatesInput.reshape({kRUNTIME_BATCH_SIZE, inputIdsLength, mDraftEngineConfig.draftModelHiddenDim});
    mLogitsOutput.reshape({kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelVocabSize});
    mDraftHiddenStatesOutput.reshape({kRUNTIME_BATCH_SIZE, mDraftEngineConfig.draftModelHiddenDim});

    // Clear garbage data in the draft hidden inputs.
    CUDA_CHECK(cudaMemsetAsync(
        mDraftHiddenStatesInput.rawPointer(), 0, mDraftHiddenStatesInput.getMemoryCapacity(), context.stream));

    // Prepare the IdsInput, we need to copy from mAcceptedTokenIds.
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), mAcceptedTokenIds.rawPointer(), inputIdsLength * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, context.stream));

    bool const acceptTokenSuccess = mDraftEngineRunner->executeEagleAcceptDecodeTokenStep(mIdsInput,
        mBaseHiddenStatesOutput, mDraftHiddenStatesInput, mLogitsOutput, mDraftHiddenStatesOutput, context.stream);
    if (!acceptTokenSuccess)
    {
        LOG_ERROR("Failed to execute accept token step for draft model.");
        return false;
    }
    // No need to do sampling, directly produce logits (mLogitsOutput) and hidden states (mDraftHiddenStatesOutput) for
    // the next step.
    return true;
}

} // namespace rt
} // namespace drivellm