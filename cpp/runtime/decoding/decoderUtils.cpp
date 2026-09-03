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

#include "runtime/decoding/decoderUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/debug/layerDebugger.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace decoder_utils
{
void zeroActiveRegion(Tensor& tensor, cudaStream_t stream)
{
    auto const bytes = static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
    CUDA_CHECK(cudaMemsetAsync(tensor.rawPointer(), 0, bytes, stream));
}

std::unique_ptr<EngineExecutor> loadDraftEngine(
    std::filesystem::path const& engineDir, DeploymentConfig const& deployment)
{
    std::filesystem::path const draftEnginePath = engineDir / "spec_draft.engine";
    std::unique_ptr<EngineExecutor> draftExecutor;
    try
    {
        draftExecutor = EngineExecutor::createForDraft(draftEnginePath, deployment);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize draft EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize draft EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Draft EngineExecutor successfully loaded from %s.", draftEnginePath.c_str());
    validateAgainstEngine(*deployment.draft, *draftExecutor, "draft");
    return draftExecutor;
}

void appendSampledTokens(DecodingInferenceContext& context, int32_t const* sampledTokenIds, int32_t activeBatchSize)
{
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            continue;
        }
        context.tokenIds[i].push_back(sampledTokenIds[i]);
        context.currentGenerateLengths[i] += 1;
    }
}

void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream, int32_t proposedDraftsPerRound,
    int32_t const* perSlotProposedDrafts)
{
    int32_t const activeBatchSize = context.activeBatchSize;

    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(hostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* hostAcceptLengthsData = hostAcceptLengths.dataPointer<int32_t>();
    int32_t* hostAcceptedTokenIdsData = hostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengthsData, deviceAcceptLength.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostAcceptedTokenIdsData, deviceAcceptedTokenIds.rawPointer(),
        activeBatchSize * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        if (context.finishedStates[batchIdx])
        {
            hostAcceptLengthsData[batchIdx] = 0;
            continue;
        }
        int32_t const acceptLength = hostAcceptLengthsData[batchIdx];
        int32_t const proposedDrafts = perSlotProposedDrafts ? perSlotProposedDrafts[batchIdx] : proposedDraftsPerRound;
        int32_t appended = 0;
        for (int32_t i = 0; i < acceptLength; i++)
        {
            int32_t const token = hostAcceptedTokenIdsData[batchIdx * maxAcceptDepth + i];
            context.tokenIds[batchIdx].push_back(token);
            context.currentGenerateLengths[batchIdx]++;
            ++appended;
            bool const shouldStop = context.shouldStopAfterAcceptedToken
                ? context.shouldStopAfterAcceptedToken(batchIdx, token)
                : token == tokenizer.getEosId();
            if (shouldStop)
            {
                break;
            }
        }
        // EOS / stop can end the slot mid-accept; write back the appended count so
        // collectSpecLogprobsFromHost skips verify rows past the end of the sequence.
        hostAcceptLengthsData[batchIdx] = appended;
        // Full accept: last token is the target-sampled bonus, not a draft, so -1.
        // EOS truncation: count drafts up to and including the EOS.
        int32_t const acceptedDrafts
            = std::min(std::max(0, appended - (appended == acceptLength ? 1 : 0)), proposedDrafts);
        context.acceptedDraftTokens[batchIdx] += acceptedDrafts;
        context.proposedDraftTokens[batchIdx] += proposedDrafts;
        LOG_DEBUG(
            "SpecRound %d batchIdx %d: generated=%d, acceptedDrafts=%d, proposedDrafts=%d, "
            "cumGenerateLength=%d",
            context.generationRound, batchIdx, appended, acceptedDrafts, proposedDrafts,
            context.currentGenerateLengths[batchIdx]);
    }
}

void clampAcceptLengthsToRemainingGeneration(
    DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& deviceAcceptLength, cudaStream_t stream)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostAcceptLengthsData = hostAcceptLengths.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengthsData, deviceAcceptLength.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool changed = false;
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int32_t const remaining = std::max(0, context.maxGenerateLength - context.currentGenerateLengths[batchIdx]);
        int32_t const clamped = std::max(0, std::min(hostAcceptLengthsData[batchIdx], remaining));
        changed |= clamped != hostAcceptLengthsData[batchIdx];
        hostAcceptLengthsData[batchIdx] = clamped;
    }

    if (changed)
    {
        CUDA_CHECK(cudaMemcpyAsync(deviceAcceptLength.rawPointer(), hostAcceptLengthsData,
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
}

void applyForcedAcceptance(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor& deviceAcceptLength, Tensor& deviceAcceptedTokenIds, std::vector<int32_t>& ownTokens, int32_t maxAcceptDepth,
    cudaStream_t stream)
{
    ownTokens.clear();
    if (context.layerDebugger == nullptr || !context.layerDebugger->hasForcedTokens())
    {
        return;
    }
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(hostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* lengths = hostAcceptLengths.dataPointer<int32_t>();
    int32_t* tokens = hostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(
        lengths, deviceAcceptLength.rawPointer(), activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tokens, deviceAcceptedTokenIds.rawPointer(),
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!context.layerDebugger->applyForcedAcceptance(context.currentGenerateLengths, context.batchIndexMapping,
            lengths, tokens, ownTokens, activeBatchSize, maxAcceptDepth))
    {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        deviceAcceptLength.rawPointer(), lengths, activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceAcceptedTokenIds.rawPointer(), tokens,
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void dumpSpecRound(DecodingInferenceContext& context, HybridCacheManager& cacheManager, KVPageTable const& pageTable,
    Tensor const& verifyLogits, Tensor const& acceptedTokenIndices, Tensor const& hostAcceptLengths,
    std::vector<int32_t> const& ownTokens, int32_t verifySize, int32_t maxAcceptDepth, cudaStream_t stream)
{
    if (context.layerDebugger == nullptr)
    {
        return;
    }
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const* lengths = hostAcceptLengths.dataPointer<int32_t>();

    // Of the verify block's rows only the last accepted one has a counterpart in a vanilla golden:
    // it is the row that predicts the token this round leaves pending, which is exactly what the
    // golden's last-token logits are.
    Tensor hostIndices(
        {activeBatchSize, maxAcceptDepth}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "spec_accepted_token_indices");
    auto const* indices = hostIndices.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostIndices.rawPointer(), acceptedTokenIndices.rawPointer(),
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Coords const shape = verifyLogits.getShape();
    int64_t const vocab = shape[shape.getNumDims() - 1];
    nvinfer1::DataType const dtype = verifyLogits.getDataType();
    size_t const rowBytes = static_cast<size_t>(vocab) * utils::getTypeSize(dtype);
    Tensor bonusLogits({activeBatchSize, vocab}, DeviceType::kGPU, dtype, "spec_bonus_logits");
    auto* dst = static_cast<char*>(bonusLogits.rawPointer());
    auto const* src = static_cast<char const*>(verifyLogits.rawPointer());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const slot = std::max(0, lengths[i] - 1);
        int32_t const row = indices[static_cast<size_t>(i) * maxAcceptDepth + slot];
        size_t const offset = (static_cast<size_t>(i) * verifySize + row) * rowBytes;
        CUDA_CHECK(cudaMemcpyAsync(dst + i * rowBytes, src + offset, rowBytes, cudaMemcpyDeviceToDevice, stream));
    }

    // The bonus token is next round's input and has no cache entry yet, so the committed cache is
    // one shorter than the token list -- matching what the vanilla path reports, where the dump
    // runs before the sampled token is pushed.
    std::vector<int32_t> validLengths(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size()) - 1;
    }
    context.layerDebugger->dumpRound(cacheManager, pageTable, bonusLogits, validLengths, context.batchIndexMapping,
        ownTokens.empty() ? nullptr : ownTokens.data(), activeBatchSize, stream);
}

void enqueueLogprobsD2H(
    Tensor const& inputLogits, int32_t rows, DecodingRuntimeContext& runtime, int32_t topK, cudaStream_t stream)
{
    check::check(runtime.logprobs.deviceLogprobsValues.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.deviceLogprobsIndices.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.hostLogprobsValues.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.hostLogprobsIndices.reshape({rows, topK}), "Tensor reshape failed");
    extractTopKLogprobs(inputLogits, runtime.logprobs.deviceLogprobsValues, runtime.logprobs.deviceLogprobsIndices,
        topK, runtime.sampling.workspace, stream);
    if (runtime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(
            runtime.logprobs.deviceLogprobsIndices, runtime.sampling.baseVocabMappingTable, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(runtime.logprobs.hostLogprobsValues.rawPointer(),
        runtime.logprobs.deviceLogprobsValues.rawPointer(), static_cast<size_t>(rows) * topK * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(runtime.logprobs.hostLogprobsIndices.rawPointer(),
        runtime.logprobs.deviceLogprobsIndices.rawPointer(), static_cast<size_t>(rows) * topK * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
}

void collectLogprobsFromHost(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, int32_t activeBatchSize, int32_t topK)
{
    float const* hostVals = runtime.logprobs.hostLogprobsValues.dataPointer<float>();
    int32_t const* hostIdx = runtime.logprobs.hostLogprobsIndices.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
            continue;
        rt::LogprobsSlot& slot = context.stepLogprobs[i];
        int32_t const base = slot.numSteps * topK;
        check::check(static_cast<size_t>(base + topK) <= slot.data.size(), "logprobs slot overflow");
        for (int32_t k = 0; k < topK; ++k)
            slot.data[base + k] = {hostIdx[i * topK + k], hostVals[i * topK + k]};
        ++slot.numSteps;
    }
}

void collectSpecLogprobsFromHost(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    int32_t activeBatchSize, int32_t rowsPerBatch, int32_t const* hostAcceptLens, int32_t topK)
{
    float const* hostVals = runtime.logprobs.hostLogprobsValues.dataPointer<float>();
    int32_t const* hostIdx = runtime.logprobs.hostLogprobsIndices.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
            continue;
        int32_t const acceptLen = hostAcceptLens[i];
        rt::LogprobsSlot& slot = context.stepLogprobs[i];
        for (int32_t j = 0; j < acceptLen; ++j)
        {
            int32_t const row = i * rowsPerBatch + j;
            int32_t const base = slot.numSteps * topK;
            check::check(static_cast<size_t>(base + topK) <= slot.data.size(), "logprobs slot overflow");
            for (int32_t k = 0; k < topK; ++k)
                slot.data[base + k] = {hostIdx[row * topK + k], hostVals[row * topK + k]};
            ++slot.numSteps;
        }
    }
}

} // namespace decoder_utils
} // namespace rt
} // namespace trt_edgellm
