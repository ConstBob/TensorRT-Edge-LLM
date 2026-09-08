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

#include "runtime/state/decodingInferenceContext.h"

#include "common/checkMacros.h"
#include "runtime/debug/layerDebugger.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

// Out-of-line so the unique_ptr<LayerDebugger> member can hold an incomplete type
// in the header; LayerDebugger is complete here. Move ops are defined too: a
// user-declared destructor suppresses the implicit move, but the context is
// returned by value (e.g. tests' makeContext), so it must stay movable. It
// remains non-copyable through the unique_ptr member.
DecodingInferenceContext::DecodingInferenceContext() = default;
DecodingInferenceContext::DecodingInferenceContext(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext& DecodingInferenceContext::operator=(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext::~DecodingInferenceContext() = default;

int32_t DecodingInferenceContext::appendSlot(SlotSeed seed)
{
    ELLM_CHECK(activeBatchSize > 0, "appendSlot joins a running batch; an empty one is built by initialize().");
    ELLM_CHECK(!seed.promptTokenIds.empty(), "appendSlot requires a non-empty prompt.");
    ELLM_CHECK(seed.prefillStart >= 0 && seed.prefillStart < static_cast<int32_t>(seed.promptTokenIds.size()),
        "appendSlot: the reused prefix must leave at least one token to prefill.");
    ELLM_CHECK(perSlotSizesConsistent(), "appendSlot found the batch already inconsistent.");
    for (int32_t const original : batchIndexMapping)
    {
        ELLM_CHECK(original != seed.originalIndex, "appendSlot: original index is held by a live slot.");
    }
    ELLM_CHECK(completedBatches.find(seed.originalIndex) == completedBatches.end(),
        "appendSlot: original index already has a collected result.");

    // The attach is the last operation that can throw, so a rejected channel (attached elsewhere)
    // leaves the batch exactly as it was. The push_backs after it only fail by bad_alloc.
    if (seed.channel)
    {
        attachStreamChannel(seed.channel, seed.originalIndex);
    }

    size_t maxStopLen = 0;
    for (auto const& stop : seed.stopStrings)
    {
        maxStopLen = std::max(maxStopLen, stop.size());
    }

    // The working history holds only what this batch must compute: everything from prefillStart
    // on. Founder slots under a context cache are seeded the same way (setUpForPrefillExecution),
    // so streaming offsets and generate-length math see one shape regardless of how a slot joined.
    size_t const promptLength = seed.promptTokenIds.size() - static_cast<size_t>(seed.prefillStart);

    systemPrompts.push_back(std::move(seed.systemPrompt));
    rawBatchedInputIds.push_back(seed.promptTokenIds);
    tokenIds.emplace_back(seed.promptTokenIds.begin() + seed.prefillStart, seed.promptTokenIds.end());
    currentGenerateLengths.push_back(0);
    samplingSeeds.push_back(seed.samplingSeed);
    // Spec decode never takes admissions, but the counters are sized for every deployment.
    acceptedDraftTokens.push_back(0);
    proposedDraftTokens.push_back(0);
    effectivePrefillLengths.push_back(0);
    finishedStates.push_back(0);
    batchIndexMapping.push_back(seed.originalIndex);
    // Reasoning trackers: a fresh slot has produced no tokens, so neither latch is set. Guided
    // admission itself is refused upstream (the GuidedDecoder's matchers are slot-numbered and the
    // seating swaps do not cover them), but the vectors stay rectangular either way.
    thinkingDone.push_back(0);
    guidedReasoningEnded.push_back(0);

    // Seeded to the prompt length so streaming and callbacks emit only generated tokens, exactly as
    // a slot set up at request start is seeded.
    callbackEmittedTokenCounts.push_back(static_cast<int32_t>(promptLength));
    SlotStreamState slotState;
    slotState.channel = std::move(seed.channel);
    slotState.sentTokenCount = promptLength;
    slotState.lastEmittedTokenCount = promptLength;
    slotState.maxStopLen = maxStopLen;
    slotStreams.push_back(std::move(slotState));

    stopStringsPerSlot.push_back(std::move(seed.stopStrings));

    logitBiasPerSlot.push_back(std::move(seed.logitBias));
    hasLogitBias = hasLogitBias || !logitBiasPerSlot.back().empty();
    // The GPU-side table is indexed by slot, so any bias anywhere means the grown table must go up
    // again -- mirroring what eviction does after it compacts.
    logitBiasGpuDirty = hasLogitBias;

    // Capacity is deployment-dependent (spec decode can accept several tokens per step), so it is
    // inherited from a slot that was sized by the code that knows -- the reason this method refuses
    // an empty batch.
    rt::LogprobsSlot logprobs;
    if (numLogprobs > 0)
    {
        logprobs.data.resize(stepLogprobs.front().data.size());
        logprobs.numSteps = 0;
    }
    stepLogprobs.push_back(std::move(logprobs));

    // Conditional vector: sized only while visual-token pruning is active. Grown when present so a
    // per-slot read stays aligned, left empty otherwise.
    if (!prunedPrefillTokens.empty())
    {
        prunedPrefillTokens.push_back(0);
    }

    ++activeBatchSize;
    ELLM_CHECK(perSlotSizesConsistent(), "appendSlot left the batch inconsistent; a vector was missed.");
    return activeBatchSize - 1;
}

void DecodingInferenceContext::swapSlots(int32_t slotA, int32_t slotB)
{
    ELLM_CHECK(slotA >= 0 && slotA < activeBatchSize && slotB >= 0 && slotB < activeBatchSize,
        "swapSlots: slot is out of range.");
    ELLM_CHECK(perSlotSizesConsistent(), "swapSlots found the batch inconsistent.");
    if (slotA == slotB)
    {
        return;
    }

    auto const a = static_cast<size_t>(slotA);
    auto const b = static_cast<size_t>(slotB);
    std::swap(systemPrompts[a], systemPrompts[b]);
    std::swap(rawBatchedInputIds[a], rawBatchedInputIds[b]);
    std::swap(tokenIds[a], tokenIds[b]);
    std::swap(currentGenerateLengths[a], currentGenerateLengths[b]);
    std::swap(effectivePrefillLengths[a], effectivePrefillLengths[b]);
    std::swap(finishedStates[a], finishedStates[b]);
    std::swap(batchIndexMapping[a], batchIndexMapping[b]);
    std::swap(thinkingDone[a], thinkingDone[b]);
    std::swap(guidedReasoningEnded[a], guidedReasoningEnded[b]);
    std::swap(callbackEmittedTokenCounts[a], callbackEmittedTokenCounts[b]);
    std::swap(slotStreams[a], slotStreams[b]);
    std::swap(stopStringsPerSlot[a], stopStringsPerSlot[b]);
    std::swap(samplingSeeds[a], samplingSeeds[b]);
    std::swap(acceptedDraftTokens[a], acceptedDraftTokens[b]);
    std::swap(proposedDraftTokens[a], proposedDraftTokens[b]);
    std::swap(logitBiasPerSlot[a], logitBiasPerSlot[b]);
    std::swap(stepLogprobs[a], stepLogprobs[b]);
    // hasLogitBias / logitBiasGpuDirty are batch-wide, but the GPU-side table is slot-indexed, so a
    // swap that moved any bias must be pushed up again before the next step reads it.
    if (hasLogitBias)
    {
        logitBiasGpuDirty = true;
    }
    if (!prunedPrefillTokens.empty())
    {
        std::swap(prunedPrefillTokens[a], prunedPrefillTokens[b]);
    }
}

bool DecodingInferenceContext::perSlotSizesConsistent() const noexcept
{
    auto const expected = static_cast<size_t>(activeBatchSize);
    return systemPrompts.size() == expected && rawBatchedInputIds.size() == expected && tokenIds.size() == expected
        && currentGenerateLengths.size() == expected && samplingSeeds.size() == expected
        && acceptedDraftTokens.size() == expected && proposedDraftTokens.size() == expected
        && effectivePrefillLengths.size() == expected && finishedStates.size() == expected
        && batchIndexMapping.size() == expected && thinkingDone.size() == expected
        && guidedReasoningEnded.size() == expected && callbackEmittedTokenCounts.size() == expected
        && slotStreams.size() == expected && stopStringsPerSlot.size() == expected
        && logitBiasPerSlot.size() == expected && stepLogprobs.size() == expected
        && (prunedPrefillTokens.empty() || prunedPrefillTokens.size() == expected);
}

void DecodingInferenceContext::initialize(int32_t batchSize, int32_t maxGenLength,
    rt::OptionalInputTensor const& visual, rt::OptionalInputTensors const& deepstack, std::string const& loraName,
    cudaStream_t cudaStream)
{
    systemPrompts.resize(batchSize);
    rawBatchedInputIds.reserve(batchSize);
    tokenIds.resize(batchSize);
    currentGenerateLengths.resize(batchSize, 0);
    samplingSeeds.resize(batchSize, kDefaultSamplingSeed);
    effectivePrefillLengths.resize(batchSize, 0);
    finishedStates.resize(batchSize, 0);
    thinkingDone.clear();
    thinkingDone.resize(batchSize, 0);
    guidedReasoningEnded.clear();
    guidedReasoningEnded.resize(batchSize, 0);
    acceptedDraftTokens.assign(batchSize, 0);
    proposedDraftTokens.assign(batchSize, 0);
    slotStreams.clear();
    slotStreams.resize(batchSize);
    stopStringsPerSlot.clear();
    stopStringsPerSlot.resize(batchSize);
    logitBiasPerSlot.clear();
    logitBiasPerSlot.resize(batchSize);
    hasLogitBias = false;
    logitBiasGpuDirty = false;
    hasGuidedDecoding = false;
    guidedUnsatisfiableSlots.clear();
    guidedMaskSuppressedPerSlot.clear();
    enableThinking = false;
    callbackEmittedTokenCounts.clear();
    callbackEmittedTokenCounts.resize(batchSize, 0);
    shouldStopAfterAcceptedToken = {};

    batchIndexMapping.resize(batchSize);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        batchIndexMapping[i] = i;
    }

    completedBatches.clear();

    // Initialize per-batch logprobs accumulator (populated only when numLogprobs > 0)
    stepLogprobs.clear();
    stepLogprobs.resize(batchSize);

    visualEmbeddings = visual;
    deepstackFeatures = deepstack;
    generationRound = 0;
    maxGenerateLength = maxGenLength;
    diffusionMaxDenoisingSteps = 0;
    activeBatchSize = batchSize;
    loraWeightsName = loraName;
    stream = cudaStream;
}

} // namespace rt
} // namespace trt_edgellm
