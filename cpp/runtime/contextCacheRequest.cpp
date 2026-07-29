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

#include "runtime/contextCacheRequest.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/streaming.h"

#include <limits>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

ContextCacheLookupPolicy contextCacheLookupPolicy(LLMGenerationRequest const& request, bool outputThinkerEmbeddings)
{
    bool const requiresBypass = request.contextCacheLookupPolicy == ContextCacheLookupPolicy::kBypass
        || request.generateAudio || outputThinkerEmbeddings;
    return requiresBypass ? ContextCacheLookupPolicy::kBypass : ContextCacheLookupPolicy::kUseCache;
}

ContextCacheSequenceAdmission makeContextCacheSequenceAdmission(
    std::vector<int32_t> const& tokenIds, std::string const& loraWeightsName)
{
    ContextCacheSequenceAdmission admission;
    admission.tokenIds = tokenIds;
    if (!loraWeightsName.empty())
    {
        // The runtime-local adapter registry is immutable after construction, so its generation remains zero.
        AdapterKey const adapter{hashOpaqueIdentity(loraWeightsName), 0};
        admission.keyExtras.adapter = adapter;
    }
    return admission;
}

bool contextCacheOperationSucceeded(ContextCacheCoordinatorStatus status, char const* operation)
{
    if (status == ContextCacheCoordinatorStatus::kOk)
    {
        return true;
    }
    LOG_ERROR("Context-cache %s failed (%s).", operation,
        status == ContextCacheCoordinatorStatus::kPoisoned ? "coordinator poisoned" : "request failure");
    return false;
}

} // namespace

std::optional<ContextCacheRequest> ContextCacheRequest::begin(ContextCacheCoordinator& coordinator,
    LLMGenerationRequest const& request, DecodingInferenceContext const& context, DecodingStrategyKind strategyKind)
{
    ELLM_CHECK(strategyKind == DecodingStrategyKind::kVanilla || strategyKind == DecodingStrategyKind::kEAGLE,
        "Context cache supports only vanilla or EAGLE request execution.");

    ContextCacheBatchAdmission admission;
    admission.executionMode = strategyKind == DecodingStrategyKind::kEAGLE ? ContextCacheExecutionMode::kEAGLE
                                                                           : ContextCacheExecutionMode::kVanilla;
    admission.lookupPolicy = contextCacheLookupPolicy(request, context.outputThinkerEmbeddings);
    admission.commitPolicy = request.contextCacheCommitPolicy;
    admission.sequences.reserve(context.rawBatchedInputIds.size());
    for (std::vector<int32_t> const& tokenIds : context.rawBatchedInputIds)
    {
        admission.sequences.push_back(makeContextCacheSequenceAdmission(tokenIds, context.loraWeightsName));
    }

    ContextCacheCoordinator::BeginRequestResult admitted = coordinator.beginRequest(admission, context.stream);
    if (!contextCacheOperationSucceeded(admitted.status, "admission") || !admitted.admission.has_value())
    {
        return std::nullopt;
    }
    return ContextCacheRequest{coordinator, std::move(*admitted.admission)};
}

ContextCacheRequest::ContextCacheRequest(
    ContextCacheCoordinator& coordinator, ContextCacheCoordinator::AdmissionResult&& admission) noexcept
    : mCoordinator(coordinator)
    , mRequest(std::move(admission.request))
    , mPrefillStarts(std::move(admission.prefillStarts))
{
}

std::vector<int32_t> const& ContextCacheRequest::prefillStarts() const noexcept
{
    return mPrefillStarts;
}

bool ContextCacheRequest::preparePrefill()
{
    return contextCacheOperationSucceeded(mCoordinator.preparePrefill(mRequest), "prefill preparation");
}

bool ContextCacheRequest::enqueuePrefillCaptures()
{
    return contextCacheOperationSucceeded(mCoordinator.enqueuePrefillCaptures(mRequest), "prefill snapshot capture");
}

bool ContextCacheRequest::completePrefill(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    std::vector<ContextCacheSequenceAdvance> progress;
    progress.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        ELLM_CHECK(context.currentGenerateLengths[slot] == 1 && !context.tokenIds[slot].empty(),
            "Managed context-cache prefill did not produce one sampled lookahead token");
        progress.push_back(ContextCacheSequenceAdvance{
            &context.tokenIds[slot].back(), 1, static_cast<int32_t>(context.rawBatchedInputIds[slot].size())});
    }
    std::vector<int32_t> const* const commonStateLengthsPtr
        = commonStateLengths.empty() ? nullptr : &commonStateLengths;
    return contextCacheOperationSucceeded(
        mCoordinator.finalizePrefillPublication(mRequest, progress, commonStateLengthsPtr), "prefill publication");
}

bool ContextCacheRequest::prepareDecodeStep(DecodingInferenceContext const& context)
{
    ELLM_CHECK(!mTokenCountsBeforeDecode.has_value(),
        "Managed context-cache decode preparation cannot overlap a pending decode step.");
    if (!contextCacheOperationSucceeded(mCoordinator.prepareDecodeStep(mRequest), "decode preparation"))
    {
        return false;
    }

    std::vector<size_t> tokenCounts;
    tokenCounts.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        tokenCounts.push_back(context.tokenIds[slot].size());
    }
    mTokenCountsBeforeDecode.emplace(std::move(tokenCounts));
    return true;
}

bool ContextCacheRequest::completeDecodeStep(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    ELLM_CHECK(
        mTokenCountsBeforeDecode.has_value(), "Managed context-cache decode completion has no prepared decode step.");
    std::vector<size_t> tokenCountsBeforeDecode = std::move(*mTokenCountsBeforeDecode);
    mTokenCountsBeforeDecode.reset();
    ELLM_CHECK(static_cast<int32_t>(tokenCountsBeforeDecode.size()) == context.activeBatchSize,
        "Managed context-cache active batch changed during a decode step.");

    std::vector<ContextCacheSequenceAdvance> progress;
    std::vector<int32_t> publishableCompletedSlots;
    progress.reserve(static_cast<size_t>(context.activeBatchSize));
    publishableCompletedSlots.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        ELLM_CHECK(!context.tokenIds[slot].empty() && context.currentGenerateLengths[slot] > 0,
            "Managed context-cache decode did not produce a sampled lookahead token");
        size_t const previousTokenCount = tokenCountsBeforeDecode[static_cast<size_t>(slot)];
        ELLM_CHECK(context.tokenIds[slot].size() > previousTokenCount
                && context.tokenIds[slot].size() - previousTokenCount
                    <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "Managed context-cache decode produced an invalid accepted-token delta");
        int32_t const acceptedTokenCount = static_cast<int32_t>(context.tokenIds[slot].size() - previousTokenCount);
        int64_t const committedStateLength = static_cast<int64_t>(context.rawBatchedInputIds[slot].size())
            + static_cast<int64_t>(context.currentGenerateLengths[slot]) - 1;
        ELLM_CHECK(committedStateLength <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "Managed context-cache committed state length exceeds int32");
        progress.push_back(ContextCacheSequenceAdvance{context.tokenIds[slot].data() + previousTokenCount,
            acceptedTokenCount, static_cast<int32_t>(committedStateLength)});

        FinishReason const terminalReason = context.slotStreams[slot].terminalReason;
        if (context.finishedStates[slot] && terminalReason != FinishReason::kCancelled
            && terminalReason != FinishReason::kError)
        {
            publishableCompletedSlots.push_back(slot);
        }
    }

    std::vector<int32_t> const* const commonStateLengthsPtr
        = commonStateLengths.empty() ? nullptr : &commonStateLengths;
    return contextCacheOperationSucceeded(
        mCoordinator.completeDecodeStep(mRequest, progress, publishableCompletedSlots, commonStateLengthsPtr),
        "decode completion");
}

bool ContextCacheRequest::beginBatchCompaction(
    std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    return contextCacheOperationSucceeded(
        mCoordinator.beginBatchCompaction(mRequest, oldToNew, newBatchSize, deviceBatchMapping),
        "batch-compaction preparation");
}

bool ContextCacheRequest::completeBatchCompaction()
{
    return contextCacheOperationSucceeded(mCoordinator.compactBatch(mRequest), "batch compaction");
}

bool ContextCacheRequest::finish()
{
    ELLM_CHECK(
        !mTokenCountsBeforeDecode.has_value(), "Managed context-cache request cannot finish during a decode step.");
    return contextCacheOperationSucceeded(mCoordinator.finish(mRequest), "request finish");
}

} // namespace rt
} // namespace trt_edgellm
