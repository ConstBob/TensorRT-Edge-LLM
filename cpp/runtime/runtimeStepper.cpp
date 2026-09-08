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

#include "runtime/runtimeStepper.h"

#include "common/checkMacros.h"
#include "common/logger.h"

#include <exception>

namespace trt_edgellm
{
namespace rt
{

RuntimeStepper::RuntimeStepper(LLMRankRuntime::GenerationSession& session)
    : mSession(session)
{
    DecodingInferenceContext const& context = mSession.mContext;
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        int32_t const originalIndex = context.batchIndexMapping[static_cast<size_t>(slot)];
        mRefs.emplace(originalIndex, acquireHandle());
        // Tokens the founding prefill produced inside beginGeneration are not yet reported; the
        // founding prefill tick claims them as its delta.
        size_t const generated = static_cast<size_t>(context.currentGenerateLengths[static_cast<size_t>(slot)]);
        mReported.emplace(originalIndex, context.tokenIds[static_cast<size_t>(slot)].size() - generated);
    }
}

ResidentRef RuntimeStepper::acquireHandle()
{
    if (!mFreeHandles.empty())
    {
        int32_t const handle = mFreeHandles.back();
        mFreeHandles.pop_back();
        return ResidentRef{handle, mHandleEpochs[static_cast<size_t>(handle)]};
    }
    int32_t const handle = static_cast<int32_t>(mHandleEpochs.size());
    mHandleEpochs.push_back(0);
    return ResidentRef{handle, 0};
}

void RuntimeStepper::releaseHandle(ResidentRef ref)
{
    ELLM_CHECK(ref.slot >= 0 && static_cast<size_t>(ref.slot) < mHandleEpochs.size()
            && mHandleEpochs[static_cast<size_t>(ref.slot)] == ref.epoch,
        "releaseHandle: the ref is not the handle's live owner.");
    // The bump is the aliasing guard: a stale copy of this ref can never equal the handle's next
    // owner, which is the invariant that made row-based refs unsound (a tail seat reused without
    // any survivor moving handed the next resident an identical ref).
    ++mHandleEpochs[static_cast<size_t>(ref.slot)];
    mFreeHandles.push_back(ref.slot);
}

AdmissionResult RuntimeStepper::admit(AdmissionIntent intent)
{
    ELLM_CHECK(!mPending.has_value(), "admit: the previous admission has not run its prefill tick yet.");

    // The same invariant admitSequence holds at its entry, for the same reason: it must fire
    // before the lease, or a violation would strand leased pages.
    ELLM_CHECK(!mSession.mContext.hasGuidedDecoding, "A guided batch cannot take an admission.");

    int32_t const originalIndex = intent.seed.originalIndex;
    LLMRankRuntime::GenerationSession::AdmissionIntent inner{std::move(intent.seed)};
    switch (mSession.reserveAdmission(inner))
    {
    case AdmitDecision::kAdmitted: break;
    case AdmitDecision::kNoCapacity: return {AdmissionResult::Status::kNoCapacity, {}, {}};
    case AdmitDecision::kFailed: return {AdmissionResult::Status::kRejected, {}, "reservation failed"};
    }

    bool const leased = mSession.mManagedRequest != nullptr;
    try
    {
        PendingAdmission pending;
        pending.seat = mSession.seatSlot(std::move(inner.seed));
        pending.ref = acquireHandle();
        pending.originalIndex = originalIndex;
        pending.leased = leased;
        mRefs.emplace(originalIndex, pending.ref);
        // The seat's working history (prompt from prefillStart on) is not generated output.
        mReported.emplace(originalIndex, mSession.mContext.tokenIds[static_cast<size_t>(pending.seat.slot)].size());
        mPending = std::move(pending);
    }
    catch (std::exception const& error)
    {
        // seatSlot already retracted the lease on the throwing path; nothing is resident.
        return {AdmissionResult::Status::kRejected, {}, error.what()};
    }
    return {AdmissionResult::Status::kAdmitted, mPending->ref, {}};
}

StepResult RuntimeStepper::prefill(ImmutablePrefillBatch const& batch)
{
    StepResult result;
    if (mPending.has_value())
    {
        ELLM_CHECK(batch.target == mPending->ref, "prefill: the view does not name the pending admission.");
        PendingAdmission pending = std::move(*mPending);
        mPending.reset();
        AdmitDecision const decision = mSession.prefillSeated(std::move(pending.seat));
        // kFailed left the slot terminal from birth; the step itself executed, and the seat's
        // (empty, kError) result surfaces through the eviction that files it.
        result.ok = true;
        result.publishedPrefix = pending.leased && decision == AdmitDecision::kAdmitted;
    }
    else
    {
        ELLM_CHECK(!mRefs.empty(), "prefill: nothing is pending and the session has no residents to prime.");
        result.ok = mSession.primeFromPrefill();
    }
    finalizeResult(result);
    return result;
}

StepResult RuntimeStepper::decode(ImmutableDecodeBatch const& batch)
{
    ELLM_CHECK(!mPending.has_value(), "decode: the pending admission must run its prefill tick first.");
    std::vector<ResidentRef> const liveResidents = residents();
    ELLM_CHECK(batch.residents == liveResidents,
        "decode: the view must name every live resident in slot order; V0 runs a single cohort.");
    StepResult result;
    result.ok = mSession.advance();
    finalizeResult(result);
    return result;
}

std::vector<ResidentRef> RuntimeStepper::residents() const
{
    DecodingInferenceContext const& context = mSession.mContext;
    std::vector<ResidentRef> refs;
    refs.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        auto const it = mRefs.find(context.batchIndexMapping[static_cast<size_t>(slot)]);
        ELLM_CHECK(it != mRefs.end(), "residents: a live slot has no ref; the table is stale.");
        refs.push_back(it->second);
    }
    return refs;
}

void RuntimeStepper::finalizeResult(StepResult& result)
{
    DecodingInferenceContext& context = mSession.mContext;

    // Deltas: tokens a surviving resident holds beyond its reported watermark. An evicted
    // resident's final tokens travel inside its BatchResult snapshot instead.
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        int32_t const originalIndex = context.batchIndexMapping[static_cast<size_t>(slot)];
        auto const reported = mReported.find(originalIndex);
        auto const ref = mRefs.find(originalIndex);
        ELLM_CHECK(reported != mReported.end() && ref != mRefs.end(), "finalizeResult: a live slot has no ref.");
        auto const& tokens = context.tokenIds[static_cast<size_t>(slot)];
        if (tokens.size() > reported->second)
        {
            TokenDelta delta;
            delta.tokenIds.assign(tokens.begin() + static_cast<ptrdiff_t>(reported->second), tokens.end());
            result.deltas.emplace_back(ref->second, std::move(delta));
            reported->second = tokens.size();
        }
    }

    // Finished: everything the operation's eviction filed. Releasing the handle here is what makes
    // the finished ref part of this commit unit: from the caller's next operation on, the ref is
    // provably stale. Dense-row compaction needs no reporting -- refs never tracked rows.
    for (auto& [originalIndex, batchResult] : mSession.takeCompletedAtOrAbove(0))
    {
        auto const it = mRefs.find(originalIndex);
        if (it == mRefs.end())
        {
            LOG_WARNING("Stepper: a result for original index %d has no ref; dropping it.", originalIndex);
            continue;
        }
        releaseHandle(it->second);
        result.finished.emplace_back(it->second, std::move(batchResult));
        mRefs.erase(it);
        mReported.erase(originalIndex);
    }
}

std::unique_ptr<SteppedRequest> SteppedRequest::begin(LLMRankRuntime& runtime, LLMGenerationRequest prepared,
    cudaStream_t stream, LLMRankRuntime::TokenBroadcastFn tokenBroadcast, int32_t parallelRank)
{
    std::unique_ptr<SteppedRequest> stepped(new SteppedRequest(runtime, std::move(prepared), stream));
    stepped->mGeneration = runtime.beginGeneration(stepped->mRequest, stepped->mScratchResponse, stream,
        /*outputThinkerEmbeddings=*/false, std::move(tokenBroadcast), parallelRank);
    if (stepped->mGeneration == nullptr)
    {
        return nullptr;
    }
    // boundarySchedulingActive: the stepped plane always schedules, so the cancellation consensus
    // runs regardless of the founding request's channels -- same flag the hook path sets.
    stepped->mSession = std::make_unique<LLMRankRuntime::GenerationSession>(runtime, stepped->mGeneration->context,
        *stepped->mGeneration->strategy, stepped->mGeneration->managedRequest(), stepped->mRequest,
        stepped->mGeneration->kvHeadroom, stream, /*boundarySchedulingActive=*/true);
    stepped->mStepper = std::make_unique<RuntimeStepper>(*stepped->mSession);
    return stepped;
}

SteppedRequest::SteppedRequest(LLMRankRuntime& runtime, LLMGenerationRequest prepared, cudaStream_t stream)
    : mRuntime(runtime)
    , mRequest(std::move(prepared))
    , mStream(stream)
{
}

SteppedRequest::~SteppedRequest() = default;

AdmissionIntent SteppedRequest::buildIntent(LLMGenerationRequest const& request, int32_t originalIndex)
{
    AdmissionIntent intent;
    intent.seed = mSession->buildAdmissionIntent(request, originalIndex).seed;
    return intent;
}

std::vector<ResidentRef> SteppedRequest::residents() const
{
    return mStepper->residents();
}

AdmissionResult SteppedRequest::admit(LLMGenerationRequest const& request, int32_t originalIndex)
{
    return mStepper->admit(buildIntent(request, originalIndex));
}

StepResult SteppedRequest::prefill(ImmutablePrefillBatch const& batch)
{
    return mStepper->prefill(batch);
}

StepResult SteppedRequest::decode(ImmutableDecodeBatch const& batch)
{
    return mStepper->decode(batch);
}

LLMGenerationResponse SteppedRequest::materialize(
    BatchResult const& result, std::vector<std::string> const& stopStrings) const
{
    return mSession->materializeResult(result, stopStrings);
}

bool SteppedRequest::finish(LLMGenerationResponse& response)
{
    // The session must be gone before finishGeneration tears the request down: it references the
    // context and the strategy the teardown closes out.
    mStepper.reset();
    mSession.reset();
    return mRuntime.finishGeneration(*mGeneration, mRequest, response, mStream);
}

} // namespace rt
} // namespace trt_edgellm
