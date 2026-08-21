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

#pragma once

#include "common/tensor.h"
#include "runtime/llmRuntimeUtils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{

namespace tokenizer
{
class Tokenizer;
}

namespace rt
{

using tokenizer::Tokenizer;

struct DecodingInferenceContext;

/*!
 * @brief Grammar-constrained decoding backed by XGrammar.
 *
 * Owns the tokenizer info, the compiled-grammar cache, and one grammar matcher per active
 * batch slot.
 *
 * Lifetime mirrors the runtime, but the per-slot matchers are request-local and must be
 * reindexed by @ref compactSlots on every batch eviction, in lockstep with the per-slot
 * vectors in DecodingInferenceContext.
 */
class GuidedDecoder
{
public:
    GuidedDecoder();
    ~GuidedDecoder(); //!< Out-of-line: Impl is incomplete here.
    GuidedDecoder(GuidedDecoder const&) = delete;
    GuidedDecoder& operator=(GuidedDecoder const&) = delete;

    /*!
     * @brief Allocate reusable buffers and record the vocabulary geometry.
     *
     * The tokenizer info is built on the first guided request instead: it costs one
     * `idToPiece` call per output-vocabulary entry.
     *
     * @param outputVocabSize   Width of the engine's logits, i.e. the reduced vocabulary
     *                          when the engine prunes, else the full vocabulary
     * @param tokenizer         Borrowed; must outlive this object
     * @param reducedToFullVocabMap Output-space index to full token ID; pass an unallocated
     *                          tensor when the engine does not prune
     * @param stream            Used for the one-off device-to-host copy of that map
     */
    void initialize(int32_t maxBatchSize, int32_t outputVocabSize, Tokenizer const* tokenizer,
        Tensor const& reducedToFullVocabMap, cudaStream_t stream);

    /*!
     * @brief Compile a grammar and install a fresh matcher for one slot.
     *
     * Runs before any GPU work for the request, so a failure marks just this slot. This is
     * the only input-driven step here, hence the only one that catches exceptions.
     *
     * @param[out] failReason Set only when the call returns false
     * @return False when the guide could not be compiled; the slot is left unconstrained
     */
    bool prepareSlot(int32_t slot, GuidedDecodingParams const& params, std::string& failReason);

    //! Drop every matcher.
    void reset();

    //! Reindex per-slot state after a batch eviction. `batchMapping[i]` is the new index
    //! of old slot `i`, or -1 when it was evicted. Must run with the same mapping, and at
    //! the same point, as the DecodingInferenceContext vectors.
    void compactSlots(std::vector<int32_t> const& batchMapping);

    bool hasAnyGrammar() const noexcept;
    bool hasGrammar(int32_t slot) const noexcept;

    //! True once the matcher has accepted the stop token. Filling a mask past that point is
    //! a hard error in XGrammar, so callers must gate on this.
    bool isTerminated(int32_t slot) const noexcept;

    /*!
     * @brief Advance a slot's grammar by one accepted token.
     * @param outputSpaceToken Token index in the engine's output vocabulary, i.e. captured
     *        before `mapReducedVocabToFullVocab`
     * @return False when the grammar rejects the token, which the caller turns into kError
     */
    bool advance(int32_t slot, int32_t outputSpaceToken);

    /*!
     * @brief Fill and upload the bitmask for one decode step.
     *
     * Each slot owns `rowsPerSlot` consecutive rows, matching the logits layout
     * `[activeBatchSize, rowsPerSlot, vocab]` that \ref applyLogitBiasRepeatedRows assumes.
     * Vanilla decode has `rowsPerSlot == 1`; speculative verification raises it because each
     * draft position sits at a different grammar state and needs its own mask.
     *
     * @param maskSuppressedPerSlot Slots to leave unconstrained this step
     * @param[out] unsatisfiableSlots Slots whose mask came out all-zero, i.e. the grammar
     *                          cannot be satisfied in this engine's vocabulary
     */
    void fillMasks(int32_t activeBatchSize, int32_t rowsPerSlot, std::vector<int8_t> const& maskSuppressedPerSlot,
        std::vector<int32_t>& unsatisfiableSlots, cudaStream_t stream);

    //! Apply the uploaded bitmask to `logits`, shaped [activeBatchSize, rowsPerSlot, vocab].
    void applyMask(Tensor& logits, int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

    //! Compiled-grammar cache footprint in bytes.
    int64_t cacheSizeBytes() const;

private:
    //! Keeps XGrammar out of this header, so a version bump or backend swap rebuilds
    //! one translation unit rather than every consumer of llmInferenceRuntime.h.
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

bool hasGuidedDecoding(LLMGenerationRequest const& request) noexcept;

/*!
 * @brief Whether the prompt leaves the reasoning block closed, i.e. whether guided decoding
 *        may constrain from the very first generated token.
 *
 * The most recent marker wins. A tokenizer carrying none of these markers has no reasoning
 * phase, and is reported as closed.
 *
 * @param startMarkers Ids opening a reasoning block; entries below zero are absent from the
 *                     tokenizer and ignored
 * @param endMarkers   Ids closing one, same convention
 */
bool reasoningClosedInPrompt(std::vector<int32_t> const& promptTokens, std::vector<int32_t> const& startMarkers,
    std::vector<int32_t> const& endMarkers) noexcept;

/*!
 * @brief Constrain one step's logits to the grammar-legal tokens.
 *
 * A slot whose grammar admits nothing ends with FinishReason::kError, keeping the text it
 * generated so far. Slots still inside their thinking block are left unconstrained, so a
 * JSON grammar cannot mask away the opening `<think>`.
 *
 * @param rowsPerSlot Logits rows owned by each slot; 1 for vanilla decode
 */
void applyGuidedDecodingMask(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

/*!
 * @brief Advance each slot's grammar by the token it just produced.
 *
 * Must run before `thinkingDone` is updated for this token; that ordering is what keeps the
 * `</think>` separator itself out of the grammar.
 *
 * @param outputSpaceIds Sampled indices in the engine's output vocabulary, captured before
 *        the reduced-vocabulary remap
 */
void advanceGuidedDecoding(
    GuidedDecoder& decoder, DecodingInferenceContext& context, int32_t const* outputSpaceIds, int32_t activeBatchSize);

/*!
 * @brief Reject guides XGrammar would accept but not honour, and oversized ones.
 *
 * Some JSON Schema keywords compile cleanly and then do nothing, so the output would violate
 * the schema while the API claims it cannot. Compilation cannot catch that, which makes this
 * the only line of defence. The blacklist is chosen by measurement against the pinned
 * XGrammar version and must be re-checked when that pin moves.
 *
 * @param[out] failReason Set only when the call returns false
 * @return False when the request must be rejected outright
 */
bool validateGuidedDecodingParams(GuidedDecodingParams const& params, std::string& failReason);

} // namespace rt
} // namespace trt_edgellm
