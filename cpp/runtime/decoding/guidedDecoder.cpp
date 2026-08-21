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

#include "runtime/decoding/guidedDecoder.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "sampler/sampling.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>

#include <xgrammar/xgrammar.h>

namespace trt_edgellm
{
namespace rt
{

bool hasGuidedDecoding(LLMGenerationRequest const& request) noexcept
{
    return std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& slotRequest) { return slotRequest.guidedDecoding.has_value(); });
}

namespace
{

//! The two conditions are independent: a request can finish while its grammar is mid-rule,
//! and a thinking model must be free to open `<think>` before the schema takes over.
void buildSlotSuppression(
    DecodingInferenceContext const& context, int32_t activeBatchSize, std::vector<int8_t>& suppressed)
{
    suppressed.assign(static_cast<size_t>(activeBatchSize), 0);
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        bool const finished
            = slot < static_cast<int32_t>(context.finishedStates.size()) && context.finishedStates[slot] != 0;
        bool const stillThinking = slot < static_cast<int32_t>(context.guidedReasoningEnded.size())
            && context.guidedReasoningEnded[slot] == 0;
        suppressed[static_cast<size_t>(slot)] = (finished || stillThinking) ? 1 : 0;
    }
}

//! Fail one slot but keep what it already produced: streaming has already handed those tokens
//! to the client, so the non-streaming path must agree.
void failSlot(DecodingInferenceContext& context, int32_t slot, char const* what)
{
    if (context.finishedStates[slot] != 0)
    {
        return; // First writer wins; a cancel may already have landed.
    }
    context.finishedStates[slot] = 1;
    context.slotStreams[slot].terminalReason = FinishReason::kError;
    LOG_ERROR("Request %d: %s; ending the request and keeping the %d tokens generated so far.", slot, what,
        context.currentGenerateLengths[slot]);
}

} // namespace

bool reasoningClosedInPrompt(std::vector<int32_t> const& promptTokens, std::vector<int32_t> const& startMarkers,
    std::vector<int32_t> const& endMarkers) noexcept
{
    auto const present = [](std::vector<int32_t> const& markers) {
        return std::any_of(markers.begin(), markers.end(), [](int32_t id) { return id >= 0; });
    };
    if (!present(startMarkers) && !present(endMarkers))
    {
        return true;
    }

    auto const matches = [](std::vector<int32_t> const& markers, int32_t token) {
        return token >= 0 && std::find(markers.begin(), markers.end(), token) != markers.end();
    };
    for (auto it = promptTokens.rbegin(); it != promptTokens.rend(); ++it)
    {
        if (matches(endMarkers, *it))
        {
            return true;
        }
        if (matches(startMarkers, *it))
        {
            return false;
        }
    }
    // No marker at all: the block has not been opened, and the model may still open one.
    return false;
}

void applyGuidedDecodingMask(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (!context.hasGuidedDecoding || activeBatchSize <= 0)
    {
        return;
    }

    buildSlotSuppression(context, activeBatchSize, context.guidedMaskSuppressedPerSlot);
    decoder.fillMasks(
        activeBatchSize, rowsPerSlot, context.guidedMaskSuppressedPerSlot, context.guidedUnsatisfiableSlots, stream);

    for (auto const slot : context.guidedUnsatisfiableSlots)
    {
        failSlot(context, slot, "the grammar admits no token in this engine's output vocabulary");
    }

    decoder.applyMask(logits, activeBatchSize, rowsPerSlot, stream);
}

void advanceGuidedDecoding(
    GuidedDecoder& decoder, DecodingInferenceContext& context, int32_t const* outputSpaceIds, int32_t activeBatchSize)
{
    if (!context.hasGuidedDecoding)
    {
        return;
    }

    buildSlotSuppression(context, activeBatchSize, context.guidedMaskSuppressedPerSlot);
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        if (!decoder.hasGrammar(slot) || context.guidedMaskSuppressedPerSlot[static_cast<size_t>(slot)] != 0)
        {
            continue;
        }
        // EOS included: the matcher only terminates by accepting the stop token.
        if (!decoder.advance(slot, outputSpaceIds[slot]))
        {
            // The mask should have made this impossible.
            failSlot(context, slot, "the grammar rejected a token the mask should have forbidden");
        }
    }
}

namespace
{

/*!
 * @brief JSON Schema keywords XGrammar accepts but does not implement correctly.
 *
 * Measured on v0.2.1 by feeding violating documents to the compiled grammar and requiring
 * both full acceptance and rule completion. multipleOf / uniqueItems / contains are ignored
 * outright; minContains / maxContains degrade to a bare element count, so a schema asking for
 * two nines accepts [1, 2]. patternProperties compiles into a grammar that rejects everything.
 *
 * The list tracks measurement, not vLLM's list: propertyNames was broken in v0.1.25 and is
 * correct in v0.2.1, so it is deliberately absent. Re-measure when the pin moves.
 */
constexpr char const* kUnsupportedSchemaKeywords[] = {
    "multipleOf",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "patternProperties",
};

//! `format` values XGrammar converts to a pattern. Anything else compiles and is then ignored.
constexpr char const* kSupportedSchemaFormats[] = {
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "json-pointer",
    "relative-json-pointer",
    "time",
    "uri",
    "uri-reference",
    "uri-template",
    "uuid",
};

//! Keywords whose sub-object is keyed by user-chosen names rather than by schema keywords.
//! Without this, a schema with a property literally named "contains" would be rejected.
constexpr char const* kNameKeyedKeywords[] = {
    "properties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
};

//! Keys off the keyword alone rather than a sibling "type": {"multipleOf": 5} carries no type
//! and would slip past a type-keyed check.
bool findUnsupportedSchemaFeature(nlohmann::json const& node, std::string& feature)
{
    if (node.is_array())
    {
        for (auto const& element : node)
        {
            if (findUnsupportedSchemaFeature(element, feature))
            {
                return true;
            }
        }
        return false;
    }
    if (!node.is_object())
    {
        return false;
    }

    for (auto const* keyword : kUnsupportedSchemaKeywords)
    {
        if (node.contains(keyword))
        {
            feature = keyword;
            return true;
        }
    }
    auto const formatIt = node.find("format");
    if (formatIt != node.end() && formatIt->is_string())
    {
        auto const formatValue = formatIt->get<std::string>();
        bool const supported = std::any_of(std::begin(kSupportedSchemaFormats), std::end(kSupportedSchemaFormats),
            [&formatValue](char const* candidate) { return formatValue == candidate; });
        if (!supported)
        {
            feature = "format: \"" + formatValue + "\"";
            return true;
        }
    }

    for (auto const& item : node.items())
    {
        bool const nameKeyed = item.value().is_object()
            && std::any_of(std::begin(kNameKeyedKeywords), std::end(kNameKeyedKeywords),
                [&item](char const* candidate) { return item.key() == candidate; });
        if (nameKeyed)
        {
            // One level down the keys are names, so only their subschemas are inspected.
            for (auto const& named : item.value().items())
            {
                if (findUnsupportedSchemaFeature(named.value(), feature))
                {
                    return true;
                }
            }
            continue;
        }
        if (findUnsupportedSchemaFeature(item.value(), feature))
        {
            return true;
        }
    }
    return false;
}

/*!
 * @brief Parse a kChoice guide into its alternatives.
 *
 * @param[out] failReason Set only when the call returns false
 */
bool parseChoiceList(std::string const& guide, std::vector<std::string>& choices, std::string& failReason)
{
    nlohmann::json document;
    try
    {
        document = nlohmann::json::parse(guide);
    }
    catch (std::exception const& e)
    {
        failReason = format::fmtstr("guided_decoding.choice is not valid JSON: %s", e.what());
        return false;
    }
    bool const wellFormed = document.is_array() && !document.empty()
        && std::all_of(document.begin(), document.end(), [](nlohmann::json const& item) { return item.is_string(); });
    if (!wellFormed)
    {
        failReason = "guided_decoding.choice must be a non-empty array of strings";
        return false;
    }
    for (auto const& item : document)
    {
        choices.push_back(item.get<std::string>());
    }
    return true;
}

} // namespace

bool validateGuidedDecodingParams(GuidedDecodingParams const& params, std::string& failReason)
{
    if (params.type != GuideType::kJsonObject && params.guide.empty())
    {
        failReason = std::string("guided_decoding.") + guideTypeName(params.type) + " must not be empty";
        return false;
    }
    if (params.guide.size() > limits::security::kMaxGuidedDecodingGuideBytes)
    {
        failReason = format::fmtstr("guided_decoding.%s is too large: %zu bytes (max: %zu). Limit defined in %s.",
            guideTypeName(params.type), params.guide.size(), limits::security::kMaxGuidedDecodingGuideBytes,
            limits::kInputLimitsLocation);
        return false;
    }

    if (params.type == GuideType::kChoice)
    {
        std::vector<std::string> choices;
        return parseChoiceList(params.guide, choices, failReason);
    }

    if (params.type == GuideType::kJsonSchema || params.type == GuideType::kStructuralTag)
    {
        nlohmann::json document;
        try
        {
            document = nlohmann::json::parse(params.guide);
        }
        catch (std::exception const& e)
        {
            failReason
                = format::fmtstr("guided_decoding.%s is not valid JSON: %s", guideTypeName(params.type), e.what());
            return false;
        }

        if (params.type == GuideType::kJsonSchema)
        {
            std::string feature;
            if (findUnsupportedSchemaFeature(document, feature))
            {
                failReason = format::fmtstr(
                    "guided_decoding.json_schema uses '%s', which XGrammar accepts but does not enforce; "
                    "the generated output would not be guaranteed to match the schema",
                    feature.c_str());
                return false;
            }
        }
    }
    return true;
}

namespace
{

//! Threads XGrammar may use to compile one grammar.
constexpr int kCOMPILER_THREADS = 8;

//! Compiled-grammar LRU budget; XGrammar leaves it unlimited by default.
constexpr int64_t kCACHE_LIMIT_BYTES = 64LL * 1024 * 1024;

//! Lower the alternatives onto XGrammar's EBNF. JSON string escaping is a subset of the
//! EBNF literal escaping, so a `dump()` of each choice is already a valid literal.
std::string choiceListToEbnf(std::vector<std::string> const& choices)
{
    std::string grammar = "root ::= ";
    for (size_t i = 0; i < choices.size(); ++i)
    {
        if (i > 0)
        {
            grammar += " | ";
        }
        grammar += nlohmann::json(choices[i]).dump();
    }
    return grammar;
}

} // namespace

struct GuidedDecoder::Impl
{
    Tokenizer const* tokenizer{nullptr};
    int32_t maxBatchSize{0};
    //! Bitmask rows allocated up front = maxBatchSize * maxRowsPerSlot.
    int32_t maxRows{0};
    int32_t outputVocabSize{0};
    int32_t bitmaskSize{0}; //!< int32 words per row, i.e. xgrammar::GetBitmaskSize(outputVocabSize)

    //! Output-space index -> full tokenizer ID, [outputVocabSize]. Unset when the engine does
    //! not prune, in which case the mapping is the identity.
    Tensor outputToFullVocab;
    bool hasReducedVocab{false};

    //! Built on first use: one idToPiece call per output-vocabulary entry is too
    //! expensive to pay for runs that never use guided decoding.
    std::optional<xgrammar::TokenizerInfo> tokenizerInfo;
    std::optional<xgrammar::GrammarCompiler> compiler;

    std::vector<std::optional<xgrammar::GrammarMatcher>> matchers;

    Tensor deviceBitmask;
    Tensor deviceRowNeedsMask;
    Tensor hostBitmask;
    Tensor hostRowNeedsMask;

    void ensureCompiler();
    xgrammar::CompiledGrammar compile(GuidedDecodingParams const& params);
};

void GuidedDecoder::Impl::ensureCompiler()
{
    if (compiler.has_value())
    {
        return;
    }
    ELLM_CHECK(tokenizer != nullptr, "GuidedDecoder used before initialize()");

    auto const start = std::chrono::steady_clock::now();

    // Null when the engine does not prune; output space is then the identity.
    int32_t const* const outputToFull = outputToFullVocab.dataPointer<int32_t>();
    std::vector<std::string> encodedVocab(static_cast<size_t>(outputVocabSize));
    for (int32_t outputId = 0; outputId < outputVocabSize; ++outputId)
    {
        int32_t const fullId = hasReducedVocab ? outputToFull[outputId] : outputId;
        encodedVocab[static_cast<size_t>(outputId)] = tokenizer->idToPiece(fullId, /*skipSpecialTokens=*/false);
    }

    // Must come from the same source the runtime uses to detect EOS; XGrammar's own
    // detection guesses by name and gets it wrong.
    std::vector<int32_t> stopTokenIds;
    for (auto const fullEosId : tokenizer->getEosIds())
    {
        int32_t outputId = static_cast<int32_t>(fullEosId);
        if (hasReducedVocab)
        {
            int32_t const* const end = outputToFull + outputVocabSize;
            auto const* const it = std::find(outputToFull, end, static_cast<int32_t>(fullEosId));
            ELLM_CHECK(it != end,
                "EOS token missing from the engine's reduced vocabulary; the engine's vocab_map is inconsistent "
                "with its tokenizer");
            outputId = static_cast<int32_t>(it - outputToFull);
        }
        ELLM_CHECK(outputId >= 0 && outputId < outputVocabSize, "EOS token falls outside the output vocabulary");
        stopTokenIds.push_back(outputId);
    }
    ELLM_CHECK(!stopTokenIds.empty(), "Guided decoding requires at least one EOS token");

    // RAW: our tokenizer already resolves byte-level and byte-fallback pieces to raw bytes at
    // load time and discards the display form, so any other vocab type would decode twice.
    tokenizerInfo.emplace(encodedVocab, xgrammar::VocabType::RAW, outputVocabSize, stopTokenIds,
        /*add_prefix_space=*/false);

    compiler.emplace(*tokenizerInfo, kCOMPILER_THREADS, /*cache_enabled=*/true, kCACHE_LIMIT_BYTES);

    auto const elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    LOG_INFO("Guided decoding: built tokenizer info over %d output-vocab entries in %.1f ms (cache limit %lld bytes)",
        outputVocabSize, elapsedMs, static_cast<long long>(kCACHE_LIMIT_BYTES));
}

xgrammar::CompiledGrammar GuidedDecoder::Impl::compile(GuidedDecodingParams const& params)
{
    // Pinned rather than left to defaults: any_whitespace keeps the newlines a model emits
    // after </think> legal, strict_mode forbids properties the schema does not mention.
    constexpr bool kANY_WHITESPACE = true;
    constexpr bool kSTRICT_MODE = true;

    switch (params.type)
    {
    case GuideType::kJsonObject:
        // Not CompileBuiltinJSONGrammar(): that admits any JSON value, not just an object.
        return compiler->CompileJSONSchema(
            R"({"type":"object"})", kANY_WHITESPACE, std::nullopt, std::nullopt, kSTRICT_MODE, std::nullopt);
    case GuideType::kJsonSchema:
        return compiler->CompileJSONSchema(
            params.guide, kANY_WHITESPACE, std::nullopt, std::nullopt, kSTRICT_MODE, std::nullopt);
    case GuideType::kRegex: return compiler->CompileRegex(params.guide);
    case GuideType::kEbnf: return compiler->CompileGrammar(params.guide, "root");
    case GuideType::kStructuralTag: return compiler->CompileStructuralTag(params.guide);
    case GuideType::kChoice:
    {
        // Validated at request admission, so the throw only guards a caller that skipped it.
        std::vector<std::string> choices;
        std::string failReason;
        if (!parseChoiceList(params.guide, choices, failReason))
        {
            throw std::runtime_error(failReason);
        }
        return compiler->CompileGrammar(choiceListToEbnf(choices), "root");
    }
    }
    throw std::runtime_error("Unhandled guided decoding guide type");
}

GuidedDecoder::GuidedDecoder()
    : mImpl(std::make_unique<Impl>())
{
}

GuidedDecoder::~GuidedDecoder() = default;

void GuidedDecoder::initialize(int32_t maxBatchSize, int32_t outputVocabSize, Tokenizer const* tokenizer,
    Tensor const& reducedToFullVocabMap, cudaStream_t stream)
{
    ELLM_CHECK(maxBatchSize > 0, "GuidedDecoder requires a positive max batch size");
    ELLM_CHECK(outputVocabSize > 0, "GuidedDecoder requires a positive output vocabulary size");
    ELLM_CHECK(tokenizer != nullptr, "GuidedDecoder requires a tokenizer");

    mImpl->tokenizer = tokenizer;
    mImpl->maxBatchSize = maxBatchSize;
    mImpl->outputVocabSize = outputVocabSize;
    mImpl->bitmaskSize = xgrammar::GetBitmaskSize(outputVocabSize);
    mImpl->matchers.assign(static_cast<size_t>(maxBatchSize), std::nullopt);

    mImpl->hasReducedVocab = false;
    if (!reducedToFullVocabMap.isEmpty())
    {
        auto const entries = static_cast<size_t>(reducedToFullVocabMap.getShape().volume());
        ELLM_CHECK(entries >= static_cast<size_t>(outputVocabSize),
            "Reduced vocabulary map is smaller than the engine output vocabulary");
        mImpl->outputToFullVocab = Tensor(
            {outputVocabSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "GuidedDecoder::outputToFullVocab");
        CUDA_CHECK(cudaMemcpyAsync(mImpl->outputToFullVocab.rawPointer(), reducedToFullVocabMap.rawPointer(),
            static_cast<size_t>(outputVocabSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        mImpl->hasReducedVocab = true;
    }

    // Vanilla decode emits one logits row per slot; speculative verification will need
    // maxBatchSize * rowsPerSlot here.
    int32_t const maxRows = maxBatchSize;
    mImpl->maxRows = maxRows;
    mImpl->deviceBitmask
        = Tensor({maxRows, mImpl->bitmaskSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "GuidedDecoder::bitmask");
    mImpl->deviceRowNeedsMask
        = Tensor({maxRows}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "GuidedDecoder::rowNeedsMask");
    mImpl->hostBitmask = Tensor(
        {maxRows, mImpl->bitmaskSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "GuidedDecoder::hostBitmask");
    mImpl->hostRowNeedsMask
        = Tensor({maxRows}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "GuidedDecoder::hostRowNeedsMask");
    std::memset(mImpl->hostBitmask.rawPointer(), 0,
        static_cast<size_t>(maxRows) * static_cast<size_t>(mImpl->bitmaskSize) * sizeof(int32_t));
    std::memset(mImpl->hostRowNeedsMask.rawPointer(), 0, static_cast<size_t>(maxRows) * sizeof(int32_t));

    LOG_DEBUG("GuidedDecoder initialized: maxBatchSize=%d outputVocabSize=%d bitmaskSize=%d reducedVocab=%s",
        maxBatchSize, outputVocabSize, mImpl->bitmaskSize, mImpl->hasReducedVocab ? "yes" : "no");
}

bool GuidedDecoder::prepareSlot(int32_t slot, GuidedDecodingParams const& params, std::string& failReason)
{
    ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(mImpl->matchers.size()),
        "GuidedDecoder::prepareSlot slot out of range");

    // The only input-driven step in this class, so the only one that catches: a malformed
    // schema must fail this slot, not the process.
    try
    {
        mImpl->ensureCompiler();

        auto const start = std::chrono::steady_clock::now();
        int64_t const cacheBefore = mImpl->compiler->GetCacheSizeBytes();
        auto compiledGrammar = mImpl->compile(params);
        auto const elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        int64_t const cacheAfter = mImpl->compiler->GetCacheSizeBytes();

        mImpl->matchers[static_cast<size_t>(slot)].emplace(compiledGrammar);

        LOG_INFO("Guided decoding: compiled %s for slot %d in %.2f ms (cache %s, now %.1f KB)",
            guideTypeName(params.type), slot, elapsedMs, cacheAfter == cacheBefore ? "hit" : "miss",
            static_cast<double>(cacheAfter) / 1024.0);
        return true;
    }
    catch (std::exception const& e)
    {
        failReason = e.what();
        mImpl->matchers[static_cast<size_t>(slot)].reset();
        return false;
    }
}

void GuidedDecoder::reset()
{
    std::fill(mImpl->matchers.begin(), mImpl->matchers.end(), std::nullopt);
}

void GuidedDecoder::compactSlots(std::vector<int32_t> const& batchMapping)
{
    if (mImpl->matchers.empty())
    {
        return;
    }
    // Missing this leaves slot i pointing at another request's matcher: the constraint
    // migrates silently, and the terminated-matcher gate starts reading the wrong object.
    std::vector<std::optional<xgrammar::GrammarMatcher>> compacted(mImpl->matchers.size(), std::nullopt);
    for (size_t oldSlot = 0; oldSlot < batchMapping.size() && oldSlot < mImpl->matchers.size(); ++oldSlot)
    {
        int32_t const newSlot = batchMapping[oldSlot];
        if (newSlot >= 0 && newSlot < static_cast<int32_t>(compacted.size()))
        {
            compacted[static_cast<size_t>(newSlot)] = std::move(mImpl->matchers[oldSlot]);
        }
    }
    mImpl->matchers = std::move(compacted);
}

bool GuidedDecoder::hasAnyGrammar() const noexcept
{
    return std::any_of(
        mImpl->matchers.begin(), mImpl->matchers.end(), [](auto const& matcher) { return matcher.has_value(); });
}

bool GuidedDecoder::hasGrammar(int32_t slot) const noexcept
{
    return slot >= 0 && slot < static_cast<int32_t>(mImpl->matchers.size())
        && mImpl->matchers[static_cast<size_t>(slot)].has_value();
}

bool GuidedDecoder::isTerminated(int32_t slot) const noexcept
{
    return hasGrammar(slot) && mImpl->matchers[static_cast<size_t>(slot)]->IsTerminated();
}

bool GuidedDecoder::advance(int32_t slot, int32_t outputSpaceToken)
{
    if (!hasGrammar(slot))
    {
        return true;
    }
    auto& matcher = *mImpl->matchers[static_cast<size_t>(slot)];
    if (matcher.IsTerminated())
    {
        return true; // Feeding a terminated matcher only produces warnings.
    }
    return matcher.AcceptToken(outputSpaceToken);
}

void GuidedDecoder::fillMasks(int32_t activeBatchSize, int32_t rowsPerSlot,
    std::vector<int8_t> const& maskSuppressedPerSlot, std::vector<int32_t>& unsatisfiableSlots, cudaStream_t stream)
{
    unsatisfiableSlots.clear();
    if (activeBatchSize <= 0 || rowsPerSlot <= 0)
    {
        return;
    }
    int32_t const totalRows = activeBatchSize * rowsPerSlot;
    ELLM_CHECK(totalRows <= mImpl->maxRows, "GuidedDecoder::fillMasks row count exceeds the allocated buffer");

    // XGrammar writes straight into row `index` of the host buffer.
    std::array<int64_t, 2> shape{static_cast<int64_t>(mImpl->maxRows), static_cast<int64_t>(mImpl->bitmaskSize)};
    DLTensor bitmaskTensor{};
    bitmaskTensor.data = mImpl->hostBitmask.rawPointer();
    bitmaskTensor.device = DLDevice{kDLCPU, 0};
    bitmaskTensor.ndim = 2;
    bitmaskTensor.dtype = xgrammar::GetBitmaskDLType();
    bitmaskTensor.shape = shape.data();

    int32_t* const rowNeedsMask = mImpl->hostRowNeedsMask.dataPointer<int32_t>();
    int32_t const* const hostBitmask = mImpl->hostBitmask.dataPointer<int32_t>();

    auto const leaveSlotUnconstrained = [&](int32_t slot) {
        for (int32_t row = 0; row < rowsPerSlot; ++row)
        {
            rowNeedsMask[slot * rowsPerSlot + row] = 0;
        }
    };

    bool anyRowNeedsMask = false;
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        bool const suppressed = slot < static_cast<int32_t>(maskSuppressedPerSlot.size())
            && maskSuppressedPerSlot[static_cast<size_t>(slot)] != 0;

        // Both gates matter and they are independent. A request can finish on max-length,
        // a stop string, or cancellation while its grammar is still mid-rule; and under
        // EDGELLM_IGNORE_EOS the matcher terminates while the request keeps going.
        // Filling a mask after termination is a hard error in XGrammar, not a no-op.
        if (!hasGrammar(slot) || suppressed || mImpl->matchers[static_cast<size_t>(slot)]->IsTerminated())
        {
            leaveSlotUnconstrained(slot);
            continue;
        }

        // One mask per row, not per slot: each row sits at its own grammar state.
        bool unsatisfiable = false;
        for (int32_t row = 0; row < rowsPerSlot && !unsatisfiable; ++row)
        {
            int32_t const rowId = slot * rowsPerSlot + row;
            mImpl->matchers[static_cast<size_t>(slot)]->FillNextTokenBitmask(&bitmaskTensor, rowId);
            rowNeedsMask[rowId] = 1;

            // All-zero means the grammar admits nothing; reachable only with a pruned
            // vocabulary that cannot spell what the schema needs.
            int32_t const* const rowMask = hostBitmask + static_cast<size_t>(rowId) * mImpl->bitmaskSize;
            int32_t accumulated = 0;
            for (int32_t word = 0; word < mImpl->bitmaskSize; ++word)
            {
                accumulated |= rowMask[word];
            }
            unsatisfiable = accumulated == 0;
        }

        if (unsatisfiable)
        {
            // The caller is about to fail this slot. Applying the all-zero mask first would
            // drive the whole row to the sentinel and make sampling uniformly random, so skip
            // it and leave the logits alone.
            leaveSlotUnconstrained(slot);
            unsatisfiableSlots.push_back(slot);
            continue;
        }
        anyRowNeedsMask = true;
    }

    // Always upload the flags, so a row left set by a previous step cannot be reapplied.
    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceRowNeedsMask.rawPointer(), mImpl->hostRowNeedsMask.rawPointer(),
        static_cast<size_t>(totalRows) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (!anyRowNeedsMask)
    {
        return;
    }

    // Unconditional: unlike logit bias, the mask changes every step.
    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceBitmask.rawPointer(), mImpl->hostBitmask.rawPointer(),
        static_cast<size_t>(totalRows) * static_cast<size_t>(mImpl->bitmaskSize) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
}

void GuidedDecoder::applyMask(Tensor& logits, int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (activeBatchSize <= 0 || rowsPerSlot <= 0)
    {
        return;
    }
    int32_t const totalRows = activeBatchSize * rowsPerSlot;

    // The kernel takes 2D logits, so flatten a speculative [batch, rowsPerSlot, vocab] tensor
    // and restore it afterwards, as applyLogitBiasRepeatedRows does with the same tensor.
    auto const originalShape = logits.getShape();
    if (originalShape.getNumDims() == 3)
    {
        ELLM_CHECK(originalShape[1] == rowsPerSlot, "Guided mask rowsPerSlot does not match the logits shape");
        ELLM_CHECK(logits.reshape({originalShape[0] * originalShape[1], originalShape[2]}), "Tensor reshape failed");
        applyTokenBitmask(logits, mImpl->deviceBitmask, mImpl->deviceRowNeedsMask, totalRows, stream);
        ELLM_CHECK(logits.reshape(originalShape), "Tensor reshape failed");
        return;
    }
    applyTokenBitmask(logits, mImpl->deviceBitmask, mImpl->deviceRowNeedsMask, totalRows, stream);
}

int64_t GuidedDecoder::cacheSizeBytes() const
{
    return mImpl->compiler.has_value() ? mImpl->compiler->GetCacheSizeBytes() : 0;
}

} // namespace rt
} // namespace trt_edgellm
