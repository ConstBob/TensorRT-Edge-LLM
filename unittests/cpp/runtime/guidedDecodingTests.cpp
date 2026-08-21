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

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

rt::GuidedDecodingParams makeGuide(rt::GuideType type, std::string guide)
{
    rt::GuidedDecodingParams params;
    params.type = type;
    params.guide = std::move(guide);
    return params;
}

// ---------------------------------------------------------------- pre-check (D6 appendix)

//! Keywords XGrammar compiles and then ignores are the dangerous case: the output would
//! violate the schema while the API claims it cannot. Rejecting up front is the only defence,
//! because compilation itself succeeds.
TEST(GuidedDecodingValidationTest, RejectsSilentlyIgnoredKeywords)
{
    for (char const* schema : {
             R"({"type":"object","properties":{"a":{"type":"integer","multipleOf":5}}})",
             R"({"type":"array","items":{"type":"integer"},"uniqueItems":true})",
             R"({"type":"array","items":{"type":"integer"},"contains":{"const":9}})",
             R"({"type":"array","items":{"type":"integer"},"contains":{"const":9},"minContains":2})",
             R"({"type":"array","items":{"type":"integer"},"maxContains":2})",
         })
    {
        std::string failReason;
        EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << "schema should have been rejected: " << schema;
        EXPECT_FALSE(failReason.empty());
    }
}

//! patternProperties compiles into a grammar that rejects every document, including matching
//! ones, which would otherwise surface much later as an unsatisfiable mask.
TEST(GuidedDecodingValidationTest, RejectsOverRestrictiveKeywords)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"type":"object","patternProperties":{"^S_":{"type":"string"}}})"),
        failReason));
}

//! The check keys off the keyword alone rather than a sibling "type". vLLM looks at the type
//! first, so a schema like this one slips past its check entirely.
TEST(GuidedDecodingValidationTest, RejectsUnsupportedKeywordWithoutSiblingType)
{
    std::string failReason;
    EXPECT_FALSE(
        rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, R"({"multipleOf": 5})"), failReason));
}

//! Unsupported keywords must be found however deeply they are buried.
TEST(GuidedDecodingValidationTest, RejectsNestedUnsupportedKeyword)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema,
            R"({"type":"object","properties":{"outer":{"type":"array","items":{"type":"object",
                "properties":{"inner":{"type":"integer","multipleOf":3}}}}}})"),
        failReason));
}

//! An unknown `format` is ignored rather than rejected by XGrammar, so it belongs on the list;
//! the fourteen it really implements must still pass.
TEST(GuidedDecodingValidationTest, RejectsUnknownFormatButAcceptsSupportedOnes)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"type":"string","format":"totally-bogus"})"), failReason));

    for (char const* format : {"email", "date-time", "uuid", "ipv4", "uri"})
    {
        auto const schema = std::string(R"({"type":"string","format":")") + format + R"("})";
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << format << ": " << failReason;
    }
}

//! propertyNames was broken in v0.1.25 and works in v0.2.1. The blacklist tracks measurement
//! against the pinned version, so this one is deliberately allowed; if a version bump
//! regresses it, this test is the reminder to re-measure.
TEST(GuidedDecodingValidationTest, AcceptsKeywordsThatWorkOnThePinnedVersion)
{
    std::string failReason;
    for (char const* schema : {
             R"({"type":"object","propertyNames":{"pattern":"^[a-z]+$"}})",
             R"({"type":"object","properties":{"a":{"type":"integer","minimum":1,"maximum":9}}})",
             R"({"type":"string","minLength":3,"maxLength":5})",
             R"({"type":"array","items":{"type":"integer"},"minItems":2})",
             R"({"type":"object","properties":{"a":{"enum":["x","y"]}}})",
         })
    {
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << schema << ": " << failReason;
    }
}

//! Under "properties" (and $defs, definitions, ...) the keys are user-chosen names, not schema
//! keywords, so a field literally named "contains" must not be mistaken for the keyword.
TEST(GuidedDecodingValidationTest, AcceptsPropertyNamesThatCollideWithBlacklistedKeywords)
{
    std::string failReason;
    for (char const* schema : {
             R"({"type":"object","properties":{"contains":{"type":"string"}},"required":["contains"]})",
             R"({"type":"object","properties":{"multipleOf":{"type":"integer"},"format":{"type":"string"}}})",
             R"({"$defs":{"uniqueItems":{"type":"string"}},"type":"object"})",
             R"({"type":"object","properties":{"a":{"type":"string"}},"required":["contains","multipleOf"]})",
         })
    {
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << schema << ": " << failReason;
    }
}

//! The name-space exemption must not hide a real keyword one level deeper.
TEST(GuidedDecodingValidationTest, StillRejectsUnsupportedKeywordInsideANamedSubschema)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema,
            R"({"type":"object","properties":{"contains":{"type":"integer","multipleOf":5}}})"),
        failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"$defs":{"ok":{"type":"array","uniqueItems":true}}})"), failReason));
}

//! A guide big enough to stall compilation is a denial-of-service vector, since compilation is
//! synchronous and runs before any GPU work.
TEST(GuidedDecodingValidationTest, RejectsOversizedGuide)
{
    std::string const huge(limits::security::kMaxGuidedDecodingGuideBytes + 1, 'a');
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, huge), failReason));
}

//! Malformed JSON must be reported as such rather than reaching the compiler.
TEST(GuidedDecodingValidationTest, RejectsMalformedJsonSchema)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, R"({"type":)"), failReason));
}

//! Regex and EBNF are not JSON and must not be parsed as such.
TEST(GuidedDecodingValidationTest, AcceptsNonJsonGuides)
{
    std::string failReason;
    EXPECT_TRUE(
        rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, R"([a-z]+@[a-z]+\.[a-z]{2,3})"), failReason))
        << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" | "b")"), failReason))
        << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"(["yes","no"])"), failReason))
        << failReason;
}

//! choice carries a JSON array rather than a grammar, so the shape is checked up front; an
//! empty array would otherwise lower to an alternation with no branches.
TEST(GuidedDecodingValidationTest, RejectsMalformedChoiceList)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, "[\"a\""), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, "[]"), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"({"a":1})"), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"(["a",1])"), failReason));
}

//! Only json_object may carry an empty guide; for the rest an empty string means "unset" and
//! must never be read as "a grammar that accepts nothing".
TEST(GuidedDecodingValidationTest, RejectsEmptyGuideExceptJsonObject)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, ""), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, ""), failReason));
}

// ---------------------------------------------------------------- request helper

TEST(GuidedDecodingRequestTest, DetectsGuidedDecodingInAnySlot)
{
    rt::LLMGenerationRequest request;
    request.requests.resize(2);
    EXPECT_FALSE(rt::hasGuidedDecoding(request));
    request.requests[1].guidedDecoding = makeGuide(rt::GuideType::kJsonObject, "");
    EXPECT_TRUE(rt::hasGuidedDecoding(request));
}

// ---------------------------------------------------------------- apply kernel (T4)

class ApplyTokenBitmaskTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }
    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! Pack a per-token allow list into the bit layout XGrammar produces: bit set = allowed.
    static std::vector<int32_t> packMask(std::vector<std::vector<int32_t>> const& allowedPerRow, int32_t vocabSize)
    {
        int32_t const words = (vocabSize + 31) / 32;
        std::vector<int32_t> packed(allowedPerRow.size() * words, 0);
        for (size_t row = 0; row < allowedPerRow.size(); ++row)
        {
            for (auto const tokenId : allowedPerRow[row])
            {
                packed[row * words + tokenId / 32] |= (1 << (tokenId % 32));
            }
        }
        return packed;
    }

    cudaStream_t mStream{};
};

//! Forbidden entries go to the sentinel; allowed entries must be left exactly as they were.
TEST_F(ApplyTokenBitmaskTest, MasksForbiddenTokensAndLeavesAllowedUntouched)
{
    constexpr int32_t kROWS = 2;
    constexpr int32_t kVOCAB = 8;
    int32_t const words = (kVOCAB + 31) / 32;

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    std::vector<float> hostLogits(kROWS * kVOCAB);
    for (size_t i = 0; i < hostLogits.size(); ++i)
    {
        hostLogits[i] = static_cast<float>(i) + 1.0F;
    }
    copyHostToDevice<float>(logits, hostLogits);

    auto const packed = packMask({{1, 3}, {0, 7}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1, 1});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(hostLogits.size());
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int32_t>> const allowed{{1, 3}, {0, 7}};
    for (int32_t row = 0; row < kROWS; ++row)
    {
        for (int32_t token = 0; token < kVOCAB; ++token)
        {
            bool const isAllowed = std::find(allowed[row].begin(), allowed[row].end(), token) != allowed[row].end();
            float const value = result[row * kVOCAB + token];
            if (isAllowed)
            {
                EXPECT_FLOAT_EQ(value, hostLogits[row * kVOCAB + token]) << "row " << row << " token " << token;
            }
            else
            {
                EXPECT_FLOAT_EQ(value, kMaskedLogitValue) << "row " << row << " token " << token;
            }
        }
    }
}

//! A cleared row flag is how unconstrained requests share a batch with constrained ones at
//! zero cost; the kernel must not touch those rows even though the bitmask holds stale bits.
TEST_F(ApplyTokenBitmaskTest, SkipsRowsWithClearedFlag)
{
    constexpr int32_t kROWS = 2;
    constexpr int32_t kVOCAB = 4;
    int32_t const words = (kVOCAB + 31) / 32;

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    std::vector<float> const hostLogits{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    copyHostToDevice<float>(logits, hostLogits);

    // Row 1's mask forbids everything, but its flag is clear, so it must survive intact.
    auto const packed = packMask({{0}, {}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1, 0});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(hostLogits.size());
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(result[0], 1.0F);
    for (int32_t token = 1; token < kVOCAB; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue);
    }
    for (int32_t token = 0; token < kVOCAB; ++token)
    {
        EXPECT_FLOAT_EQ(result[kVOCAB + token], hostLogits[kVOCAB + token]);
    }
}

//! The sentinel must stay finite after the sampler divides by temperature; -FLT_MAX would
//! overflow back to -inf and make the softmax denominator zero.
TEST_F(ApplyTokenBitmaskTest, MaskedValueSurvivesTemperatureScaling)
{
    constexpr float kLOWEST_TEMPERATURE = 0.01F;
    float const scaled = kMaskedLogitValue / kLOWEST_TEMPERATURE;
    EXPECT_TRUE(std::isfinite(scaled));
    EXPECT_LT(scaled, 0.0F);
}

//! A vocabulary that is not a multiple of 32 exercises the partial trailing word.
TEST_F(ApplyTokenBitmaskTest, HandlesVocabularyNotAlignedToWordBoundary)
{
    constexpr int32_t kROWS = 1;
    constexpr int32_t kVOCAB = 35;
    int32_t const words = (kVOCAB + 31) / 32;
    ASSERT_EQ(words, 2);

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    std::vector<float> const hostLogits(kVOCAB, 5.0F);
    copyHostToDevice<float>(logits, hostLogits);

    auto const packed = packMask({{34}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[34], 5.0F);
    for (int32_t token = 0; token < 34; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue) << "token " << token;
    }
}

// ---------------------------------------------------------------- reasoning gate seed

//! The chat template, not the request flag, decides where the reasoning block stands, and the
//! prompt is the only place that shows it. Cases are named after the template shapes in use.
TEST(GuidedDecodingReasoningGateTest, SeedsFromThePromptRatherThanARequestFlag)
{
    constexpr int32_t kSTART_THINK = 100;
    constexpr int32_t kEND_THINK = 101;
    constexpr int32_t kSTART_CHANNEL = 102;
    constexpr int32_t kEND_CHANNEL = 103;
    constexpr int32_t kTEXT = 7;
    std::vector<int32_t> const starts{kSTART_CHANNEL, kSTART_THINK};
    std::vector<int32_t> const ends{kEND_CHANNEL, kEND_THINK};

    // Template opened and closed the block: constrain from the first generated token. The end
    // marker is never generated, so a gate waiting for one would never open.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kTEXT, kSTART_THINK, kEND_THINK}, starts, ends));

    // Template left the block open: the model is mid-reasoning.
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kSTART_THINK}, starts, ends));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kSTART_CHANNEL}, starts, ends));

    // Template omitted the block: the model may still open one itself.
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kTEXT}, starts, ends));

    // Only the most recent marker counts; an earlier turn's block must not leak.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kSTART_THINK, kEND_THINK, kTEXT, kSTART_THINK, kEND_THINK}, starts, ends));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kSTART_THINK, kEND_THINK, kTEXT, kSTART_THINK}, starts, ends));

    // Marker families are independent: a channel block closes on the channel end marker.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kSTART_CHANNEL, kEND_CHANNEL}, starts, ends));
}

//! A model with no reasoning markers has no reasoning phase, so it must not sit behind a gate
//! that can never open. Absent tokens come back from the tokenizer as -1.
TEST(GuidedDecodingReasoningGateTest, ModelWithoutMarkersIsNeverGated)
{
    std::vector<int32_t> const absent{-1, -1};
    EXPECT_TRUE(rt::reasoningClosedInPrompt({7, 8, 9}, absent, absent));
    EXPECT_TRUE(rt::reasoningClosedInPrompt({}, absent, absent));

    // -1 must not match a padding or placeholder id that happens to be negative.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({-1, 7}, absent, absent));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({-1, 7}, {-1, 100}, {-1, 101}));
}

// ---------------------------------------------------------------- GuidedDecoder lifecycle

//! A tiny real tokenizer: enough vocabulary to express `{"a":1}` plus an EOS, so the grammar
//! machinery can be exercised end to end without loading a model.
class GuidedDecoderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = std::filesystem::temp_directory_path() / "edgellm_guided_decoder_test";
        std::filesystem::remove_all(mDir);
        std::filesystem::create_directories(mDir);

        // IDs 0..9 spell the JSON we need; 10 is EOS.
        std::ofstream(mDir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {
    "{": 0, "}": 1, "\"": 2, ":": 3, "a": 4, "1": 5, "2": 6, ",": 7, "b": 8, " ": 9, "<eos>": 10
  }, "merges": []},
  "added_tokens": [{"id": 10, "content": "<eos>"}],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
        std::ofstream(mDir / "tokenizer_config.json") << R"JSON({"eos_token": {"content": "<eos>"}})JSON";
        std::ofstream(mDir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {"system": {"prefix": "", "suffix": ""}, "user": {"prefix": "", "suffix": ""},
            "assistant": {"prefix": "", "suffix": ""}},
  "generation_prompt": ""
})JSON";
        ASSERT_TRUE(mTokenizer.loadFromHF(mDir));
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! No reduced-vocabulary map: output space is the identity over the full vocabulary.
    void initDecoder(rt::GuidedDecoder& decoder, int32_t maxBatchSize)
    {
        rt::Tensor emptyMap;
        decoder.initialize(maxBatchSize, kVOCAB_SIZE, &mTokenizer, emptyMap, mStream);
    }

    static constexpr int32_t kVOCAB_SIZE = 11;
    static constexpr int32_t kEOS_ID = 10;

    std::filesystem::path mDir;
    tokenizer::Tokenizer mTokenizer;
    cudaStream_t mStream{};
};

TEST_F(GuidedDecoderTest, CompilesEveryGuideType)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    struct Case
    {
        rt::GuideType type;
        char const* guide;
    };
    std::vector<Case> const cases{
        {rt::GuideType::kJsonObject, ""},
        {rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"},
        {rt::GuideType::kRegex, "a+"},
        {rt::GuideType::kEbnf, R"(root ::= "a" | "b")"},
        {rt::GuideType::kStructuralTag,
            R"({"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<f="],
                "tags":[{"begin":"<f=g>","content":{"type":"json_schema","json_schema":{"type":"object"}},
                "end":"</f>"}]}})"},
        {rt::GuideType::kChoice, R"(["a","b"])"},
    };
    for (auto const& testCase : cases)
    {
        std::string failReason;
        EXPECT_TRUE(decoder.prepareSlot(0, makeGuide(testCase.type, testCase.guide), failReason))
            << rt::guideTypeName(testCase.type) << ": " << failReason;
        EXPECT_TRUE(decoder.hasGrammar(0));
    }
}

//! A guide that cannot compile must fail only its own slot, leaving the rest of the batch to
//! run: compilation is the one input-driven step, so it is the one that must not be fatal.
TEST_F(GuidedDecoderTest, FailedCompilationIsIsolatedToItsSlot)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    EXPECT_FALSE(decoder.prepareSlot(1, makeGuide(rt::GuideType::kEbnf, "this is not valid ebnf ((("), failReason));
    EXPECT_FALSE(failReason.empty());

    EXPECT_TRUE(decoder.hasGrammar(0));
    EXPECT_FALSE(decoder.hasGrammar(1));
    EXPECT_TRUE(decoder.hasAnyGrammar());
}

//! The three-stage termination model: mid-rule, root complete but stop token not yet accepted,
//! and finally terminated. Filling a mask past the last stage aborts inside XGrammar, which is
//! why the runtime gates on isTerminated().
TEST_F(GuidedDecoderTest, TerminatesOnlyAfterAcceptingTheStopToken)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    // Spell out {"a":1 -- still mid-rule.
    for (int32_t token : {0, 2, 4, 2, 3, 5})
    {
        ASSERT_TRUE(decoder.advance(0, token)) << "token " << token;
        EXPECT_FALSE(decoder.isTerminated(0));
    }
    // Closing brace completes the root rule, but termination needs the stop token.
    ASSERT_TRUE(decoder.advance(0, 1));
    EXPECT_FALSE(decoder.isTerminated(0));

    ASSERT_TRUE(decoder.advance(0, kEOS_ID));
    EXPECT_TRUE(decoder.isTerminated(0));
}

//! The grammar must actually forbid things: a letter where a value belongs is rejected.
TEST_F(GuidedDecoderTest, RejectsTokensTheGrammarForbids)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    for (int32_t token : {0, 2, 4, 2, 3})
    {
        ASSERT_TRUE(decoder.advance(0, token)) << "token " << token;
    }
    EXPECT_FALSE(decoder.advance(0, 4)) << "'a' is not a legal start for an integer value";
}

//! Missing this reindexing is the failure mode where a constraint silently migrates onto a
//! different request; after an eviction slot 0 must carry what was slot 1's grammar.
TEST_F(GuidedDecoderTest, CompactSlotsMovesMatchersWithTheirRequests)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 3);

    std::string failReason;
    // Only slots 1 and 2 are constrained, and they are given different grammars so the test can
    // tell which matcher survived where.
    ASSERT_TRUE(decoder.prepareSlot(1, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a")"), failReason)) << failReason;
    ASSERT_TRUE(decoder.prepareSlot(2, makeGuide(rt::GuideType::kEbnf, R"(root ::= "b")"), failReason)) << failReason;
    ASSERT_FALSE(decoder.hasGrammar(0));

    // Evict old slot 0; old slot 1 -> new 0, old slot 2 -> new 1.
    decoder.compactSlots({-1, 0, 1});

    EXPECT_TRUE(decoder.hasGrammar(0));
    EXPECT_TRUE(decoder.hasGrammar(1));
    EXPECT_FALSE(decoder.hasGrammar(2));
    // New slot 0 is the "a"-only grammar; new slot 1 is the "b"-only one.
    EXPECT_TRUE(decoder.advance(0, 4)) << "new slot 0 should accept 'a'";
    EXPECT_FALSE(decoder.advance(1, 4)) << "new slot 1 should reject 'a'";
}

TEST_F(GuidedDecoderTest, ResetDropsEveryMatcher)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);
    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    ASSERT_TRUE(decoder.hasAnyGrammar());
    decoder.reset();
    EXPECT_FALSE(decoder.hasAnyGrammar());
    EXPECT_FALSE(decoder.hasGrammar(0));
}

//! Repeating a schema must hit the LRU rather than recompiling.
TEST_F(GuidedDecoderTest, RepeatedGuideReusesTheCompiledGrammar)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    auto const guide = makeGuide(
        rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})");
    ASSERT_TRUE(decoder.prepareSlot(0, guide, failReason)) << failReason;
    int64_t const cacheAfterFirst = decoder.cacheSizeBytes();
    ASSERT_GT(cacheAfterFirst, 0);

    ASSERT_TRUE(decoder.prepareSlot(1, guide, failReason)) << failReason;
    EXPECT_EQ(decoder.cacheSizeBytes(), cacheAfterFirst) << "an identical guide should not grow the cache";
}

//! The mask must mark exactly the legal continuations. At the very start of an object grammar
//! only `{` is legal, so the row has a single set bit.
TEST_F(GuidedDecoderTest, FillMasksMarksOnlyLegalTokens)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_TRUE(unsatisfiable.empty());

    // Apply the mask to a flat row and read back which entries survived.
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[0], 1.0F) << "'{' must be allowed at the start of an object";
    for (int32_t token = 1; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue) << "token " << token << " should be forbidden";
    }
}

//! choice has no XGrammar primitive behind it: it is lowered to an EBNF alternation, so the
//! test that matters is that only the first character of some alternative is legal at step 0.
TEST_F(GuidedDecoderTest, ChoiceAllowsOnlyTheFirstCharacterOfAnAlternative)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kChoice, R"(["a","b"])"), failReason)) << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        bool const allowed = (token == 4 || token == 8); // 'a' and 'b'
        EXPECT_FLOAT_EQ(result[token], allowed ? 1.0F : kMaskedLogitValue) << "token " << token;
    }
}

//! An alternative is embedded in the generated grammar as a literal, so any character that
//! terminates an EBNF literal must survive the round trip rather than truncate the branch.
TEST_F(GuidedDecoderTest, ChoiceEscapesQuotesInsideAnAlternative)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kChoice, R"(["\"a\"","b"])"), failReason))
        << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        bool const allowed = (token == 2 || token == 8); // '"' and 'b'
        EXPECT_FLOAT_EQ(result[token], allowed ? 1.0F : kMaskedLogitValue) << "token " << token;
    }
}

//! The reasoning gate must open only on the thinking-end marker. `thinkingDone` also flips
//! when the first generated token is not a thinking marker, and that token is answer content:
//! opening on it would let it past the mask and out of the matcher's prefix, so the request
//! returns `<stray token>{...}` while the matcher believes it started from an empty prefix.
TEST_F(GuidedDecoderTest, ReasoningGateIgnoresTheFirstTokenHeuristic)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;

    rt::DecodingInferenceContext context;
    context.initialize(/*batchSize=*/1, /*maxSeqLen=*/4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    context.hasGuidedDecoding = true;
    context.enableThinking = true;
    context.finishedStates = {0};
    context.currentGenerateLengths = {1};
    // What the first-token heuristic does, and what it must not be allowed to imply here.
    context.thinkingDone = {1};
    context.guidedReasoningEnded = {0};

    auto applyAndRead = [&]() {
        rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
        copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 3.0F));
        rt::applyGuidedDecodingMask(decoder, context, logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> result(kVOCAB_SIZE);
        CUDA_CHECK(
            cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    };

    std::vector<float> const duringReasoning = applyAndRead();
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(duringReasoning[token], 3.0F) << "token " << token << " must be free while reasoning";
    }

    // The end marker is what opens the gate; `{` is then the only legal start of an object.
    context.guidedReasoningEnded = {1};
    std::vector<float> const afterReasoning = applyAndRead();
    EXPECT_FLOAT_EQ(afterReasoning[0], 3.0F) << "'{' must be allowed once the gate opens";
    for (int32_t token = 1; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(afterReasoning[token], kMaskedLogitValue) << "token " << token;
    }
}

//! A suppressed slot -- finished, or still inside its thinking block -- must be left alone even
//! though it carries a grammar. This is also what keeps a terminated matcher from being asked
//! for another mask, which XGrammar treats as a hard error.
TEST_F(GuidedDecoderTest, SuppressedSlotIsLeftUnconstrained)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;

    std::vector<int8_t> const suppressed{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);

    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 3.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], 3.0F) << "token " << token;
    }
}

//! A slot without a grammar shares the batch with a constrained one and must be untouched.
TEST_F(GuidedDecoderTest, MixedBatchLeavesUnconstrainedSlotsAlone)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int8_t> const suppressed{0, 0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/2, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);

    rt::Tensor logits({2, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(2 * kVOCAB_SIZE, 2.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/2, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(2 * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[0], 2.0F) << "'{' stays legal for the constrained slot";
    EXPECT_FLOAT_EQ(result[1], kMaskedLogitValue) << "'}' is illegal for the constrained slot";
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[kVOCAB_SIZE + token], 2.0F) << "unconstrained slot, token " << token;
    }
}

//! A grammar needing bytes the vocabulary cannot spell produces an all-zero row. Only we act
//! on this, because only we support a pruned base vocabulary where it is reachable.
TEST_F(GuidedDecoderTest, ReportsUnsatisfiableGrammarAsAnAllZeroRow)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    // 'z' exists in no token of this vocabulary, so nothing can ever satisfy the rule.
    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "zzz")"), failReason)) << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    ASSERT_EQ(unsatisfiable.size(), 1U);
    EXPECT_EQ(unsatisfiable[0], 0);
}

} // namespace
