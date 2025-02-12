/*
 * Copyright 2024 The TensorRT-LLM Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
 
#pragma once

/// @brief Represents the inputs to the decoder.
/// @details This input type is assumed immutable. It represents whatever the
/// decoder received initially, and can always be referred to as such.
#include "memoryUtils.h"
#include <cstdint>
#include <curand_kernel.h>
#include <memory>
#include <optional>

class FinishedState
{
public:
    static auto constexpr empty()
    {
        return FinishedState{0};
    }

    static auto constexpr finished()
    {
        return FinishedState{kFinished};
    }

    static auto constexpr skipDecoding()
    {
        return FinishedState{kSkipDecoding};
    }

    static auto constexpr finishedEOS()
    {
        return FinishedState{kFinishedEos};
    }

    static auto constexpr finishedMaxLength()
    {
        return FinishedState{kFinishedMaxLength};
    }

    static auto constexpr finishedStopWords()
    {
        return FinishedState{kFinishedStopWords};
    }

    __host__ __device__ void constexpr setFinishedEOS()
    {
        mState |= kFinishedEos;
    }

    __host__ __device__ bool constexpr isFinishedEOS()
    {
        return anyBitSet(kFinishedEos);
    }

    __host__ __device__ void constexpr setFinishedStopWords()
    {
        mState |= kFinishedStopWords;
    }

    __host__ __device__ bool constexpr isFinishedStopWords()
    {
        return anyBitSet(kFinishedStopWords);
    }

    __host__ __device__ void constexpr setFinishedMaxLength()
    {
        mState |= kFinishedMaxLength;
    }

    __host__ __device__ bool constexpr isFinishedMaxLength()
    {
        return anyBitSet(kFinishedMaxLength);
    }

    __host__ __device__ void constexpr setFinished()
    {
        mState |= kFinished;
    }

    __host__ __device__ bool constexpr isFinished() const
    {
        return anyBitSet(kFinished);
    }

    __host__ __device__ void constexpr setSkipDecoding()
    {
        mState = kSkipDecoding;
    }

    __host__ __device__ bool constexpr isSkipDecoding() const
    {
        return anyBitSet(kSkipDecoding);
    }

    using UnderlyingType = uint8_t;

private:
    // The default state is interpreted as not finished.
    __host__ __device__ constexpr FinishedState(UnderlyingType state)
        : mState(state)
    {
    }

    // Request has finished based on the generation of EOS token
    static UnderlyingType constexpr kFinishedEos{1u << 0};
    // Request has finished based on the generation of stop words
    static UnderlyingType constexpr kFinishedStopWords{1u << 1};
    // Request has finished based on reaching max sequence length
    static UnderlyingType constexpr kFinishedMaxLength{1u << 2};
    // Finished by any condition
    static UnderlyingType constexpr kFinished{kFinishedEos | kFinishedStopWords | kFinishedMaxLength};
    // Skip decoding. E.g. used for not accepted tokens in speculative decoding
    static UnderlyingType constexpr kSkipDecoding{1u << 3};

    __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    UnderlyingType mState{};
};

static_assert(!FinishedState::empty().isFinished());
static_assert(!FinishedState::empty().isSkipDecoding());
static_assert(FinishedState::finished().isFinished());
static_assert(FinishedState::skipDecoding().isSkipDecoding());
static_assert(FinishedState::finishedEOS().isFinishedEOS());
static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

class DecoderDomain
{
public:
    DecoderDomain(std::int64_t batchSize, std::int64_t beamWidth, std::int64_t vocabSize,
        std::optional<std::int64_t> vocabSizePadded = std::nullopt)
        : mBatchSize(batchSize)
        , mBeamWidth(beamWidth)
        , mVocabSize(vocabSize)
        , mVocabSizePadded(vocabSizePadded.value_or(vocabSize))
    {
    }

    [[nodiscard]] std::int64_t getBatchSize() const
    {
        return mBatchSize;
    }

    [[nodiscard]] std::int64_t getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::int64_t getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] std::int64_t getVocabSizePadded() const
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] std::int32_t getMaxDecodingTokens() const
    {
        return 1;
    }

private:
    std::int64_t mBatchSize;
    std::int64_t mBeamWidth;
    std::int64_t mVocabSize;
    std::int64_t mVocabSizePadded;
};

class BaseSetupParams
{
public:
    virtual ~BaseSetupParams() = default;
};

class DecodingSetupParams : public BaseSetupParams
{
public:
    virtual ~DecodingSetupParams() = default;

    std::optional<std::vector<uint64_t>> randomSeed; // [1] or [setupBatchSize] on cpu
    std::optional<std::vector<bool>> outputLogProbs; // [setupBatchSize]
    std::optional<std::vector<bool>> cumLogProbs;    // [setupBatchSize]
};

class SamplingSetupParams : public DecodingSetupParams
{
public:
    // baseSamplingLayer
    std::optional<std::vector<std::int32_t>> runtimeTopK; // [1] or [setupBatchSize] on cpu
    std::optional<std::vector<float>> runtimeTopP;        // [1] or [setupBatchSize] on cpu

    // topPSamplingLayer
    std::optional<std::vector<float>> topPDecay;           // [setupBatchSize], must between [0, 1]
    std::optional<std::vector<float>> topPMin;             // [setupBatchSize], must between [0, 1]
    std::optional<std::vector<std::int32_t>> topPResetIds; // [setupBatchSize]
    std::optional<bool> normalizeLogProbs;
};

class BaseDecodingInputs
{
public:
    BaseDecodingInputs(std::int32_t localBatchSize)
        : localBatchSize(localBatchSize)
    {
    }

    virtual ~BaseDecodingInputs() = default;

    std::int32_t localBatchSize;
};

class BaseDecodingOutputs
{
public:
    explicit BaseDecodingOutputs(TensorPtr outputIds)
        : outputIds{std::move(outputIds)}
    {
    }

    virtual ~BaseDecodingOutputs() = default;

    // mandatory parameters
    TensorPtr outputIds; // [maxBatchSize, maxSeqLen]
    std::int64_t maxSeqLen;

    // optional parameters
    //! [maxBatchSize * maxBeamWidth], optional
    std::optional<TensorPtr> finished;
    //! [maxBatchSize * maxBeamWidth], optional
    std::optional<TensorPtr> sequenceLength;
    //! [maxBatchSize * maxBeamWidth], necessary in beam search
    std::optional<TensorPtr> cumLogProbs;
    //! [maxBatchSize, maxBeamWidth, maxSeqLen], must be float*, optional
    std::optional<TensorPtr> outputLogProbs;
    //! [maxBatchSize, maxBeamWidth, maxSeqLen], necessary in beam search
    std::optional<TensorPtr> parentIds;

    //! [maxBatchSize] int* (2-d array), each int* has [maxBeamWidth, maxSeqLen]
    TensorPtr outputIdsPtr;
    //! [maxBatchSize] int* (2-d array), each int* has [maxBeamWidth, maxSeqLen]
    TensorPtr parentIdsPtr;

    // Tokens predicted at current iteration.
    TensorPtr newTokens; // [maxBatchSize, maxBeamWidth]

    // optional parameters
    //! Number of tokens predicted at current iteration.
    //! [maxBatchSize]
    std::optional<TensorPtr> numNewTokens;
    //! [1] in pinned host memory
    std::optional<TensorPtr> finishedSum;
    //! [maxSeqLen, maxBatchSize, maxBeamWidth], must be float*
    std::optional<TensorPtr> outputLogProbsTiled;
};

class DefaultDecodingParams
{
public:
    [[nodiscard]] __host__ __device__ static constexpr float getTemperature()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getRepetitionPenalty()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getPresencePenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getFrequencyPenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr std::int32_t getMinLength()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t getSeed()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr std::int32_t getTopK()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopP()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPDecay()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPMin()
    {
        return 1.0e-6f;
    }

    [[nodiscard]] __host__ __device__ static constexpr std::int32_t getTopPResetId()
    {
        return -1;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getBeamSearchDiversity()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getLengthPenalty()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr std::int32_t getEarlyStopping()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr bool getNormalizeLogProbs()
    {
        return false;
    }

    [[nodiscard]] static std::vector<std::int32_t> getTopKMedusaHeads()
    {
        return {};
    }

    [[nodiscard]] __host__ __device__ static constexpr std::int32_t getNoRepeatNgramSize()
    {
        return 1 << 30;
    }
};

class BanWordsDecodingInputs : public BaseDecodingInputs
{
public:
    BanWordsDecodingInputs(std::int32_t localBatchSize)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    std::int32_t maxBadWordsLen{0};
    //! [maxBatchSize][2, bad_words_length], on gpu
    std::optional<TensorConstPtr> badWordsPtr;
    //! [maxBatchSize], on gpu
    std::optional<TensorConstPtr> badWordsLengths;
};

class DecodingInputs : public BaseDecodingInputs
{
public:
    DecodingInputs(std::int32_t localBatchSize = 0)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    //! [maxBatchSize]
    std::optional<TensorConstPtr> endIds;

    //! One of these two fields has to be set
    //! DynamicDecodeLayer::forward checks for it
    //! Need both of these fields to support legacy code during transition period
    //! to the batched decoder [forwardBatchSize, beamWidth, vocabSizePadded]
    std::optional<TensorConstPtr> logits;
    //! [forwardBatchSize][beamWidth, vocabSizePadded], on gpu
    std::optional<std::vector<TensorPtr>> logitsVec;

    // optional parameters
    //! the indices of the selected beams, mandatory for beam search, on gpu
    //! [forwardBatchSize, maxBeamWidth, maxSeqLen]
    std::optional<TensorPtr> srcCacheIndirection;
    //! [vocabSizePadded], on gpu
    std::optional<TensorConstPtr> embeddingBias;
    //! [maxBatchSize, maxBeamWidth], on gpu
    std::optional<TensorConstPtr> inputLengths;
    //! [forwardBatchSize], on pinned memory
    std::optional<TensorConstPtr> batchSlots;
    //! [maxBatchSize, maxBeamWidth]
    std::optional<TensorConstPtr> finished;
    //! [maxBatchSize], on gpu
    std::optional<TensorPtr> curTokensPerStep;

    std::shared_ptr<BanWordsDecodingInputs> banWordsInputs;
};

class SamplingInputs : public DecodingInputs
{
public:
    explicit SamplingInputs(std::int32_t localBatchSize)
        : DecodingInputs{localBatchSize}
    {
    }

    //! optional parameters
    //! [localBatchSize]
    curandState_t* curandStates{};
    //! Pointer to the workspace for sampling computation
    void* samplingWorkspace{};
    //! Flag to mark that logits tensor contains probabilities
    bool probsComputed{};
};