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

#include "common/common.h"
#include "decodingParams.h"
#include <cstdint>
#include <curand_kernel.h>
#include <vector>

static constexpr std::int32_t TOP_K_MAX = 1024;

template <typename T>
struct TopKSamplingKernelParams
{

    //! Input buffer [batchSize, maxTokensPerStep, vocabSizePadded].
    //! Log probabilities of each token in the vocab. If logitsHasProbs is true,
    //! logProbs must contain **just** probabilities instead of log probabilities.
    T const* logProbs{nullptr};
    //! input buffer [batchSize][vocabSizePadded] array of pointers to logits.
    //! If nullptr, logProbs is used. Only maxTokensPerStep == 1 is supported.
    T const* const* logProbsPtrs{nullptr};

    //! output buffer [maxBatchSize][maxSeqLen], optional. Contains pointers to
    //! rows with output tokens per request. If nullptr, outputIds must be
    //! provided.
    std::int64_t** outputIdsPtrs{nullptr};
    //! output buffer [maxBatchSize, maxSeqLen], optional. Tensor to store output
    //! tokens. Not used if outputIdsPtrs != nullptr
    std::int64_t* outputIds{nullptr};

    //! Required. Pointer to the workspace of size returned by
    //! getTopKWorkspaceSize. Has to be pre-allocated by caller. Function does not
    //! take ownership of the buffer
    void* workspace{nullptr};

    //! input buffer [maxBatchSize], optional. EOS token ids per request
    std::int32_t const* endIds{nullptr};

    //! input/output buffer [maxBatchSize], optional. If nullptr, seqLen is 0
    //! Current sequence length of the request. Set up to, but excluding endId
    //! token.
    std::int32_t* sequenceLengths{nullptr};
    //! input buffer[batchSize], optional. Indices of rows of data in memory pool.
    //! Linear indexing (batchIdx) is used if nullptr.
    std::int32_t const* batchSlots{nullptr};
    //! input buffer [maxBatchSize], optional. Number of tokens per step for each
    //! request. It is assumed that all requests have maxTokensPerStep tokens per
    //! step if nullptr.
    std::int32_t const* tokensPerStep{nullptr};

    //! input buffer [maxBatchSize], optional. If true, request exits early.
    FinishedState const* finishedInput{nullptr};
    //! output buffer [maxBatchSize], optional.
    //! Set to true if sequence has finished (if finished || outputId == endId).
    FinishedState* finishedOutput{nullptr};
    //! input buffer [maxBatchSize]. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    //! input/output buffer [maxBatchSize], optional.
    //! Cumulative log probability of selected tokens. Ignored if nullptr
    float* cumLogProbs{nullptr};
    //! output buffer [maxBatchSize]. Log probs is the probability induced by the
    //! top-k sampling. If normalizeLogProbs is true, we normalize the probability
    //! 'expLogit' of the selected token by the probability 's_sum' of a set of
    //! top-k tokens, meaning the logProb is the probability of the selected
    //! token, conditioned on the event that it is selected, i.e., log_prob = log
    //! P(i | i is in top-k) = log(expLogit / s_sum). Ignored if nullptr.
    float* outputLogProbs{nullptr};

    //! input buffer [maxBatchSize]. Initialized curand states
    curandState_t* curandState{nullptr};
    //! input buffer [maxBatchSize]. K for topK sampling per request.
    //! Supported K is in range [1; 1024]. Where K=1 is greedy search.
    //! If nullptr maxTopK is used for all requests.
    std::int32_t const* topKs{nullptr};
    //! input buffer [maxBatchSize]. Probability for topP sampling per request.
    //! Supported P is in range (0.0, 1.0]. If nullptr, topP is used for all
    //! requests
    float const* topPs{nullptr};
    //! maximum among all topKs K for topK sampling
    std::int32_t maxTopK{TOP_K_MAX};
    //! probability for topP sampling.
    float maxTopP{1.0f};

    std::int32_t batchSize{-1};
    std::int32_t maxBatchSize{-1};
    std::int32_t vocabSizePadded{-1};
    std::int32_t maxTokensPerStep{-1};
    std::int32_t maxSeqLen{-1};

    //! when set to True outputLogProbs are normalized to TopK
    bool normalizeLogProbs{false};
    //! flag to highlight that logProbs contains probabilities
    bool logitsHasProbs{false};
    //! flag to return all selectedTopK results
    bool returnAllTopK{false};

    void checkParams() const
    {
        assert(batchSize > 0);
        assert(maxBatchSize > 0);
        assert(maxBatchSize >= batchSize);
        assert(vocabSizePadded > 0);
        assert(maxTokensPerStep > 0);

        assert(logProbs || logProbsPtrs);
        assert(outputIds || outputIdsPtrs);

        if (maxTokensPerStep > 1)
        {
            assert(tokensPerStep);
        }

        if (outputIds)
        {
            assert(maxSeqLen > 0);
        }

        assert(workspace);
        assert(curandState);

        if (cumLogProbs != nullptr || outputLogProbs != nullptr)
        {
            assert(maxTokensPerStep == 1 && !returnAllTopK);
        }

        assert(0 < maxTopP && maxTopP <= 1.f);
        assert(0 <= maxTopK && maxTopK <= TOP_K_MAX);
    }
};

template <typename T>
void invokeBatchTopKSampling(TopKSamplingKernelParams<T> const& params, cudaStream_t stream);