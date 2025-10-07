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

#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/eagleDraftEngineRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

namespace drivellm
{
namespace rt
{

// Data structure to hold execution context within spec-decode runtime to store
// execution information and intermediate meta data.
struct SpecDecodeInferenceContext
{
    std::vector<int32_t> tokenIds;
    rt::OptionalInputTensor multimodalEmbeddings;
    int32_t generationRound;
    int32_t maxGenerateLength;
    int32_t currentGenerateLength;
    cudaStream_t stream;
};

// Drafting configuration we want to use to drive eagle3 spec-decoding.
// draftingTopK: Tokens to select from one predecessor to build next level of draft tree.
// draftingStep: Number of drafting steps to perform with the draft model.
// verifyTreeSize: Number of tokens we collect for the base model to verify.
struct EagleDraftingConfig
{
    int32_t draftingTopK;
    int32_t draftingStep;
    int32_t verifyTreeSize;
};

static constexpr int32_t kRUNTIME_BATCH_SIZE{1};

class LLMInferenceSpecDecodeRuntime
{
public:
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        EagleDraftingConfig const& draftingConfig, cudaStream_t stream);

    ~LLMInferenceSpecDecodeRuntime() = default;
    bool captureDraftProposalCudaGraph(cudaStream_t stream);

    bool captureDraftAcceptDecodeTokenCudaGraph(cudaStream_t stream);

    bool captureBaseVerificationCudaGraph(cudaStream_t stream);

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mPrefillMetrics;
    }

    //! Get Eagle generation stage metrics
    metrics::EagleGenerationMetrics const& getEagleGenerationMetrics() const
    {
        return mEagleGenerationMetrics;
    }

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mMultimodalRunner ? mMultimodalRunner->getMultimodalMetrics() : metrics::MultimodalMetrics{};
    }

private:
    EagleDraftingConfig mDraftingConfig;
    LLMEngineRunnerConfig mBaseEngineConfig;
    EagleDraftEngineRunnerConfig mDraftEngineConfig;

    std::unique_ptr<LLMEngineRunner> mBaseEngineRunner;
    std::unique_ptr<EagleDraftEngineRunner> mDraftEngineRunner;
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] I/O Tensors to work with base and eagle draft engine.
    rt::Tensor mIdsInput;
    rt::Tensor mContextLengthsInput;
    rt::Tensor mLogitsOutput;
    rt::Tensor mDraftTreeSize;
    rt::Tensor mDraftTreeMask;
    rt::Tensor mBaseHiddenStatesOutput;
    // Distinguish draft hidden states input and output since we cannot easily
    // Perform inplace update for hidden states between drafting steps.
    rt::Tensor mDraftHiddenStatesInput;
    rt::Tensor mDraftHiddenStatesOutput;

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;

    // [3] Data structures used during Draft tree constructions.
    // Data tables that store the data structure that can completely describe a multi-layer draft tree.
    rt::Tensor mDraftTokenIdsFullTable;
    rt::Tensor mDraftTokenScoreFullTable;
    rt::Tensor mDraftTokenPredecessorFullTable;
    // Store conversion table (offset) to map from draft-model vocab token id to the original token id.
    // base_id = draft_id + mapping_table[draft_id]
    rt::Tensor mDraftVocabMappingTable;

    rt::Tensor mDraftTreeRootTokenId;
    rt::Tensor mDraftTokenIdsTable;
    rt::Tensor mDraftTokenScoresTable;
    rt::Tensor mDraftTokenIntermediateScores;
    rt::Tensor mDraftTokenIntermediateParents;

    // [4] Data structures that used during base model verification.
    rt::Tensor mAcceptedTokenIds;
    rt::Tensor mAcceptedTokenIndices;
    rt::Tensor mAcceptLength;

    // Key functions to drive the spec-decode runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    bool runBaseModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the base model hidden states and input token of the sequence. Produce the draft hidden states and logits
    // for the last token of the sequence.
    bool runDraftModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the draft hidden states and logits for the last token of the sequence. Produce a speculative draft tree
    // that described by a sequence of draft tokens and tree mask that describe the tree structure.
    bool constructDraftTree(SpecDecodeInferenceContext& context);

    // Consume the speulative draft tree, produce selected tokens and corresponding hidden states.
    bool runBaseModelVerification(SpecDecodeInferenceContext& context);

    // Consume the selected tokens and base model hidden state, produce the draft hidden states and logits for the last
    // token of the accepted sequence.
    bool runDraftModelAcceptToken(SpecDecodeInferenceContext& context);

    // Helper function to load draft vocab mapping table from file. To be removed by SafeTensor loader.
    bool loadDraftVocabMappingTable(std::filesystem::path const& draftVocPath, cudaStream_t stream);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::EagleGenerationMetrics mEagleGenerationMetrics;
};

} // namespace rt
} // namespace drivellm
