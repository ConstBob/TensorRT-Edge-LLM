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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"

#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class DFlashDecoder final : public DecodingStrategy
{
public:
    DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kDFlash;
    }

    char const* name() const noexcept override
    {
        return "dflash";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    char const* unsupportedReason(LLMGenerationRequest const& request) const noexcept override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const& key) const override;
    void restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream) override;
    bool runSystemPromptPrefill(DecodingInferenceContext& context) override;
    void saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
        std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    bool runDraftForward(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);

    DecodingRuntimeContext& mRuntime;

    //! Draft KV cache manager (shared resource index 1)
    HybridCacheManager& mDraftCacheManager;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;

    //! Draft engine I/O tensors
    Tensor mDraftInputsEmbeds; //!< [B, BS, draftHiddenSize] FP16
    Tensor mDraftTargetHidden; //!< [B, deltaLen, baseOutputHiddenDim] FP16 — target hidden delta only
    Tensor mDraftOutputLogits; //!< [B, BS, vocabSize] FP32

    //! Proposal attention inputs (prepared by DFlash prep kernel)
    Tensor mDraftPackedAttentionMask; //!< [B, BS, divUp(BS,32)] INT32
    Tensor mDraftAttentionPosId;      //!< [B, BS] INT32
    Tensor mDraftContextLengths;      //!< [B] INT32
    Tensor mDraftDeltaLenCommit;      //!< [B] INT32 — pre-allocated for draft cache commit
    Tensor mDraftDeltaLens;           //!< [B] INT32 — per-batch delta lengths for KV cache update plugin

    //! Draft/verify tokens
    Tensor mDraftTokenIds;          //!< [B, BS] INT32
    Tensor mVerifyTokenIds;         //!< [B, BS] INT32
    Tensor mAcceptedTokenIds;       //!< [B, BS] INT32
    Tensor mAcceptLength;           //!< [B] INT32
    Tensor mHostAcceptLengths;      //!< [B] INT32 (CPU)
    Tensor mHostAcceptedTokenIds;   //!< [B, BS] INT32 (CPU)
    Tensor mHostDraftInputIds;      //!< [B, BS] INT32 (CPU)
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 (CPU)
    Tensor mHostDeltaLens;          //!< [B] INT32 (CPU)

    //! Pre-allocated argmax scratch buffer for dflashSequentialAccept [maxBatch * BS] INT32
    Tensor mArgmaxScratch;

    //! Last accepted token per batch [maxBatch] INT32 (GPU)
    Tensor mLastAcceptedTokens;

    //! System prompt KV cache for draft target KV
    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;

    //! DFlash-specific parameters
    //! mBlockSize is the runtime proposal/verify block size, set from specConfig->verifySize.
    //! This is the number of tokens the draft engine generates per round and base verifies.
    //! It may be smaller than the DFlash checkpoint's training block_size (e.g., verify=8
    //! on a block_size=16 checkpoint for Qwen3.5 4B with GDN intermediate-state constraints).
    int32_t mBlockSize{16};
    int32_t mMaskTokenId{248070};
    int32_t mDraftHiddenSize{0};
    int32_t mBaseOutputHiddenDim{0};
    int32_t mDraftVocabSize{0};
};

} // namespace rt
} // namespace trt_edgellm
