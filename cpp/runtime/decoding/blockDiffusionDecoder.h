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

#include "runtime/decoding/decodingStrategy.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class BlockDiffusionDecoder final : public DecodingStrategy
{
public:
    explicit BlockDiffusionDecoder(
        DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kBlockDiffusion;
    }

    char const* name() const noexcept override
    {
        return "block_diffusion";
    }

    bool isSpeculative() const noexcept override
    {
        return false;
    }

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const override
    {
        return false;
    }
    void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t) override {}
    bool runSystemPromptPrefill(DecodingInferenceContext&) override
    {
        return true;
    }
    void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&, std::vector<tokenizer::Rank> const&,
        int32_t, cudaStream_t) override
    {
    }

    void resetForNewSequences(Tensor&, cudaStream_t) override {}
    void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) override {}

private:
    struct DenoiseStepParams
    {
        int32_t batchSize{0};
        int32_t canvasLen{0};
        int32_t step{0};
        std::vector<int32_t> const* validCanvasLengths{nullptr};
        float selfConditioningTemperature{0.0F};
        cudaStream_t stream{};
    };

    bool initializeCanvas(int32_t batchSize, int32_t canvasLen, cudaStream_t stream);
    bool prepareCanvasMetadata(int32_t batchSize, int32_t canvasLen, bool denoisePhase, cudaStream_t stream,
        std::vector<int32_t> const* contextLengths = nullptr);
    bool updateSelfConditioningTemperature(float temperature, cudaStream_t stream);
    bool prepareUnifiedConditioning(
        int32_t batchSize, int32_t canvasLen, int32_t step, float temperature, cudaStream_t stream);
    void restoreUnifiedBackboneBindings();
    Tensor& currentDenoiseLogits() noexcept;
    bool runDenoiseStep(DenoiseStepParams const& params);
    bool sampleCanvasEntropyBound(
        int32_t batchSize, int32_t canvasLen, int32_t step, int32_t maxDenoisingSteps, cudaStream_t stream);
    int32_t effectiveMaxDenoisingSteps(DecodingInferenceContext const& context) const noexcept;
    float denoiseTemperature(int32_t step, int32_t maxDenoisingSteps) const noexcept;
    float denoiseTemperature(int32_t step) const noexcept;
    int32_t readAcceptedPrefixLength(int32_t batchSize, int32_t canvasLen, cudaStream_t stream);
    bool copyCanvasToHost(int32_t batchSize, int32_t canvasLen, cudaStream_t stream);
    std::vector<int32_t> getCommittedLengths(DecodingInferenceContext const& context) const;
    bool compactCommitCanvas(int32_t batchSize, int32_t canvasLen, int32_t maxBlockLen,
        std::vector<int32_t> const& commitLengths, cudaStream_t stream);
    bool commitBlock(DecodingInferenceContext& context, std::vector<int32_t> const& commitLengths, int32_t canvasLen);

    DecodingRuntimeContext& mRuntime;
    Tensor mCanvasIds;
    Tensor mArgmaxCanvasIds;
    Tensor mSampledCanvasIds;
    Tensor mCommitCanvasIds;
    Tensor mCommitLengths;
    Tensor mSelfConditioningEmbedsA;
    Tensor mSelfConditioningEmbedsB;
    Tensor mSelfConditioningTemperature;
    Tensor mHostSelfConditioningTemperature;
    Tensor mPreviousArgmaxIds;
    Tensor mStableCounts;
    Tensor mAcceptedMask;
    Tensor mPrefixLengths;
    std::vector<int32_t> mHostPrefixLengths;
    Tensor* mCurrentDenoiseLogits{nullptr};
    uint64_t mRandomOffset{0};
    int32_t mCanvasLen{0};
    int32_t mMaxConditioningSeqLen{0};
    int32_t mMaxDenoisingSteps{0};
    int32_t mPrefixCheckInterval{1};
};

} // namespace rt
} // namespace trt_edgellm
