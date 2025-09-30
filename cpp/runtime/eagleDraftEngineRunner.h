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

#pragma once

#include "common/tensor.h"
#include "runtime/linearKVCache.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>

namespace drivellm
{
namespace rt
{

struct EagleDraftEngineRunnerConfig
{
    int32_t numDecoderLayers{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t maxSupportedInputLength{};
    int32_t kvCacheCapacityLength{};
    int32_t draftModelVocabSize{};
    int32_t maxDraftTreeSize{};
    int32_t baseModelHiddenDim{};
    int32_t draftModelHiddenDim{};
};

// Disable clang-format to explicitly format the class interface documentation.
// clang-format off
class EagleDraftEngineRunner
{
public:
    EagleDraftEngineRunner(
        std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream);

    ~EagleDraftEngineRunner() = default;

    // Get internal resources for the eagle draft engine.
    rt::Tensor& getRopeCosSinCacheTensor();
    rt::LinearKVCache& getLinearKVCache();

    EagleDraftEngineRunnerConfig getDraftEngineConfig() const;

    // API entry to execute prefill step for the eagle draft engine. By definition, eagle operates on feature level with
    // formulation of f_n = F_proj(f_{n}, token_{n+1}). The API will takes hidden states input from base model and
    // token_ids of [1 ~ N] as input, output logits and (draft) hidden states for the "last entry" to be used in
    // following draft proposal step. Currently, we only support batch size of 1. Inputs:
    //     inputIds [GPU, Int32]: Input token_ids for the draft model with shape [1, N] denoting the token_ids of [1 ~N]. 
    //     baseModelHiddenStates [GPU, Float16]: Hidden states input from base model with shape [1, N, base-Hidden-dim],
    //          denote hidden states corresponding to token_ids of [1 ~ N-1].
    //     draftModelHiddenStates [GPU, Float16]: The input [1, N, draft-Hidden-input-dim] is unused in the prefill step,
    //          but it is required by the engine execution. The input shall be set to all zeros to ensure correctness.
    //     stream: The CUDA stream to execute the prefill step.
    // Outputs:
    //     outputLogits [GPU, Float16]: The output logits with shape [1, draft-Vocab-Size].
    //     outputHiddenStates [GPU]: The output hidden states with shape [1, draft-hidden-dim].
    bool executeEaglePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        cudaStream_t stream);

    // API entry to execute the draft proposal step for the eagle draft engine. The API will takes a draft tree of
    // input_token_ids and hidden-states from the draft model. DraftTreeMask denote the relationship between the draft
    // tree nodes, draft tree length denote the "real" length of the draft tree. To efficiently use cuda graph and
    // reduce implementation complexity, the input length will be padded to accommodate the maximum draft tree size.
    // Inputs:
    //     draftTreeInputIds [GPU, Int32]: Input token_ids for the draft model with shape [1, padded-draft-Tree-Size].
    //     draftModelHiddenStates [GPU, Float16]: Hidden states input from draft model with shape [1, padded-draft-Tree-Size, draft-Hidden-Dim],
    //          denote hidden states corresponding to token_ids of [1 ~ draft-Tree-Size].
    //     baseModelHiddenStates [GPU, Float16]: The input [1, padded-draft-Tree-Size, base-Hidden-Dim] is unused in the
    //          draft proposal step, but it is required by the engine execution. The input shall be set to all zeros to ensure correctness.
    //     draftTreeLength [GPU, Int32]: Denote the "real" length of the draft tree with shape [1]
    //     draftTreeMask [GPU, Int32]: Denote the relationship between the draft tree nodes with shape [1, padded-draft-Tree-Size, padded-draft-Tree-Size].
    //     stream: The CUDA stream to execute the draft proposal step.
    // Outputs:
    //     outputLogits [GPU, Float16]: The output logits with shape [topK, draft-Vocab-Size].
    //     outputHiddenStates [GPU]: The output hidden states with shape [topK, draft-hidden-dim].
    // Note: The API will automatically collect the "last" topK logits and hidden-states counting from the tail of
    // "real" draft tree size. Caller shall
    //       specify the topK parameter through tensor dimension. Also this API will NOT "commit" the KVCache during
    //       execution.
    bool executeEagleDraftProposalStep(rt::Tensor const& draftTreeInputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    // API entry for the eagle draft model to accept the "committed" token from the base model. The functionality is
    // similar to the prefill step where this API will operates based on the previous committed KVCache. Output logits
    // and hidden-states will be collected from the last accepted token. Inputs:
    //     acceptedTokens [GPU, Int32]: The accepted tokens with shape [1, N_accepted].
    //     baseModelHiddenStates [GPU, Float16]: Hidden states input from base model with shape [1, N_accepted, base-Hidden-Dim].
    //     draftModelHiddenStates [GPU, Float16]: The input [1, N_accepted, draft-Hidden-Dim] is unused in the accept decode token step,
    //          but it is required by the engine execution. The input shall be set to all zeros to ensure correctness.
    //     stream: The CUDA stream to execute the accept decode token step.
    // Outputs:
    //     outputLogits [GPU, Float16]: The output logits with shape [1, draft-Vocab-Size].
    //     outputHiddenStates [GPU]: The output hidden states with shape [1, draft-hidden-dim].
    // Note: This API will "commit" the KVCache for the accepted tokens.
    bool executeEagleAcceptDecodeTokenStep(rt::Tensor const& acceptedTokens, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        cudaStream_t stream);

private:
    EagleDraftEngineRunnerConfig mConfig{};

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;

    rt::LinearKVCache mLinearKVCache{};

    // (GPU, Float32) to store the CosSinCache for rotary positional encoding.
    rt::Tensor mPosEncCosSinCache{};
    // (GPU, Int64) to store the select token indices that will be outputted from the model.
    rt::Tensor mSelectTokenIndices{};
    // (GPU, Int32) to store the sequence context lengths input that will be used by the TensorRT Engine.
    rt::Tensor mSequenceContextLengths{};
    // (GPU, Int32) to store the draft tree position ids within the sequence that used by positional encoding.
    rt::Tensor mDraftTreePositionIds{};
    // (GPU, Int32) to store the packed tree mask to indicate the attention relationship between the draft tree nodes.
    rt::Tensor mPackedTreeMask{};
    // (GPU, Int32) to store a GPU buffer as dummy input for the engine when shape is set to zero. TensorRT doesn't
    // allow binding address to be nullptr.
    rt::Tensor mDummyInput{};

    bool bindKVCacheToEngine(int32_t activeBatchSize);

    void initializeConfigFromEngine();

    bool prefillStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates);

    bool draftProposalStepInputValidation(rt::Tensor const& draftTreeInputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask,
        rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates);

    bool acceptDecodeTokenStepInputValidation(rt::Tensor const& acceptedTokens, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates);
};

// clang-format on

} // namespace rt
} // namespace drivellm