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

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace drivellm
{
namespace rt
{
using Json = nlohmann::json;

struct LLMEngineRunnerConfig
{
    bool enableReuseKVCache{false};
    bool useContextDependentRope{false};
    int32_t numDecoderLayers{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t rotaryDim{};
    int32_t maxSupportedBatchSize{};
    int32_t minSupportedInputLength{};
    int32_t maxSupportedInputLength{};
    int32_t maxSequenceLength{};
    int32_t vocabSize{};
    int32_t maxSupportedLoraRank{};
};

//! The class wraps the TensorRT engine built for auto-regressive style decoder model.
//! The LLMEngineRunner define the interface for upper level runtime to execute engine actions to drive
//!     autoregressive decoding with/without speculative decoding for edge inference scenarios. Current design
//!     assume prefill and decoding operations are synchronous so a batched requests need to perform prefill and
//!     decoding at the same time (no continuous batching).
//! The LLMEngineRunner will:
//!     1. Hold TensorRT resources of the LLM engine (TRT IRuntime, CUDA Engine, Execution Contexts).
//!     2. Hold the LinearKVCache resources that support till maxSupportedBatchSize and maxSequenceLength.
//!     3. Hold the Rope CosSinCache tensor required for positional encoding.
class LLMEngineRunner
{
public:
    LLMEngineRunner(std::filesystem::path const& enginePath, std::filesystem::path const& configPath,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    ~LLMEngineRunner();

    //! API entry to get the Rope CosSinCache tensor.
    //! The API is useful when the rope cos/sin cache depends on the context which cannot be initialized
    //! in advance when creating the LLMEngineRunner instance.
    rt::Tensor& getRopeCosSinCacheTensor();

    rt::LinearKVCache& getLinearKVCache();

    LLMEngineRunnerConfig getEngineConfig() const;

    //! API entry to execute one prefill engine action for a batched request. The API will clear existing KVCache for
    //! last
    //!     batch of requests and perform prefill operations to fill the KVCache and produce the output logits.
    //! Inputs:
    //!     inputIds [GPU]: The input token_ids for the batch of new requests.
    //!     contextLengths [CPU]: The context lengths for each sequence in the batch.
    //!     multimodalEmbeddings [GPU]: Optional. The multimodal embeddings for the batch of requests.
    //!     outputLogits [GPU]: The output logits for the batch of requests..
    //!     stream: The CUDA stream to execute the prefill stp.
    //! Returns:
    //!     True if the prefill step is successful, false otherwise.
    bool executePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& contextLengths,
        rt::Tensor const& multimodalEmbeddings, rt::Tensor& outputLogits, cudaStream_t stream);

    //! API entry to execute one vanilla decoding engine action for a batched request. The API will perform decoding
    //!     operations fill the KVCache of the new generated tokens and produce the output logits. The decoding
    //!     operation shall be performed after the prefill step is completed.
    //! Inputs:
    //!     inputIds [GPU]: The input token_ids for the batch of new requests.
    //!     multimodalEmbeddings [GPU]: Optional. The multimodal embeddings for the batch of requests.
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //!     stream: The CUDA stream to execute the decoding step.
    //! Returns:
    //!     True if the decoding step is successful, false otherwise.
    bool executeVanillaDecodingStep(rt::Tensor const& inputIds, rt::Tensor const& multimodalEmbeddings,
        rt::Tensor& outputLogits, cudaStream_t stream);

    //! API entry to capture the CUDA graph for the decoding step. If CUDA graph capture is successful, later
    //!     call to executeVanillaDecodingStep() will always launch the captured CUDA graph.
    //! Inputs:
    //!     inputIds [GPU]: The input token_ids for the batch of new requests.
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //!     loraWeightsName: The name to the LoRA weights. Empty string if no LoRA weights.
    //!     stream: The CUDA stream to execute the decoding step.
    //! Returns:
    //!     True if the CUDA graph capture is successful, false otherwise.
    bool captureVanillaDecodingCudaGraph(
        rt::Tensor const& inputIds, rt::Tensor& outputLogits, std::string const& loraWeightsName, cudaStream_t stream);

    //! API entry to switch the LoRA weights of the LLM engine.
    //! Inputs:
    //!     loraWeightsName: The name of the LoRA weights.
    //!     stream: The CUDA stream to execute the switch step.
    //! Returns:
    //!     True if the LoRA weights switch is successful, false otherwise.
    bool switchLoraWeights(std::string const& loraWeightsName, cudaStream_t stream);

    //! API entry to get the active LoRA weights name.
    //! Returns:
    //!     The active LoRA weights name.
    std::string getActiveLoraWeightsName() const;

    //! API entry to get the LoRA weights.
    //! Returns:
    //!     The LoRA weights names.
    std::vector<std::string> getAvailableLoraWeights() const;

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;
    //! Holds the CUDA graph captured for the decoding step. Each CUDA graph is associated with a unique hash value
    //! which denote the input/output shapes and other execution properties like LoRA weights.
    std::unordered_map<size_t, std::pair<cudaGraph_t, cudaGraphExec_t>> mCudaGraphs;

    //! Holds the LoRA weights for the LLM engine.
    std::unordered_map<std::string, std::vector<rt::Tensor>> mLoraWeights{};
    std::string mActiveLoraWeightsName{};

    LLMEngineRunnerConfig mConfig{};

    //! The Rope CosSinCache tensor that pre-computed prior to engine execution.
    //! The design is to produce better performance and accommodate complex context dependent rope.
    rt::Tensor mPosEncCosSinCache{};

    //! The select token indices tensor is used to select indices from hidden states to pass to
    //! the LM head of LLM model. Enforce to be int64_t to align with ONNX Gather-ND specification.
    rt::Tensor mSelectTokenIndices{};

    //! The tensor has different meaning for prefill and decoding phase due to implementation of
    //! the AttentionPlugin. Used as LLM engine input.
    //! For prefill phase, the field denotes the actual content length of input_ids for each sequence.
    //! For decoding phase, this field denotes the cumulative length of the sequence length of prefill
    //!     plus generated tokens (including the length in "current" run).
    rt::Tensor mSequenceContextLengths{};

    //! The LinearKVCache tensor that carried for the LLM model execution.
    rt::LinearKVCache mKVCache{};

    //! The dummy LoRA weights tensor is used to bind the LoRA weights to the LLM engine. TensorRT does not support
    //! nullptr for binding, even when the LoRA rank is 0.
    rt::Tensor mDummyLoraWeightsTensor{};

    //! Initialize the configuration from the JSON file.
    bool initializeConfigFromJson(Json const& configJson);

    //! Validate the configuration from the engine.
    bool validateConfigFromEngine();

    //! The Function is used to bind the KVCache to the LLM engine for a new set of requests.
    bool bindKVCacheToEngine(int32_t activeBatchSize);

    bool prefillStepInputValidation(
        rt::Tensor const& inputIds, rt::Tensor const& contextLengths, rt::Tensor const& outputLogits);

    bool vanlliaDecodingStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& outputLogits);

    //! The Function is used to add a LoRA weights to the LLM engine.
    bool addLoraWeights(std::string const& loraWeightsName, std::string const& loraWeightsPath, cudaStream_t stream);

    //! The Function is used to reset the LoRA weights of the LLM engine to dummy tensors with rank 0.
    bool resetLoraWeights(cudaStream_t stream);

    //! The Function is used to get the tensor names of the LoRA weights of the LLM engine.
    //! Returns:
    //!     The tensor names of the LoRA weights.
    std::vector<std::string> getLoraWeightsTensorNames() const;

    bool isLoraWeightsSupported() const;
};

} // namespace rt
} // namespace drivellm