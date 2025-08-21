/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "common/tensor.h"
#include "runtime/linearKVCache.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace drivellm
{
namespace rt
{
struct LLMEngineRunnerConfig
{
    bool enableDynamicShape{false};
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
    LLMEngineRunner(
        std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream);

    ~LLMEngineRunner() = default;

    //! API entry to get the Rope CosSinCache tensor.
    //! The API is useful when the rope cos/sin cache depends on the context which cannot be initialized
    //! in advance when creating the LLMEngineRunner instance.
    rt::Tensor& getRopeCosSinCacheTensor();

    LLMEngineRunnerConfig getEngineConfig() const;

    //! API entry to execute one prefill engine action for a batched request. The API will clear existing KVCache for
    //! last
    //!     batch of requests and perform prefill operations to fill the KVCache and produce the output logits.
    //! Inputs:
    //!     inputIds [GPU]: The input token_ids for the batch of new requests.
    //!     contextLengths [CPU]: The context lengths for each sequence in the batch.
    //!     outputLogits [GPU]: The output logits for the batch of requests..
    //!     stream: The CUDA stream to execute the prefill stp.
    //! Returns:
    //!     True if the prefill step is successful, false otherwise.
    bool executePrefillStep(
        rt::Tensor const& inputIds, rt::Tensor const& contextLengths, rt::Tensor& outputLogits, cudaStream_t stream);

    //! API entry to execute one vanilla decoding engine action for a batched request. The API will perform decoding
    //!     operations fill the KVCache of the new generated tokens and produce the output logits. The decoding
    //!     operation shall be performed after the prefill step is completed.
    //! Inputs:
    //!     inputIds [GPU]: The input token_ids for the batch of new requests.
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //!     stream: The CUDA stream to execute the decoding step.
    //! Returns:
    //!     True if the decoding step is successful, false otherwise.
    bool executeVanillaDecodingStep(rt::Tensor const& inputIds, rt::Tensor& outputLogits, cudaStream_t stream);

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;

    LLMEngineRunnerConfig mConfig{};

    //! The Rope CosSinCache tensor that pre-computed prior to engine execution.
    //! The design is to produce better performance and accommodate complex context dependent rope.
    rt::Tensor mPosEncCosSinCache{};

    //! The select token indices tensor is used to select indices from hidden states to pass to
    //! the LM head of LLM model. Enforce to be int64_t to align with ONNX Gather-ND specification.
    rt::Tensor mSelectTokenIndices{};

    //! The LinearKVCache tensor that carried for the LLM model execution.
    rt::LinearKVCache mKVCache{};

    void initializeConfigFromEngine();

    //! The Function is used to bind the KVCache to the LLM engine for a new set of requests.
    bool bindKVCacheToEngine(int32_t activeBatchSize);

    bool prefillStepInputValidation(
        rt::Tensor const& inputIds, rt::Tensor const& contextLengths, rt::Tensor const& outputLogits);

    bool vanlliaDecodingStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& outputLogits);
};

} // namespace rt
} // namespace drivellm