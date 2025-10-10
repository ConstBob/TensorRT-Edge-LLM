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
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace drivellm
{
namespace rt
{

class MultimodalRunner
{
public:
    MultimodalRunner() = default;
    MultimodalRunner(std::string const& engineDir, cudaStream_t stream);
    virtual ~MultimodalRunner() = default;

    // Static factory method to create appropriate MultimodalRunner instance
    static std::unique_ptr<MultimodalRunner> create(std::string const& multimodalEngineDir, cudaStream_t stream);

    virtual bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
        = 0;

    virtual std::string preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
        rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
        = 0;

    // Multimodal inference
    virtual bool infer(cudaStream_t stream) = 0;

    // Get multimodal output embeddings.
    virtual rt::Tensor& getOutputEmbedding();

    // Initialize random inputs for benchmark purpose
    virtual void initRandomInputs(std::vector<int32_t>& inputIds, int const batchSize, int const imageTokenLength,
        int const inputLength, cudaStream_t stream)
        = 0;

    // Parse and fill config from config file and engine
    virtual bool validateAndFillConfig(std::string const& configPath) = 0;

    // Allocate device buffer
    virtual bool allocateBuffer() = 0;

    virtual void* getConfig() = 0;

    // Get model type at runtime
    virtual std::string getModelType() const
    {
        return mModelType;
    }

    //! Get multimodal metrics for this runner
    metrics::MultimodalMetrics const& getMultimodalMetrics() const
    {
        return mMultimodalMetrics;
    }

protected:
    // Common members
    std::string mModelType;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    rt::Tensor mOutputEmbedding;
    metrics::MultimodalMetrics mMultimodalMetrics;
};

} // namespace rt
} // namespace drivellm
