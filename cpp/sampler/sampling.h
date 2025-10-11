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

#include "common/logger.h"
#include "common/tensor.h"
#include <cstdint>
#include <stdexcept>

namespace drivellm
{

// Structure to hold sampling parameters
struct SamplingParams
{
    int32_t batchSize;
    int32_t vocabSize;
    float temperature;
    int32_t topK;
    float topP;
    bool useTopK;
    bool useTopP;

    // Constructor with default values
    SamplingParams(
        int32_t batchSize_, int32_t vocabSize_, float temperature_ = 1.0f, int32_t topK_ = 0, float topP_ = 1.0f)
        : batchSize(batchSize_)
        , vocabSize(vocabSize_)
        , temperature(temperature_)
        , topK(topK_)
        , topP(topP_)
        , useTopK(topK_ > 0)
        , useTopP(topP_ < 1.0f)
    {
        if (!useTopK && !useTopP)
        {
            throw std::invalid_argument("Either topK or topP must be set");
        }

        if (temperature < 0.0f)
        {
            throw std::invalid_argument("Temperature must be greater than 0.0f");
        }

        if (temperature < 1e-3f)
        {
            if (topK != 1 || topP != 1.0f)
            {
                LOG_WARNING(
                    "Temperature is 0.0f, but topK is not 1 or topP is not 1.0f, this may cause numerical instability. "
                    "Setting topK to 1 and topP to 1.0f");
                topK = 1;
                topP = 1.0f;
                useTopK = true;
                useTopP = false;
            }
        }
    }
};

// Forward declaration for internal workspace structure
struct SamplingWorkspace;

// Main sampling function for top-K and top-P sampling from logits.
// Inputs:
//     logits [GPU, Float]: Input logits tensor with shape [batch-size, vocab-size].
//     params: Sampling parameters including batch size, vocab size, temperature, top-K, and top-P values.
//     workspace [GPU, Int8]: Workspace buffer for intermediate computations.
//     stream: The CUDA stream to execute the kernel.
//     philoxSeed: Random seed for sampling (default: 42).
//     philoxOffset: Random offset for sampling (default: 0).
// Outputs:
//     selectedIndices [GPU, Int32]: Selected token indices with shape [batch-size].
void topKtopPSamplingFromLogits(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingParams const& params,
    rt::Tensor& workspace, cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);

// Select all top-K elements from input tensor.
// Returns topK indices and raw values from input (no transformations applied).
// Inputs:
//     input [GPU, Float]: Input tensor with shape [batch-size, vocab-size].
//     topK: Number of top elements to select.
//     workspace [GPU, Int8]: Workspace buffer for intermediate computations.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     topKValues [GPU, Float]: Optional top-K values with shape [batch-size, top-K]. Can be std::nullopt if values not
//     needed. topKIndices [GPU, Int32]: Top-K indices with shape [batch-size, top-K].
void selectAllTopK(rt::Tensor const& input, rt::OptionalOutputTensor topKValues, rt::Tensor& topKIndices, int32_t topK,
    rt::Tensor& workspace, cudaStream_t stream);

// Get workspace size for sampling (FP32 only).
// Inputs:
//     batchSize: Batch size for sampling.
//     vocabSize: Vocabulary size.
//     params: Sampling parameters.
// Returns:
//     Required workspace size in bytes.
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

// Get workspace size for selectAllTopK (FP32 only).
// Inputs:
//     batchSize: Batch size for selection.
//     vocabSize: Vocabulary size.
//     topK: Number of top elements to select.
// Returns:
//     Required workspace size in bytes.
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK);

} // namespace drivellm
