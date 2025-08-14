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
    SamplingParams(int32_t batchSize, int32_t vocabSize, float temperature = 1.0f, int32_t topK = 0, float topP = 1.0f)
        : batchSize(batchSize)
        , vocabSize(vocabSize)
        , temperature(temperature)
        , topK(topK)
        , topP(topP)
        , useTopK(topK > 0)
        , useTopP(topP < 1.0f)
    {
        if (!useTopK && !useTopP)
        {
            throw std::invalid_argument("Either topK or topP must be set");
        }
    }
};

// Forward declaration for internal workspace structure
struct SamplingWorkspace;

// ========================================================================
// MAIN SAMPLING FUNCTIONS
// ========================================================================

// Main sampling function with workspace (FP32 only)
void topKtopPSamplingFromLogits(float const* logits, int32_t* selectedIndices, SamplingParams const& params,
    void* workspace, size_t workspaceSize, cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);

// Select all top-K elements with workspace (FP32 only)
// TODO: The definition of logits will be formalized in the next release
void selectAllTopKFromLogits(float const* input, float* topKValues, int32_t* topKIndices, int32_t batchSize,
    int32_t vocabSize, int32_t topK, void* workspace, size_t workspaceSize, cudaStream_t stream,
    bool returnLogProbs = false, bool normalizeLogProbs = true, bool inputHasProbs = false);

// ========================================================================
// WORKSPACE SIZE CALCULATION
// ========================================================================

// Get workspace size for sampling (FP32 only)
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

// Get workspace size for selectAllTopK (FP32 only)
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK);

} // namespace drivellm
