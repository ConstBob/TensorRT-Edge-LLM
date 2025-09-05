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

#include <common/logger.h>
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
