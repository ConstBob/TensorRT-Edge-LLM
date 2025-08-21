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

#include <cstdint>
#include <nlohmann/json.hpp>

namespace drivellm
{
namespace rt
{

enum class RopeType
{
    // Default 1-D RoPE that specified by the original paper.
    kDefault,
    // Dynamic RoPE type used by InternVL-3.
    kDynamic,
    // Long RoPE type used by Phi-4.
    kLongRope,
    // MRope type used by Qwen2-VL
    kMRope,
};

struct RopeCommonConfig
{
    RopeType type{};
    // Instantiate the RopeConfig with common default values.
    float rotaryScale{1.0F};
    float rotaryTheta{100000.0F};
    int32_t maxPositionEmbeddings{32768};
};

// Collect basic rope configuration from the model config.
// Input:
//     config [JSON]: The model config file supplied with the model.
// Output:
//     The parsed rope configuration. Default values are used if certain fields are not
//     specified in the model config.
RopeCommonConfig collectBaseRopeConfig(nlohmann::json const& config);

// Initialize the rope cos/sin cache tensor for persistent type of RoPE (default, longrope)
// Input:
//     cosSinCache [GPU]: The tensor to store the rope cos/sin cache.
//     config [RopeCommonConfig]: The basic rope configuration.
//     modelConfig [JSON]: Model config json that can supply additional information for the rope initialization.
//     stream [CUDA stream]: The stream to execute the initialization.
// Returns:
//     True if the initialization is successful, false otherwise.
bool initializeRopeCosSinCache(
    rt::Tensor& cosSinCache, RopeCommonConfig const& config, nlohmann::json const& modelConfig, cudaStream_t stream);

} // namespace rt
} // namespace drivellm