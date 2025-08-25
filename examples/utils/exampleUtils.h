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
#include "engine/llm_engine.h"
#include "llm_param.h"
#include <vector>

std::unique_ptr<drivellm::rt::LLMEngine> getLLMEngine(int32_t batchSize, BaseParams const& baseParams,
    EagleParams const& eagleParams, LoraWeights const& loraWeights, cudaStream_t stream);