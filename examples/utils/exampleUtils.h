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
#include "multimodal/multimodalRunner.h"
#include <vector>

// Image loading and resizing helper functions
drivellm::rt::ImageData loadImageFromFile(std::string const& path);
drivellm::rt::ImageData loadImageFromMemory(unsigned char const* data, size_t size);
drivellm::rt::ImageData resizeImage(
    drivellm::rt::ImageData const& image, int newWidth, int newHeight, bool isThumbnail = false);

std::unique_ptr<drivellm::rt::MultimodalRunner> getMultimodalRunner(
    VLMRunParams const& vlmRunParams, cudaStream_t stream);

std::unique_ptr<LLMEngine> getLLMEngine(int32_t batchSize, BaseParams const& baseParams, EagleParams const& eagleParams,
    LoraWeights const& loraWeights, cudaStream_t stream);