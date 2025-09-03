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

#include "logger.h"
#include "tensor.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

namespace drivellm
{
namespace rt
{
namespace safetensors
{

// Save a vector of tensors to a safetensors file
// Each tensor in the vector must have a unique name associated with it
bool saveSafetensors(std::filesystem::path const& filePath, std::vector<Tensor> const& tensors, cudaStream_t stream);

// Load tensors from a safetensors file
// Tensors are loaded into the provided vector, each tensor owns its memory
// Returns true on success, false on failure
bool loadSafetensors(std::filesystem::path const& filePath, std::vector<Tensor>& tensors, cudaStream_t stream);

} // namespace safetensors
} // namespace rt
} // namespace drivellm
