/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

//! Get all supported aspect ratios (width_ratio, height_ratio) bounded by tile counts
std::vector<std::pair<int64_t, int64_t>> getAllSupportedAspectRatios(int64_t minImageTiles, int64_t maxImageTiles);

//! Compute resized image size (height, width) based on token/tile constraints
std::tuple<int64_t, int64_t> computeBestBlockGridForResize(int64_t height, int64_t width,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t blockImageSizeH, int64_t blockImageSizeW);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
