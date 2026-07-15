/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multimodal/imageUtils.h"
#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

namespace
{

//! Banker's rounding (round-half-to-even) to match Python's round() used by the HF reference.
int64_t roundByFactor(int64_t value, int64_t factor)
{
    int64_t q = value / factor;
    int64_t r = value - q * factor;
    int64_t twoR = 2 * r;
    if (twoR > factor || (twoR == factor && (q & 1)))
        ++q;
    return q * factor;
}

int64_t floorByFactor(int64_t value, int64_t factor)
{
    return std::floor(static_cast<double>(value) / factor) * factor;
}

int64_t ceilByFactor(int64_t value, int64_t factor)
{
    return std::ceil(static_cast<double>(value) / factor) * factor;
}

} // namespace

std::vector<std::pair<int64_t, int64_t>> getAllSupportedAspectRatios(int64_t minImageTiles, int64_t maxImageTiles)
{
    std::vector<std::pair<int64_t, int64_t>> aspectRatios;
    for (int64_t width = 1; width <= maxImageTiles; ++width)
    {
        for (int64_t height = 1; height <= maxImageTiles; ++height)
        {
            if (width * height <= maxImageTiles && width * height >= minImageTiles)
            {
                aspectRatios.emplace_back(width, height);
            }
        }
    }
    std::sort(aspectRatios.begin(), aspectRatios.end(),
        [](std::pair<int64_t, int64_t> const& a, std::pair<int64_t, int64_t> const& b) {
            return a.first * a.second < b.first * b.second;
        });
    return aspectRatios;
}

/*!
 * @brief Choose target resize (H, W) for InternVL/Phi-4-multimodal style vision frontends.
 *
 * Given an input image (height, width) and the allowed token range, this function:
 * 1) Converts token bounds to tile bounds (each tile produces 256 tokens; we also add a thumbnail, so subtract 1).
 * 2) Enumerates candidate aspect-ratio grids (tw, th) within [minTiles, maxTiles].
 * 3) Picks the grid whose aspect ratio tw/th is closest to the original width/height.
 * 4) Tie-breaker: if two grids are equally close by ratio, prefer the one whose pixel capacity
 *    (tw*th * blockImageSizeH * blockImageSizeW) is “sufficiently large” for the current image
 *    (area > 0.5 * blockPixelArea * tw*th), reducing excessive scaling/letterboxing.
 *
 * Return value is the resized (height, width): (th * blockImageSizeH, tw * blockImageSizeW).
 * Note: This version uses floating-point for readability; near-ties are rare in practice.
 */
std::tuple<int64_t, int64_t> computeBestBlockGridForResize(int64_t height, int64_t width,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t blockImageSizeH, int64_t blockImageSizeW)
{
    // -1 to reserve space for a potential thumbnail (always added for Phi4MM; skipped for single-block images in
    // InternVL)
    int64_t const minImageTiles = std::max<int64_t>(1, minImageTokensPerImage / 256 - 1);
    int64_t const maxImageTiles = std::max<int64_t>(1, maxImageTokensPerImage / 256 - 1);
    auto const targetRatios = getAllSupportedAspectRatios(minImageTiles, maxImageTiles);
    double const aspectRatio = static_cast<double>(width) / static_cast<double>(height);
    int64_t const area = width * height;

    double bestRatioDiff = std::numeric_limits<double>::max();
    std::pair<int64_t, int64_t> bestRatio = {1, 1};
    for (auto const& ratio : targetRatios)
    {
        double const targetAspectRatio = static_cast<double>(ratio.first) / static_cast<double>(ratio.second);
        double const ratioDiff = std::abs(aspectRatio - targetAspectRatio);

        if (ratioDiff < bestRatioDiff)
        {
            bestRatioDiff = ratioDiff;
            bestRatio = ratio;
        }
        else if (ratioDiff == bestRatioDiff)
        {
            // Tie-breaker: prefer grids whose capacity better matches the current image area
            int64_t const baseBlockArea = blockImageSizeH * blockImageSizeW;
            int64_t const targetTileArea = ratio.first * ratio.second;
            int64_t const thresholdArea = (baseBlockArea / 2) * targetTileArea;
            if (area > thresholdArea)
            {
                bestRatio = ratio;
            }
        }
    }
    // return (height, width)
    return {bestRatio.second * blockImageSizeH, bestRatio.first * blockImageSizeW};
}

std::tuple<int64_t, int64_t> qwenSmartResize(int64_t height, int64_t width, int64_t patchSize, int64_t mergeSize,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    int64_t const factor = patchSize * mergeSize;
    int64_t const minPixels = minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = maxImageTokensPerImage * factor * factor;

    ELLM_CHECK(std::max(height, width) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        // Clamp to >= factor: a heavily-downscaled image must not yield a sub-factor (or zero) dimension.
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

std::tuple<int64_t, int64_t> qwenSmartResize3D(int64_t numFrames, int64_t height, int64_t width, int64_t patchSize,
    int64_t mergeSize, int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t temporalPatchSize,
    int64_t maxRatio)
{
    // Mirrors HF Qwen3-VL smart_resize: 3D temporal-aware budget (t_bar = ceil(N/TPS)*TPS) — the base 2D body plus
    // numFrames counted into the budget. NOTE (HF parity): beta is derived from the raw numFrames*height*width (not
    // tBar), and the minPixels branch intentionally has no >= factor clamp, matching the reference implementation.
    int64_t const factor = patchSize * mergeSize;
    int64_t const minPixels = minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = maxImageTokensPerImage * factor * factor;

    ELLM_CHECK(std::max(height, width) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    int64_t tBar = (numFrames > 1) ? ceilByFactor(numFrames, temporalPatchSize) : 1;

    int64_t const budget = tBar * hBar * wBar;
    if (budget > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(numFrames * height * width) / maxPixels);
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (budget < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (numFrames * height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

std::tuple<int64_t, int64_t> gemma4ResizeTarget(
    int64_t height, int64_t width, int64_t maxImageTokensPerImage, int64_t poolingKernelSize, int64_t patchSize)
{
    ELLM_CHECK(height > 0 && width > 0, "Gemma4 image height/width must be positive");
    int64_t const maxPatches = maxImageTokensPerImage * poolingKernelSize * poolingKernelSize;
    double const totalPx = static_cast<double>(height) * static_cast<double>(width);
    double const targetPx = static_cast<double>(maxPatches) * patchSize * patchSize;
    double const factor = std::sqrt(targetPx / totalPx);
    double const idealHeight = factor * static_cast<double>(height);
    double const idealWidth = factor * static_cast<double>(width);
    int64_t const sideMult = poolingKernelSize * patchSize;
    auto floorBySideMult = [sideMult](double value) {
        return static_cast<int64_t>(std::floor(value / static_cast<double>(sideMult))) * sideMult;
    };
    auto roundBySideMult = [sideMult](double value) {
        return static_cast<int64_t>(std::round(value / static_cast<double>(sideMult))) * sideMult;
    };

    int64_t targetHeight = floorBySideMult(idealHeight);
    int64_t targetWidth = floorBySideMult(idealWidth);
    ELLM_CHECK(targetHeight != 0 || targetWidth != 0, "Gemma4 target image size rounded to 0x0");

    int64_t const maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult;
    if (targetHeight == 0)
    {
        targetHeight = sideMult;
        int64_t const maxWidth = std::min(maxSideLength, floorBySideMult(targetPx / targetHeight));
        targetWidth = std::clamp(roundBySideMult(idealWidth), sideMult, maxWidth);
    }
    else if (targetWidth == 0)
    {
        targetWidth = sideMult;
        int64_t const maxHeight = std::min(maxSideLength, floorBySideMult(targetPx / targetWidth));
        targetHeight = std::clamp(roundBySideMult(idealHeight), sideMult, maxHeight);
    }

    ELLM_CHECK(
        static_cast<double>(targetHeight) * targetWidth <= targetPx, "Gemma4 target image size exceeds patch budget");
    return {targetHeight, targetWidth};
}

std::tuple<int64_t, int64_t> gemma4UnifiedResizeTarget(
    int64_t height, int64_t width, int64_t maxPatchesPerImage, int64_t modelPatchSize, int64_t positionEmbeddingSize)
{
    ELLM_CHECK(height > 0 && width > 0, "Gemma4 Unified image dimensions must be positive");
    double const targetPixels = static_cast<double>(maxPatchesPerImage) * modelPatchSize * modelPatchSize;
    double const scale = std::sqrt(targetPixels / (static_cast<double>(height) * width));
    double const idealHeight = scale * height;
    double const idealWidth = scale * width;
    int64_t const sideMultiple = modelPatchSize;
    auto floorToMultiple = [sideMultiple](double value) {
        return static_cast<int64_t>(std::floor(value / sideMultiple)) * sideMultiple;
    };
    int64_t const maxSide = std::min(maxPatchesPerImage, positionEmbeddingSize) * sideMultiple;
    int64_t targetHeight = std::min(floorToMultiple(idealHeight), maxSide);
    int64_t targetWidth = std::min(floorToMultiple(idealWidth), maxSide);
    ELLM_CHECK(targetHeight != 0 || targetWidth != 0, "Gemma4 Unified resized image rounded to 0x0");
    if (targetHeight == 0)
    {
        targetHeight = sideMultiple;
        targetWidth
            = std::min(static_cast<int64_t>(std::floor(static_cast<double>(width) / height)) * sideMultiple, maxSide);
    }
    else if (targetWidth == 0)
    {
        targetWidth = sideMultiple;
        targetHeight
            = std::min(static_cast<int64_t>(std::floor(static_cast<double>(height) / width)) * sideMultiple, maxSide);
    }
    ELLM_CHECK(static_cast<double>(targetHeight) * targetWidth <= targetPixels,
        "Gemma4 Unified resized image exceeds per-image patch budget");
    return {targetHeight, targetWidth};
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
