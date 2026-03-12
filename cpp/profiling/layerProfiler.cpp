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

#include "layerProfiler.h"
#include "common/logger.h"

namespace trt_edgellm
{
namespace layerProfiler
{

LayerProfiler& LayerProfiler::getInstance()
{
    static LayerProfiler instance;
    return instance;
}

LayerProfiler::LayerProfiler(std::string const& stageName)
    : mStageName(stageName)
{
}

void LayerProfiler::setEnabled(bool enabled) noexcept
{
    mEnabled = enabled;
}

bool LayerProfiler::isEnabled() const noexcept
{
    return mEnabled;
}

void LayerProfiler::reset() noexcept
{
    mLayers.clear();
    mIndex = 0;
    mIterationCount = 0;
}

void LayerProfiler::reportLayerTime(char const* layerName, float timeMs) noexcept
{
    if (!mEnabled)
    {
        return;
    }

    if (mIndex >= mLayers.size())
    {
        bool const first = !mLayers.empty() && mLayers[0].name == layerName;
        mIterationCount += mLayers.empty() || first;
        if (first)
        {
            mIndex = 0;
            mLayers[mIndex].stageName = mStageName;
        }
        else
        {
            mLayers.emplace_back();
            mLayers.back().name = layerName;
            mLayers.back().stageName = mStageName;
            mIndex = mLayers.size() - 1;
        }
    }
    else
    {
        if (mLayers[mIndex].name != layerName)
        {
            LOG_ERROR(
                "LayerProfiler consistency error: expected layer '%s' but got '%s'. "
                "Profile data may be corrupted due to dynamic control flow.",
                mLayers[mIndex].name.c_str(), layerName);
        }
    }

    mLayers[mIndex].timeMs.push_back(timeMs);
    ++mIndex;
}

LayerProfilerMetrics LayerProfiler::getMetrics() const
{
    LayerProfilerMetrics metrics;
    metrics.stageName = mStageName;
    metrics.layers = mLayers;
    metrics.iterationCount = mIterationCount;
    metrics.enabled = mEnabled;
    return metrics;
}

void enableLayerProfilers()
{
    LayerProfiler::getInstance().setEnabled(true);
}

void disableLayerProfilers()
{
    LayerProfiler::getInstance().setEnabled(false);
}

} // namespace layerProfiler
} // namespace trt_edgellm