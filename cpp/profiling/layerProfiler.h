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

#pragma once

#include "metrics.h"
#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace layerProfiler
{

//! Layer profile information - pure data structure
struct LayerProfile
{
    std::string name;
    std::string stageName;
    std::vector<float> timeMs;

    LayerProfile() = default;
};

//! Layer profiler metrics
struct LayerProfilerMetrics
{
    std::string stageName;
    std::vector<LayerProfile> layers;
    int32_t iterationCount{0};
    bool enabled{false};
};

//! Stage-specific layer profiler
class LayerProfiler : public nvinfer1::IProfiler
{
public:
    //! Get the singleton instance
    static LayerProfiler& getInstance();

    LayerProfiler(LayerProfiler const&) = delete;
    LayerProfiler& operator=(LayerProfiler const&) = delete;

    ~LayerProfiler() noexcept override = default;

    //! Enable or disable layer profiling recording
    void setEnabled(bool enabled) noexcept;
    bool isEnabled() const noexcept;

    //! Reset all profiling data
    void reset() noexcept;

    //! TensorRT IProfiler interface implementation
    void reportLayerTime(char const* layerName, float timeMs) noexcept override;

    //! Get metrics
    LayerProfilerMetrics getMetrics() const;

private:
    LayerProfiler(std::string const& stageName = metrics::StageNames::kLLM_LAYER);

    bool mEnabled{false};
    std::string mStageName;
    std::vector<LayerProfile> mLayers;
    size_t mIndex{0};
    int32_t mIterationCount{0};
};

//! Enable all layer profilers
void enableLayerProfilers();

//! Disable all layer profilers
void disableLayerProfilers();

} // namespace layerProfiler
} // namespace trt_edgellm