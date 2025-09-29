/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/checkMacros.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace drivellm
{
namespace timer
{

//! Lazy timer pair for CUDA event-based timing
struct TimerPair
{
    cudaEvent_t gpuStart{nullptr};
    cudaEvent_t gpuEnd{nullptr};
    bool hasStarted{false};
    bool isInitialized{false};

    void initialize()
    {
        if (!isInitialized)
        {
            CUDA_CHECK(cudaEventCreate(&gpuStart));
            CUDA_CHECK(cudaEventCreate(&gpuEnd));
            isInitialized = true;
        }
    }

    ~TimerPair()
    {
        if (isInitialized)
        {
            if (gpuStart)
            {
                cudaEventDestroy(gpuStart);
            }
            if (gpuEnd)
            {
                cudaEventDestroy(gpuEnd);
            }
        }
    }

    TimerPair() = default;
    TimerPair(TimerPair const&) = delete;
    TimerPair& operator=(TimerPair const&) = delete;

    TimerPair(TimerPair&& other) noexcept
        : gpuStart(other.gpuStart)
        , gpuEnd(other.gpuEnd)
        , hasStarted(other.hasStarted)
        , isInitialized(other.isInitialized)
    {
        other.gpuStart = nullptr;
        other.gpuEnd = nullptr;
        other.hasStarted = false;
        other.isInitialized = false;
    }

    TimerPair& operator=(TimerPair&& other) noexcept
    {
        if (this != &other)
        {
            if (isInitialized)
            {
                if (gpuStart)
                {
                    cudaEventDestroy(gpuStart);
                }
                if (gpuEnd)
                {
                    cudaEventDestroy(gpuEnd);
                }
            }

            gpuStart = other.gpuStart;
            gpuEnd = other.gpuEnd;
            hasStarted = other.hasStarted;
            isInitialized = other.isInitialized;

            other.gpuStart = nullptr;
            other.gpuEnd = nullptr;
            other.hasStarted = false;
            other.isInitialized = false;
        }
        return *this;
    }
};

//! RAII session for automatic timing cleanup.
class TimerSession
{
public:
    TimerSession(std::function<void()> onEnd)
        : mOnEnd(std::move(onEnd))
        , mActive(true)
    {
    }

    TimerSession(std::nullptr_t)
        : mOnEnd(nullptr)
        , mActive(false)
    {
    }

    ~TimerSession()
    {
        if (mActive && mOnEnd)
        {
            mOnEnd();
        }
    }

    TimerSession(TimerSession const&) = delete;
    TimerSession& operator=(TimerSession const&) = delete;

    TimerSession(TimerSession&& other) noexcept
        : mOnEnd(std::move(other.mOnEnd))
        , mActive(other.mActive)
    {
        other.mActive = false;
    }

    TimerSession& operator=(TimerSession&& other) noexcept
    {
        if (this != &other)
        {
            if (mActive && mOnEnd)
            {
                mOnEnd();
            }
            mOnEnd = std::move(other.mOnEnd);
            mActive = other.mActive;
            other.mActive = false;
        }
        return *this;
    }

private:
    std::function<void()> mOnEnd;
    bool mActive;
};

//! Simplified stage timing data - stores raw measurements and calculates derived values on-demand
struct StageTimingData
{
    std::vector<float> gpuTimesMs;

    void addTiming(float timeMs)
    {
        gpuTimesMs.push_back(timeMs);
    }

    void reset()
    {
        gpuTimesMs.clear();
    }

    // On-demand calculations
    float getTotalGpuTimeMs() const
    {
        return std::accumulate(gpuTimesMs.begin(), gpuTimesMs.end(), 0.0f);
    }

    float getAverageTimeMs() const
    {
        return gpuTimesMs.empty() ? 0.0f : getTotalGpuTimeMs() / gpuTimesMs.size();
    }

    int64_t getTotalRuns() const
    {
        return static_cast<int64_t>(gpuTimesMs.size());
    }
};

//! Simple timer for CUDA timing with RAII and deferred calculation
class Timer
{
public:
    Timer() = default;
    ~Timer() = default;

    //! Start/stop timing
    void startTiming();
    void stopTiming();

    //! Reset all timing data
    void reset();

    //! Start timing stage with automatic cleanup
    TimerSession startStage(std::string const& stageId, cudaStream_t stream = 0);

    //! Get timing data for a stage (triggers deferred calculation if needed)
    std::optional<StageTimingData> getTimingData(std::string const& stageId) const;

    //! Get all timing data (triggers deferred calculations if needed)
    std::unordered_map<std::string, StageTimingData> const& getAllTimingData() const;

private:
    bool mTimingActive{false};
    mutable std::unordered_map<std::string, StageTimingData> mTimingData;

    // Simple timer management - one timer per stage
    mutable std::unordered_map<std::string, TimerPair> mTimers;
    mutable std::unordered_map<std::string, std::vector<float>> mTimingResults;
    mutable std::unordered_set<std::string> mPendingTimings;

    void startTimer(std::string const& stageId, cudaStream_t stream);
    void endTimer(std::string const& stageId, cudaStream_t stream);
    void recordTiming(std::string const& stageId) const;
    void onStageComplete(std::string const& stageId);
};

//! Convenience macro for RAII-based timing
#define TIME_STAGE(stageId, stream) auto _session = drivellm::gTimer.startStage(stageId, stream)

} // namespace timer

//! Global timer instance
inline timer::Timer gTimer;

} // namespace drivellm