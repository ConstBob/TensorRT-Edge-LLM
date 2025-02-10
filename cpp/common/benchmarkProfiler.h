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

#include "common.h"
#include "cudaEvent.h"
#include "cudaUtils.h"
#include "logger.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <future>
#include <limits>
#include <numeric>
#include <string>
#include <sys/sysinfo.h>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

inline size_t monitorDeviceMemory(std::atomic_bool& done)
{
    // A simple memory monitor function that monitors peak GPU memory usage
    size_t peakMem = 0;
    while (!done)
    {
        auto const [freeMem, totalMem] = getDeviceMemoryInfo();
        if (totalMem - freeMem > peakMem)
        {
            peakMem = totalMem - freeMem;
        }
        // Sleep for 50 ms to avoid spamming getDeviceMemoryInfo to reduce overhead
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return peakMem;
}

inline size_t monitorHostMemory(std::atomic_bool& done)
{
    // A simple memory monitor function that monitors peak CPU memory usage
    size_t peakMem = 0;
    size_t availableMemory, totalMemory = 0;
    while (!done)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;

        if (meminfo.is_open())
        {
            while (std::getline(meminfo, line))
            {
                if (line.find("MemAvailable:") == 0)
                {
                    availableMemory = strtoul(line.c_str() + 13, nullptr, 10);

                    if (totalMemory - availableMemory > peakMem)
                    {
                        peakMem = totalMemory - availableMemory;
                    }
                    break;
                }
                else if (totalMemory == 0 && line.find("MemTotal:") == 0)
                {
                    totalMemory = strtoul(line.c_str() + 9, nullptr, 10);
                }
            }
            meminfo.close();
        }
        else
        {
            LOG_ERROR("Unable to open /proc/meminfo");
        }
        // Sleep for 50 ms to reduce overhead
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return peakMem;
}

class BenchmarkProfiler
{
public:
    BenchmarkProfiler() = default;

    auto recordDeviceStart(std::string const& timer)
    {
        recordDeviceHelper(timer + "::start");
    }

    auto recordDeviceEnd(std::string const& timer)
    {
        recordDeviceHelper(timer + "::end");
        if (cudaResults.find(timer) == cudaResults.end())
        {
            cudaResults[timer] = {};
        }

        if (!timingStarted)
        {
            return;
        }

        auto& startEvent = cudaEventMap[timer + "::start"];
        auto& endEvent = cudaEventMap[timer + "::end"];
        startEvent.synchronize();
        endEvent.synchronize();

        float result;
        CUDA_CHECK(cudaEventElapsedTime(&result, startEvent.get(), endEvent.get()));

        cudaResults[timer].emplace_back(result);
    }

    auto recordDeviceMemStart()
    {
        // Stop the previous recording
        if (peakDeviceMem.valid())
        {
            deviceMemDone = true;
            peakDeviceMem.get();
        }
        deviceMemDone = false;
        peakDeviceMem = std::async(std::launch::async, monitorDeviceMemory, std::ref(deviceMemDone));
    }

    auto recordDeviceMemEnd()
    {
        deviceMemDone = true;
        check(peakDeviceMem.valid(), "Must call `recordDeviceMemStart` before calling `recordDeviceMemEnd`. ");
        return peakDeviceMem.get();
    }

    auto recordHostStart(std::string const& timer)
    {
        recordHostHelper(timer + "::start");
    }

    auto recordHostEnd(std::string const& timer)
    {
        recordHostHelper(timer + "::end");
        if (hostResults.find(timer) == hostResults.end())
        {
            hostResults[timer] = {};
        }
        if (!timingStarted)
        {
            return;
        }
        hostResults[timer].emplace_back(
            std::chrono::duration<float, std::milli>(hostEventMap[timer + "::end"] - hostEventMap[timer + "::start"])
                .count());
    }

    auto recordHostMemStart()
    {
        // Stop the previous recording
        if (peakHostMem.valid())
        {
            hostMemDone = true;
            peakHostMem.get();
        }
        hostMemDone = false;
        peakHostMem = std::async(std::launch::async, monitorHostMemory, std::ref(hostMemDone));
    }

    auto recordHostMemEnd()
    {
        hostMemDone = true;
        check(peakHostMem.valid(), "Must call `recordHostMemStart` before calling `recordHostMemEnd`. ");
        return peakHostMem.get();
    }

    auto getDeviceElapsedTimeMs(std::string const& timer)
    {
        return getElapsedTimeHelper(cudaResults[timer]);
    }

    auto getHostElapsedTimeMs(std::string const& timer)
    {
        return getElapsedTimeHelper(hostResults[timer]);
    }

    auto startTiming()
    {
        timingStarted = true;
    }

    auto stopTiming()
    {
        timingStarted = false;
    }

private:
    void recordDeviceHelper(std::string const& event)
    {
        if (cudaEventMap.find(event) == cudaEventMap.end())
        {
            cudaEventMap[event] = CudaEvent();
        }
        if (!timingStarted)
        {
            return;
        }
        CUDA_CHECK(cudaEventRecord(cudaEventMap[event].get()));
    }

    void recordHostHelper(std::string const& event)
    {
        hostEventMap[event] = std::chrono::steady_clock::now();
    }

    std::tuple<float, float, std::vector<float> const&> getElapsedTimeHelper(std::vector<float> const& vec)
    {
        if (vec.size() == 0)
        {
            return {0, 0, vec};
        }
        auto sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return {sum / vec.size(), sum, vec};
    }

    std::unordered_map<std::string, CudaEvent> cudaEventMap;
    std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> hostEventMap;
    std::unordered_map<std::string, std::vector<float>> cudaResults;
    std::unordered_map<std::string, std::vector<float>> hostResults;
    std::future<size_t> peakHostMem, peakDeviceMem;
    std::atomic_bool deviceMemDone{false};
    std::atomic_bool hostMemDone{false};
    bool timingStarted{false};
};