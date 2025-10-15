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

#include "memoryMonitor.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <chrono>
#include <cuda_runtime.h>
#include <exception>
#include <thread>

using namespace trt_edgellm;

//! Get current GPU memory usage
namespace
{
size_t getCurrentGpuMemoryUsage()
{
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    return totalMem - freeMem;
}
} // namespace

void MemoryMonitor::start()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }

    if (mPeakMemory == 0)
    {
        mPeakMemory = getCurrentGpuMemoryUsage();
    }

    mActive = true;
    mTask = std::async(std::launch::async, [this]() { monitor(); });
}

void MemoryMonitor::stop()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }
}

size_t MemoryMonitor::getPeakMemory() const
{
    return mPeakMemory;
}

void MemoryMonitor::monitor()
{
    while (mActive.load())
    {
        try
        {
            size_t usedMem = getCurrentGpuMemoryUsage();
            size_t currentPeak = mPeakMemory.load();
            while (usedMem > currentPeak && !mPeakMemory.compare_exchange_weak(currentPeak, usedMem))
            {
                // Keep trying until we successfully update or find a larger value
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Error monitoring GPU memory: %s", e.what());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}
