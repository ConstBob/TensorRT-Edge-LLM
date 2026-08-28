/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "allReducePluginTestUtils.h"

#include "common/cudaUtils.h"
#include "kernels/multiDeviceKernels/shmAllReduce.h"
#include "runtime/multiDevice/backends/nccl/tensorParallelNcclResources.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <exception>
#include <gtest/gtest.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernels;
using namespace trt_edgellm::rt;
using namespace trt_edgellm::test;

namespace
{

constexpr int32_t kTpSize{2};
constexpr int64_t kMaxElements{8192};
constexpr int64_t kNumElements{4096};
constexpr int64_t kSmallThreshold{1024};
constexpr float kTolerance{1e-2F};

using RegisterShmPathFn = bool (*)(void*, int, int);
using UnregisterShmPathFn = bool (*)(void*, int, int);

RegisterShmPathFn registerShmPathSymbol() noexcept
{
    if (loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    return reinterpret_cast<RegisterShmPathFn>(dlsym(RTLD_DEFAULT, "edgellmRegisterShmAllReduceForPlugin"));
}

UnregisterShmPathFn unregisterShmPathSymbol() noexcept
{
    if (loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    return reinterpret_cast<UnregisterShmPathFn>(dlsym(RTLD_DEFAULT, "edgellmUnregisterShmAllReduceForPlugin"));
}

class ThreadBarrier
{
public:
    explicit ThreadBarrier(int32_t count)
        : mThreshold(count)
        , mCount(count)
    {
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mAborted)
        {
            throw std::runtime_error("peer rank failed");
        }
        int32_t const generation = mGeneration;
        if (--mCount == 0)
        {
            ++mGeneration;
            mCount = mThreshold;
            mCv.notify_all();
            return;
        }
        mCv.wait(lock, [this, generation]() { return mAborted || mGeneration != generation; });
        if (mAborted)
        {
            throw std::runtime_error("peer rank failed");
        }
    }

    void abort()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mAborted = true;
        mCv.notify_all();
    }

private:
    int32_t const mThreshold;
    int32_t mCount;
    int32_t mGeneration{0};
    bool mAborted{false};
    std::mutex mMutex;
    std::condition_variable mCv;
};

struct RankOutcome
{
    std::vector<float> output{};
    std::string failure{};
};

//! Drive one rank of a TP=2 AllReduce through the plugin. Every rank contributes
//! rank + 1, so a correct reduction leaves 3 in every element.
void runRank(int32_t rank, ShmAllReduceState* state, ThreadBarrier& barrier, RankOutcome& outcome) noexcept
{
    auto const dataType = nvinfer1::DataType::kHALF;
    UnregisterShmPathFn const unregisterPath = unregisterShmPathSymbol();
    bool registered = false;

    try
    {
        RegisterShmPathFn const registerPath = registerShmPathSymbol();
        if (registerPath == nullptr || unregisterPath == nullptr)
        {
            throw std::runtime_error("Cannot resolve the SHM registration symbols in the plugin library");
        }

        if (cudaSetDevice(rank) != cudaSuccess)
        {
            throw std::runtime_error("cudaSetDevice failed");
        }

        barrier.wait();

        if (!registerPath(state, rank, rank))
        {
            throw std::runtime_error("SHM path registration failed");
        }
        registered = true;

        size_t const bytes = static_cast<size_t>(kNumElements) * dataTypeSize(dataType);
        DeviceBuffer input(bytes);
        DeviceBuffer output(bytes);
        if (input.get() == nullptr || output.get() == nullptr)
        {
            throw std::runtime_error("cudaMalloc failed");
        }

        std::vector<uint8_t> const hostInput = makeHostBuffer(dataType, kNumElements, static_cast<float>(rank + 1));
        if (cudaMemcpy(input.get(), hostInput.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("Host to device copy failed");
        }

        AllReducePluginOwner plugin(kTpSize);
        if (!plugin.valid())
        {
            throw std::runtime_error("AllReducePlugin creation failed");
        }

        cudaStream_t stream{};
        if (cudaStreamCreate(&stream) != cudaSuccess)
        {
            throw std::runtime_error("cudaStreamCreate failed");
        }

        nvinfer1::PluginTensorDesc const desc = makeLinearDesc(dataType, kNumElements);
        int32_t const status = enqueueAllReduce(plugin.runtime(), desc, input.get(), output.get(), stream);
        cudaError_t const syncError = cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        if (status != 0)
        {
            throw std::runtime_error("enqueue returned " + std::to_string(status));
        }
        if (syncError != cudaSuccess)
        {
            throw std::runtime_error(std::string("Stream sync failed: ") + cudaGetErrorString(syncError));
        }

        std::vector<uint8_t> hostOutput(bytes, 0);
        if (cudaMemcpy(hostOutput.data(), output.get(), bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("Device to host copy failed");
        }
        outcome.output = readHostBuffer(dataType, hostOutput, kNumElements);

        barrier.wait();
    }
    catch (std::exception const& e)
    {
        outcome.failure = "[rank " + std::to_string(rank) + "] " + e.what();
        barrier.abort();
    }
    catch (...)
    {
        outcome.failure = "[rank " + std::to_string(rank) + "] unknown failure";
        barrier.abort();
    }

    if (registered && unregisterPath != nullptr)
    {
        unregisterPath(state, rank, rank);
    }
}

void expectReducedOutputs(std::vector<RankOutcome> const& outcomes)
{
    for (int32_t rank = 0; rank < kTpSize; ++rank)
    {
        SCOPED_TRACE("rank=" + std::to_string(rank));
        ASSERT_TRUE(outcomes[rank].failure.empty()) << outcomes[rank].failure;
        ASSERT_EQ(outcomes[rank].output.size(), static_cast<size_t>(kNumElements));
        for (size_t index = 0; index < outcomes[rank].output.size(); ++index)
        {
            ASSERT_NEAR(outcomes[rank].output[index], 3.0F, kTolerance) << "Mismatch at element " << index;
        }
    }
}

//! Allocate one SHM state on the main thread and share it with both ranks, which
//! is the ownership shape shmAllReduceTests.cu uses.
bool runBothRanks(int64_t elementThreshold, std::vector<RankOutcome>& outcomes)
{
    if (cudaSetDevice(0) != cudaSuccess)
    {
        return false;
    }
    ShmAllReduceState* state
        = shmAllReduceInit(kTpSize, kMaxElements, elementThreshold, kDefaultShmFp8SmallPathElementThreshold);
    if (state == nullptr)
    {
        return false;
    }

    ThreadBarrier barrier(kTpSize);
    std::vector<std::thread> workers;
    workers.reserve(kTpSize);
    for (int32_t rank = 0; rank < kTpSize; ++rank)
    {
        workers.emplace_back([rank, state, &barrier, &outcomes] { runRank(rank, state, barrier, outcomes[rank]); });
    }
    for (auto& worker : workers)
    {
        worker.join();
    }
    shmAllReduceDestroy(state);
    return true;
}

} // namespace

TEST(AllReducePluginShmTest, RegistrationValidatesArguments)
{
    if (detectCudaDeviceCount() < 1)
    {
        GTEST_SKIP() << "No CUDA device is available.";
    }

    RegisterShmPathFn const registerPath = registerShmPathSymbol();
    UnregisterShmPathFn const unregisterPath = unregisterShmPathSymbol();
    ASSERT_NE(registerPath, nullptr) << "Cannot resolve the SHM registration symbol in " << pluginLibraryPath();
    ASSERT_NE(unregisterPath, nullptr) << "Cannot resolve the SHM unregistration symbol in " << pluginLibraryPath();

    // Registration only reads tpSize and stores the pointer, so stack state is
    // enough to exercise validation without allocating shared memory.
    ShmAllReduceState valid{};
    valid.tpSize = kTpSize;
    ShmAllReduceState other{};
    other.tpSize = kTpSize;
    ShmAllReduceState wrongTpSize{};
    wrongTpSize.tpSize = 1;

    EXPECT_FALSE(registerPath(nullptr, 0, 0));
    EXPECT_FALSE(registerPath(&valid, 0, -1));
    EXPECT_FALSE(registerPath(&valid, 0, kTpSize));
    EXPECT_FALSE(registerPath(&valid, detectCudaDeviceCount(), 0));
    EXPECT_FALSE(registerPath(&wrongTpSize, 0, 0));

    ASSERT_TRUE(registerPath(&valid, 0, 0));
    EXPECT_FALSE(unregisterPath(&other, 0, 0));
    EXPECT_FALSE(unregisterPath(&valid, 0, 1));
    EXPECT_FALSE(unregisterPath(&valid, 1, 0));
    EXPECT_TRUE(unregisterPath(&valid, 0, 0));
    EXPECT_FALSE(unregisterPath(&valid, 0, 0));
}

TEST(AllReducePluginShmTest, SumsAcrossTwoRanks)
{
    if (detectCudaDeviceCount() < kTpSize)
    {
        GTEST_SKIP() << "The SHM AllReduce path needs " << kTpSize << " CUDA devices, found "
                     << detectCudaDeviceCount();
    }

    ASSERT_NE(loadPluginLibrary(), nullptr) << "Failed to load " << pluginLibraryPath() << ": " << dlerror();

    std::vector<RankOutcome> outcomes(kTpSize);
    ASSERT_TRUE(runBothRanks(0, outcomes)) << "shmAllReduceInit failed";
    expectReducedOutputs(outcomes);
}

TEST(AllReducePluginShmTest, FallsBackToNcclAboveThreshold)
{
    if (detectCudaDeviceCount() < kTpSize)
    {
        GTEST_SKIP() << "The AllReduce fallback test needs " << kTpSize << " CUDA devices, found "
                     << detectCudaDeviceCount();
    }

    ASSERT_NE(loadPluginLibrary(), nullptr) << "Failed to load " << pluginLibraryPath() << ": " << dlerror();

    try
    {
        NcclCollectiveBackend::load();
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NCCL runtime is unavailable: " << e.what();
    }

    auto resources = createTensorParallelNcclResources(kTpSize, {0, 1}, {0, 1}, {}, true);
    ASSERT_NE(resources, nullptr);
    ASSERT_TRUE(resources->registered());

    // kNumElements is above kSmallThreshold, so the SHM path reports itself
    // unavailable and the registered NCCL path must produce the result.
    static_assert(kNumElements > kSmallThreshold, "The payload must exceed the SHM threshold");

    std::vector<RankOutcome> outcomes(kTpSize);
    ASSERT_TRUE(runBothRanks(kSmallThreshold, outcomes)) << "shmAllReduceInit failed";
    expectReducedOutputs(outcomes);
}
