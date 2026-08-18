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

/*
 * Threaded GTest coverage for the SHM AllReduce infrastructure used by
 * multi-device plugin resources. The two TP ranks run as worker threads in
 * one process, matching Edge-LLM's unit-test build flow.
 */

#include <gtest/gtest.h>

#include "kernels/multiDeviceKernels/shmAllReduce.h"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace trt_edgellm::kernels;

namespace
{

constexpr int32_t kWorldSize = 2;
constexpr int64_t kMaxElements = 5120;
constexpr int32_t kVisibilityElements = 16;
constexpr float kRank0VisibilityValue = 1.0F;
constexpr float kRank1VisibilityValue = 2.0F;
constexpr float kRank0ReduceValue = 3.0F;
constexpr float kRank1ReduceValue = 5.0F;
constexpr float kReduceExpectedValue = kRank0ReduceValue + kRank1ReduceValue;
constexpr float kTolerance = 0.01F;

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

void checkCuda(cudaError_t err, int32_t rank, char const* file, int32_t line)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("[Rank ") + std::to_string(rank) + "] CUDA error at " + file + ":"
            + std::to_string(line) + ": " + cudaGetErrorString(err));
    }
}

#define CHECK_CUDA_RANK(call, rank) checkCuda((call), (rank), __FILE__, __LINE__)

struct RankResult
{
    bool passed{false};
    std::string error;
};

half* rankSlot(ShmAllReduceState* state, int32_t rank)
{
    return (rank == 0) ? state->shmBuf : state->shmBuf1;
}

half* peerSlot(ShmAllReduceState* state, int32_t rank)
{
    return (rank == 0) ? state->shmBuf1 : state->shmBuf;
}

void runVisibilityTest(ShmAllReduceState* state, int32_t rank, ThreadBarrier& barrier)
{
    half hostBuf[kVisibilityElements];
    float const writeVal = (rank == 0) ? kRank0VisibilityValue : kRank1VisibilityValue;
    for (int32_t i = 0; i < kVisibilityElements; ++i)
    {
        hostBuf[i] = __float2half(writeVal);
    }

    CHECK_CUDA_RANK(
        cudaMemcpy(rankSlot(state, rank), hostBuf, kVisibilityElements * sizeof(half), cudaMemcpyHostToDevice), rank);
    CHECK_CUDA_RANK(cudaDeviceSynchronize(), rank);

    barrier.wait();

    half peerBuf[kVisibilityElements];
    CHECK_CUDA_RANK(
        cudaMemcpy(peerBuf, peerSlot(state, rank), kVisibilityElements * sizeof(half), cudaMemcpyDeviceToHost), rank);

    float const expected = (rank == 0) ? kRank1VisibilityValue : kRank0VisibilityValue;
    for (int32_t i = 0; i < kVisibilityElements; ++i)
    {
        float const value = __half2float(peerBuf[i]);
        if (std::fabs(value - expected) > kTolerance)
        {
            throw std::runtime_error("peer visibility mismatch at element " + std::to_string(i) + ": got "
                + std::to_string(value) + ", expected " + std::to_string(expected));
        }
    }

    barrier.wait();
}

void runReduceTest(ShmAllReduceState* state, int32_t rank, ThreadBarrier& barrier)
{
    std::vector<half> hostData(kMaxElements);
    float const writeVal = (rank == 0) ? kRank0ReduceValue : kRank1ReduceValue;
    for (half& value : hostData)
    {
        value = __float2half(writeVal);
    }

    CHECK_CUDA_RANK(
        cudaMemcpy(rankSlot(state, rank), hostData.data(), kMaxElements * sizeof(half), cudaMemcpyHostToDevice), rank);
    CHECK_CUDA_RANK(cudaDeviceSynchronize(), rank);

    barrier.wait();

    half* dOutput{nullptr};
    CHECK_CUDA_RANK(cudaMalloc(&dOutput, kMaxElements * sizeof(half)), rank);
    CHECK_CUDA_RANK(cudaMemset(dOutput, 0, kMaxElements * sizeof(half)), rank);

    shmAllReduceMultiCtaFp16(state, dOutput, kMaxElements, rank, /*stream=*/0);
    CHECK_CUDA_RANK(cudaDeviceSynchronize(), rank);
    CHECK_CUDA_RANK(cudaGetLastError(), rank);

    std::vector<half> hostOut(kMaxElements);
    CHECK_CUDA_RANK(cudaMemcpy(hostOut.data(), dOutput, kMaxElements * sizeof(half), cudaMemcpyDeviceToHost), rank);
    CHECK_CUDA_RANK(cudaFree(dOutput), rank);

    for (int32_t i = 0; i < kMaxElements; ++i)
    {
        float const value = __half2float(hostOut[i]);
        if (std::fabs(value - kReduceExpectedValue) > kTolerance)
        {
            throw std::runtime_error("allreduce mismatch at element " + std::to_string(i) + ": got "
                + std::to_string(value) + ", expected " + std::to_string(kReduceExpectedValue));
        }
    }

    barrier.wait();
}

void runRank(int32_t rank, char const* shmName, ThreadBarrier& barrier, RankResult& result)
{
    ShmAllReduceState* state{nullptr};
    try
    {
        CHECK_CUDA_RANK(cudaSetDevice(rank), rank);
        state = shmAllReduceInit(kWorldSize, kMaxElements, /*allReduceElementThreshold=*/0,
            kDefaultShmFp8SmallPathElementThreshold, shmName);
        if (state == nullptr)
        {
            throw std::runtime_error("shmAllReduceInit failed");
        }

        barrier.wait();
        runVisibilityTest(state, rank, barrier);
        runReduceTest(state, rank, barrier);
        barrier.wait();

        result.passed = true;
    }
    catch (std::exception const& e)
    {
        result.error = e.what();
        barrier.abort();
    }
    catch (...)
    {
        result.error = "unknown failure";
        barrier.abort();
    }

    if (state != nullptr)
    {
        shmAllReduceDestroy(state);
    }
}

} // namespace

TEST(ShmAllReduceTest, ThreadedFp16MultiCta)
{
    int deviceCount{0};
    CHECK_CUDA_RANK(cudaGetDeviceCount(&deviceCount), 0);
    if (deviceCount < kWorldSize)
    {
        GTEST_SKIP() << "requires at least " << kWorldSize << " CUDA devices, found " << deviceCount;
    }

    char shmName[128];
    std::snprintf(shmName, sizeof(shmName), "/edgellm_shm_ar_threaded_%d", static_cast<int>(getpid()));
    ThreadBarrier barrier(kWorldSize);
    std::vector<RankResult> results(kWorldSize);
    std::vector<std::thread> workers;
    workers.reserve(kWorldSize);
    for (int32_t rank = 0; rank < kWorldSize; ++rank)
    {
        workers.emplace_back([rank, shmName, &barrier, &results]() { runRank(rank, shmName, barrier, results[rank]); });
    }

    for (std::thread& worker : workers)
    {
        worker.join();
    }

    for (int32_t rank = 0; rank < kWorldSize; ++rank)
    {
        EXPECT_TRUE(results[rank].passed) << "rank " << rank << " failed: " << results[rank].error;
    }
}
