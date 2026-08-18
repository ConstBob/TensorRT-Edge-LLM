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

#include "common/cudaUtils.h"
#include "kernels/multiDeviceKernels/shmAllReduce.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cuda_fp16.h>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

void runGraphReplayTest(ShmAllReduceState* state, int32_t rank, ThreadBarrier& barrier)
{
    constexpr int32_t kReplayCount{8};
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    half* dOutput{nullptr};

    CHECK_CUDA_RANK(cudaStreamCreate(&stream), rank);
    CHECK_CUDA_RANK(cudaMalloc(&dOutput, kMaxElements * sizeof(half)), rank);
    CHECK_CUDA_RANK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), rank);
    CHECK_CUDA_RANK(shmAllReduceMultiCtaFp16(state, dOutput, kMaxElements, rank, stream), rank);
    CHECK_CUDA_RANK(cudaStreamEndCapture(stream, &graph), rank);
    CHECK_CUDA_RANK(trt_edgellm::instantiateCudaGraph(&graphExec, graph), rank);

    barrier.wait();
    for (int32_t replay = 0; replay < kReplayCount; ++replay)
    {
        CHECK_CUDA_RANK(cudaGraphLaunch(graphExec, stream), rank);
        CHECK_CUDA_RANK(cudaStreamSynchronize(stream), rank);
        barrier.wait();
    }

    std::vector<half> hostOut(kMaxElements);
    CHECK_CUDA_RANK(cudaMemcpy(hostOut.data(), dOutput, kMaxElements * sizeof(half), cudaMemcpyDeviceToHost), rank);
    for (int32_t i = 0; i < kMaxElements; ++i)
    {
        float const value = __half2float(hostOut[i]);
        if (std::fabs(value - kReduceExpectedValue) > kTolerance)
        {
            throw std::runtime_error("graph replay allreduce mismatch at element " + std::to_string(i) + ": got "
                + std::to_string(value) + ", expected " + std::to_string(kReduceExpectedValue));
        }
    }

    CHECK_CUDA_RANK(cudaGraphExecDestroy(graphExec), rank);
    CHECK_CUDA_RANK(cudaGraphDestroy(graph), rank);
    CHECK_CUDA_RANK(cudaFree(dOutput), rank);
    CHECK_CUDA_RANK(cudaStreamDestroy(stream), rank);
    barrier.wait();
}

void runRank(int32_t rank, ShmAllReduceState* state, ThreadBarrier& barrier, RankResult& result)
{
    try
    {
        CHECK_CUDA_RANK(cudaSetDevice(rank), rank);
        barrier.wait();
        runVisibilityTest(state, rank, barrier);
        runReduceTest(state, rank, barrier);
        // RuntimeCoordinator captures TP ranks in lockstep. Mixed captured/eager peers are unsupported because
        // collectives must execute the same path and launch order on every rank, so this test intentionally captures
        // both ranks.
        runGraphReplayTest(state, rank, barrier);
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
}

} // namespace

TEST(ShmAllReduceTest, HostRendezvousTimeoutPublishesFailureToLatePeer)
{
    alignas(uint64_t) uint64_t barrierState{};
    ShmAllReduceState state{};
    state.hostBarrierState = &barrierState;

    constexpr uint64_t kFailureBit{uint64_t{1} << 63};
    EXPECT_FALSE(syncShmHostBarrier(&state, 0, std::chrono::milliseconds{1}));
    EXPECT_NE(barrierState & kFailureBit, uint64_t{0});
    uint64_t const failedState = barrierState;

    auto const start = std::chrono::steady_clock::now();
    // A published failure lets a late peer return immediately instead of waiting for its own timeout.
    EXPECT_FALSE(syncShmHostBarrier(&state, 1, std::chrono::seconds{1}));
    auto const elapsed
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    EXPECT_LT(elapsed.count(), 500);
    EXPECT_EQ(barrierState, failedState);
}

TEST(ShmAllReduceTest, HostRendezvousCompletesWhenBothRanksArrive)
{
    alignas(uint64_t) uint64_t barrierState{};
    ShmAllReduceState state{};
    state.hostBarrierState = &barrierState;

    bool rank0Result{false};
    bool rank1Result{false};
    std::thread rank0([&]() { rank0Result = syncShmHostBarrier(&state, 0, std::chrono::seconds{1}); });
    std::thread rank1([&]() { rank1Result = syncShmHostBarrier(&state, 1, std::chrono::seconds{1}); });
    rank0.join();
    rank1.join();

    EXPECT_TRUE(rank0Result);
    EXPECT_TRUE(rank1Result);
    EXPECT_EQ(barrierState, uint64_t{2});
}

TEST(ShmAllReduceTest, ThreadedFp16MultiCtaAndGraphReplay)
{
    int deviceCount{0};
    CHECK_CUDA_RANK(cudaGetDeviceCount(&deviceCount), 0);
    if (deviceCount < kWorldSize)
    {
        GTEST_SKIP() << "requires at least " << kWorldSize << " CUDA devices, found " << deviceCount;
    }

    CHECK_CUDA_RANK(cudaSetDevice(0), 0);
    ShmAllReduceState* state = shmAllReduceInit(
        kWorldSize, kMaxElements, /*allReduceElementThreshold=*/0, kDefaultShmFp8SmallPathElementThreshold);
    ASSERT_NE(state, nullptr);

    ThreadBarrier barrier(kWorldSize);
    std::vector<RankResult> results(kWorldSize);
    std::vector<std::thread> workers;
    workers.reserve(kWorldSize);
    for (int32_t rank = 0; rank < kWorldSize; ++rank)
    {
        workers.emplace_back([rank, state, &barrier, &results]() { runRank(rank, state, barrier, results[rank]); });
    }

    for (std::thread& worker : workers)
    {
        worker.join();
    }

    for (int32_t rank = 0; rank < kWorldSize; ++rank)
    {
        EXPECT_TRUE(results[rank].passed) << "rank " << rank << " failed: " << results[rank].error;
    }
    shmAllReduceDestroy(state);
}
