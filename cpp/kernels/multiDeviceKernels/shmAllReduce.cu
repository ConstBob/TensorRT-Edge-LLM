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

#include "shmAllReduce.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace trt_edgellm
{
namespace kernels
{

namespace
{
constexpr uint64_t kShmHostBarrierFailureBit{uint64_t{1} << 63};

cudaError_t allocatePinnedOnNumaNode(void** hostPtr, size_t size, int32_t node, char const* label) noexcept
{
    *hostPtr = nullptr;
    constexpr unsigned int kHostFlags = cudaHostAllocMapped | cudaHostAllocPortable;
#if defined(SYS_get_mempolicy) && defined(SYS_set_mempolicy)
    if (node >= 0 && node < static_cast<int32_t>(sizeof(unsigned long) * 8))
    {
        int oldMode = MPOL_DEFAULT;
        unsigned long oldMask = 0;
        long const getPolicyRc = syscall(SYS_get_mempolicy, &oldMode, &oldMask, sizeof(oldMask) * 8, nullptr, 0UL);
        if (getPolicyRc == 0)
        {
            unsigned long const nodeMask = 1UL << node;
            long const setPolicyRc = syscall(SYS_set_mempolicy, MPOL_BIND, &nodeMask, sizeof(nodeMask) * 8);
            if (setPolicyRc == 0)
            {
                cudaError_t const error = cudaHostAlloc(hostPtr, size, kHostFlags);
                long const restoreRc = syscall(
                    SYS_set_mempolicy, oldMode, oldMode == MPOL_DEFAULT ? nullptr : &oldMask, sizeof(oldMask) * 8);
                if (restoreRc != 0)
                {
                    LOG_WARNING("ShmAllReduce: failed to restore NUMA policy after allocating %s: %s", label,
                        std::strerror(errno));
                }
                if (error == cudaSuccess)
                {
                    memset(*hostPtr, 0, size);
                }
                return error;
            }
            LOG_WARNING("ShmAllReduce: set_mempolicy(%s, node=%d) failed: %s; using default placement.", label, node,
                std::strerror(errno));
        }
        else
        {
            LOG_WARNING("ShmAllReduce: get_mempolicy before allocating %s failed: %s; using default placement.", label,
                std::strerror(errno));
        }
    }
#else
    (void) node;
#endif

    cudaError_t const error = cudaHostAlloc(hostPtr, size, kHostFlags);
    if (error == cudaSuccess)
    {
        memset(*hostPtr, 0, size);
    }
    return error;
}

void freePinnedAllocations(ShmAllReduceState* state) noexcept
{
    if (state->controlHost != nullptr)
    {
        cudaFreeHost(state->controlHost);
        state->controlHost = nullptr;
    }
    for (int32_t rank = 0; rank < kShmAllReduceWorldSize; ++rank)
    {
        if (state->shmBufHost[rank] != nullptr)
        {
            cudaFreeHost(state->shmBufHost[rank]);
            state->shmBufHost[rank] = nullptr;
        }
    }
}

bool mapDevicePointer(void** devicePtr, void* hostPtr, char const* label) noexcept
{
    cudaError_t const err = cudaHostGetDevicePointer(devicePtr, hostPtr, 0);
    if (err != cudaSuccess)
    {
        LOG_ERROR("ShmAllReduce: cudaHostGetDevicePointer(%s) failed: %s", label, cudaGetErrorString(err));
        return false;
    }
    return true;
}

int checkedShmElementCount(int64_t count, char const* label) noexcept
{
    if (count <= 0)
    {
        return 0;
    }
    if (count > std::numeric_limits<int>::max())
    {
        LOG_ERROR(
            "ShmAllReduce: %s element count %lld exceeds int32 launch capacity", label, static_cast<long long>(count));
        return 0;
    }
    return static_cast<int>(count);
}

int64_t normalizeShmAllReduceElementThreshold(int64_t threshold, int64_t maxElements)
{
    return threshold > 0 ? std::min(threshold, maxElements) : maxElements;
}

int64_t normalizeShmFp8SmallPathElementThreshold(int64_t threshold)
{
    return threshold >= 0 ? threshold : kDefaultShmFp8SmallPathElementThreshold;
}

} // namespace

bool syncShmHostBarrier(ShmAllReduceState* state, int32_t rank, std::chrono::milliseconds timeout) noexcept
{
    if (rank < 0 || rank >= kShmAllReduceWorldSize || state == nullptr || state->hostBarrierState == nullptr
        || timeout <= std::chrono::milliseconds::zero())
    {
        LOG_ERROR("Cannot synchronize SHM pre-enqueue rendezvous: invalid state, rank %d, or timeout %lld ms.", rank,
            static_cast<long long>(timeout.count()));
        return false;
    }

    uint64_t volatile* barrierState = state->hostBarrierState;

    uint64_t const initialState = __atomic_load_n(barrierState, __ATOMIC_ACQUIRE);
    if ((initialState & kShmHostBarrierFailureBit) != 0)
    {
        LOG_ERROR("SHM pre-enqueue rendezvous is already in a failed state on rank %d.", rank);
        return false;
    }

    uint64_t const previousState = __atomic_fetch_add(barrierState, uint64_t{1}, __ATOMIC_ACQ_REL);
    if ((previousState & kShmHostBarrierFailureBit) != 0)
    {
        LOG_ERROR("SHM pre-enqueue rendezvous failed while rank %d was arriving.", rank);
        return false;
    }

    uint64_t const ticket = previousState + 1;
    uint64_t const round = (ticket + 1) / 2;
    if ((ticket & uint64_t{1}) == 0)
    {
        return true;
    }

    uint64_t const completionTicket = ticket + 1;
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (true)
    {
        uint64_t const observedState = __atomic_load_n(barrierState, __ATOMIC_ACQUIRE);
        if ((observedState & kShmHostBarrierFailureBit) != 0)
        {
            LOG_ERROR("SHM pre-enqueue rendezvous failed while rank %d waited for round %llu.", rank,
                static_cast<unsigned long long>(round));
            return false;
        }
        if (observedState >= completionTicket)
        {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            uint64_t expectedState = ticket;
            uint64_t const failedState = kShmHostBarrierFailureBit | ticket;
            if (__atomic_compare_exchange_n(
                    barrierState, &expectedState, failedState, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            {
                LOG_ERROR(
                    "Timed out after %lld ms waiting for SHM peer before enqueue on rank %d "
                    "(round %llu, barrier state %llu). The SHM state is now failed.",
                    static_cast<long long>(timeout.count()), rank, static_cast<unsigned long long>(round),
                    static_cast<unsigned long long>(ticket));
                return false;
            }
            if ((expectedState & kShmHostBarrierFailureBit) != 0)
            {
                LOG_ERROR("SHM pre-enqueue rendezvous failed while rank %d waited for round %llu.", rank,
                    static_cast<unsigned long long>(round));
                return false;
            }
            if (expectedState >= completionTicket)
            {
                return true;
            }
        }
#if defined(__aarch64__)
        __asm__ volatile("yield");
#elif defined(__x86_64__)
        __asm__ volatile("pause");
#endif
    }
}

// ---------------------------------------------------------------------------
// System-scope memory operations via PTX inline assembly.
// Required for cross-GPU visibility on Thor where each GPU has its own L2
// cache despite sharing physical LPDDR5 memory.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int ldSys(unsigned int volatile* addr)
{
    unsigned int val;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ unsigned int atomicAddSys(unsigned int* addr, unsigned int val)
{
    unsigned int old;
    asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    return old;
}

/// Release store: all prior reads/writes are visible system-wide before this store.
/// Replaces the heavyweight __threadfence_system() + atom.relaxed.sys pattern.
/// Single-writer barrier counters don't need atomic RMW — a plain release store suffices.
__device__ __forceinline__ void stReleaseSys(unsigned int* addr, unsigned int val)
{
    asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}

// ---------------------------------------------------------------------------
// Single-block AllReduce kernel for 2 GPUs via shared memory.
//
// Protocol (per invocation):
//   1. Wait: peer finished reading our previous data (read_flag[peer] >= write_flag[me])
//   2. Copy input → shmBuf[rank * maxElements]
//   3. Fence + increment write_flag[rank]
//   4. Wait: peer finished writing (write_flag[peer] >= write_flag[rank])
//   5. Reduce: output = shmBuf[0] + shmBuf[1]
//   6. Fence + increment read_flag[rank]
//
// Barrier counters are monotonically increasing, so the protocol is correct
// across arbitrary numbers of CUDA Graph replays.
// ---------------------------------------------------------------------------

__global__ void shmAllReduceKernel(half const* __restrict__ input, half* __restrict__ output, half* __restrict__ mySlot,
    half const* __restrict__ otherSlot, unsigned int* __restrict__ barriers, int rank, int otherRank, int numElements)
{
    // --- Phase 1: wait for peer to finish reading our previous data ---
    if (threadIdx.x == 0)
    {
        unsigned int myWrites = ldSys(&barriers[rank]);
        while (ldSys(&barriers[2 + otherRank]) < myWrites)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // --- Phase 2: copy input to our shared slot (vectorized half2) ---
    int const numHalf2 = numElements >> 1;
    half2 const* __restrict__ in2 = reinterpret_cast<half2 const*>(input);
    half2* __restrict__ slot2 = reinterpret_cast<half2*>(mySlot);

    for (int i = threadIdx.x; i < numHalf2; i += blockDim.x)
    {
        slot2[i] = in2[i];
    }
    if ((numElements & 1) && threadIdx.x == 0)
    {
        mySlot[numElements - 1] = input[numElements - 1];
    }

    // --- Phase 3: signal write done (release ensures copy visible) ---
    if (threadIdx.x == 0)
    {
        unsigned int newWriteCount = ldSys(&barriers[rank]) + 1;
        stReleaseSys(&barriers[rank], newWriteCount);
        while (ldSys(&barriers[otherRank]) < newWriteCount)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // --- Phase 4: reduce (sum both slots) ---
    half2 const* __restrict__ other2 = reinterpret_cast<half2 const*>(otherSlot);
    half2* __restrict__ out2 = reinterpret_cast<half2*>(output);

    for (int i = threadIdx.x; i < numHalf2; i += blockDim.x)
    {
        out2[i] = __hadd2(slot2[i], other2[i]);
    }
    if ((numElements & 1) && threadIdx.x == 0)
    {
        output[numElements - 1] = __hadd(mySlot[numElements - 1], otherSlot[numElements - 1]);
    }

    // --- Phase 5: signal read done ---
    if (threadIdx.x == 0)
    {
        unsigned int newReadCount = ldSys(&barriers[2 + rank]) + 1;
        stReleaseSys(&barriers[2 + rank], newReadCount);
    }
}

// ---------------------------------------------------------------------------
// Reduce-only kernel for fused GEMM+AllReduce.
// Assumes data is already in shmBuf[rank * maxElements] (written by cuBLASLt).
// Skips Phase 2 (copy) — goes directly to fence+signal+wait+reduce.
// ---------------------------------------------------------------------------

__global__ void shmReduceOnlyKernel(half* __restrict__ output, half const* __restrict__ mySlot,
    half const* __restrict__ otherSlot, unsigned int* __restrict__ barriers, int rank, int otherRank, int numElements)
{

    // --- Phase 1: wait for peer to finish reading our previous data ---
    if (threadIdx.x == 0)
    {
        unsigned int myWrites = ldSys(&barriers[rank]);
        while (ldSys(&barriers[2 + otherRank]) < myWrites)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // --- Phase 3: signal write done (release ensures GEMM writes visible) ---
    if (threadIdx.x == 0)
    {
        unsigned int newWriteCount = ldSys(&barriers[rank]) + 1;
        stReleaseSys(&barriers[rank], newWriteCount);
        while (ldSys(&barriers[otherRank]) < newWriteCount)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // --- Phase 4: reduce (sum both slots) ---

    int const numHalf2 = numElements >> 1;
    half2 const* __restrict__ my2 = reinterpret_cast<half2 const*>(mySlot);
    half2 const* __restrict__ other2 = reinterpret_cast<half2 const*>(otherSlot);
    half2* __restrict__ out2 = reinterpret_cast<half2*>(output);

    for (int i = threadIdx.x; i < numHalf2; i += blockDim.x)
    {
        out2[i] = __hadd2(my2[i], other2[i]);
    }
    if ((numElements & 1) && threadIdx.x == 0)
    {
        output[numElements - 1] = __hadd(mySlot[numElements - 1], otherSlot[numElements - 1]);
    }

    // --- Phase 5: signal read done ---
    if (threadIdx.x == 0)
    {
        unsigned int newReadCount = ldSys(&barriers[2 + rank]) + 1;
        stReleaseSys(&barriers[2 + rank], newReadCount);
    }
}

#if SUPPORTS_FP8

// ---------------------------------------------------------------------------
// FP8 reduce-only kernel for fused GEMM(FP8 output)+AllReduce.
// Reads FP8 E4M3 from shmBuf, accumulates in FP32, outputs FP16.
// shmBuf layout: each rank wrote M*N FP8 bytes starting at byte offset
//   rank * maxSlotBytes. maxSlotBytes = maxElements * sizeof(half).
// ---------------------------------------------------------------------------

__global__ void shmReduceOnlyFp8Kernel(half* __restrict__ output, uint8_t const* __restrict__ mySlotFp8,
    uint8_t const* __restrict__ otherSlotFp8, unsigned int* __restrict__ barriers, int rank, int otherRank,
    int numElements)
{

    // Phase 1: wait for peer to finish reading our previous data.
    if (threadIdx.x == 0)
    {
        unsigned int myWrites = ldSys(&barriers[rank]);
        while (ldSys(&barriers[2 + otherRank]) < myWrites)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // Phase 3: signal our GEMM write is done, wait for peer's write.
    if (threadIdx.x == 0)
    {
        unsigned int newWriteCount = ldSys(&barriers[rank]) + 1;
        stReleaseSys(&barriers[rank], newWriteCount);
        while (ldSys(&barriers[otherRank]) < newWriteCount)
        {
            __nanosleep(100);
        }
    }
    __syncthreads();

    // Phase 4: FP8 reduce
    int const numVec4 = numElements >> 2;
    uint32_t const* __restrict__ myVec = reinterpret_cast<uint32_t const*>(mySlotFp8);
    uint32_t const* __restrict__ otherVec = reinterpret_cast<uint32_t const*>(otherSlotFp8);
    half2* __restrict__ out2 = reinterpret_cast<half2*>(output);

    for (int i = threadIdx.x; i < numVec4; i += blockDim.x)
    {
        uint32_t a4 = myVec[i];
        uint32_t b4 = otherVec[i];
        __nv_fp8_e4m3 const* a = reinterpret_cast<__nv_fp8_e4m3 const*>(&a4);
        __nv_fp8_e4m3 const* b = reinterpret_cast<__nv_fp8_e4m3 const*>(&b4);
        float s0 = static_cast<float>(a[0]) + static_cast<float>(b[0]);
        float s1 = static_cast<float>(a[1]) + static_cast<float>(b[1]);
        float s2 = static_cast<float>(a[2]) + static_cast<float>(b[2]);
        float s3 = static_cast<float>(a[3]) + static_cast<float>(b[3]);
        out2[i * 2 + 0] = __halves2half2(__float2half(s0), __float2half(s1));
        out2[i * 2 + 1] = __halves2half2(__float2half(s2), __float2half(s3));
    }

    int const tail = numVec4 << 2;
    for (int i = tail + threadIdx.x; i < numElements; i += blockDim.x)
    {
        float va = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&mySlotFp8[i]));
        float vb = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&otherSlotFp8[i]));
        output[i] = __float2half(va + vb);
    }

    // Phase 5: signal read done
    if (threadIdx.x == 0)
    {
        unsigned int newReadCount = ldSys(&barriers[2 + rank]) + 1;
        stReleaseSys(&barriers[2 + rank], newReadCount);
    }
}

#endif // SUPPORTS_FP8

// Barrier-only kernel for multi-CTA SHM AllReduce. The actual reduce is done
// by a separate multi-CTA kernel launched after this one on the same stream.

__global__ void shmBarrierKernel(unsigned int* __restrict__ barriers, int rank, int otherRank)
{
    if (threadIdx.x == 0)
    {
        unsigned int myWrites = ldSys(&barriers[rank]);
        while (ldSys(&barriers[2 + otherRank]) < myWrites)
            __nanosleep(100);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        unsigned int newWriteCount = ldSys(&barriers[rank]) + 1;
        stReleaseSys(&barriers[rank], newWriteCount);
        while (ldSys(&barriers[otherRank]) < newWriteCount)
            __nanosleep(100);
    }
}

// Phase 1 only: wait for peer to finish reading previous data.
// Used before DMA copy in the split copy+reduce path.
__global__ void shmWaitReadKernel(unsigned int* __restrict__ barriers, int rank, int otherRank)
{
    if (threadIdx.x == 0)
    {
        unsigned int myWrites = ldSys(&barriers[rank]);
        while (ldSys(&barriers[2 + otherRank]) < myWrites)
            __nanosleep(100);
    }
}

// Phase 3 only: signal write done, wait for peer's write.
// Used after DMA copy in the split copy+reduce path.
__global__ void shmSignalWriteKernel(unsigned int* __restrict__ barriers, int rank, int otherRank)
{
    if (threadIdx.x == 0)
    {
        unsigned int newWriteCount = ldSys(&barriers[rank]) + 1;
        stReleaseSys(&barriers[rank], newWriteCount);
        while (ldSys(&barriers[otherRank]) < newWriteCount)
            __nanosleep(100);
    }
}

// Multi-CTA FP8 reduce kernel for large payloads. The last block to finish
// signals read completion through completionCounter.

#if SUPPORTS_FP8

__global__ void shmReduceMultiCtaFp8Kernel(half* __restrict__ output, uint8_t const* __restrict__ mySlotFp8,
    uint8_t const* __restrict__ otherSlotFp8, unsigned int* __restrict__ barriers,
    unsigned int* __restrict__ completionCounter, int rank, int numElements, int numBlocks)
{
    int const numVec4 = numElements >> 2;
    uint32_t const* __restrict__ myVec = reinterpret_cast<uint32_t const*>(mySlotFp8);
    uint32_t const* __restrict__ otherVec = reinterpret_cast<uint32_t const*>(otherSlotFp8);
    half2* __restrict__ out2 = reinterpret_cast<half2*>(output);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numVec4; i += stride)
    {
        uint32_t a4 = myVec[i];
        uint32_t b4 = otherVec[i];
        __nv_fp8_e4m3 const* a = reinterpret_cast<__nv_fp8_e4m3 const*>(&a4);
        __nv_fp8_e4m3 const* b = reinterpret_cast<__nv_fp8_e4m3 const*>(&b4);
        float s0 = static_cast<float>(a[0]) + static_cast<float>(b[0]);
        float s1 = static_cast<float>(a[1]) + static_cast<float>(b[1]);
        float s2 = static_cast<float>(a[2]) + static_cast<float>(b[2]);
        float s3 = static_cast<float>(a[3]) + static_cast<float>(b[3]);
        out2[i * 2 + 0] = __halves2half2(__float2half(s0), __float2half(s1));
        out2[i * 2 + 1] = __halves2half2(__float2half(s2), __float2half(s3));
    }

    int const tail = numVec4 << 2;
    for (int i = tail + tid; i < numElements; i += stride)
    {
        float va = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&mySlotFp8[i]));
        float vb = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&otherSlotFp8[i]));
        output[i] = __float2half(va + vb);
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int completed = atomicAdd(completionCounter, 1u);
        if (completed == static_cast<unsigned int>(numBlocks - 1))
        {
            unsigned int newReadCount = ldSys(&barriers[2 + rank]) + 1;
            stReleaseSys(&barriers[2 + rank], newReadCount);
            atomicExch(completionCounter, 0u);
        }
    }
}

#endif // SUPPORTS_FP8

// Multi-CTA FP16 reduce kernel for large payloads. Same structure as the FP8
// variant, but reads and writes FP16 directly.

__global__ void shmReduceMultiCtaFp16Kernel(half* __restrict__ output, half const* __restrict__ mySlot,
    half const* __restrict__ otherSlot, unsigned int* __restrict__ barriers,
    unsigned int* __restrict__ completionCounter, int rank, int numElements, int numBlocks)
{
    int const numHalf2 = numElements >> 1;
    half2 const* __restrict__ my2 = reinterpret_cast<half2 const*>(mySlot);
    half2 const* __restrict__ other2 = reinterpret_cast<half2 const*>(otherSlot);
    half2* __restrict__ out2 = reinterpret_cast<half2*>(output);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numHalf2; i += stride)
    {
        out2[i] = __hadd2(my2[i], other2[i]);
    }
    if ((numElements & 1) && tid == 0)
    {
        output[numElements - 1] = __hadd(mySlot[numElements - 1], otherSlot[numElements - 1]);
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int completed = atomicAdd(completionCounter, 1u);
        if (completed == static_cast<unsigned int>(numBlocks - 1))
        {
            unsigned int newReadCount = ldSys(&barriers[2 + rank]) + 1;
            stReleaseSys(&barriers[2 + rank], newReadCount);
            atomicExch(completionCounter, 0u);
        }
    }
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

ShmAllReduceState* shmAllReduceInit(
    int32_t tpSize, int64_t maxElements, int64_t allReduceElementThreshold, int64_t fp8SmallPathElementThreshold)
{
    if (tpSize != kShmAllReduceWorldSize)
    {
        LOG_ERROR("ShmAllReduce: only tpSize=%d is supported, got %d", kShmAllReduceWorldSize, tpSize);
        return nullptr;
    }
    auto* state = new ShmAllReduceState();
    state->tpSize = tpSize;
    state->maxElements = maxElements;
    state->allReduceElementThreshold = normalizeShmAllReduceElementThreshold(allReduceElementThreshold, maxElements);
    state->fp8SmallPathElementThreshold = normalizeShmFp8SmallPathElementThreshold(fp8SmallPathElementThreshold);

    // Compute max tiles for tile-pipelined reduce (128x128 tile shape).
    // For maxInputLen=2048, hidden=4096: ceil(2048/128)*ceil(4096/128) = 16*32 = 512.
    // Use a generous upper bound to cover all GEMM output shapes.
    constexpr int kTileN = 128;
    // Estimate: max possible M (sequence length) and max N (hidden_size)
    // from maxElements. N <= sqrt(maxElements) is too small; use maxElements/kTileN
    // as upper bound for tiles along N, and similar for M.
    int32_t maxTilesN = static_cast<int32_t>((maxElements + kTileN - 1) / kTileN);
    if (maxTilesN > 256)
        maxTilesN = 256;
    int32_t maxTilesM = 32; // Generous bound for 4096-token inputs.
    int32_t maxTiles = maxTilesM * maxTilesN;
    if (maxTiles < 1)
        maxTiles = 1;
    state->maxTiles = maxTiles;

    size_t const perRankBytes = static_cast<size_t>(maxElements) * sizeof(half);
    size_t const barrierBytes = 4 * sizeof(unsigned int);
    size_t const tileBarBytes = static_cast<size_t>(state->maxTiles) * 2 * sizeof(unsigned int);
    size_t const hostBarrierBytes = sizeof(uint64_t);
    size_t const hostBarrierUnalignedOffset
        = barrierBytes + tileBarBytes + tileBarBytes + 2 * sizeof(unsigned int); // completionCounter[2]
    size_t const hostBarrierPadding
        = (alignof(uint64_t) - hostBarrierUnalignedOffset % alignof(uint64_t)) % alignof(uint64_t);
    size_t const controlBytes = hostBarrierUnalignedOffset + hostBarrierPadding + hostBarrierBytes;

    cudaError_t error = allocatePinnedOnNumaNode(&state->shmBufHost[0], perRankBytes, 0, "rank-0 slot");
    if (error == cudaSuccess)
    {
        error = allocatePinnedOnNumaNode(&state->shmBufHost[1], perRankBytes, 1, "rank-1 slot");
    }
    if (error == cudaSuccess)
    {
        error = allocatePinnedOnNumaNode(&state->controlHost, controlBytes, 0, "control data");
    }
    if (error != cudaSuccess)
    {
        LOG_ERROR("ShmAllReduce: pinned allocation failed: %s", cudaGetErrorString(error));
        freePinnedAllocations(state);
        delete state;
        return nullptr;
    }

    char* cursor = static_cast<char*>(state->controlHost);
    unsigned int* barr = reinterpret_cast<unsigned int*>(cursor);
    cursor += barrierBytes;
    unsigned int* tileBars = reinterpret_cast<unsigned int*>(cursor);
    cursor += tileBarBytes;
    unsigned int* readBars = reinterpret_cast<unsigned int*>(cursor);
    cursor += tileBarBytes;
    unsigned int* compCtr = reinterpret_cast<unsigned int*>(cursor);
    cursor += 2 * sizeof(unsigned int);
    cursor += hostBarrierPadding;
    uint64_t volatile* hostBarr = reinterpret_cast<uint64_t volatile*>(cursor);
    cursor += hostBarrierBytes;

    bool const mapped = mapDevicePointer(reinterpret_cast<void**>(&state->shmBuf), state->shmBufHost[0], "shmBuf")
        && mapDevicePointer(reinterpret_cast<void**>(&state->shmBuf1), state->shmBufHost[1], "shmBuf1")
        && mapDevicePointer(reinterpret_cast<void**>(&state->barriers), barr, "barriers")
        && mapDevicePointer(reinterpret_cast<void**>(&state->tileBarriers), tileBars, "tileBarriers")
        && mapDevicePointer(reinterpret_cast<void**>(&state->readBarriers), readBars, "readBarriers")
        && mapDevicePointer(reinterpret_cast<void**>(&state->completionCounter), compCtr, "completionCounter");
    if (!mapped)
    {
        freePinnedAllocations(state);
        delete state;
        return nullptr;
    }

    state->hostBarrierState = hostBarr;

    LOG_INFO(
        "ShmAllReduce: initialized (%s): tpSize=%d, maxElements=%ld, allReduceElementThreshold=%ld, "
        "fp8SmallPathElementThreshold=%ld, bufSize=%.1f KB, maxTiles=%d",
        "NUMA-local pinned memory", tpSize, static_cast<long>(maxElements),
        static_cast<long>(state->allReduceElementThreshold), static_cast<long>(state->fp8SmallPathElementThreshold),
        perRankBytes * 2 / 1024.0, maxTiles);

    return state;
}

cudaError_t shmAllReduceExec(
    ShmAllReduceState* state, void const* input, void* output, int64_t numElements, int32_t rank, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    half* mySlot = (rank == 0) ? state->shmBuf : state->shmBuf1;
    half* otherSlot = (rank == 0) ? state->shmBuf1 : state->shmBuf;
    int const elementCount = checkedShmElementCount(numElements, "shmAllReduceExec");
    if (elementCount == 0)
    {
        return numElements == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }

    if (elementCount <= kShmSingleCtaMaxElements)
    {
        // Small payload: single-kernel copy plus reduce.
        shmAllReduceKernel<<<1, kBlockSize, 0, stream>>>(static_cast<half const*>(input), static_cast<half*>(output),
            mySlot, otherSlot, state->barriers, rank, 1 - rank, elementCount);
        return cudaGetLastError();
    }

    // Large payload (prefill): Phase1 wait → DMA copy → Phase3 signal → multi-CTA reduce.
    // Phase 1: wait for peer to finish reading previous slot data.
    shmWaitReadKernel<<<1, 1, 0, stream>>>(state->barriers, rank, 1 - rank);
    CUDA_CHECK(cudaGetLastError());
    // DMA copy: device → host-pinned SHM slot (safe — peer finished reading).
    cudaError_t error
        = cudaMemcpyAsync(mySlot, input, static_cast<size_t>(elementCount) * sizeof(half), cudaMemcpyDefault, stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    // Phase 3: signal write done + wait for peer's write.
    shmSignalWriteKernel<<<1, 1, 0, stream>>>(state->barriers, rank, 1 - rank);
    CUDA_CHECK(cudaGetLastError());

    // Phase 4+5: multi-CTA reduce.
    constexpr int kMaxBlocks = 8;
    int numHalf2 = elementCount >> 1;
    int numBlocks = (numHalf2 + kBlockSize - 1) / kBlockSize;
    if (numBlocks > kMaxBlocks)
        numBlocks = kMaxBlocks;
    if (numBlocks < 1)
        numBlocks = 1;
    shmReduceMultiCtaFp16Kernel<<<numBlocks, kBlockSize, 0, stream>>>(static_cast<half*>(output), mySlot, otherSlot,
        state->barriers, &state->completionCounter[rank], rank, elementCount, numBlocks);
    return cudaGetLastError();
}

cudaError_t shmAllReduceExecFused(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    half const* mySlot = (rank == 0) ? state->shmBuf : state->shmBuf1;
    half const* otherSlot = (rank == 0) ? state->shmBuf1 : state->shmBuf;
    int const elementCount = checkedShmElementCount(numElements, "shmAllReduceExecFused");
    if (elementCount == 0)
    {
        return numElements == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }

    shmReduceOnlyKernel<<<1, kBlockSize, 0, stream>>>(
        static_cast<half*>(output), mySlot, otherSlot, state->barriers, rank, 1 - rank, elementCount);
    return cudaGetLastError();
}

cudaError_t shmAllReduceWaitRead(ShmAllReduceState* state, int32_t rank, cudaStream_t stream)
{
    shmWaitReadKernel<<<1, 1, 0, stream>>>(state->barriers, rank, 1 - rank);
    return cudaGetLastError();
}

cudaError_t shmAllReduceExecFusedFp8(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream)
{
#if SUPPORTS_FP8
    constexpr int kBlockSize = 256;
    int const elementCount = checkedShmElementCount(numElements, "shmAllReduceExecFusedFp8");
    if (elementCount == 0)
    {
        return numElements == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }
    uint8_t const* mySlotFp8 = reinterpret_cast<uint8_t const*>((rank == 0) ? state->shmBuf : state->shmBuf1);
    uint8_t const* otherSlotFp8 = reinterpret_cast<uint8_t const*>((rank == 0) ? state->shmBuf1 : state->shmBuf);

    shmReduceOnlyFp8Kernel<<<1, kBlockSize, 0, stream>>>(
        static_cast<half*>(output), mySlotFp8, otherSlotFp8, state->barriers, rank, 1 - rank, elementCount);
    return cudaGetLastError();
#else
    (void) state;
    (void) output;
    (void) numElements;
    (void) rank;
    (void) stream;
    LOG_ERROR("ShmAllReduce: FP8 SHM AllReduce requires CUDA_VERSION >= 11080 (cuda_fp8.h unavailable).");
    return cudaErrorNotSupported;
#endif
}

cudaError_t shmAllReduceMultiCtaFp8(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream)
{
#if SUPPORTS_FP8
    constexpr int kBlockSize = 256;
    constexpr int kMaxBlocks = 8;
    int const elementCount = checkedShmElementCount(numElements, "shmAllReduceMultiCtaFp8");
    if (elementCount == 0)
    {
        return numElements == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }
    int numVec4 = (elementCount + 3) >> 2;
    int numBlocks = (numVec4 + kBlockSize - 1) / kBlockSize;
    if (numBlocks > kMaxBlocks)
        numBlocks = kMaxBlocks;
    if (numBlocks < 1)
        numBlocks = 1;

    shmBarrierKernel<<<1, kBlockSize, 0, stream>>>(state->barriers, rank, 1 - rank);
    CUDA_CHECK(cudaGetLastError());

    uint8_t const* mySlotFp8 = reinterpret_cast<uint8_t const*>((rank == 0) ? state->shmBuf : state->shmBuf1);
    uint8_t const* otherSlotFp8 = reinterpret_cast<uint8_t const*>((rank == 0) ? state->shmBuf1 : state->shmBuf);

    shmReduceMultiCtaFp8Kernel<<<numBlocks, kBlockSize, 0, stream>>>(static_cast<half*>(output), mySlotFp8,
        otherSlotFp8, state->barriers, &state->completionCounter[rank], rank, elementCount, numBlocks);
    return cudaGetLastError();
#else
    (void) state;
    (void) output;
    (void) numElements;
    (void) rank;
    (void) stream;
    LOG_ERROR("ShmAllReduce: FP8 SHM AllReduce requires CUDA_VERSION >= 11080 (cuda_fp8.h unavailable).");
    return cudaErrorNotSupported;
#endif
}

cudaError_t shmAllReduceMultiCtaFp16(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    constexpr int kMaxBlocks = 8;
    int const elementCount = checkedShmElementCount(numElements, "shmAllReduceMultiCtaFp16");
    if (elementCount == 0)
    {
        return numElements == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }
    int numHalf2 = elementCount >> 1;
    int numBlocks = (numHalf2 + kBlockSize - 1) / kBlockSize;
    if (numBlocks > kMaxBlocks)
        numBlocks = kMaxBlocks;
    if (numBlocks < 1)
        numBlocks = 1;

    half const* mySlot = (rank == 0) ? state->shmBuf : state->shmBuf1;
    half const* otherSlot = (rank == 0) ? state->shmBuf1 : state->shmBuf;

    shmBarrierKernel<<<1, kBlockSize, 0, stream>>>(state->barriers, rank, 1 - rank);
    CUDA_CHECK(cudaGetLastError());

    shmReduceMultiCtaFp16Kernel<<<numBlocks, kBlockSize, 0, stream>>>(static_cast<half*>(output), mySlot, otherSlot,
        state->barriers, &state->completionCounter[rank], rank, elementCount, numBlocks);
    return cudaGetLastError();
}

void shmAllReduceDestroy(ShmAllReduceState* state)
{
    if (state == nullptr)
        return;

    freePinnedAllocations(state);
    delete state;
}

} // namespace kernels
} // namespace trt_edgellm
