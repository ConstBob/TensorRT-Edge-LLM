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

#include <chrono>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernels
{

constexpr int64_t kShmSingleCtaMaxElements{8192};
inline constexpr int32_t kShmAllReduceWorldSize{2};
constexpr int64_t kDefaultShmFp8SmallPathElementThreshold{8192};

/// Opaque state for SHM-based AllReduce on Thor unified memory.
/// Bypasses NCCL by using NUMA-local mapped pinned buffers and device-side barriers.
/// The monotonic barrier and completion counters remain correct across CUDA
/// Graph replays when both ranks launch matching graph instances in lockstep.
struct ShmAllReduceState
{
    half* shmBuf;                          //!< Rank 0's SHM slot [maxElements], NUMA node 0
    half* shmBuf1;                         //!< Rank 1's SHM slot [maxElements], NUMA node 1
    unsigned int* barriers;                //!< [4] pinned — write[0..1], read[0..1] counters
    unsigned int* tileBarriers;            //!< [maxTiles * 2] pinned — per-tile write-done counters
    unsigned int* readBarriers;            //!< [maxTiles * 2] pinned — per-tile read-done counters
    unsigned int* completionCounter;       //!< [2] pinned — multi-CTA reduce completion (one per rank)
    int64_t maxElements;                   //!< Max half elements per AllReduce call
    int64_t allReduceElementThreshold;     //!< Max FP16 elements routed to generic SHM AllReduce
    int64_t fp8SmallPathElementThreshold;  //!< Attention-output FP8 SHM threshold; 0 disables FP8 SHM
    int32_t tpSize;                        //!< Number of ranks (must be 2)
    int32_t maxTiles;                      //!< Max output tiles for tile-pipelined reduce
    void* shmBufHost[2]{nullptr, nullptr}; //!< Host pointers for the rank-local pinned slots
    void* controlHost{nullptr};            //!< Host pointer for the pinned control allocation
    // Host-side barrier shared by the two rank worker threads.
    // The high bit is a terminal failure marker; lower bits are arrival tickets.
    uint64_t volatile* hostBarrierState{nullptr}; //!< Shared, aligned atomic ticket/failure state
};

/// Bound the host-only pre-enqueue SHM rendezvous.
///
/// This runs during normal host enqueue and graph capture, but not graph replay;
/// replay synchronization is provided by the monotonic device barriers. A timeout
/// marks the shared SHM state failed so both ranks reject subsequent SHM work.
/// @param state Initialized SHM state
/// @param rank This GPU's rank (0 or 1)
/// @param timeout Maximum time to wait for the peer rank
bool syncShmHostBarrier(ShmAllReduceState* state, int32_t rank,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{30000}) noexcept;

/// Allocate shared buffers for 2-GPU AllReduce.
/// @param tpSize Must be 2
/// @param maxElements Largest payload in FP16 elements (e.g., 256K for 32B model prefill)
/// @param allReduceElementThreshold Largest generic FP16 payload routed to SHM; <=0 means maxElements
/// @param fp8SmallPathElementThreshold Largest attention-output payload using FP8 SHM; 0 disables FP8 SHM
/// @return Heap-allocated state (caller owns). nullptr on failure.
ShmAllReduceState* shmAllReduceInit(
    int32_t tpSize, int64_t maxElements, int64_t allReduceElementThreshold, int64_t fp8SmallPathElementThreshold);

/// Launch the SHM AllReduce kernel on the given stream.
/// Payloads up to kShmSingleCtaMaxElements use the graph-safe single-CTA path.
/// Larger payloads use a multi-CTA path whose graph replay requires rank lockstep.
/// @param state Initialized SHM state
/// @param input Device pointer to this rank's input (FP16)
/// @param output Device pointer for result (may equal input for in-place)
/// @param numElements Number of FP16 elements. Must be <= state->maxElements.
/// @param rank This GPU's rank (0 or 1)
/// @param stream CUDA stream
/// @return cudaSuccess, or the first CUDA launch/submission error
cudaError_t shmAllReduceExec(
    ShmAllReduceState* state, void const* input, void* output, int64_t numElements, int32_t rank, cudaStream_t stream);

/// Launch the fused SHM AllReduce (reduce-only, no copy phase).
/// The caller has already written data to shmBuf[rank * maxElements] (e.g.,
/// via cuBLASLt GEMM with output targeting the SHM buffer). This function
/// only does: fence -> signal -> wait -> reduce -> signal_read_done.
/// @param state Initialized SHM state
/// @param output Device pointer for reduced result
/// @param numElements Number of FP16 elements written to shmBuf[rank]
/// @param rank This GPU's rank (0 or 1)
/// @param stream CUDA stream
/// @return cudaSuccess, or the CUDA kernel launch error
cudaError_t shmAllReduceExecFused(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream);

/// Wait until the peer has finished reading this rank's previous SHM slot contents.
/// Call before an external writer, such as a GEMM kernel, overwrites shmBuf[rank].
/// @return cudaSuccess, or the CUDA kernel launch error
cudaError_t shmAllReduceWaitRead(ShmAllReduceState* state, int32_t rank, cudaStream_t stream);

/// Launch the fused SHM AllReduce for FP8 data (reduce-only, no copy phase).
/// The caller has already written FP8 E4M3 data to shmBuf[rank * maxElements * sizeof(half)]
/// (reinterpreted as bytes). This function reads FP8, accumulates in FP32, and
/// outputs FP16 to the output buffer.
/// @param state Initialized SHM state (shmBuf is reinterpreted as FP8 storage)
/// @param output Device pointer for FP16 reduced result
/// @param numElements Number of FP8 elements written per rank to shmBuf
/// @param rank This GPU's rank (0 or 1)
/// @param stream CUDA stream
/// @return cudaSuccess, or the CUDA kernel launch error
cudaError_t shmAllReduceExecFusedFp8(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream);

/// Multi-CTA FP8 AllReduce for large payloads.
/// Two-kernel approach: barrier kernel (1 CTA) + grid-stride reduce (N CTAs).
/// Graph replay requires both ranks to launch matching graph instances in lockstep.
/// @return cudaSuccess, or the first CUDA kernel launch error
cudaError_t shmAllReduceMultiCtaFp8(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream);

/// Multi-CTA FP16 AllReduce for large payloads.
/// Same two-kernel approach as FP8 but operates on FP16 data in shmBuf.
/// Graph replay requires both ranks to launch matching graph instances in lockstep.
/// @return cudaSuccess, or the first CUDA kernel launch error
cudaError_t shmAllReduceMultiCtaFp16(
    ShmAllReduceState* state, void* output, int64_t numElements, int32_t rank, cudaStream_t stream);

/// Free shared buffers.
void shmAllReduceDestroy(ShmAllReduceState* state);

} // namespace kernels
} // namespace trt_edgellm
