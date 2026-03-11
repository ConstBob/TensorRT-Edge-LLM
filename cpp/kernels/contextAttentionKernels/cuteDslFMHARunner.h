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

#include "cuteDSLArtifact/fmha_d128.h"    // CuTe DSL generated header (head_dim=128)
#include "cuteDSLArtifact/fmha_d128_sw.h" // CuTe DSL generated header (head_dim=128, sliding window)
#include "cuteDSLArtifact/fmha_d64.h"     // CuTe DSL generated header (head_dim=64)
#include "cuteDSLArtifact/fmha_d64_sw.h"  // CuTe DSL generated header (head_dim=64, sliding window)

#include <climits>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

/**
 * @brief Runner class for CuTe DSL compiled FMHA kernels (Blackwell SM100+)
 *
 * Supports head_dim=64 and head_dim=128 via separate AOT-compiled kernels.
 * Supports sliding window attention via separate AOT-compiled kernel variants.
 * At runtime, dispatches to the correct kernel based on head dimension and
 * whether sliding window is enabled (slidingWindowSize < INT_MAX).
 * The kernel expects a combined KV tensor with layout [B, 2, H_kv, S, D].
 */
class CuteDslFMHARunner
{
public:
    CuteDslFMHARunner(int32_t b, int32_t s_q, int32_t kvCacheCapacity, int32_t h_q, int32_t h_k, int32_t d);

    ~CuteDslFMHARunner() = default;

    CuteDslFMHARunner(CuteDslFMHARunner const&) = delete;
    CuteDslFMHARunner& operator=(CuteDslFMHARunner const&) = delete;

    /**
     * @brief Load all CuTe DSL FMHA kernel modules (thread-safe, idempotent)
     * @return true if initialization succeeded
     */
    static bool loadKernelModule();

    /**
     * @brief Unload all CuTe DSL FMHA kernel modules
     */
    static void unloadKernelModule();

    /**
     * @brief Check if the CuTe DSL FMHA kernel supports the given configuration
     *
     * @param headSize Head dimension (supports 64, 128)
     * @param smVersion CUDA SM version (requires SM100+)
     * @return true if the kernel can handle this configuration
     */
    static bool canImplement(int32_t headSize, int32_t smVersion);

    /**
     * @brief Run the FMHA kernel with combined KV tensor
     *
     * @param qPtr Query tensor [B, S_q, H_q, D]
     * @param kvPtr Combined KV cache tensor [B, 2, H_kv, Cap, D]
     * @param oPtr Output tensor [B, S_q, H_q, D]
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1] (int32, device pointer)
     * @param stream CUDA stream
     * @param slidingWindowSize Sliding window size (INT_MAX = no sliding window)
     */
    void run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens, cudaStream_t stream,
        int32_t slidingWindowSize = INT_MAX);

private:
    int32_t mBatchSize;
    int32_t mSeqLenQ;
    int32_t mKVCacheCapacity;
    int32_t mNumHeadsQ;
    int32_t mNumHeadsK;
    int32_t mHeadDim;

    // Non-sliding-window kernel modules
    static fmha_d64_Kernel_Module_t sKernelModule_d64;
    static fmha_d128_Kernel_Module_t sKernelModule_d128;
    // Sliding window kernel modules
    static fmha_d64_sw_Kernel_Module_t sKernelModule_d64_sw;
    static fmha_d128_sw_Kernel_Module_t sKernelModule_d128_sw;

    static bool sModuleLoaded;
    static std::mutex sLoadMutex;
};

} // namespace trt_edgellm
