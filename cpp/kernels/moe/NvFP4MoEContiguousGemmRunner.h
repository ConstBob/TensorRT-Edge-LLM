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

// C++ runner for the contiguous grouped GEMM kernel (FC1 in decomposed pipeline).
//
// Replaces the bucketed NvFP4MoEGroupedGemmRunner with a simpler runner that:
// - Takes contiguous gathered input + 3D stacked weights
// - Uses runtime lookup tables (no compile-time group_count)
// - Needs only 2 AOT variants per activation (n128, n256) vs 16 bucketed
// - Produces bit-identical output to the original grouped GEMM kernel

#pragma once

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
#include "cutedsl_nvfp4_moe_all.h"
#endif

#include "NvFP4MoEUtils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_moe
{

class NvFP4MoEContiguousGemmRunner
{
public:
    /// @param numLocalExperts  Number of local experts (L)
    /// @param topK             Routing factor
    /// @param n                Intermediate size (N)
    /// @param k                Hidden size (K)
    /// @param tileSize         Tile size (128)
    NvFP4MoEContiguousGemmRunner(int32_t numLocalExperts, int32_t topK, int32_t n, int32_t k, int32_t tileSize = 128,
        Activation activation = Activation::kIdentity);

    static bool loadKernelModules();
    static void unloadKernelModules();

    /// Run the contiguous grouped GEMM with fused alpha + activation.
    ///
    /// Unlike the bucketed runner, this takes the layout directly — no
    /// per-group metadata construction needed.  Alpha scaling and activation
    /// are applied inside the kernel epilogue in float32.
    ///
    /// @param gatheredFP4    [permutedM, K/2] float4_e2m1fn_x2 on device
    /// @param weight         [L, N, K/2] float4_e2m1fn_x2 on device (3D stacked)
    /// @param gatheredSF     atom-layout SF buffer on device (input A scales)
    /// @param weightSF       atom-layout SF buffer on device (weight B scales)
    /// @param output         [permutedM, N_out] bfloat16 on device (output)
    /// @param alpha          [L] float32 per-expert scaling on device
    /// @param layout         MoE layout (tile metadata + permutation indices)
    /// @param permutedM      Total permuted rows
    /// @param stream         CUDA stream
    void run(void const* gatheredFP4, void const* weight, void const* gatheredSF, void const* weightSF, void* output,
        void const* alpha, MoELayout const& layout, int64_t permutedM, cudaStream_t stream);

private:
    int32_t mNumLocalExperts;
    int32_t mTopK;
    int32_t mN;
    int32_t mK;
    int32_t mTileSize;
    Activation mActivation;

    static int32_t selectTactic(int64_t n, int64_t k);

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    static nvfp4_moe_fc1_identity_n128_Kernel_Module_t sIdentityN128;
    static nvfp4_moe_fc1_identity_n256_Kernel_Module_t sIdentityN256;
    static nvfp4_moe_fc1_relu2_n128_Kernel_Module_t sRelu2N128;
    static nvfp4_moe_fc1_relu2_n256_Kernel_Module_t sRelu2N256;
    static nvfp4_moe_fc1_swiglu_n128_Kernel_Module_t sSwigluN128;
    static nvfp4_moe_fc1_swiglu_n256_Kernel_Module_t sSwigluN256;
#endif
    static bool sLoaded;
    static std::mutex sMutex;
};

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
