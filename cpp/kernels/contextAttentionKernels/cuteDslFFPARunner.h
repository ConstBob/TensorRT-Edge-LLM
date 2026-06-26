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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 12.8+ defines cudaLibrary_t / cudaLibraryUnload in <cuda_runtime.h>;
// older toolkits still require the cuLibrary* driver API.  Provide a typedef
// and a forward declaration so the generated AOT header (which references both
// the type and the unload function) compiles against CUDA 12.0-12.7.  The
// matching cudaLibrary* function symbol is supplied at link time by
// cpp/kernels/gdnKernels/cutedsl_cuda_runtime_library_shim.c (weak definition,
// linked through trt_edgellm_cutedsl_cudart_shim).
#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif

#include "cutedsl_all.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

//! Parameters for the generated FFPA CuTe DSL kernel.
struct CuteDslFFPAParams
{
    void const* q{nullptr};
    void const* k{nullptr};
    void const* v{nullptr};
    void* o{nullptr};
    int32_t batchSize{0};
    int32_t seqlenQ{0};
    int32_t seqlenK{0};
    int32_t numQHeads{0};
    int32_t numKVHeads{0};
    int32_t headDim{0};
    float softmaxScale{0.0F};
};

//! Runner for the generated BF16 FFPA d=512 causal kernel.
//! Supports MHA and GQA dynamically via numKVHeads runtime argument.
class CuteDslFFPARunner
{
public:
    static bool canImplement(int32_t headDim, int32_t smVersion);

    static bool loadKernelModule();

    static void unloadKernelModule();

    static int run(CuteDslFFPAParams const& params, cudaStream_t stream);

private:
    static ffpa_d512_causal_Kernel_Module_t sD512CausalModule;
    static bool sLoaded;
    static std::mutex sMutex;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FFPA_ENABLED)
