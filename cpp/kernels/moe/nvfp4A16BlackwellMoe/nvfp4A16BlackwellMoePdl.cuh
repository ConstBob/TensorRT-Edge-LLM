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
#pragma once

//! Programmatic Dependent Launch (PDL) helpers for the Thor W4A16 MoE kernels.
//!
//! Every kernel of the plugin calls pdlWait() before its first read of data a
//! previous kernel (or the preceding TensorRT layer) produced, and pdlTrigger()
//! once its own outputs are written, so the next kernel in the stream can be
//! launched with cudaLaunchAttributeProgrammaticStreamSerialization and start
//! its prologue while this one drains.  Both instructions are no-ops when the
//! grid was launched without the attribute, so the kernels carry them
//! unconditionally and the runner gates PDL at the launch (toolchain macro,
//! EDGELLM_ENABLE_PDL, benchmark A/B).

#include "common/cudaMacros.h"

#include <cuda_runtime.h>

#include <utility>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Wait until every prerequisite grid has completed and flushed its memory.
__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

//! Allow the dependent grid to be scheduled (its own pdlWait() still orders the data).
__device__ __forceinline__ void pdlTrigger()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

//! Launch @p kernel with the programmatic-stream-serialization attribute when
//! @p enablePdl is set (and the toolchain supports it).  cudaLaunchKernelEx
//! coerces the actual arguments to the kernel's parameter types.
template <typename... KernelArgs, typename... ActualArgs>
inline cudaError_t launchKernelPdl(void (*kernel)(KernelArgs...), dim3 const grid, dim3 const block,
    size_t const smemBytes, cudaStream_t const stream, bool const enablePdl, ActualArgs&&... args)
{
    static_assert(sizeof...(KernelArgs) == sizeof...(ActualArgs), "kernel argument count mismatch");
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    config.attrs = enablePdl ? &attribute : nullptr;
    config.numAttrs = enablePdl ? 1U : 0U;
    return cudaLaunchKernelEx(&config, kernel, std::forward<ActualArgs>(args)...);
#else
    (void) enablePdl;
    kernel<<<grid, block, smemBytes, stream>>>(std::forward<ActualArgs>(args)...);
    return cudaGetLastError();
#endif
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
