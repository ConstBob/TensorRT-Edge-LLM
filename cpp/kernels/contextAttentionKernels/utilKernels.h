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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace drivellm
{
namespace kernel
{

//! \brief Host-side wrapper that launches a lightweight CUDA kernel to build several prefix-sum
//! buffers needed by context-attention.
//!
//! \param[in]  seqLenDev         Device pointer – int32_t[B].  Actual token length of each request.
//! \param[out] cuSeqLensDev      Device pointer – int32_t[B+1]. Exclusive prefix-sum of seqLenDev.
//!                                cuSeqLensDev[0] is set to 0 inside the kernel.
//! \param[in]  kvCacheStartIdxs  Device pointer – int32_t[B].  Start index of KV cache for each request.
//! \param[out] cuKvCacheLensDev  Device pointer – int32_t[B+1]. Exclusive prefix-sum of kvCacheEndIdxs.
//!                                cuKvCacheLensDev[0] is set to 0 inside the kernel.
//! \param[out] kvCacheEndIdxsDev Device pointer – int32_t[B].  Each element equals
//!                                kvCacheStartIdxs[i] + seqLenDev[i]. For invoking
//!                                launchApplyRopeWriteContinuousQAndKVCache.
//! \param[in]  runtimeSeqLen     Runtime sequence length(with padding).
//! \param[in]  B                 Batch size.
//! \param[in]  stream            CUDA stream used to launch the kernel. Should be the same stream that
//!                                later launches the FMHA-v2 kernel.

void calCuQCuKVSeqLensAndKVEndIdxs(int32_t const* seqLenDev, int32_t* cuSeqLensDev, int32_t const* kvCacheStartIdxs,
    int32_t* cuKvCacheLensDev, int32_t* kvCacheEndIdxsDev, int32_t runtimeSeqLen, int32_t B, cudaStream_t stream);

//! \brief Converts KV cache layout from XQA format to FMHA format
//!
//! Converts an input tensor in [B, 2, H, S, D] into [B, S, 2, H, D].
//!
//! \tparam T Element type (e.g. float, half, bfloat16, etc.).
//!
//! \param[in] src    Device pointer to the padded input tensor.
//! \param[out] dst   Device pointer to the destination compact tensor.
//! \param[in] B      Batch size.
//! \param[in] S      Maximum (padded) sequence length.
//! \param[in] H      Number of attention heads.
//! \param[in] D      Hidden dimension per head.
//! \param[in] stream CUDA stream to launch the kernel on, shall be the same stream to launch FMHA-v2 kernel.

template <typename T>
void cvtKVCachelayoutXQAToFMHA(T const* src, T* dst, int32_t B, int32_t S, int32_t H, int32_t D, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm
