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

#include "gdnKernelUtils.cuh"

namespace trt_edgellm
{

/**
 * Single-thread prefix-sum kernel: converts context_lengths[N] to cu_seqlens[N+1].
 *
 * cu_seqlens[0] = 0
 * cu_seqlens[i+1] = cu_seqlens[i] + context_lengths[i]
 */
__global__ void gdnCalCuSeqLensKernel(int32_t const* context_lengths, // [N]
    int32_t* cu_seqlens,                                              // [N+1]  output
    int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cu_seqlens[0] = 0;
        int32_t running = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            running += context_lengths[i];
            cu_seqlens[i + 1] = running;
        }
    }
}

void launchGdnCalCuSeqLens(void const* context_lengths, void* cu_seqlens, int32_t batchSize, cudaStream_t stream)
{
    gdnCalCuSeqLensKernel<<<1, 1, 0, stream>>>(
        static_cast<int32_t const*>(context_lengths), static_cast<int32_t*>(cu_seqlens), batchSize);
}

} // namespace trt_edgellm
