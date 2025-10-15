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

#include "utilKernels.h"

namespace trt_edgellm
{
namespace kernel
{

__global__ void calCuQCuKVSeqLensAndKVEndIdxsKernel(int32_t const* seqLenDev, int32_t* cuSeqLensDev,
    int32_t const* kvCacheStartIdxs, int32_t* cuKvCacheLensDev, int32_t* kvCacheEndIdxsDev, int32_t runtimeSeqLen,
    int32_t B)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cuSeqLensDev[0] = 0;
        if (kvCacheStartIdxs != nullptr)
        {
            cuKvCacheLensDev[0] = 0;
        }

        int32_t runningCuSeqLen = 0;
        int32_t runningCuKvCacheLen = 0;
        for (int32_t i = 0; i < B; ++i)
        {
            runningCuSeqLen += seqLenDev[i];
            cuSeqLensDev[i + 1] = runningCuSeqLen;

            if (kvCacheStartIdxs != nullptr)
            {
                runningCuKvCacheLen += (kvCacheStartIdxs[i] + seqLenDev[i]);
                cuKvCacheLensDev[i + 1] = runningCuKvCacheLen;
                // To keep semantic consistency with the packed QKV layout for RoPE, use runtimeSeqLen here.
                kvCacheEndIdxsDev[i] = kvCacheStartIdxs[i] + runtimeSeqLen;
            }
        }
    }
}

// ===== kernel: produce [B, S, 2, H, D] (FMHA expected padded layout) =====
template <typename T>
__global__ void cvtKVCachelayoutXQAToFMHAKernel(T const* __restrict__ src, // [B, 2, H, S, D]
    T* __restrict__ dst,                                                   // [B, S, 2, H, D]
    int32_t B, int32_t S, int32_t H, int32_t D)
{
    // Thread mapping identical to paddedLayoutToCompactKernel but without cuSeqLens.
    //   x-dim: feature dimension  D
    //   y-dim: sequence/token     S
    //   z-dim: (batch, headPair)  batch * numHpBlocks + hpBlock

    uint32_t const token = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. S-1
    uint32_t const d = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. D-1

    // Decode batch index and head-pair tile from z-dimension
    uint32_t const numHpBlocks = (2 * H + blockDim.z - 1) / blockDim.z; // number of head-pair tiles per batch
    uint32_t const batch = blockIdx.z / numHpBlocks;                    // 0 .. B-1
    uint32_t const hpTile = blockIdx.z % numHpBlocks;                   // 0 .. numHpBlocks-1
    uint32_t const headPair = hpTile * blockDim.z + threadIdx.z;        // 0 .. 2*H-1

    if (batch >= B || headPair >= 2 * H || d >= D || token >= S)
        return;

    // ---------- flat indices ----------
    // src layout: [B, 2, H, S, D] -> ((((b * 2 + kv) * H + h) * S + token) * D + d)
    uint32_t const kv = headPair / H; // 0 = K, 1 = V
    uint32_t const h = headPair % H;  // head index
    size_t srcIdx = (((((size_t) batch * 2 + kv) * H + h) * S + token) * D + d);

    // dst layout: [B, S, 2, H, D] -> ((((b * S + token) * 2 + kv) * H + h) * D + d)
    size_t dstIdx = (((((size_t) batch * S + token) * 2 + kv) * H + h) * D + d);

    dst[dstIdx] = src[srcIdx];
}

// ---------- convenience launcher ----------

void calCuQCuKVSeqLensAndKVEndIdxs(int32_t const* seqLenDev, int32_t* cuSeqLensDev, int32_t const* kvCacheStartIdxs,
    int32_t* cuKvCacheLensDev, int32_t* cuKvCacheEndIdxsDev, int32_t runtimeSeqLen, int32_t B, cudaStream_t stream)
{
    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(
        seqLenDev, cuSeqLensDev, kvCacheStartIdxs, cuKvCacheLensDev, cuKvCacheEndIdxsDev, runtimeSeqLen, B);
}

template <typename T>
void cvtKVCachelayoutXQAToFMHA(T const* src, T* dst, int32_t B, int32_t S, int32_t H, int32_t D, cudaStream_t stream)
{
    // Block config with safe thread count (≤ 1024)
    uint32_t const tx = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    uint32_t const ty = 4; // token dimension per block
    uint32_t const tz = 1; // process one head-pair per thread in z
    dim3 block(tx, ty, tz);

    // Grid config
    uint32_t const hpTilesPerBatch = (2 * H + tz - 1) / tz; // z-blocks needed per batch for head-pairs
    dim3 grid((D + tx - 1) / tx,                            // x : feature dim
        (S + ty - 1) / ty,                                  // y : token dim
        hpTilesPerBatch * B);                               // z : (batch, headPair)

    cvtKVCachelayoutXQAToFMHAKernel<<<grid, block, 0, stream>>>(src, dst, B, S, H, D);
}

} // namespace kernel
} // namespace trt_edgellm

/// @cond EXCLUDE_FROM_DOCS
template void trt_edgellm::kernel::cvtKVCachelayoutXQAToFMHA<half>(
    half const*, half*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
/// @endcond