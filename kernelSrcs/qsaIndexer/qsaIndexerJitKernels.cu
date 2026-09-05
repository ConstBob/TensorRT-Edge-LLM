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

// QSA indexer device kernels, NVRTC dialect. Compiled at runtime by
// cpp/kernels/qsaIndexer/qsaIndexerJitCompiler.cpp; the host launchers live in
// cpp/kernels/qsaIndexer/qsaIndexerKernels.cpp. The exact numerics documented in
// cpp/kernels/qsaIndexer/qsaIndexerKernels.h are pinned by unit tests: keep the
// device code bit-identical when editing either side.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if !defined(QSA_INDEXER_DATA_TYPE) || !defined(QSA_INDEXER_SOURCE_ABI)
#error "QSA indexer JIT configuration is incomplete"
#endif

#if QSA_INDEXER_SOURCE_ABI != 1
#error "Unsupported QSA indexer source ABI"
#endif

// NVRTC has no <cstdint>/<cfloat>; spell out the fixed-width types and FLT_MAX.
using int32_t = int;
using int64_t = long long;
using uint32_t = unsigned int;
static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");

namespace
{

#if QSA_INDEXER_DATA_TYPE == 0
using QsaType = half;
#elif QSA_INDEXER_DATA_TYPE == 1
using QsaType = __nv_bfloat16;
#else
#error "Unsupported QSA indexer data type"
#endif

//! FLT_MAX; masked logits must compare equal to the host's -FLT_MAX bit for bit.
constexpr float kFloatMax = 3.402823466e+38F;

//! QSA indexer geometry; must match the kQSA_* constants in
//! cpp/kernels/qsaIndexer/qsaIndexerKernels.h (the host side keeps its own copy).
constexpr int32_t kQSA_INDEXER_NUM_HEADS = 4;  //!< Index-Q heads.
constexpr int32_t kQSA_INDEXER_HEAD_DIM = 128; //!< Index-Q / index-K head dimension.
constexpr int32_t kQSA_COMPRESS_RATIO = 4;     //!< Raw index-K tokens averaged per compressed block.
constexpr int32_t kQSA_BLOCK_TOPK = 512;       //!< Blocks selected per query row.
constexpr int32_t kQSA_INDEX_WIDTH = 2051;     //!< Expanded token indices per query row.
constexpr int32_t kQSA_INDEXER_ROTARY_DIM = 64;

constexpr int32_t kWarpSize = 32;
constexpr int32_t kDimsPerLane = kQSA_INDEXER_HEAD_DIM / kWarpSize;                        // 4
constexpr int32_t kIndexQkRowWidth = (kQSA_INDEXER_NUM_HEADS + 1) * kQSA_INDEXER_HEAD_DIM; // 640
constexpr int32_t kQNormedRowWidth = kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM;       // 512
constexpr int32_t kRawKColumnOffset = kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM;      // 512
constexpr int32_t kHalfRotaryDim = kQSA_INDEXER_ROTARY_DIM / 2;                            // 32
constexpr uint32_t kFullWarpMask = 0xffffffffU;

template <typename T>
__device__ __forceinline__ float toFloat(T const& v);

template <>
__device__ __forceinline__ float toFloat<half>(half const& v)
{
    return __half2float(v);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 const& v)
{
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ half fromFloat<half>(float v)
{
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v)
{
    return __float2bfloat16(v);
}

//! Butterfly warp reduction; returns the 32-lane sum replicated to every lane.
__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(kFullWarpMask, v, offset);
    }
    return v;
}

//! Neox (half-rotation) partial rope on one FP32 head staged in shared memory.
//! Dim d < 32: y = x[d]*cos[d] - x[d+32]*sin[d]; 32 <= d < 64: y = x[d]*cos[d-32] + x[d-32]*sin[d-32];
//! d >= 64: passthrough. cosSinRow points at cosSin[pos][0] (cos [0:32], sin [32:64]).
__device__ __forceinline__ float applyNeoxRopeFromSmem(float const* headSmem, float const* cosSinRow, int32_t d)
{
    if (d < kHalfRotaryDim)
    {
        return headSmem[d] * cosSinRow[d] - headSmem[d + kHalfRotaryDim] * cosSinRow[kHalfRotaryDim + d];
    }
    if (d < kQSA_INDEXER_ROTARY_DIM)
    {
        int32_t const j = d - kHalfRotaryDim;
        return headSmem[d] * cosSinRow[j] + headSmem[j] * cosSinRow[kHalfRotaryDim + j];
    }
    return headSmem[d];
}

} // namespace

//! K1a: see launchQsaIndexQPrep. grid(B*S), block(128); warp h owns q head h, 4 dims/lane.
extern "C" __global__ void qsa_indexer_q_prep(QsaType* qNormed, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wQ, float rmsEps, int32_t seqLen)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x); // b * seqLen + t
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const head = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;

    QsaType* out = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth + head * kQSA_INDEXER_HEAD_DIM;

    if (tokenIdx >= contextLengths[batchIdx])
    {
        // Padding row: exact zeros, inputs never read.
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            out[lane * kDimsPerLane + i] = fromFloat<QsaType>(0.0f);
        }
        return;
    }

    __shared__ float sQ[kQSA_INDEXER_NUM_HEADS][kQSA_INDEXER_HEAD_DIM];

    QsaType const* in = indexQk + static_cast<int64_t>(row) * kIndexQkRowWidth + head * kQSA_INDEXER_HEAD_DIM;

    // FP32 sum-of-squares over the 128 head dims (4 elements per lane + warp reduction).
    float x[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        x[i] = toFloat(in[lane * kDimsPerLane + i]);
        sumSq = fmaf(x[i], x[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);

    // Gemma norm with raw checkpoint gamma: y = x * invRms * (1 + w), all FP32, staged in smem.
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sQ[head][d] = x[i] * invRms * (1.0f + toFloat(wQ[d]));
    }
    __syncwarp();

    // Partial neox rope at position tokenIdx; single cast to QsaType at the very end.
    float const* cosSinRow = cosSin + static_cast<int64_t>(tokenIdx) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sQ[head], cosSinRow, d));
    }
}

//! K1b: see launchQsaIndexKCompress. grid(B * numGroups), block(32); one warp per (b, g).
extern "C" __global__ void qsa_indexer_k_compress(QsaType* kbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wK, float rmsEps, int32_t seqLen, int32_t numBlocks,
    int32_t numGroups, int32_t blockBegin, int32_t pastLen)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x) / numGroups;
    int32_t const blockId = blockBegin + static_cast<int32_t>(blockIdx.x) % numGroups;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);

    QsaType* out = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;

    int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
    bool const valid = (firstToken + kQSA_COMPRESS_RATIO - 1 < contextLengths[batchIdx]) && (firstToken >= pastLen);
    if (!valid)
    {
        // Incomplete/out-of-range block: exact zeros, inputs never read.
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            out[lane * kDimsPerLane + i] = fromFloat<QsaType>(0.0f);
        }
        return;
    }

    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];

    // Raw index-K rows of the 4 block tokens (columns [512, 640) of indexQk).
    QsaType const* k0 = indexQk + static_cast<int64_t>(batchIdx) * seqLen * kIndexQkRowWidth
        + static_cast<int64_t>(firstToken - pastLen) * kIndexQkRowWidth + kRawKColumnOffset;

    float m[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        // FP32 mean in FIXED sequential order, then the bit-compatibility-critical cast to QsaType.
        float acc = toFloat(k0[d]);
        acc = acc + toFloat(k0[kIndexQkRowWidth + d]);
        acc = acc + toFloat(k0[2 * kIndexQkRowWidth + d]);
        acc = acc + toFloat(k0[3 * kIndexQkRowWidth + d]);
        acc = acc * 0.25f;
        m[i] = toFloat(fromFloat<QsaType>(acc));
        sumSq = fmaf(m[i], m[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);

#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sK[d] = m[i] * invRms * (1.0f + toFloat(wK[d]));
    }
    __syncwarp();

    // Rope at the position of the block's first token; single cast to QsaType at the very end.
    float const* cosSinRow = cosSin + static_cast<int64_t>(firstToken) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sK, cosSinRow, d));
    }
}

//! K2: see launchQsaIndexScores. grid(numRows, ceilDiv(numBlocks, 64)), block(256).
//! The CTA stages the row's 4x128 q in FP32 smem; each of the 8 warps owns 8 consecutive
//! block-columns of the 64-column tile and evaluates them one at a time.
extern "C" __global__ void qsa_indexer_scores(float* logits, QsaType const* qNormed, QsaType const* kbar,
    int32_t const* contextLengths, int32_t seqLen, int32_t numBlocks, int32_t rowStart)
{
    constexpr int32_t kColumnsPerCta = 64;
    constexpr int32_t kColumnsPerWarp = 8;

    int32_t const chunkRow = static_cast<int32_t>(blockIdx.x);
    int32_t const row = rowStart + chunkRow;
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    int32_t const columnBase = static_cast<int32_t>(blockIdx.y) * kColumnsPerCta + warp * kColumnsPerWarp;

    int32_t const contextLen = contextLengths[batchIdx];
    bool const isPadding = (tokenIdx >= contextLen) || (contextLen == 0);
    int32_t const numVisible = isPadding ? 0 : (tokenIdx + 1) / kQSA_COMPRESS_RATIO;

    float* outRow = logits + static_cast<int64_t>(chunkRow) * numBlocks;

    __shared__ float sQ[kQNormedRowWidth];
    if (!isPadding)
    {
        QsaType const* q = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth;
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQNormedRowWidth; i += static_cast<int32_t>(blockDim.x))
        {
            sQ[i] = toFloat(q[i]);
        }
    }
    __syncthreads();

    for (int32_t it = 0; it < kColumnsPerWarp; ++it)
    {
        int32_t const blockId = columnBase + it;
        if (blockId >= numBlocks)
        {
            break;
        }

        float score = -kFloatMax;
        if (blockId < numVisible)
        {
            QsaType const* kb = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;
            float partial[kQSA_INDEXER_NUM_HEADS] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int32_t i = 0; i < kDimsPerLane; ++i)
            {
                int32_t const d = lane + i * kWarpSize;
                float const kf = toFloat(kb[d]);
#pragma unroll
                for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
                {
                    partial[h] = fmaf(sQ[h * kQSA_INDEXER_HEAD_DIM + d], kf, partial[h]);
                }
            }
#pragma unroll
            for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
            {
                partial[h] = warpReduceSum(partial[h]);
            }
            // Relu PER HEAD, sum over heads (ascending h), scale AFTER the sum.
            float sum = fmaxf(0.0f, partial[0]);
            sum += fmaxf(0.0f, partial[1]);
            sum += fmaxf(0.0f, partial[2]);
            sum += fmaxf(0.0f, partial[3]);
            score = sum * (1.0f / sqrtf(static_cast<float>(kQSA_INDEXER_HEAD_DIM)));
        }
        if (lane == 0)
        {
            outRow[blockId] = score;
        }
    }
}

//! ids[r * numBlocks + c] = c for the segmented sort values.
extern "C" __global__ void qsa_indexer_ids_fill(int32_t* ids, int64_t numItems, int32_t numBlocks)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numItems; i += stride)
    {
        ids[i] = static_cast<int32_t>(i % numBlocks);
    }
}

//! K4: see launchQsaIndexExpand. grid(numRows), block(256), threads stride the 2051 outputs.
extern "C" __global__ void qsa_indexer_expand(int32_t* outIdx, int32_t const* sortedIds, int32_t const* contextLengths,
    int32_t seqLen, int32_t numBlocks, int32_t rowStart)
{
    int32_t const chunkRow = static_cast<int32_t>(blockIdx.x);
    int32_t const row = rowStart + chunkRow;
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const contextLen = contextLengths[batchIdx];
    bool const isPadding = (tokenIdx >= contextLen) || (contextLen == 0);

    int32_t numVisible = 0;
    int32_t numSelected = 0;
    int32_t tailCount = 0;
    if (!isPadding)
    {
        numVisible = (tokenIdx + 1) / kQSA_COMPRESS_RATIO;
        numSelected = min(kQSA_BLOCK_TOPK, numVisible);
        tailCount = (tokenIdx + 1) % kQSA_COMPRESS_RATIO;
    }

    int32_t const* rowIds = sortedIds + static_cast<int64_t>(chunkRow) * numBlocks;
    int32_t* out = outIdx + static_cast<int64_t>(row) * kQSA_INDEX_WIDTH;

    int32_t const expandedCount = numSelected * kQSA_COMPRESS_RATIO;
    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < kQSA_INDEX_WIDTH; j += static_cast<int32_t>(blockDim.x))
    {
        int32_t value = -1;
        if (!isPadding)
        {
            if (j < expandedCount)
            {
                // Padding rows never reach this read: sortedIds may be garbage there.
                value = rowIds[j / kQSA_COMPRESS_RATIO] * kQSA_COMPRESS_RATIO + j % kQSA_COMPRESS_RATIO;
            }
            else if (j < expandedCount + tailCount)
            {
                value = numVisible * kQSA_COMPRESS_RATIO + (j - expandedCount);
            }
        }
        out[j] = value;
    }
}
