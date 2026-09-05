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

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Backend selection for Nvfp4A16BlackwellMoePlugin (plugin attribute `backend`).
enum class Backend : int32_t
{
    kAuto = 0,    //!< token-count policy below
    kDecode = 1,  //!< force the fused CUDA-core decode kernels
    kPrefill = 2, //!< force the tcgen05 grouped GEMM
};

//! Token (MMA-N) tiles baked into the nvfp4_a16_blackwell_moe AOT group. The
//! tile doubles as the per-expert row padding granularity of the permuted
//! activation buffer.
enum class TokenTile : int32_t
{
    kTn8 = 8,
    kTn16 = 16,
    kTn32 = 32,
    kTn64 = 64,
    kTn128 = 128,
};

inline constexpr int32_t kLargestTokenTile{128};

//! Upper bound on token tiles per grouped-GEMM launch: the kernel stages
//! tile_group_idx into shared memory once per CTA (MAX_TOKEN_TILES in
//! nvfp4_a16_blackwell_moe_gemm.py). validShape rejects profiles whose padded
//! row count at the selected tile would exceed it (T > ~19k at E=128, topK=6).
inline constexpr int32_t kMaxTokenTiles{1024};

// ---------------------------------------------------------------------------
// Fixed production policy for Thor SM110, Nemotron 3.5 Lightning shape
// (E=128, topK=6, H=2688, I=1856, FP16). Sealed on 2026-09-05 from the
// committed plugin benchmark (unittests/cpp/plugins/nvfp4A16BlackwellMoePlugin/
// nvfp4A16MoePluginBenchTests.cpp, genMoeBenchData.py inputs) on Jetson AGX
// Thor (MAXN, GPU 1575 MHz, 20 SMs): CUDA-graph replay, cold L2 (256 MB flush
// before every layer, flush cost subtracted), median of 7 batches of 50
// layers, both plugins driven through the same enqueue path.
//
//   Layer time in us (Blackwell plugin / Marlin plugin), cold L2, graph replay:
//
//   tokens | uniform routing            | skewed routing (hot experts)
//      1   |  decode 196-206 / 212-214  |  decode 196-200 / 211-214
//      2   |  tn8   364 / 363           |  tn8   347 / 337
//      4   |  tn8   605 / 590           |  tn8   459 / 454
//      8   |  tn8  1015 / 986           |  tn8   773 / 759
//     16   |  tn8  1601 / 1574          |  tn8   887 / 879
//     32   |  tn16 2061 / 2002          |  tn16 1065 / 1043
//     64   |  tn16 2281 / 2236          |  tn16 1259 / 1195
//    128   |  tn32 3100 / 2951          |  tn32 1542 / 1603
//    256   |  tn32 3115 / 3096          |  tn32 2068 / 2258
//    512   |  tn64 3543 / 4081          |  tn64 2419 / 3408
//   1024   |  tn64 4102 / 6156          |  tn64 3317 / 5666
//   2048   |  tn64 5865 / 10783         |  tn64 5396 / 10545
//
//   * decode kernels win only at T=1 (cold L2, the 23-layer reality): every
//     routed row is its own CUDA-core GEMV, so from T=2 tokens sharing an
//     expert re-read its weights while the grouped GEMM streams each expert
//     once. FC2 split-K 8 is the best of {1,2,4,8} at T=1, FC1 split-K 1.
//   * grouped path from T=2. The token tile is the per-expert padding
//     granularity: small tiles remove the pad rows FC1 streams and writes
//     (tn8 vs tn32 at T=16: -8 MB DRAM, -7% FC1 time) but a hot expert with
//     more rows than the tile is re-streamed once per extra N tile (through
//     L2, yet the 32-byte TMA requests make that costly: tn8 at skewed T=64
//     was 1.25x Marlin). Thresholds are the best worst-case over both
//     routings from the tile sweep (EDGELLM_MOE_FORCE_TILE):
//       T=32:  tn8 1.01/1.05, tn16 1.03/1.02, tn32 1.07/1.05 (uniform/skewed vs Marlin)
//       T=64:  tn8 1.02/1.25, tn16 1.02/1.05, tn32 1.06/1.03
//       T=128: tn16 1.02/1.10, tn32 1.05/0.96, tn64 1.12/0.99
//       T=256: tn16 1.07/1.20, tn32 1.01/0.92, tn64 1.07/0.85
//       T=512: tn32 0.90/0.85, tn64 0.87/0.71
//   * residual: T=2..128 trails Marlin by 1-5% on uniform routing (weights
//     stream at ~230 GB/s vs Marlin's ~243 GB/s; the A/scale TMA boxes are
//     32-byte rows, 1.1 L2 sectors per request versus Marlin's 128-byte
//     cp.async lines). Tracked as the follow-up of issue #944.
// Values are policy hints, not support gates; the runner validates shapes
// independently.
// ---------------------------------------------------------------------------
inline constexpr int32_t kDecodeMaxTokens{1};
inline constexpr int32_t kDecodeFc1SplitK{1};
inline constexpr int32_t kDecodeFc2SplitK{8};

constexpr Backend resolveBackend(Backend const requested, int32_t const numTokens) noexcept
{
    if (requested != Backend::kAuto)
    {
        return requested;
    }
    return numTokens <= kDecodeMaxTokens ? Backend::kDecode : Backend::kPrefill;
}

constexpr TokenTile selectTokenTile(int32_t const numTokens) noexcept
{
    // Expected rows per expert is T * topK / E (0.75 at T=16, 12 at T=256 for
    // the Nemotron shape); the tile is the padding granularity, so small tiles
    // remove the pad rows FC1 reads/writes while weights dominate the bytes.
    if (numTokens <= 16)
    {
        return TokenTile::kTn8;
    }
    if (numTokens <= 64)
    {
        return TokenTile::kTn16;
    }
    if (numTokens <= 256)
    {
        return TokenTile::kTn32;
    }
    if (numTokens <= 2048)
    {
        return TokenTile::kTn64;
    }
    return TokenTile::kTn128;
}

//! Conservative padded-row capacity of the permuted activation buffer for a
//! given tile: every expert may waste tile-1 rows. Multiple of `tile`.
constexpr int64_t maxRowsPadded(
    int64_t const numTokens, int64_t const topK, int64_t const numExperts, int64_t const tile) noexcept
{
    int64_t const rows = numTokens * topK + numExperts * (tile - 1);
    return (rows + tile - 1) / tile * tile;
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
