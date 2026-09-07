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
    kDecode = 1,  //!< force the CUDA-core decode kernels
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
// (E=128, topK=6, H=2688, I=1856, FP16). Sealed on 2026-09-05 from a
// Marlin-vs-Blackwell plugin benchmark (a gtest harness kept outside the
// repository; random codes, real shapes) on Jetson AGX Thor (MAXN, GPU
// 1575 MHz, 20 SMs): CUDA-graph replay, cold L2 (256 MB flush
// before every layer, flush cost subtracted), median of 7 batches of 50
// layers, both plugins driven through the same enqueue path.
//
//   Layer time in us (Blackwell plugin / Marlin plugin), cold L2, graph replay,
//   weights streamed as 2 KB-row TMA boxes over the pre-swizzled layout:
//
//   tokens | uniform routing            | skewed routing (hot experts)
//      1   |  decode  169 / 225         |  decode  161 / 211
//      2   |  tn8     347 / 361         |  tn8     335 / 336
//      4   |  tn8     572 / 589         |  tn8     437 / 450
//      8   |  tn8     950 / 995         |  tn8     730 / 753
//     16   |  tn8    1504 / 1580        |  tn8     845 / 880
//     32   |  tn16   1954 / 2001        |  tn16   1033 / 1056
//     64   |  tn32   2258 / 2227        |  tn32   1182 / 1203
//    128   |  tn32   2932 / 2951        |  tn32   1483 / 1599
//    256   |  tn32   2959 / 3092        |  tn32   2008 / 2257
//    512   |  tn64   3614 / 4081        |  tn64   2444 / 3405
//   1024   |  tn64   4335 / 6153        |  tn64   3448 / 5665
//   2048   |  tn64   6244 / 10806       |  tn64   5607 / 10549
//   (cold-mode numbers at T >= 512 vary by up to ~7% run to run in this
//   protocol, e.g. 5810-6244 us at T=2048; warm-mode 5769 us is stable.
//   The T=1 row is the 2026-09-05 re-measurement after the decode kernels
//   lost the fused routing (routing kernel + FC1 split-K 2 + FC2 split-K 8,
//   launch bounds 256x2 / 256x3); it was 199/212 and 197/209 before.)
//
//   * decode kernels win only at T=1 (cold L2, the 23-layer reality): every
//     routed row is its own CUDA-core GEMV, so from T=2 tokens sharing an
//     expert re-read its weights while the grouped GEMM streams each expert
//     once. FC2 split-K 8 and FC1 split-K 2 are the best of {1,2,4,8} at T=1
//     (FC1 swept in the engine, see kDecodeFc1SplitK). The plugin-only bench
//     overstates decode: inside the engine Marlin's 13 support kernels hide
//     under the shared-expert GEMV on TensorRT's aux stream, so only GEMM
//     streaming efficiency counts there (nsys graph-mode trace, 2026-09-05).
//   * grouped path from T=2. The token tile is the per-expert padding
//     granularity: small tiles remove the pad rows FC1 streams and writes
//     (tn8 vs tn32 at T=16: -8 MB DRAM, -7% FC1 time) but a hot expert with
//     more rows than the tile is re-streamed once per extra N tile (through
//     L2). Thresholds are the best worst-case over both routings from the
//     tile sweep (EDGELLM_MOE_FORCE_TILE) on the 2 KB-row build:
//       T=32:  tn8 0.96/1.01, tn16 0.97/0.97, tn32 1.01/1.00 (uniform/skewed vs Marlin)
//       T=64:  tn8 0.98/1.21, tn16 0.97/1.03, tn32 1.01/0.98, tn64 1.09/1.06
//       T=128: tn16 0.97/1.05, tn32 0.99/0.92, tn64 1.08/0.96
//       T=256: tn16 1.03/1.17, tn32 0.95/0.89, tn64 1.03/0.83
//   * weight streaming: the A operand is fetched as one (256 x 2) uint64 TMA
//     box per 4 KB row tile (the layout carries the K_SW32 smem image). With
//     the original 128 x 32-byte box rows the GEMM streamed ~230 GB/s against
//     Marlin's ~243 GB/s and trailed it by 1-7% at T=2..128; Thor caps
//     32-byte TMA rows at ~230 GB/s regardless of in-flight depth while 64 B+
//     rows reach 258-270 GB/s (kernelSrcs/nvfp4_a16_blackwell_moe/tma_bw_probe.cu).
//     Residual after the change: uniform T=64 1.01x (tn32; tn16 would trade it
//     for skewed 1.03x), everything else at or below Marlin.
// Values are policy hints, not support gates; the runner validates shapes
// independently.
// ---------------------------------------------------------------------------
inline constexpr int32_t kDecodeMaxTokens{1};
//! Decode FC1 split-K: FC1 has only N1_pad/128 * topK = 90 row tiles at T=1
//! (2.25 waves of 40 resident CTAs on 20 SMs); split-K 2 doubles the CTA count
//! and was the best of {1,2,4,8} in the engine (decode step 11.57 / 11.41 /
//! 11.49 / 11.63 ms for split-K 1 / 2 / 4 / 8 at pastKV 128; 1/2/4 are means of
//! three interleaved rounds, 8 a single run; 2 < 4 < 1 held in every round and
//! at pastKV 2048). The fp32 partials (2 x 6 x 1920) cost one ~3 us reduce;
//! decode stays deterministic.
inline constexpr int32_t kDecodeFc1SplitK{2};
//! Largest FC1 split-K the benchmark override may select; the decode workspace
//! is sized for it so the size recorded at engine build never depends on the
//! environment.
inline constexpr int32_t kDecodeFc1MaxSplitK{8};
static_assert(kDecodeFc1SplitK >= 1 && kDecodeFc1SplitK <= kDecodeFc1MaxSplitK, "sealed FC1 split-K out of range");
inline constexpr int32_t kDecodeFc2SplitK{8};
//! Decode FC2 slots whose weight tiles are copied into shared memory with
//! cp.async before the kernel's griddepcontrol.wait (their experts are routing
//! results, complete when FC2 starts).  Sealed to 0: each staged slot costs
//! stagedTiles x 4.5 KB of shared memory per CTA (Nemotron 18 KB), and the
//! larger carve-out keeps FC2 CTAs from co-residing with the shared-expert
//! GEMV that TensorRT runs on its auxiliary stream (kernels with different
//! shared-memory configurations wait for the SM to drain), so FC2 started
//! ~10 us later and the layer got slower: decode step 11.44 / 11.51 / 11.52 ms
//! for 0 / 1 / 2 slots at pastKV 128 (three interleaved rounds, PDL on; 11.42
//! with PDL off), 11.54 / 11.58 / 11.58 at pastKV 2048.  The path stays
//! available for other shapes (EDGELLM_MOE_DECODE_FC2_PREFETCH=1|2 or the
//! runner parameter) and is exercised by the unit tests.
inline constexpr int32_t kDecodeFc2PrefetchSlots{0};
inline constexpr int32_t kDecodeFc2MaxPrefetchSlots{2};

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
    if (numTokens <= 32)
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
