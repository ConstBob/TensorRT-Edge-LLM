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

/*
 * Whisper-style fbank GPU kernel suite: mono FP32 PCM -> log-mel F16
 * spectrogram. Five kernels (pcmToFramesAndWindow, stftR2C400FusedMagsq,
 * melLinearGemmFp16TC, log10MaxReduce, logMelNormalizeAndCastF16) compose the
 * on-GPU feature extractor; signatures are expressed in rt::Tensor with
 * check::check tensor-contract validation.
 */

#include "audioFbankKernels.h"
#include "common/checkMacros.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace kernel
{
namespace
{

// ============================================================================
// Kernels
// ============================================================================

// Fused input pipeline: FP32 PCM → (logical reflect-padded) → frame + Hann.
// Each thread emits PCM_M elements of framed[t, :] strided by blockDim.x: it
// computes its logical position in the (would-be) padded buffer, reflect-folds
// to a valid index into the raw FP32 input, and multiplies by hann[n]. The
// input is already mono FP32 in [-1, 1] (decoded host-side by the C++
// audioLoader; see AudioPCM::samples), so no S16→F32 normaliser is folded into
// the window — a single FP mul per element. No intermediate padded buffer is
// materialised.
//
// PCM_M = 4 work-coarsening gives each thread 8 back-to-back independent LDGs
// (pcm + hann), hiding L1 latency on Thor sm_110 via ILP, and lets one block
// cover a whole frame (no partially-active tail block).
//
// Three index branches handle the reflect fold (pos < 0 / pos in [0,N) /
// pos >= N) via `pos = -pos` and `pos = 2N - pos - 2`; a final clamp guards
// pathologically short inputs where pad > N.
//
// Launch contract: block(128) × grid(T_full, 1), one block per frame on
// grid.x. Frames index grid.x (not grid.y) so long-form clips with
// T_full > 65535 (~11 min @ 16 kHz) stay under the 65535 grid.y limit.
// Requires blockDim.x * PCM_M ≥ nFft (4*128 = 512 ≥ Whisper-v3 nFft = 400).
__global__ void pcmToFramedKernel(float const* __restrict__ pcmIn, float const* __restrict__ hann, int64_t N, int p,
    int nFft, int hop, int T_full, float* __restrict__ framedOut)
{
    constexpr int PCM_M = 4;
    int const tid = static_cast<int>(threadIdx.x);
    int const t = blockIdx.x;
    if (t >= T_full)
        return;

    int64_t const t_hop_minus_p = static_cast<int64_t>(t) * hop - p;
    int64_t const t_row_off = static_cast<int64_t>(t) * nFft;

#pragma unroll
    for (int m = 0; m < PCM_M; ++m)
    {
        int const n = tid + m * static_cast<int>(blockDim.x);
        if (n < nFft)
        {
            int64_t pos = t_hop_minus_p + n;
            if (pos < 0)
                pos = -pos;
            if (pos >= N)
                pos = 2 * N - pos - 2;
            pos = max(static_cast<int64_t>(0), min(static_cast<int64_t>(N) - 1, pos));

            framedOut[t_row_off + n] = pcmIn[pos] * hann[n];
        }
    }
}

// Self-written N=400 R2C FFT kernel with |·|^2 + FP16 cast fused into
// Stage 4 (mixed-radix Cooley-Tukey, fp32 internal). Writes directly into
// the T-major FP16 GEMM B-buffer instead of materialising a separate fp32
// complex spectrum + a follow-up magsq kernel — the fp32 spectrum is
// already resident in smem A[] after Stage 3, so magsq costs only 2 muls +
// 1 add + 1 cvt per element in-register (no DRAM round-trip).
//
// Replaces cufftExecR2C — removes the libcufft.so runtime dependency.
//
// Decomposition: N = 400 = N1 × N2 = 16 × 25.
//   Input x[n] viewed as X2d[n1, n2] = x[n1*N2 + n2], row-major.
//   Stage 1 (col DFT, size N1=16):  Y[k1, n2]    = DFT_N1(X2d[:, n2])
//   Stage 2 (twiddle W_N^(k1*n2)):  Z[k1, n2]    = Y[k1, n2] * W_N^(k1*n2)
//   Stage 3 (row DFT, size N2=25):  Xfreq[k1,k2] = DFT_N2(Z[k1, :])
//   Stage 4 (output index + magsq): mag[frame, k1 + k2*N1] = |X|^2 (FP16)
//                                   for k < nFreq=201
//
// Launch: 1 warp (32 threads) per frame; grid_x = T_out (the last STFT frame
// is the HuggingFace `stft[..., :-1]` trim and never consumed downstream;
// writing past T_out would corrupt the GEMM padding rows that the caller's
// upstream cudaMemsetAsync keeps at zero).
//
// Shared memory layout: A[N] = 400 * 8 = 3200 B per block. Reused in-place
// across stages 0/1/3 (input → column-DFT in place → row-DFT natural-bin
// sink). The outer twiddle table W_N (3.2 KB) is read from global directly
// (L1-cached) rather than staged into per-block smem.

__device__ __forceinline__ float2 fft_cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 fft_cadd(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 fft_csub(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

// In-place radix-5 butterfly (kissFFT kf_bfly5 style). 5 complex inputs →
// 5 complex outputs. ~52 fp ops vs naive 200 ops.
// Constants: ya = W_5^1, yb = W_5^2 (rest derived by conjugation).
__device__ __forceinline__ void dft5(float2& x0, float2& x1, float2& x2, float2& x3, float2& x4)
{
    constexpr float ya_r = 0.30901699437494742f;
    constexpr float ya_i = -0.95105651629515357f;
    constexpr float yb_r = -0.80901699437494742f;
    constexpr float yb_i = -0.58778525229247312f;

    float2 const s0 = x0;
    float2 const s7 = fft_cadd(x1, x4);
    float2 const s10 = fft_csub(x1, x4);
    float2 const s8 = fft_cadd(x2, x3);
    float2 const s9 = fft_csub(x2, x3);

    x0 = make_float2(s0.x + s7.x + s8.x, s0.y + s7.y + s8.y);

    float2 const s5 = make_float2(s0.x + s7.x * ya_r + s8.x * yb_r, s0.y + s7.y * ya_r + s8.y * yb_r);
    float2 const s6 = make_float2(s10.y * ya_i + s9.y * yb_i, -s10.x * ya_i - s9.x * yb_i);

    x1 = fft_csub(s5, s6);
    x4 = fft_cadd(s5, s6);

    float2 const s11 = make_float2(s0.x + s7.x * yb_r + s8.x * ya_r, s0.y + s7.y * yb_r + s8.y * ya_r);
    float2 const s12 = make_float2(-s10.y * yb_i + s9.y * ya_i, s10.x * yb_i - s9.x * ya_i);

    x2 = fft_cadd(s11, s12);
    x3 = fft_csub(s11, s12);
}

// 25-point DFT via 5x5 nested radix-5 (Cooley-Tukey N=N1*N2 with N1=N2=5).
// W_25^k = exp(-2*pi*i*k/25) precomputed in fp64 then cast to fp32 — keeps
// table accuracy near fp32 floor.
__device__ __forceinline__ void dft25(float2 x[25])
{
    constexpr float W25_r[25] = {1.0000000000000000f, 0.9685831611286311f, 0.8763066800438636f, 0.7289686274214116f,
        0.5358267949789965f, 0.3090169943749474f, 0.0627905195293134f, -0.1873813145857247f, -0.4257792915650728f,
        -0.6374239897486896f, -0.8090169943749473f, -0.9297764858882515f, -0.9921147013144779f, -0.9921147013144779f,
        -0.9297764858882516f, -0.8090169943749475f, -0.6374239897486897f, -0.4257792915650729f, -0.1873813145857249f,
        0.0627905195293134f, 0.3090169943749472f, 0.5358267949789964f, 0.7289686274214113f, 0.8763066800438634f,
        0.9685831611286311f};
    constexpr float W25_i[25] = {0.0000000000000000f, -0.2486898871648548f, -0.4817536741017153f, -0.6845471059286887f,
        -0.8443279255020151f, -0.9510565162951535f, -0.9980267284282716f, -0.9822872507286887f, -0.9048270524660196f,
        -0.7705132427757893f, -0.5877852522924731f, -0.3681245526846781f, -0.1253332335643041f, 0.1253332335643043f,
        0.3681245526846783f, 0.5877852522924730f, 0.7705132427757891f, 0.9048270524660196f, 0.9822872507286887f,
        0.9980267284282716f, 0.9510565162951536f, 0.8443279255020152f, 0.6845471059286890f, 0.4817536741017156f,
        0.2486898871648551f};

    // Stage A: 5 col DFT_5 over n1 axis, with stage-B twiddle W_25^(k1*n2)
    // applied inline (saves one pass).
    float2 col[5];
#pragma unroll
    for (int n2 = 0; n2 < 5; ++n2)
    {
        col[0] = x[0 * 5 + n2];
        col[1] = x[1 * 5 + n2];
        col[2] = x[2 * 5 + n2];
        col[3] = x[3 * 5 + n2];
        col[4] = x[4 * 5 + n2];
        dft5(col[0], col[1], col[2], col[3], col[4]);
#pragma unroll
        for (int k1 = 0; k1 < 5; ++k1)
        {
            int const idx = k1 * n2;
            float2 const w = make_float2(W25_r[idx], W25_i[idx]);
            x[k1 * 5 + n2] = fft_cmul(col[k1], w);
        }
    }

    // Stage C: 5 row DFT_5 over n2 axis. Natural bin = k1_outer + k2*5.
    float2 tmp[25];
#pragma unroll
    for (int k1 = 0; k1 < 5; ++k1)
    {
        float2 r0 = x[k1 * 5 + 0], r1 = x[k1 * 5 + 1], r2 = x[k1 * 5 + 2];
        float2 r3 = x[k1 * 5 + 3], r4 = x[k1 * 5 + 4];
        dft5(r0, r1, r2, r3, r4);
        tmp[k1 + 0 * 5] = r0;
        tmp[k1 + 1 * 5] = r1;
        tmp[k1 + 2 * 5] = r2;
        tmp[k1 + 3 * 5] = r3;
        tmp[k1 + 4 * 5] = r4;
    }
#pragma unroll
    for (int i = 0; i < 25; ++i)
        x[i] = tmp[i];
}

// 16-point DFT via 4-stage radix-2 DIT (Cooley-Tukey).
// Bit-reverse permutation up front, then 4 stages of butterflies (L=2,4,8,16).
// Total: 4 stages × 8 butterflies = 32 butterflies, ~320 fp ops (vs naive 2048).
__device__ __forceinline__ void dft16(float2 x[16])
{
#define BR_SWAP(I, J)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        float2 _t = x[I];                                                                                              \
        x[I] = x[J];                                                                                                   \
        x[J] = _t;                                                                                                     \
    } while (0)
    BR_SWAP(1, 8);
    BR_SWAP(2, 4);
    BR_SWAP(3, 12);
    BR_SWAP(5, 10);
    BR_SWAP(7, 14);
    BR_SWAP(11, 13);
#undef BR_SWAP

    constexpr float W16_r[8] = {1.0000000000000000f, 0.9238795325112867f, 0.7071067811865476f, 0.3826834323650898f,
        0.0000000000000000f, -0.3826834323650898f, -0.7071067811865475f, -0.9238795325112867f};
    constexpr float W16_i[8] = {0.0000000000000000f, -0.3826834323650898f, -0.7071067811865475f, -0.9238795325112867f,
        -1.0000000000000000f, -0.9238795325112867f, -0.7071067811865476f, -0.3826834323650899f};

    // Stage 0 (L=2): pairs, all twiddle = 1.
#pragma unroll
    for (int g = 0; g < 16; g += 2)
    {
        float2 const t = x[g + 1];
        x[g + 1] = fft_csub(x[g], t);
        x[g + 0] = fft_cadd(x[g], t);
    }
    // Stage 1 (L=4): step in W_16 = 4 (W_4^j = W_16^(4j)).
#pragma unroll
    for (int g = 0; g < 16; g += 4)
    {
        {
            float2 const t = x[g + 0 + 2];
            x[g + 0 + 2] = fft_csub(x[g + 0], t);
            x[g + 0] = fft_cadd(x[g + 0], t);
        }
        // j=1: W=W_16^4 = (0, -1) i.e. -i
        {
            float2 const a = x[g + 1];
            float2 const b = x[g + 1 + 2];
            float2 const t = make_float2(b.y, -b.x);
            x[g + 1 + 2] = fft_csub(a, t);
            x[g + 1] = fft_cadd(a, t);
        }
    }
    // Stage 2 (L=8): step in W_16 = 2 (W_8^j = W_16^(2j)).
#pragma unroll
    for (int g = 0; g < 16; g += 8)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float2 const a = x[g + j];
            float2 const b = x[g + j + 4];
            float2 const w = make_float2(W16_r[2 * j], W16_i[2 * j]);
            float2 const t = fft_cmul(b, w);
            x[g + j + 4] = fft_csub(a, t);
            x[g + j] = fft_cadd(a, t);
        }
    }
    // Stage 3 (L=16): one group, step = 1.
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        float2 const a = x[j];
        float2 const b = x[j + 8];
        float2 const w = make_float2(W16_r[j], W16_i[j]);
        float2 const t = fft_cmul(b, w);
        x[j + 8] = fft_csub(a, t);
        x[j] = fft_cadd(a, t);
    }
}

// Smem footprint: A[N] = 400 * 8 = 3200 B per block. A reused in-place across
// stages 0/1/3. Stage 4 writes FP16 |·|² into the caller's [N_pad, K_pad]
// GEMM B buffer at row=frame, cols [0, NFREQ); cols [NFREQ, K_pad) stay zero
// from the caller's pre-launch memset.
__global__ void stftR2CFusedMagsqKernel(
    float const* __restrict__ framed, float2 const* __restrict__ twiddle, __half* __restrict__ mag, int K_pad)
{
    constexpr int N = 400;
    constexpr int N1 = 16;
    constexpr int N2 = 25;
    constexpr int NFREQ = N / 2 + 1; // 201

    extern __shared__ float2 smem[];
    float2* A = smem; // size N: input → stage-1 in-place → stage-3 natural-bin sink
    int const frame = blockIdx.x;
    int const tid = threadIdx.x;
    int const nthr = blockDim.x; // 32

    // Stage 0: load real input as complex (imag=0) into A.
    float const* in = framed + static_cast<size_t>(frame) * N;
    for (int i = tid; i < N; i += nthr)
    {
        A[i] = make_float2(in[i], 0.0f);
    }
    __syncthreads();

    // Stage 1 + outer twiddle: 25 active threads, each does one column DFT_16.
    // Thread n2 ∈ [0, 25): col[n1] = A[n1*N2 + n2], run dft16, apply twiddle
    // W_N^(k1*n2) per output (read directly from `twiddle` — global, L1-cached),
    // write back in-place to A[k1*N2 + n2]. Each thread reads and writes ONLY
    // the indices {k1*N2 + n2 : k1} for its fixed n2 = tid — a disjoint mod-N2
    // stripe of A with no cross-lane aliasing — so the in-place reuse is
    // race-free with no warp barrier (unlike Stage 3, which permutes strides).
    if (tid < N2)
    {
        int const n2 = tid;
        float2 col[N1];
#pragma unroll
        for (int n1 = 0; n1 < N1; ++n1)
            col[n1] = A[n1 * N2 + n2];
        dft16(col);
#pragma unroll
        for (int k1 = 0; k1 < N1; ++k1)
        {
            // k1*n2 maxes at 15*24=360 < 400 — direct global index, L1-cached.
            A[k1 * N2 + n2] = fft_cmul(col[k1], twiddle[k1 * n2]);
        }
    }
    __syncthreads();

    // Stage 3: 16 active threads, each does one row DFT_25. row[k2] = X[k1, k2]
    // in outer Cooley-Tukey coords. Store to A in natural-bin order
    // `bin = k1 + k2*N1` so Stage 4 reads contiguous addresses.
    //
    // In-place stride change (read N2=25, write N1=16): one lane's write index
    // `k1 + k2*N1` aliases another lane's not-yet-read input range (e.g. lane 0
    // writes A[64], within lane 2's read range [50, 74]). All active lanes must
    // finish reading A[] into `row[]` before any lane writes back; Volta+
    // independent thread scheduling does not guarantee warp lockstep, so an
    // explicit barrier over the 16 active lanes (mask 0xFFFF) is required.
    if (tid < N1)
    {
        int const k1 = tid;
        float2 row[N2];
#pragma unroll
        for (int n2 = 0; n2 < N2; ++n2)
            row[n2] = A[k1 * N2 + n2];
        dft25(row);
        __syncwarp(0xFFFFu); // all 16 lanes done reading A[] before any write below
#pragma unroll
        for (int k2 = 0; k2 < N2; ++k2)
            A[k1 + k2 * N1] = row[k2];
    }
    __syncthreads();

    // Stage 4: pack first NFREQ bins to global as FP16 magnitude-squared.
    // A[bin] holds the natural-bin fp32 complex output; bins >= NFREQ are
    // conjugate-symmetric and dropped. K_pad is the row stride of the mag
    // buffer (round_up(NFREQ, 16) for the AOT GEMM K alignment); padded
    // cols [NFREQ, K_pad) stay zero from the upstream memset.
    __half* out = mag + static_cast<size_t>(frame) * K_pad;
    for (int idx = tid; idx < NFREQ; idx += nthr)
    {
        float2 const v = A[idx];
        out[idx] = __float2half_rn(v.x * v.x + v.y * v.y);
    }
}

// Custom 2-phase global-max reducer for log10(max(mel_power, mel_floor)).
// The 2D grid feeds (m, t) directly into thread coordinates, avoiding the
// 1D→2D index div/mod a cub::DeviceReduce::Max + thrust functor would need
// and removing the libcub / libthrust runtime dependency.
//
// FP16-input variant: mel_power is the padded [nMel, N_pad] FP16 GEMM C
// output; this kernel walks the compact (m, t) space via N_pad stride and
// bound-checks t < T_out so the zero padding cols [T_out, N_pad) never enter
// the max.

__device__ __forceinline__ void atomicMaxFp32(float* addr, float val)
{
    int* ai = reinterpret_cast<int*>(addr);
    // atomicCAS needs an int*, so the cast is unavoidable; read the initial
    // value through the __float_as_int intrinsic rather than the punned pointer.
    int old = __float_as_int(*addr);
    while (true)
    {
        float cur = __int_as_float(old);
        if (val <= cur)
            return;
        int prev = atomicCAS(ai, old, __float_as_int(val));
        if (prev == old)
            return;
        old = prev;
    }
}

__global__ void melMaxLogInitKernel(float* maxLogOut)
{
    *maxLogOut = -INFINITY;
}

// grid: (ceilDiv(T_out, BLOCK_X * ELEMS_PER_THREAD), nMel),  block: (BLOCK_X)
// row_off uses N_pad stride since melPower is the GEMM output [nMel, N_pad].
template <int BLOCK_X, int ELEMS_PER_THREAD>
__global__ void melMaxReduceFp16Kernel(
    __half const* __restrict__ base, int N_pad, int T_out, float floor_, float* __restrict__ maxLogOut)
{
    static_assert(BLOCK_X % 32 == 0, "BLOCK_X must be a multiple of 32.");
    constexpr int NWARPS = BLOCK_X / 32;
    constexpr int TILE_T = BLOCK_X * ELEMS_PER_THREAD;
    int const m = blockIdx.y;
    int const t_base = blockIdx.x * TILE_T;
    int64_t const row_off = static_cast<int64_t>(m) * N_pad;

    float v = -INFINITY;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e)
    {
        int const t = t_base + e * BLOCK_X + static_cast<int>(threadIdx.x);
        if (t < T_out)
        {
            float const x = __half2float(base[row_off + t]);
            v = fmaxf(v, log10f(fmaxf(x, floor_)));
        }
    }

    // Intra-warp reduce.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
    }

    __shared__ float warpMax[NWARPS];
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0)
        warpMax[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        v = (lane < NWARPS) ? warpMax[lane] : -INFINITY;
#pragma unroll
        for (int off = NWARPS / 2; off > 0; off >>= 1)
        {
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
        }
        if (lane == 0)
            atomicMaxFp32(maxLogOut, v);
    }
}

// Final scale + reshape, written to caller's out buffer as half. FP16-input
// variant: reads melPower [nMel, N_pad] FP16 via N_pad stride, writes only
// the compact [1, nMel, T_out] window.
// Per element:
//   y = log10(max(x, floor)); y = max(y, max_log - 8);
//   out = half(y * invScale + shiftScaled).
// log10 is recomputed inline rather than read back from log10MaxReduce.
//
// 2D grid: (ceilDiv(T_out, BLOCK_X * ELEMS_PER_THREAD), nMel),  block: (BLOCK_X).
// (v + shift) / scale is expressed as `v * invScale + shiftScaled` so it fuses
// into a single FFMA. With kShift = kScale = 4, invScale = 0.25 and
// shiftScaled = 1.0 are both powers of two, so the result is bit-identical to
// (v + shift) / scale within FP32 rounding.
template <int BLOCK_X, int ELEMS_PER_THREAD>
__global__ void logMelNormalizeFp16Kernel(__half const* __restrict__ powerIn, half* __restrict__ out, int T_out,
    int N_pad, float const* __restrict__ maxLogPtr, float floor_, float maxMinus, float invScale, float shiftScaled)
{
    constexpr int TILE_T = BLOCK_X * ELEMS_PER_THREAD;
    int const m = blockIdx.y;
    int const t_base = blockIdx.x * TILE_T;
    int64_t const row_in = static_cast<int64_t>(m) * N_pad;
    int64_t const row_out = static_cast<int64_t>(m) * T_out;
    float const threshold = *maxLogPtr - maxMinus;

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e)
    {
        int const t = t_base + e * BLOCK_X + static_cast<int>(threadIdx.x);
        if (t < T_out)
        {
            float const x = __half2float(powerIn[row_in + t]);
            float v = log10f(fmaxf(x, floor_));
            v = fmaxf(v, threshold);
            out[row_out + t] = __float2half(v * invScale + shiftScaled);
        }
    }
}

// Default normalisation constants for Whisper / Qwen3-Omni (match
// transformers WhisperFeatureExtractor):
constexpr float kMaxMinus = 8.0f;
constexpr float kShift = 4.0f;
constexpr float kScale = 4.0f;

} // anonymous namespace

// ============================================================================
// Wrappers
// ============================================================================

void pcmToFramesAndWindow(rt::Tensor const& pcmF32, rt::Tensor const& hannWindow, rt::Tensor& framedF32,
    int32_t const nFft, int32_t const hopLength, int32_t const padLength, cudaStream_t stream)
{
    check::check(pcmF32.getDeviceType() == rt::DeviceType::kGPU && hannWindow.getDeviceType() == rt::DeviceType::kGPU
            && framedF32.getDeviceType() == rt::DeviceType::kGPU,
        "pcmToFramesAndWindow: all tensors must be GPU.");
    // Input is mono FP32 PCM in [-1, 1] (decoded host-side by the C++
    // audioLoader; see AudioPCM::samples) — already float, no S16/int16
    // reinterpret.
    check::check(pcmF32.getDataType() == DataType::kFLOAT && hannWindow.getDataType() == DataType::kFLOAT
            && framedF32.getDataType() == DataType::kFLOAT,
        "pcmToFramesAndWindow: pcmF32, hannWindow and framedF32 must be Float.");
    check::check(pcmF32.getShape().getNumDims() == 1, "pcmToFramesAndWindow: pcmF32 must be [N] (mono FP32 samples).");
    check::check(hannWindow.getShape().getNumDims() == 1 && hannWindow.getShape()[0] == nFft,
        "pcmToFramesAndWindow: hannWindow shape must be [nFft].");
    check::check(framedF32.getShape().getNumDims() == 2 && framedF32.getShape()[1] == nFft,
        "pcmToFramesAndWindow: framedF32 shape must be [T_full, nFft].");

    // Launch invariant: each thread emits PCM_M=4 elements strided by
    // blockDim.x=128, so blockDim.x * PCM_M = 512 must cover nFft, otherwise
    // frame columns [512, nFft) are never written (stale DRAM → wrong spectrum).
    check::check(nFft <= 512, "pcmToFramesAndWindow: nFft must be <= blockDim.x*PCM_M (512).");

    int64_t const N = pcmF32.getShape()[0];
    int32_t const T_full = static_cast<int32_t>(framedF32.getShape()[0]);

    // One block per frame on grid.x, PCM_M = 4 elements per thread (see the
    // kernel comment). Frames index grid.x (not grid.y) so long-form clips
    // stay under the 65535 grid.y limit.
    dim3 const block(128, 1);
    dim3 const grid(T_full, 1);
    pcmToFramedKernel<<<grid, block, 0, stream>>>(pcmF32.dataPointer<float>(), hannWindow.dataPointer<float>(), N,
        padLength, nFft, hopLength, T_full, framedF32.dataPointer<float>());
    CUDA_CHECK(cudaGetLastError());
}

void stftR2C400FusedMagsq(rt::Tensor const& framedF32, rt::Tensor const& fftTwiddle, rt::Tensor& magFp16,
    int32_t const T_out, int32_t const K_pad, cudaStream_t stream)
{
    constexpr int kN = 400;

    check::check(framedF32.getDeviceType() == rt::DeviceType::kGPU && fftTwiddle.getDeviceType() == rt::DeviceType::kGPU
            && magFp16.getDeviceType() == rt::DeviceType::kGPU,
        "stftR2C400FusedMagsq: all tensors must be GPU.");
    check::check(framedF32.getDataType() == DataType::kFLOAT && fftTwiddle.getDataType() == DataType::kFLOAT,
        "stftR2C400FusedMagsq: framedF32 and fftTwiddle must be Float.");
    check::check(magFp16.getDataType() == DataType::kHALF, "stftR2C400FusedMagsq: magFp16 must be Half.");
    check::check(framedF32.getShape().getNumDims() == 2 && framedF32.getShape()[1] == kN,
        "stftR2C400FusedMagsq: framedF32 shape must be [T_full, 400].");
    check::check(
        fftTwiddle.getShape().getNumDims() == 2 && fftTwiddle.getShape()[0] == kN && fftTwiddle.getShape()[1] == 2,
        "stftR2C400FusedMagsq: fftTwiddle shape must be [400, 2] (interleaved complex).");
    check::check(magFp16.getShape().getNumDims() == 2 && magFp16.getShape()[1] == K_pad,
        "stftR2C400FusedMagsq: magFp16 shape must be [N_pad, K_pad].");
    check::check(T_out > 0 && T_out <= magFp16.getShape()[0], "stftR2C400FusedMagsq: T_out must be in (0, N_pad].");
    check::check(T_out <= framedF32.getShape()[0], "stftR2C400FusedMagsq: T_out must not exceed framedF32 T_full.");

    dim3 const grid(T_out, 1, 1);
    dim3 const block(32, 1, 1);
    // Smem: A[N=400] reused in-place across stages 0/1/3. Outer twiddle table
    // read from global directly (L1-cached), not staged into smem.
    size_t const smemBytes = kN * sizeof(float2);
    stftR2CFusedMagsqKernel<<<grid, block, smemBytes, stream>>>(framedF32.dataPointer<float>(),
        reinterpret_cast<float2 const*>(fftTwiddle.dataPointer<float>()), magFp16.dataPointer<half>(), K_pad);
    CUDA_CHECK(cudaGetLastError());
}

void melLinearGemmFp16TC(
    rt::Tensor const& melFilterFp16Kmajor, rt::Tensor const& magFp16, rt::Tensor& melPowerFp16, cudaStream_t stream)
{
#ifdef CUTE_DSL_GEMM_ENABLED
    check::check(melFilterFp16Kmajor.getDeviceType() == rt::DeviceType::kGPU
            && magFp16.getDeviceType() == rt::DeviceType::kGPU && melPowerFp16.getDeviceType() == rt::DeviceType::kGPU,
        "melLinearGemmFp16TC: all tensors must be GPU.");
    check::check(melFilterFp16Kmajor.getDataType() == DataType::kHALF && magFp16.getDataType() == DataType::kHALF
            && melPowerFp16.getDataType() == DataType::kHALF,
        "melLinearGemmFp16TC: all tensors must be Half.");
    check::check(melFilterFp16Kmajor.getShape().getNumDims() == 2 && magFp16.getShape().getNumDims() == 2
            && melPowerFp16.getShape().getNumDims() == 2,
        "melLinearGemmFp16TC: all tensors must be 2-D.");

    int32_t const M = static_cast<int32_t>(melFilterFp16Kmajor.getShape()[0]); // nMel
    int32_t const K = static_cast<int32_t>(melFilterFp16Kmajor.getShape()[1]); // K_pad
    int32_t const N = static_cast<int32_t>(magFp16.getShape()[0]);             // N_pad

    check::check(
        magFp16.getShape()[1] == K, "melLinearGemmFp16TC: magFp16.shape[1] must equal melFilter.shape[1] (K_pad).");
    check::check(melPowerFp16.getShape()[0] == M && melPowerFp16.getShape()[1] == N,
        "melLinearGemmFp16TC: melPowerFp16 shape must equal [nMel, N_pad].");

    bool const ok = CuteDslGemmRunner::run(
        melFilterFp16Kmajor.rawPointer(), magFp16.rawPointer(), melPowerFp16.rawPointer(), M, N, K, stream);
    check::check(ok, "melLinearGemmFp16TC: CuteDslGemmRunner::run dispatch failed.");
#else
    // GEMM variant not compiled (ENABLE_CUTE_DSL=OFF); signature kept so audioUtils.cpp links.
    check::check(
        false, "melLinearGemmFp16TC: CuTe DSL GEMM not compiled. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL).");
#endif
}

void log10MaxReduce(rt::Tensor const& melPowerFp16, int32_t const T_out, rt::Tensor& maxLogScalar, float const melFloor,
    cudaStream_t stream)
{
    check::check(
        melPowerFp16.getDeviceType() == rt::DeviceType::kGPU && maxLogScalar.getDeviceType() == rt::DeviceType::kGPU,
        "log10MaxReduce: all tensors must be GPU.");
    check::check(melPowerFp16.getDataType() == DataType::kHALF && maxLogScalar.getDataType() == DataType::kFLOAT,
        "log10MaxReduce: melPowerFp16 must be Half, maxLogScalar must be Float.");
    check::check(melPowerFp16.getShape().getNumDims() == 2, "log10MaxReduce: melPowerFp16 must be 2-D [nMel, N_pad].");

    int32_t const nMel = static_cast<int32_t>(melPowerFp16.getShape()[0]);
    int32_t const N_pad = static_cast<int32_t>(melPowerFp16.getShape()[1]);
    check::check(nMel > 0 && nMel <= 65535, "log10MaxReduce: nMel must be in (0, 65535].");
    check::check(T_out > 0 && T_out <= N_pad, "log10MaxReduce: T_out must be in (0, N_pad].");

    __half const* dMelPower = melPowerFp16.dataPointer<half>();
    float* dMaxLog = maxLogScalar.dataPointer<float>();

    // Stage 1: set the global sentinel to -INFINITY (1 thread).
    melMaxLogInitKernel<<<1, 1, 0, stream>>>(dMaxLog);
    CUDA_CHECK(cudaGetLastError());

    // Stage 2: 2D-grid reduce with ELEMS=8 ILP amortisation. Each block covers
    // a TILE_T = BLOCK_X * ELEMS_PER_THREAD = 2048-wide t-tile of one mel row.
    constexpr int kReduceBlock = 256;
    constexpr int kReduceElems = 8;
    constexpr int kTileT = kReduceBlock * kReduceElems;
    dim3 const grid((T_out + kTileT - 1) / kTileT, nMel);
    dim3 const block(kReduceBlock);
    melMaxReduceFp16Kernel<kReduceBlock, kReduceElems>
        <<<grid, block, 0, stream>>>(dMelPower, N_pad, T_out, melFloor, dMaxLog);
    CUDA_CHECK(cudaGetLastError());
}

void logMelNormalizeAndCastF16(rt::Tensor const& melPowerFp16, int32_t const T_out, rt::Tensor const& maxLogScalar,
    rt::Tensor& melOutF16, float const melFloor, cudaStream_t stream)
{
    check::check(melPowerFp16.getDeviceType() == rt::DeviceType::kGPU
            && maxLogScalar.getDeviceType() == rt::DeviceType::kGPU
            && melOutF16.getDeviceType() == rt::DeviceType::kGPU,
        "logMelNormalizeAndCastF16: all tensors must be GPU.");
    check::check(melPowerFp16.getDataType() == DataType::kHALF && maxLogScalar.getDataType() == DataType::kFLOAT,
        "logMelNormalizeAndCastF16: melPowerFp16 must be Half, maxLogScalar must be Float.");
    check::check(melOutF16.getDataType() == DataType::kHALF, "logMelNormalizeAndCastF16: melOutF16 must be Half.");
    check::check(melPowerFp16.getShape().getNumDims() == 2,
        "logMelNormalizeAndCastF16: melPowerFp16 must be 2-D [nMel, N_pad].");
    check::check(melOutF16.getShape().getNumDims() == 3 && melOutF16.getShape()[0] == 1
            && melOutF16.getShape()[1] == melPowerFp16.getShape()[0] && melOutF16.getShape()[2] == T_out,
        "logMelNormalizeAndCastF16: melOutF16 shape must be [1, nMel, T_out].");

    int32_t const nMel = static_cast<int32_t>(melPowerFp16.getShape()[0]);
    int32_t const N_pad = static_cast<int32_t>(melPowerFp16.getShape()[1]);
    check::check(nMel > 0 && nMel <= 65535, "logMelNormalizeAndCastF16: nMel must be in (0, 65535].");
    check::check(T_out > 0 && T_out <= N_pad, "logMelNormalizeAndCastF16: T_out must be in (0, N_pad].");
    constexpr int kNormBlock = 256;
    constexpr int kNormElems = 4;
    constexpr int kTileT = kNormBlock * kNormElems; // 1024
    constexpr float kInvScale = 1.0f / kScale;      // 0.25  (power-of-2, bit-exact fuse)
    constexpr float kShiftScaled = kShift / kScale; // 1.0
    dim3 const grid((T_out + kTileT - 1) / kTileT, nMel);
    dim3 const block(kNormBlock);
    logMelNormalizeFp16Kernel<kNormBlock, kNormElems><<<grid, block, 0, stream>>>(melPowerFp16.dataPointer<half>(),
        melOutF16.dataPointer<half>(), T_out, N_pad, maxLogScalar.dataPointer<float>(), melFloor, kMaxMinus, kInvScale,
        kShiftScaled);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
