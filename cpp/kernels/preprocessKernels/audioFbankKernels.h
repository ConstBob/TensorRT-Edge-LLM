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

#include "common/tensor.h"
#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Whisper-style fbank GPU kernel suite.
//!
//! Pipeline (caller invokes in order; see audioUtils::fbankWhisper):
//!     memset(mag, 0) → FP32 PCM → frame+Hann → R2C FFT+|·|^2+FP16 cast →
//!     mel CuTe DSL GEMM (FP16 in/out, FP32 accum) → log10 max → log-mel
//!     normalize + reshape.
//!
//! Six GPU compute launches (memset doesn't count): three single-purpose
//! custom CUDA kernels (frame+Hann, FFT+magsq, log-mel normalize) + one
//! AOT-compiled CuTe DSL FP16+TC GEMM + a 2-phase custom global-max reducer
//! (init + 2D-grid reduce, two kernels). Numerically matches HuggingFace
//! transformers WhisperFeatureExtractor with cos_sim >= 0.9998 (abs_max
//! ~0.02-0.08 mel units on the delivery clips); the residual is dominated by
//! the FP16 Tensor-Core mel GEMM accumulation and sits well above the F16
//! output-quantisation floor (which would be ~1 LSB). No libcufft / libcublas
//! / libcutlass / libcub / libthrust runtime dependency — the CuTe DSL gemm
//! artifact is an AOT static archive built by kernelSrcs/build_cutedsl.py.
//!
//! Mel pipeline shape contract — set by the AOT GEMM tile alignment:
//!     N_pad = round_up(T_out, 128)      —  GEMM N-axis (no residue handling)
//!     K_pad = round_up(nFreq, 16) = 208 —  GEMM K-axis (cluster-tile aligned)
//!     mag        [N_pad, K_pad]  Half — GEMM B-matrix; first T_out rows × nFreq
//!                                       cols written, padding stays zero
//!     melFilter  [nMel,  K_pad]  Half — GEMM A-matrix; pre-transposed +
//!                                       K-padded by caller at ctor time
//!     melPower   [nMel,  N_pad]  Half — GEMM C-matrix; only [nMel, T_out)
//!                                       window is downstream-visible

//! Fused FP32 PCM → reflect-pad (virtual) → frame → Hann window.
//! Each output element framed[t, n] = pcm[reflectIdx(t*hop + n - pad)] * hann[n].
//!     The input is already mono FP32 in [-1, 1] (decoded host-side by the C++
//!     audioLoader; see AudioPCM::samples), so no S16→F32 normaliser is applied
//!     — the kernel does a single FP mul per element. The reflect padding is
//!     virtual — no intermediate F32 waveform or padded buffer is materialized
//!     in DRAM.
//! Inputs:
//!     pcmF32 [GPU, Float]: [N]                — raw mono FP32 PCM in [-1, 1],
//!         N samples (uploaded from AudioPCM::samples by the runner).
//!     hannWindow [GPU, Float]: [nFft]         — periodic Hann window
//!         (torch.hann_window(periodic=True); see audioRunner::initFbankResources).
//!     nFft / hopLength / padLength            — STFT parameters
//!         (Whisper-v3: 400 / 160 / 200)
//! Outputs:
//!     framedF32 [GPU, Float]: [T_full, nFft] row-major,
//!         where T_full = (N + 2*padLength - nFft) / hopLength + 1.
//! \throws std::runtime_error on invalid tensor shape / dtype / location.
void pcmToFramesAndWindow(rt::Tensor const& pcmF32, rt::Tensor const& hannWindow, rt::Tensor& framedF32,
    int32_t const nFft, int32_t const hopLength, int32_t const padLength, cudaStream_t stream);

//! Self-written R2C FFT, N=400 = 16×25 mixed-radix Cooley-Tukey, fp32, with
//! |·|^2 + FP16 cast fused into Stage 4. Writes directly into the T-major
//! [N_pad, K_pad] FP16 GEMM B-buffer instead of materialising a separate
//! fp32 complex spectrum + a follow-up magsq kernel.
//!
//! One warp (32 threads) per frame; stage 1 uses 25 threads (column DFT_16),
//! stage 3 uses 16 threads (row DFT_25). Each active thread computes an
//! entire small DFT in registers via radix butterflies. 3200 B shared
//! memory per block. Bit-equivalent to cuFFT R2C N=400 within F16 LSB.
//!
//! Launch grid carries T_out (NOT T_full): the last STFT frame is the
//! HuggingFace WhisperFeatureExtractor `stft[..., :-1]` trim and is not
//! consumed downstream. Caller MUST pre-zero the magFp16 buffer (e.g.
//! cudaMemsetAsync) so the padding rows in [T_out, N_pad) and padding cols
//! in [nFreq, K_pad) stay zero — the AOT GEMM has no residue handling.
//!
//! Inputs:
//!     framedF32 [GPU, Float]: [T_full, 400]
//!     fftTwiddle [GPU, Float]: [400, 2]       — outer-stage W_N^k table
//!         (k=0..399) stored as float2 interleaved (real, imag). Precomputed
//!         once by caller in fp64 then cast to fp32.
//!     T_out / K_pad                            — first T_out frames × first
//!         nFreq=201 bins are written; K_pad is the row stride.
//! Outputs:
//!     magFp16 [GPU, Half]: [N_pad, K_pad]     — caller-pre-zeroed.
//! \throws std::runtime_error on invalid tensor shape / dtype / location.
void stftR2C400FusedMagsq(rt::Tensor const& framedF32, rt::Tensor const& fftTwiddle, rt::Tensor& magFp16,
    int32_t const T_out, int32_t const K_pad, cudaStream_t stream);

//! Mel-filter projection via AOT-compiled CuTe DSL FP16+TC GEMM.
//! Logical: melPower = melFilter @ mag^T, with the GEMM ABI
//!     C[M=nMel, N=N_pad] = A[M=nMel, K=K_pad] @ B[N=N_pad, K=K_pad]^T
//! All FP16 in/out, FP32 accumulate. Runtime-dispatched to the right
//! Blackwell tile variant by trt_edgellm::CuteDslGemmRunner::run based on
//! SM count + M: small (64×128 cluster (1,2)) on high-SM datacentres, default
//! (128×128) on Thor, 2-CTA (256×256) on extremely M-heavy paths (not us).
//! Caller must have already loaded the kernel module (idempotent + thread-safe;
//! audioRunner::initFbankResources does this).
//! Inputs:
//!     melFilterFp16Kmajor [GPU, Half]: [nMel, K_pad]  row-major, K contiguous
//!     magFp16             [GPU, Half]: [N_pad, K_pad] row-major, K contiguous
//! Outputs:
//!     melPowerFp16        [GPU, Half]: [nMel, N_pad]  row-major
//! \throws std::runtime_error on invalid tensor shape / dtype / location, or
//!     if CuteDslGemmRunner::run reports dispatch failure.
void melLinearGemmFp16TC(
    rt::Tensor const& melFilterFp16Kmajor, rt::Tensor const& magFp16, rt::Tensor& melPowerFp16, cudaStream_t stream);

//! 2-phase global-max reducer over log10(max(x, melFloor)). Reads FP16
//! melPower in the padded [nMel, N_pad] layout via N_pad stride, only the
//! active (m, t) window [0, nMel) × [0, T_out) contributes.
//!
//! Stage 1 (1-thread init): set the scalar sentinel to -INFINITY.
//! Stage 2 (2D-grid reduce): each block covers a (T_out tile, mel row) pair;
//! ELEMS_PER_THREAD=8 work-coarsening amortises L1TEX latency via ILP; intra-
//! warp shuffle + smem-staged block reduce + atomicMaxFp32 to the scalar.
//! Inputs:
//!     melPowerFp16 [GPU, Half]: [nMel, N_pad]   — nMel must be in (0, 65535].
//!     T_out                                      — active T-window width
//!         (N_pad - T_out trailing cols are zero from upstream memset, would
//!         pollute the max otherwise after log10).
//!     melFloor                                   — log10 input floor
//!         (Whisper default 1e-10f).
//! Outputs:
//!     maxLogScalar [GPU, Float]: [1]
//! \throws std::runtime_error on invalid tensor shape / dtype / location.
void log10MaxReduce(rt::Tensor const& melPowerFp16, int32_t const T_out, rt::Tensor& maxLogScalar, float const melFloor,
    cudaStream_t stream);

//! Final log-mel scale + reshape to [1, nMel, T_out]. Reads FP16 melPower in
//! the padded [nMel, N_pad] layout via N_pad stride and writes only the
//! compact [nMel, T_out] active window.
//! Per element:
//!     y = max(log10(max(x, melFloor)), maxLog - 8.0);
//!     out = half(y * invScale + shiftScaled).
//! Output shape/dtype matches the existing loadMelSpectrogramFromFile
//! contract, so downstream preprocessAudioForEncoder() requires no change.
//! Inputs:
//!     melPowerFp16 [GPU, Half]: [nMel, N_pad]
//!     T_out                                       — active T-window width
//!     maxLogScalar [GPU, Float]: [1]              — output of log10MaxReduce
//!     melFloor                                    — log10 input floor
//! Outputs:
//!     melOutF16 [GPU, Half]: [1, nMel, T_out]
//! \throws std::runtime_error on invalid tensor shape / dtype / location.
void logMelNormalizeAndCastF16(rt::Tensor const& melPowerFp16, int32_t const T_out, rt::Tensor const& maxLogScalar,
    rt::Tensor& melOutF16, float const melFloor, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
