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
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

//! Chunk metadata for Qwen3-Omni audio preprocessing
struct ChunkInfo
{
    int64_t numChunks;
    std::vector<int64_t> chunkLengths;
    std::vector<int64_t> chunkOffsets;
    int64_t maxChunkLength;
};

//! Cast a host FP32 mel ``[H, W]`` to FP16 and upload to a fresh GPU tensor
//! shaped ``[1, H, W]``. Shared by audio runners so the same FP16-cast +
//! cudaMemcpyAsync staging path isn't duplicated.
bool uploadHostMelFp32ToFp16Gpu(
    rt::Tensor const& hostMel, rt::Tensor& devOut, cudaStream_t stream, std::string const& debugName);

//! Upload host mono FP32 PCM samples into the pre-allocated ``[N]`` Float GPU
//! staging tensor — the input side of the online GPU fbank path
//! (``fbankWhisper`` consumes this tensor). ``devOut`` is metadata-reshaped to
//! the clip length within its pre-allocated capacity; initFbankResources sizes
//! it for the longest PCM the engine kMAX profile can consume, and
//! tryOnlineGpuFbank gates clip length against that bound before calling.
//! @param hostPcm Mono FP32 PCM in [-1, 1] (typically ``AudioPCM::samples``).
//! @param devOut  Pre-allocated GPU tensor, Float; reshaped to ``[N]``.
//! @param stream  CUDA stream for the async H2D copy.
//! @return true on success, false on empty input or insufficient capacity.
bool uploadHostPcmF32ToGpu(std::vector<float> const& hostPcm, rt::Tensor& devOut, cudaStream_t stream);

//! Cast a host FP32 mel filter ``[nMel, nFreq]`` (row-major) into the
//! ``[nMel, kPad]`` Half K-major layout the CuTe DSL mel GEMM A-matrix
//! expects, and upload it into the pre-allocated ``out`` (fill-only; the
//! caller — initFbankResources — owns allocation). The weights are the
//! Slaney filter exposed by ``MelExtractor::melFilterBank()``.
//! @param melFilterF32 Host row-major ``[nMel, nFreq]`` Slaney weights.
//! @param nMel  Mel bins (rows).
//! @param nFreq FFT bins == nFft/2+1 (active columns; must be <= kPad).
//!              Trailing ``kPad - nFreq`` columns are zero-padded.
//! @param kPad  K-axis padding (>= nFreq) of the GEMM A-matrix layout.
//! @param out   Pre-allocated ``[nMel, kPad]`` Half GPU tensor, filled here.
//! @param stream CUDA stream for the async H2D copy.
//! @return true on success, false on bad input or mis-allocated ``out``.
bool fillMelFilterFp16Kmajor(
    float const* melFilterF32, int32_t nMel, int32_t nFreq, int32_t kPad, rt::Tensor& out, cudaStream_t stream);

//! Build the FFT twiddle table ``W_N^k = exp(-2*pi*i*k/N)`` (k=0..N-1, N == nFft)
//! into ``twoChan`` as ``[nFft, 2]`` float2 interleaved (real, imag); computed in
//! fp64, stored as fp32.
void makeFftTwiddleHost(int32_t nFft, std::vector<float>& twoChan);

//! Compute CNN output length (three 2x downsampling layers)
int64_t computeFeatExtractOutputLength(int64_t inputLength, int32_t nWindow);

//! Compute chunk split information for audio features
ChunkInfo computeChunkInfo(int64_t featureLength, int32_t nWindow);

//! Chunk and pad audio features to uniform size
bool chunkAndPadFeatures(
    rt::Tensor const& melSpectrogram, ChunkInfo const& chunkInfo, rt::Tensor& paddedFeature, cudaStream_t stream);

//! Create validity mask for tokens after CNN downsampling
bool createPaddedMask(ChunkInfo const& chunkInfo, int32_t nWindow, rt::Tensor& paddedMask,
    std::vector<int64_t>& afterCNNLens, cudaStream_t stream);

//! Preprocess audio for Qwen3-Omni encoder: chunk, pad, and create masks
bool preprocessAudioForEncoder(rt::Tensor const& melSpectrogram, int32_t nWindow, rt::Tensor& paddedFeature,
    rt::Tensor& paddedMaskAfterCNN, std::vector<int64_t>& afterCNNLens, cudaStream_t stream);

//! Convert boolean mask to nonzero indices (equivalent to torch.nonzero)
//! This function implements the NonZero operation that was removed from the ONNX model.
//! It converts a 2D boolean mask into indices of nonzero elements.
//!
//! @param paddedMask Input boolean mask [num_chunks, max_len_after_cnn]
//! @param paddedMaskIndices Output indices [num_valid_elements, 2] where each row is [chunk_idx, position_idx]
//! @param stream CUDA stream for async operations
//! @return true on success, false on failure
//!
//! Example:
//!   Input mask: [[1, 1, 0], [1, 0, 0]]
//!   Output indices: [[0, 0], [0, 1], [1, 0]]
bool convertMaskToIndices(rt::Tensor const& paddedMask, rt::Tensor& paddedMaskIndices, cudaStream_t stream);

//! Create block-diagonal attention mask matching _prepare_attention_mask + cu_seqlens logic.
//! Merges per-chunk after-CNN lengths into larger windows using n_window_infer,
//! then builds a block-diagonal mask where each window allows bidirectional attention.
//!
//! @param afterCNNLens Per-chunk after-CNN lengths
//! @param nWindow Audio encoder n_window parameter (default 50)
//! @param nWindowInfer Audio encoder n_window_infer parameter (default 200)
//! @param attentionMask Output attention mask [total_len, total_len]
//! @param stream CUDA stream for async operations
//! @return true on success, false on failure
bool createChunkwiseAttentionMask(std::vector<int64_t> const& afterCNNLens, int32_t nWindow, int32_t nWindowInfer,
    rt::Tensor& attentionMask, cudaStream_t stream);

//! Persistent state for fbankWhisper: constant weight/table tensors, scalar
//! STFT/mel params, and the reusable device workspace. Every tensor is
//! allocated exactly once by Qwen3OmniAudioRunner::initFbankResources — the
//! weights at their natural shapes, the workspace at the engine kMAX profile
//! bound (``maxFrames``) — so the per-clip steady state performs no device
//! allocation: fbankWhisper only rewrites workspace shape metadata via
//! capacity-checked reshape.
//!
//! The scalar STFT/mel params are populated from the bound CPU
//! MelExtractor's MelExtractorConfig (single source of truth — no Whisper
//! constants hard-coded in this layer; initFbankResources validates the
//! config is Whisper-spec before filling these).
struct FbankResources
{
    // Constant weights / tables, filled once at init.
    rt::Tensor melFilterFp16Kmajor; //!< [nMel, K_pad] Half — Slaney mel weights, K-padded (from MelExtractor)
    rt::Tensor hannWindow;          //!< [nFft] Float — periodic Hann window (no S16 normaliser baked in)
    rt::Tensor fftTwiddle;          //!< [nFft, 2] Float — outer-stage W_N^k table

    // Device workspace, pre-allocated at the maxFrames bound and reshaped to
    // the active clip size by fbankWhisper.
    rt::Tensor framedF32;    //!< [T_full, nFft]            Float
    rt::Tensor magFp16;      //!< [N_pad, K_pad]            Half — GEMM B-matrix; zeroed per clip
    rt::Tensor melPowerFp16; //!< [nMel, N_pad]             Half — GEMM C-matrix
    rt::Tensor maxLogScalar; //!< [1]                       Float — global log10 max (reduce output)

    int32_t nFft{0};      //!< STFT size (== MelExtractorConfig::nFFT; kernels require 400)
    int32_t hopLength{0}; //!< STFT hop (== MelExtractorConfig::hopLength)
    int32_t padLength{0}; //!< reflect-pad each side (== nFft / 2)
    int32_t nMel{0};      //!< mel bins (== MelExtractorConfig::nMel)
    int32_t nFreq{0};     //!< FFT bins == nFft / 2 + 1 (active K-window before K-pad)
    int32_t maxFrames{0}; //!< workspace capacity in mel frames (engine kMAX profile bound)
    float melFloor{0.0f}; //!< log10 input floor (== MelExtractorConfig::logFloor)
};

//! Compute the number of output mel frames T_out for a given PCM length.
//!   T_out = (N + 2*padLength - nFft) / hopLength + 1 - 1
//! The final -1 matches the Whisper `stft[..., :-1]` trim that
//! HuggingFace WhisperFeatureExtractor applies. nFft/hopLength/padLength are
//! taken from FbankResources (derived from MelExtractorConfig) so this stays
//! in lockstep with fbankWhisper's internal framing — no duplicated constants.
int32_t computeNumMelFrames(int64_t numPcmSamples, int32_t nFft, int32_t hopLength, int32_t padLength);

//! K-axis padding for the AOT CuTe DSL Blackwell GEMM: round_up(nFreq=201, 16) = 208.
constexpr int32_t kFbankKPad = 208;

//! N-axis padding for the AOT CuTe DSL Blackwell GEMM (tcgen05 + TMA store
//! has no N-residue handling and the cluster-tile alignment requires it).
inline int32_t fbankNPad(int32_t T_out)
{
    return ((T_out + 127) / 128) * 128;
}

//! Whisper-style online fbank: mono FP32 PCM → log-mel F16 spectrogram.
//!
//! Numerically matches HuggingFace transformers WhisperFeatureExtractor with
//! cos_sim >= 0.9998 (abs_max ~ 0.02-0.08 mel units); the residual is dominated
//! by the FP16 Tensor-Core mel GEMM accumulation, well above the F16
//! output-quantisation floor. Composes the six GPU kernel launches declared in
//! kernels/preprocessKernels/audioFbankKernels.h:
//!     memset(mag, 0) → pcmToFramesAndWindow → stftR2C400FusedMagsq →
//!     melLinearGemmFp16TC (AOT CuTe DSL) → log10MaxReduce (init + reduce) →
//!     logMelNormalizeAndCastF16.
//!
//! Output shape/dtype matches the CPU MelExtractor → uploadHostMelFp32ToFp16Gpu
//! contract ([1, nMel, T_out] Half), so downstream preprocessAudioForEncoder
//! requires no change — the GPU fbank and the CPU fallback are interchangeable.
//!
//! Caller must have already loaded the CuTe DSL gemm module (idempotent +
//! thread-safe; audioRunner::initFbankResources does this).
//!
//! @param pcmF32              Input [N] Float GPU tensor — mono FP32 PCM in
//!                            [-1, 1] (uploaded from AudioPCM::samples).
//! @param resources           Persistent fbank state (weights, params, and the
//!                            init-time pre-allocated workspace); non-const:
//!                            the workspace shape metadata is reshaped to the
//!                            active clip size. See FbankResources /
//!                            initFbankResources.
//! @param melOutF16           [1, nMel, T_out] Half GPU output — an owned
//!                            tensor or a non-owning view over a pre-allocated
//!                            backing store (tryOnlineGpuFbank passes a view).
//! @param stream              CUDA stream.
//! @return true on success, false on validation / dispatch failure.
bool fbankWhisper(rt::Tensor const& pcmF32, FbankResources& resources, rt::Tensor& melOutF16, cudaStream_t stream);

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
