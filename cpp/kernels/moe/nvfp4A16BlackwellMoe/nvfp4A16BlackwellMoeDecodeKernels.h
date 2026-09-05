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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Decode-path (small token count) kernels of Nvfp4A16BlackwellMoePlugin.
//!
//! Both kernels stream the BLACKWELL_MOE_N128_K64_V1 weight layout -- the same
//! buffer the tcgen05 grouped prefill GEMM reads -- with CUDA cores:
//!   qweight      int8 [E, N_pad/128, K/64, 128, 32]
//!   block_scales int8 [E, N_pad/128, K/64, 128, 4]   (raw E4M3)
//!   global_scale fp32 [E]
//! FC1 fuses the sigmoid group-top-k routing (recomputed per CTA from the router
//! logits), the dequantized GEMV of one (token, slot) row against its expert,
//! the per-expert alpha and ReLU^2.  FC2 loops over the token's top-k slots,
//! folds alpha * router weight into the fp32 accumulation and writes the token
//! output directly (deterministic; no atomics).  Split-K variants write fp32
//! partials that a reduce kernel finalizes.
struct DecodeMoeParams
{
    // Routing inputs
    float const* routerLogits{nullptr};   //!< [numTokens, numExperts] fp32
    float const* correctionBias{nullptr}; //!< [numExperts] fp32 or nullptr
    int32_t numExperts{0};
    int32_t topK{0};
    int32_t nGroup{1};
    int32_t topkGroup{1};
    bool normTopkProb{true};
    float routedScalingFactor{1.0f};
    // Routing outputs (written by FC1, read by FC2)
    int32_t* topkIndices{nullptr}; //!< [numTokens, topK] int32
    float* topkWeights{nullptr};   //!< [numTokens, topK] fp32
    // Problem shape
    int32_t numTokens{0};
    int32_t hiddenSize{0};      //!< H: FC1 K and FC2 N (multiple of 128)
    int32_t interSize{0};       //!< I: FC2 K (multiple of 64)
    int32_t interSizePadded{0}; //!< FC1 N (I rounded up to 128)
    // Activations
    void const* hiddenStates{nullptr}; //!< [numTokens, H] FP16/BF16
    void* fc1Output{nullptr};          //!< [numTokens*topK, interSizePadded] FP16/BF16
    void* output{nullptr};             //!< [numTokens, H] FP16/BF16
    // Weights (BLACKWELL_MOE_N128_K64_V1)
    void const* fc1QWeights{nullptr};
    void const* fc1BlockScales{nullptr};
    float const* fc1GlobalScales{nullptr};
    void const* fc2QWeights{nullptr};
    void const* fc2BlockScales{nullptr};
    float const* fc2GlobalScales{nullptr};
    // Split-K
    int32_t fc1SplitK{1};
    int32_t fc2SplitK{1};
    float* fc1Partials{nullptr}; //!< [fc1SplitK, numTokens*topK, interSizePadded] fp32
    float* fc2Partials{nullptr}; //!< [fc2SplitK, numTokens, H] fp32
};

//! Supported activation dtypes.
enum class DecodeDtype : int32_t
{
    kFP16 = 0,
    kBF16 = 1,
};

//! Bytes of fp32 partials required for the given split-K settings (0 when both are 1).
size_t getDecodeFc1PartialBytes(DecodeMoeParams const& params);
size_t getDecodeFc2PartialBytes(DecodeMoeParams const& params);

//! Host-side validation; returns nullptr on success or a static reason string.
char const* validateDecodeParams(DecodeMoeParams const& params, DecodeDtype dtype);

//! Launch routing + FC1 (+ reduce when fc1SplitK > 1). Returns cudaGetLastError().
cudaError_t launchDecodeFc1(DecodeMoeParams const& params, DecodeDtype dtype, cudaStream_t stream);

//! Launch FC2 (+ reduce when fc2SplitK > 1). Returns cudaGetLastError().
cudaError_t launchDecodeFc2(DecodeMoeParams const& params, DecodeDtype dtype, cudaStream_t stream);

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
