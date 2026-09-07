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

#include "nvfp4A16BlackwellMoeDecodeKernels.h"
#include "nvfp4A16BlackwellMoeDispatchPolicy.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! Runtime parameters of one Nvfp4A16BlackwellMoePlugin forward.
//!
//! Logical contract (Thor SM110, NVFP4 weights / FP16 activations, ReLU2,
//! sigmoid group-top-k routing):
//!   routerLogits [numTokens, numExperts] fp32, correctionBias [numExperts] fp32
//!   hiddenStates [numTokens, hiddenSize], output [numTokens, hiddenSize]
//!   fc1: qweight [E, interSizePadded/128, hiddenSize/64, 128, 32] int8,
//!        blockScales [E, interSizePadded/128, hiddenSize/64, 128, 4] int8,
//!        globalScales [E] fp32   (BLACKWELL_MOE_N128_K64_V1)
//!   fc2: qweight [E, hiddenSize/128, interSize/64, 128, 32] int8, ... likewise
//! One weight buffer per projection serves both backends.
struct Nvfp4A16BlackwellMoeParams
{
    nvfp4_a16_blackwell_moe::DecodeDtype dtype{nvfp4_a16_blackwell_moe::DecodeDtype::kFP16};
    nvfp4_a16_blackwell_moe::Backend backend{nvfp4_a16_blackwell_moe::Backend::kAuto};
    //! Programmatic Dependent Launch for every kernel of the layer (the runner
    //! still requires toolchain support; shared grouped-routing kernels launch
    //! without the attribute and simply serialize).
    bool enablePdl{false};
    int32_t numTokens{0};
    int32_t numExperts{0};
    int32_t topK{0};
    int32_t hiddenSize{0};
    int32_t interSize{0};
    int32_t interSizePadded{0};
    int32_t nGroup{1};
    int32_t topkGroup{1};
    bool normTopkProb{true};
    float routedScalingFactor{1.0f};

    float const* routerLogits{nullptr};
    float const* correctionBias{nullptr};
    void const* hiddenStates{nullptr};
    void const* fc1QWeights{nullptr};
    void const* fc1BlockScales{nullptr};
    float const* fc1GlobalScales{nullptr};
    void const* fc2QWeights{nullptr};
    void const* fc2BlockScales{nullptr};
    float const* fc2GlobalScales{nullptr};
    void* output{nullptr};
};

//! Thor (SM110) W4A16 routed-MoE runner: CUDA-core decode kernels for
//! small token counts, tcgen05 grouped GEMM (AOT group nvfp4_a16_blackwell_moe)
//! otherwise, over ONE weight layout.  Mirrors Nvfp4A16BlackwellGemmRunner:
//! prepare() from onShapeChange loads exactly the AOT variants the profile can
//! dispatch, enqueue() never loads modules or queries the device.
class Nvfp4A16BlackwellMoeRunner
{
public:
    //! Shape/dtype support on this build and SM (routing softmax, SwiGLU and
    //! BF16 are rejected; the plugin keeps Marlin for those exports).
    static bool isSupported(int32_t smVersion, Nvfp4A16BlackwellMoeParams const& shape) noexcept;

    //! Load the AOT prefill variants reachable for token counts up to
    //! shape.numTokens (no-op for decode-only profiles). Call outside CUDA
    //! graph capture. Device queries happen here, not in run().
    static cudaError_t prepare(Nvfp4A16BlackwellMoeParams const& shape, cudaStream_t stream) noexcept;

    //! Workspace bytes for token counts up to shape.numTokens.
    static size_t getWorkspaceSize(Nvfp4A16BlackwellMoeParams const& shape) noexcept;

    //! Run one forward. workspace must hold getWorkspaceSize(maxShape) bytes.
    static cudaError_t run(
        Nvfp4A16BlackwellMoeParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream) noexcept;

    //! Number of GPU operations run() issues for this token count (test aid).
    static int32_t numGpuOps(Nvfp4A16BlackwellMoeParams const& params) noexcept;
};

} // namespace kernel
} // namespace trt_edgellm
