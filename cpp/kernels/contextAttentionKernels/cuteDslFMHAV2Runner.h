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

#if defined(CUTE_DSL_FMHA_V2_ENABLED)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif

namespace trt_edgellm::detail
{

inline void checkCuteDslCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CuTe DSL CUDA error ") + cudaGetErrorName(error) + ": " + cudaGetErrorString(error));
    }
}

} // namespace trt_edgellm::detail

// Generated module loaders otherwise only print cudaLibrary errors and return
// success. Make those errors observable so runtime selection can fall back.
#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::checkCuteDslCudaError(error)
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <NvInferRuntime.h>
#include <climits>
#include <cstdint>
#include <mutex>

namespace trt_edgellm
{

//! Mask contracts implemented by the FMHA-v2 CuTe DSL context kernels.
enum class CuteDslFMHAV2MaskType
{
    kCAUSAL,
    kSLIDING_CAUSAL,
    kPADDING,
    kVISION_BLOCK
};

//! Runner for the CuTe DSL replacement of the legacy FMHA-v2 kernels.
//!
//! LLM kernels consume separate BSND Q/K/V tensors. ViT kernels consume packed,
//! separate Q/K/V tensors. The runner is intentionally kept
//! separate from CuteDslFMHARunner because target-specific AOT packs contain
//! either the optimized SM100/101/110 family or this FMHA-v2 family.
class CuteDslFMHAV2Runner
{
public:
    CuteDslFMHAV2Runner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize = 0,
        int32_t seqLenQ = 0, int32_t kvSeqLen = 0, bool useSmallD64 = true);

    ~CuteDslFMHAV2Runner() = default;
    CuteDslFMHAV2Runner(CuteDslFMHAV2Runner const&) = delete;
    CuteDslFMHAV2Runner& operator=(CuteDslFMHAV2Runner const&) = delete;

    //! Returns whether the target AOT family covers this LLM context shape.
    static bool canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
        nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers this packed ViT shape.
    static bool canImplementViT(int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType);

    static bool loadLLMKernelModule();
    static void unloadLLMKernelModule();
    static bool loadViTKernelModule();
    static void unloadViTKernelModule();

    //! Runs causal or sliding-causal LLM context attention.
    bool run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuKVSeqLens,
        cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX);

    //! Runs dense non-causal padded context attention with independent logical Q/KV lengths.
    bool runPadding(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuQSeqLens,
        int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale);

    //! Runs Gemma4 vision-block attention: sliding-causal OR same-image-block.
    bool runVisionBlock(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuKVSeqLens,
        int32_t const* blockBegin, int32_t const* blockEnd, cudaStream_t stream, float attentionScale,
        int32_t slidingWindowSize);

    //! Runs packed varlen, bidirectional ViT attention.
    bool run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
        int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale);

private:
    int32_t mBatchSize{};
    int32_t mSeqLenQ{};
    int32_t mKVSeqLen{};
    int32_t mNumHeadsQ{};
    int32_t mNumHeadsKV{};
    int32_t mHeadDim{};
    bool mUseSmallD64{true};

    static fmha_v2_d64_Kernel_Module_t sLLM_d64;
    static fmha_v2_d64_small_Kernel_Module_t sLLM_d64Small;
    static fmha_v2_d128_Kernel_Module_t sLLM_d128;
    static fmha_v2_d256_Kernel_Module_t sLLM_d256;
    static fmha_v2_d256_padding_Kernel_Module_t sLLM_d256Padding;
    static fmha_v2_d64_sw_Kernel_Module_t sLLM_d64Sw;
    static fmha_v2_d128_sw_Kernel_Module_t sLLM_d128Sw;
    static fmha_v2_d256_sw_Kernel_Module_t sLLM_d256Sw;
    static fmha_v2_d256_visionblock_Kernel_Module_t sLLM_d256VisionBlock;
    static bool sLLMLoaded;
    static std::mutex sLLMMutex;

    static fmha_v2_vit_d64_Kernel_Module_t sViT_d64;
    static fmha_v2_vit_d72_Kernel_Module_t sViT_d72;
    static fmha_v2_vit_d80_Kernel_Module_t sViT_d80;
    static fmha_v2_vit_d128_Kernel_Module_t sViT_d128;
    static bool sViTLoaded;
    static std::mutex sViTMutex;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FMHA_V2_ENABLED)
