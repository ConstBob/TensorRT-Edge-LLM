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

#if defined(CUTE_DSL_FMHA_V2_ENABLED)

#include "cuteDslFMHAV2Runner.h"

#include "attentionScaleUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cmath>
#include <stdexcept>

namespace trt_edgellm
{

fmha_v2_d64_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64{};
fmha_v2_d64_small_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64Small{};
fmha_v2_d128_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128{};
fmha_v2_d256_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256{};
fmha_v2_d256_padding_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Padding{};
fmha_v2_d64_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64Sw{};
fmha_v2_d128_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128Sw{};
fmha_v2_d256_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Sw{};
fmha_v2_d256_visionblock_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256VisionBlock{};
bool CuteDslFMHAV2Runner::sLLMLoaded{false};
std::mutex CuteDslFMHAV2Runner::sLLMMutex;

fmha_v2_vit_d64_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d64{};
fmha_v2_vit_d72_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d72{};
fmha_v2_vit_d80_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d80{};
fmha_v2_vit_d128_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d128{};
bool CuteDslFMHAV2Runner::sViTLoaded{false};
std::mutex CuteDslFMHAV2Runner::sViTMutex;

namespace
{

bool isFMHAV2SM(int32_t smVersion)
{
    return smVersion == 80 || smVersion == 86 || smVersion == 87 || smVersion == 89 || smVersion == 100
        || smVersion == 101 || smVersion == 110 || smVersion == 120 || smVersion == 121;
}

template <typename Module, typename Loader>
void loadModule(Module& module, Loader loader, char const* name)
{
    loader(&module);
    if (module.module == nullptr)
    {
        throw std::runtime_error(std::string("CuTe DSL module loader returned a null handle for ") + name);
    }
}

template <typename Module>
void unloadModule(Module& module, char const* name) noexcept
{
    if (module.module == nullptr)
    {
        return;
    }
    cudaError_t const status = cudaLibraryUnload(module.module);
    if (status != cudaSuccess)
    {
        LOG_ERROR("Failed to unload CuTe DSL module %s: %s", name, cudaGetErrorString(status));
    }
    module.module = nullptr;
}

} // namespace

bool CuteDslFMHAV2Runner::canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType)
{
    if (!isFMHAV2SM(smVersion) || dataType != nvinfer1::DataType::kHALF || numQHeads <= 0 || numKVHeads <= 0
        || numQHeads < numKVHeads || numQHeads % numKVHeads != 0)
    {
        return false;
    }

    switch (maskType)
    {
    case CuteDslFMHAV2MaskType::kCAUSAL:
    case CuteDslFMHAV2MaskType::kSLIDING_CAUSAL: return headSize == 64 || headSize == 128 || headSize == 256;
    case CuteDslFMHAV2MaskType::kVISION_BLOCK: return headSize == 256;
    case CuteDslFMHAV2MaskType::kPADDING: return headSize == 256;
    }
    return false;
}

bool CuteDslFMHAV2Runner::canImplementViT(int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType)
{
    return isFMHAV2SM(smVersion) && dataType == nvinfer1::DataType::kHALF
        && (headSize == 64 || headSize == 72 || headSize == 80 || headSize == 128);
}

bool CuteDslFMHAV2Runner::loadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock{sLLMMutex};
    if (sLLMLoaded)
    {
        return true;
    }

    try
    {
        loadModule(sLLM_d64, fmha_v2_d64_Kernel_Module_Load, "fmha_v2_d64");
        loadModule(sLLM_d64Small, fmha_v2_d64_small_Kernel_Module_Load, "fmha_v2_d64_small");
        loadModule(sLLM_d128, fmha_v2_d128_Kernel_Module_Load, "fmha_v2_d128");
        loadModule(sLLM_d256, fmha_v2_d256_Kernel_Module_Load, "fmha_v2_d256");
        loadModule(sLLM_d256Padding, fmha_v2_d256_padding_Kernel_Module_Load, "fmha_v2_d256_padding");
        loadModule(sLLM_d64Sw, fmha_v2_d64_sw_Kernel_Module_Load, "fmha_v2_d64_sw");
        loadModule(sLLM_d128Sw, fmha_v2_d128_sw_Kernel_Module_Load, "fmha_v2_d128_sw");
        loadModule(sLLM_d256Sw, fmha_v2_d256_sw_Kernel_Module_Load, "fmha_v2_d256_sw");
        loadModule(sLLM_d256VisionBlock, fmha_v2_d256_visionblock_Kernel_Module_Load, "fmha_v2_d256_visionblock");
        sLLMLoaded = true;
        LOG_DEBUG("FMHA-v2 CuTe DSL LLM FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        unloadModule(sLLM_d64, "fmha_v2_d64");
        unloadModule(sLLM_d64Small, "fmha_v2_d64_small");
        unloadModule(sLLM_d128, "fmha_v2_d128");
        unloadModule(sLLM_d256, "fmha_v2_d256");
        unloadModule(sLLM_d256Padding, "fmha_v2_d256_padding");
        unloadModule(sLLM_d64Sw, "fmha_v2_d64_sw");
        unloadModule(sLLM_d128Sw, "fmha_v2_d128_sw");
        unloadModule(sLLM_d256Sw, "fmha_v2_d256_sw");
        unloadModule(sLLM_d256VisionBlock, "fmha_v2_d256_visionblock");
        LOG_ERROR("Failed to load FMHA-v2 CuTe DSL LLM FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHAV2Runner::unloadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock{sLLMMutex};
    if (!sLLMLoaded)
    {
        return;
    }

    unloadModule(sLLM_d64, "fmha_v2_d64");
    unloadModule(sLLM_d64Small, "fmha_v2_d64_small");
    unloadModule(sLLM_d128, "fmha_v2_d128");
    unloadModule(sLLM_d256, "fmha_v2_d256");
    unloadModule(sLLM_d256Padding, "fmha_v2_d256_padding");
    unloadModule(sLLM_d64Sw, "fmha_v2_d64_sw");
    unloadModule(sLLM_d128Sw, "fmha_v2_d128_sw");
    unloadModule(sLLM_d256Sw, "fmha_v2_d256_sw");
    unloadModule(sLLM_d256VisionBlock, "fmha_v2_d256_visionblock");
    sLLMLoaded = false;
}

bool CuteDslFMHAV2Runner::loadViTKernelModule()
{
    std::lock_guard<std::mutex> lock{sViTMutex};
    if (sViTLoaded)
    {
        return true;
    }

    try
    {
        loadModule(sViT_d64, fmha_v2_vit_d64_Kernel_Module_Load, "fmha_v2_vit_d64");
        loadModule(sViT_d72, fmha_v2_vit_d72_Kernel_Module_Load, "fmha_v2_vit_d72");
        loadModule(sViT_d80, fmha_v2_vit_d80_Kernel_Module_Load, "fmha_v2_vit_d80");
        loadModule(sViT_d128, fmha_v2_vit_d128_Kernel_Module_Load, "fmha_v2_vit_d128");
        sViTLoaded = true;
        LOG_DEBUG("FMHA-v2 CuTe DSL ViT FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        unloadModule(sViT_d64, "fmha_v2_vit_d64");
        unloadModule(sViT_d72, "fmha_v2_vit_d72");
        unloadModule(sViT_d80, "fmha_v2_vit_d80");
        unloadModule(sViT_d128, "fmha_v2_vit_d128");
        LOG_ERROR("Failed to load FMHA-v2 CuTe DSL ViT FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHAV2Runner::unloadViTKernelModule()
{
    std::lock_guard<std::mutex> lock{sViTMutex};
    if (!sViTLoaded)
    {
        return;
    }

    unloadModule(sViT_d64, "fmha_v2_vit_d64");
    unloadModule(sViT_d72, "fmha_v2_vit_d72");
    unloadModule(sViT_d80, "fmha_v2_vit_d80");
    unloadModule(sViT_d128, "fmha_v2_vit_d128");
    sViTLoaded = false;
}

CuteDslFMHAV2Runner::CuteDslFMHAV2Runner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize,
    int32_t seqLenQ, int32_t kvSeqLen, bool useSmallD64)
    : mBatchSize(batchSize)
    , mSeqLenQ(seqLenQ)
    , mKVSeqLen(kvSeqLen)
    , mNumHeadsQ(numQHeads)
    , mNumHeadsKV(numKVHeads)
    , mHeadDim(headDim)
    , mUseSmallD64(useSmallD64)
{
}

// clang-format off
#define CALL_FMHA_V2_LLM(PREFIX, MODULE, WINDOW_SIZE_LEFT)                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = mBatchSize;                                                                         \
        qTensor.dynamic_shapes[1] = mSeqLenQ;                                                                           \
        qTensor.dynamic_shapes[2] = mNumHeadsQ;                                                                         \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;                           \
        qTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;                                      \
                                                                                                                       \
        PREFIX##_Tensor_k_tensor_t kTensor{};                                                                            \
        kTensor.data = const_cast<void*>(kPtr);                                                                          \
        kTensor.dynamic_shapes[0] = mBatchSize;                                                                           \
        kTensor.dynamic_shapes[1] = mKVSeqLen;                                                                           \
        kTensor.dynamic_shapes[2] = mNumHeadsKV;                                                                          \
        kTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;                         \
        kTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;                                      \
                                                                                                                       \
        PREFIX##_Tensor_v_tensor_t vTensor{};                                                                            \
        vTensor.data = const_cast<void*>(vPtr);                                                                          \
        vTensor.dynamic_shapes[0] = mBatchSize;                                                                           \
        vTensor.dynamic_shapes[1] = mKVSeqLen;                                                                           \
        vTensor.dynamic_shapes[2] = mNumHeadsKV;                                                                          \
        vTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;                         \
        vTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;                                      \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                           \
        oTensor.data = oPtr;                                                                                            \
        oTensor.dynamic_shapes[0] = mBatchSize;                                                                          \
        oTensor.dynamic_shapes[1] = mSeqLenQ;                                                                            \
        oTensor.dynamic_shapes[2] = mNumHeadsQ;                                                                          \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;                            \
        oTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;                                       \
                                                                                                                       \
        PREFIX##_Tensor_cum_seqlen_k_t cumSeqlenK{};                                                                    \
        cumSeqlenK.data = const_cast<int32_t*>(cuKVSeqLens);                                                           \
        cumSeqlenK.dynamic_shapes[0] = mBatchSize + 1;                                                                  \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(&(MODULE), &qTensor, &kTensor, &vTensor, &oTensor, &cumSeqlenK,             \
            (WINDOW_SIZE_LEFT), attentionScale, scaleQ, scaleK, scaleV, invScaleO,                                    \
            getDeviceMultiProcessorCount(), stream);                                                                   \
    } while (0)
// clang-format on

bool CuteDslFMHAV2Runner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA kernel module not loaded.");
        return false;
    }

    validateAttentionScale(attentionScale);
    float const scaleQ = 1.0F;
    float const scaleK = 1.0F;
    float const scaleV = 1.0F;
    float const invScaleO = 1.0F;
    int32_t constexpr kNO_LIMIT = 1 << 30;
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;
    int32_t const windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNO_LIMIT;
    int32_t ret = -1;

    if (mHeadDim == 64)
    {
        if (useSlidingWindow)
        {
            CALL_FMHA_V2_LLM(fmha_v2_d64_sw, sLLM_d64Sw, windowSizeLeft);
        }
        else if (mUseSmallD64 && mSeqLenQ <= 512)
        {
            CALL_FMHA_V2_LLM(fmha_v2_d64_small, sLLM_d64Small, windowSizeLeft);
        }
        else
        {
            CALL_FMHA_V2_LLM(fmha_v2_d64, sLLM_d64, windowSizeLeft);
        }
    }
    else if (mHeadDim == 128)
    {
        if (useSlidingWindow)
        {
            CALL_FMHA_V2_LLM(fmha_v2_d128_sw, sLLM_d128Sw, windowSizeLeft);
        }
        else
        {
            CALL_FMHA_V2_LLM(fmha_v2_d128, sLLM_d128, windowSizeLeft);
        }
    }
    else if (mHeadDim == 256)
    {
        if (useSlidingWindow)
        {
            CALL_FMHA_V2_LLM(fmha_v2_d256_sw, sLLM_d256Sw, windowSizeLeft);
        }
        else
        {
            CALL_FMHA_V2_LLM(fmha_v2_d256, sLLM_d256, windowSizeLeft);
        }
    }
    else
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim);
        return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA kernel (d=%d, sw=%s) failed with error code: %d", mHeadDim,
            useSlidingWindow ? "true" : "false", ret);
    }
    return ret == 0;
}

#undef CALL_FMHA_V2_LLM

bool CuteDslFMHAV2Runner::runPadding(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuQSeqLens, int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA kernel module not loaded.");
        return false;
    }
    if (mHeadDim != 256)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA requires head_dim=256.");
        return false;
    }

    check::check(qPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA qPtr must not be null.");
    check::check(kPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA kPtr must not be null.");
    check::check(vPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA vPtr must not be null.");
    check::check(oPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA oPtr must not be null.");
    check::check(cuQSeqLens != nullptr, "FMHA-v2 CuTe DSL padding FMHA cuQSeqLens must not be null.");
    check::check(cuKVSeqLens != nullptr, "FMHA-v2 CuTe DSL padding FMHA cuKVSeqLens must not be null.");
    check::check(mBatchSize > 0 && mSeqLenQ > 0 && mKVSeqLen > 0 && mNumHeadsQ > 0 && mNumHeadsKV > 0,
        "FMHA-v2 CuTe DSL padding FMHA requires positive tensor extents.");
    check::check(mNumHeadsQ >= mNumHeadsKV && mNumHeadsQ % mNumHeadsKV == 0,
        "FMHA-v2 CuTe DSL padding FMHA requires Q heads to be divisible by KV heads.");

    validateAttentionScale(attentionScale);
    fmha_v2_d256_padding_Tensor_mQ_t qTensor{};
    qTensor.data = const_cast<void*>(qPtr);
    qTensor.dynamic_shapes[0] = mBatchSize;
    qTensor.dynamic_shapes[1] = mSeqLenQ;
    qTensor.dynamic_shapes[2] = mNumHeadsQ;
    qTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;
    qTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;

    fmha_v2_d256_padding_Tensor_mK_t kTensor{};
    kTensor.data = const_cast<void*>(kPtr);
    kTensor.dynamic_shapes[0] = mBatchSize;
    kTensor.dynamic_shapes[1] = mKVSeqLen;
    kTensor.dynamic_shapes[2] = mNumHeadsKV;
    kTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;
    kTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;

    fmha_v2_d256_padding_Tensor_mV_t vTensor{};
    vTensor.data = const_cast<void*>(vPtr);
    vTensor.dynamic_shapes[0] = mBatchSize;
    vTensor.dynamic_shapes[1] = mKVSeqLen;
    vTensor.dynamic_shapes[2] = mNumHeadsKV;
    vTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;
    vTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;

    fmha_v2_d256_padding_Tensor_mO_t oTensor{};
    oTensor.data = oPtr;
    oTensor.dynamic_shapes[0] = mBatchSize;
    oTensor.dynamic_shapes[1] = mSeqLenQ;
    oTensor.dynamic_shapes[2] = mNumHeadsQ;
    oTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;
    oTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;

    fmha_v2_d256_padding_Tensor_mCuSeqLenQ_t cumSeqlenQ{};
    cumSeqlenQ.data = const_cast<int32_t*>(cuQSeqLens);
    cumSeqlenQ.dynamic_shapes[0] = mBatchSize + 1;

    fmha_v2_d256_padding_Tensor_mCuSeqLenK_t cumSeqlenK{};
    cumSeqlenK.data = const_cast<int32_t*>(cuKVSeqLens);
    cumSeqlenK.dynamic_shapes[0] = mBatchSize + 1;

    int32_t const ret = cute_dsl_fmha_v2_d256_padding_wrapper(&sLLM_d256Padding, &qTensor, &kTensor, &vTensor, &oTensor,
        &cumSeqlenQ, &cumSeqlenK, attentionScale, mNumHeadsKV, stream);
    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA kernel failed with error code: %d", ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::runVisionBlock(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuKVSeqLens, int32_t const* blockBegin, int32_t const* blockEnd, cudaStream_t stream,
    float attentionScale, int32_t slidingWindowSize)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL vision-block FMHA kernel module not loaded.");
        return false;
    }
    if (mHeadDim != 256 || slidingWindowSize < 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL vision-block FMHA requires head_dim=256 and a non-negative left window.");
        return false;
    }

    validateAttentionScale(attentionScale);
    float const scaleQ = 1.0F;
    float const scaleK = 1.0F;
    float const scaleV = 1.0F;
    float const invScaleO = 1.0F;

    fmha_v2_d256_visionblock_Tensor_q_tensor_t qTensor{};
    qTensor.data = const_cast<void*>(qPtr);
    qTensor.dynamic_shapes[0] = mBatchSize;
    qTensor.dynamic_shapes[1] = mSeqLenQ;
    qTensor.dynamic_shapes[2] = mNumHeadsQ;
    qTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;
    qTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;

    fmha_v2_d256_visionblock_Tensor_k_tensor_t kTensor{};
    kTensor.data = const_cast<void*>(kPtr);
    kTensor.dynamic_shapes[0] = mBatchSize;
    kTensor.dynamic_shapes[1] = mKVSeqLen;
    kTensor.dynamic_shapes[2] = mNumHeadsKV;
    kTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;
    kTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;

    fmha_v2_d256_visionblock_Tensor_v_tensor_t vTensor{};
    vTensor.data = const_cast<void*>(vPtr);
    vTensor.dynamic_shapes[0] = mBatchSize;
    vTensor.dynamic_shapes[1] = mKVSeqLen;
    vTensor.dynamic_shapes[2] = mNumHeadsKV;
    vTensor.dynamic_strides[0] = static_cast<int64_t>(mKVSeqLen) * mNumHeadsKV * mHeadDim;
    vTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;

    fmha_v2_d256_visionblock_Tensor_o_tensor_t oTensor{};
    oTensor.data = oPtr;
    oTensor.dynamic_shapes[0] = mBatchSize;
    oTensor.dynamic_shapes[1] = mSeqLenQ;
    oTensor.dynamic_shapes[2] = mNumHeadsQ;
    oTensor.dynamic_strides[0] = static_cast<int64_t>(mSeqLenQ) * mNumHeadsQ * mHeadDim;
    oTensor.dynamic_strides[1] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;

    fmha_v2_d256_visionblock_Tensor_cum_seqlen_k_t cumSeqlenK{};
    cumSeqlenK.data = const_cast<int32_t*>(cuKVSeqLens);
    cumSeqlenK.dynamic_shapes[0] = mBatchSize + 1;

    fmha_v2_d256_visionblock_Tensor_block_begin_t blockBeginTensor{};
    blockBeginTensor.data = const_cast<int32_t*>(blockBegin);
    blockBeginTensor.dynamic_shapes[0] = mBatchSize;
    blockBeginTensor.dynamic_shapes[1] = mSeqLenQ;
    blockBeginTensor.dynamic_strides[0] = mSeqLenQ;

    fmha_v2_d256_visionblock_Tensor_block_end_t blockEndTensor{};
    blockEndTensor.data = const_cast<int32_t*>(blockEnd);
    blockEndTensor.dynamic_shapes[0] = mBatchSize;
    blockEndTensor.dynamic_shapes[1] = mSeqLenQ;
    blockEndTensor.dynamic_strides[0] = mSeqLenQ;

    int32_t const ret = cute_dsl_fmha_v2_d256_visionblock_wrapper(&sLLM_d256VisionBlock, &qTensor, &kTensor, &vTensor,
        &oTensor, &cumSeqlenK, &blockBeginTensor, &blockEndTensor, slidingWindowSize, attentionScale, scaleQ, scaleK,
        scaleV, invScaleO, getDeviceMultiProcessorCount(), stream);
    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL vision-block FMHA kernel failed with error code: %d", ret);
    }
    return ret == 0;
}

// clang-format off
#define CALL_FMHA_V2_VIT(PREFIX, MODULE)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = totalSeqLen;                                                                       \
        qTensor.dynamic_shapes[1] = mNumHeadsQ;                                                                         \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;                                      \
                                                                                                                       \
        PREFIX##_Tensor_k_tensor_t kTensor{};                                                                          \
        kTensor.data = const_cast<void*>(kPtr);                                                                        \
        kTensor.dynamic_shapes[0] = totalSeqLen;                                                                       \
        kTensor.dynamic_shapes[1] = mNumHeadsKV;                                                                        \
        kTensor.dynamic_strides[0] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;                                     \
                                                                                                                       \
        PREFIX##_Tensor_v_tensor_t vTensor{};                                                                          \
        vTensor.data = const_cast<void*>(vPtr);                                                                        \
        vTensor.dynamic_shapes[0] = totalSeqLen;                                                                       \
        vTensor.dynamic_shapes[1] = mNumHeadsKV;                                                                        \
        vTensor.dynamic_strides[0] = static_cast<int64_t>(mNumHeadsKV) * mHeadDim;                                     \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                          \
        oTensor.data = oPtr;                                                                                           \
        oTensor.dynamic_shapes[0] = totalSeqLen;                                                                       \
        oTensor.dynamic_shapes[1] = mNumHeadsQ;                                                                         \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(mNumHeadsQ) * mHeadDim;                                      \
                                                                                                                       \
        PREFIX##_Tensor_cu_seqlens_t cuSeqlensTensor{};                                                                \
        cuSeqlensTensor.data = const_cast<int32_t*>(cuSeqLens);                                                       \
        cuSeqlensTensor.dynamic_shapes[0] = batchSize + 1;                                                             \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(&(MODULE), &qTensor, &kTensor, &vTensor, &oTensor, &cuSeqlensTensor,         \
            maxSeqLen, scaleSoftmaxLog2, attentionScale, scaleOutput, getDeviceMultiProcessorCount(), stream);         \
    } while (0)
// clang-format on

bool CuteDslFMHAV2Runner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuSeqLens, int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream,
    float attentionScale)
{
    if (!sViTLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA kernel module not loaded.");
        return false;
    }

    validateAttentionScale(attentionScale);
    float const scaleSoftmaxLog2 = attentionScale * static_cast<float>(M_LOG2E);
    float const scaleOutput = 1.0F;
    int32_t ret = -1;

    if (mHeadDim == 64)
    {
        CALL_FMHA_V2_VIT(fmha_v2_vit_d64, sViT_d64);
    }
    else if (mHeadDim == 72)
    {
        CALL_FMHA_V2_VIT(fmha_v2_vit_d72, sViT_d72);
    }
    else if (mHeadDim == 80)
    {
        CALL_FMHA_V2_VIT(fmha_v2_vit_d80, sViT_d80);
    }
    else if (mHeadDim == 128)
    {
        CALL_FMHA_V2_VIT(fmha_v2_vit_d128, sViT_d128);
    }
    else
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA: unsupported head_dim=%d", mHeadDim);
        return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA kernel (d=%d) failed with error code: %d", mHeadDim, ret);
    }
    return ret == 0;
}

#undef CALL_FMHA_V2_VIT

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FMHA_V2_ENABLED)
