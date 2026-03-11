/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef CUTE_DSL_FMHA_ENABLED

#include "cuteDslFMHARunner.h"

#include "common/logger.h"

#include <climits>
#include <cmath>

namespace trt_edgellm
{

// Static member initialization
fmha_d64_Kernel_Module_t CuteDslFMHARunner::sKernelModule_d64 = {};
fmha_d128_Kernel_Module_t CuteDslFMHARunner::sKernelModule_d128 = {};
fmha_d64_sw_Kernel_Module_t CuteDslFMHARunner::sKernelModule_d64_sw = {};
fmha_d128_sw_Kernel_Module_t CuteDslFMHARunner::sKernelModule_d128_sw = {};
bool CuteDslFMHARunner::sModuleLoaded = false;
std::mutex CuteDslFMHARunner::sLoadMutex;

bool CuteDslFMHARunner::loadKernelModule()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);

    if (sModuleLoaded)
    {
        return true;
    }

    try
    {
        fmha_d64_Kernel_Module_Load(&sKernelModule_d64);
        LOG_DEBUG("CuTe DSL FMHA kernel module (d=64) loaded successfully");

        fmha_d128_Kernel_Module_Load(&sKernelModule_d128);
        LOG_DEBUG("CuTe DSL FMHA kernel module (d=128) loaded successfully");

        fmha_d64_sw_Kernel_Module_Load(&sKernelModule_d64_sw);
        LOG_DEBUG("CuTe DSL FMHA kernel module (d=64, sliding window) loaded successfully");

        fmha_d128_sw_Kernel_Module_Load(&sKernelModule_d128_sw);
        LOG_DEBUG("CuTe DSL FMHA kernel module (d=128, sliding window) loaded successfully");

        sModuleLoaded = true;
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Exception occurred while loading CuTe DSL FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHARunner::unloadKernelModule()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);

    if (sModuleLoaded)
    {
        fmha_d64_Kernel_Module_Unload(&sKernelModule_d64);
        fmha_d128_Kernel_Module_Unload(&sKernelModule_d128);
        fmha_d64_sw_Kernel_Module_Unload(&sKernelModule_d64_sw);
        fmha_d128_sw_Kernel_Module_Unload(&sKernelModule_d128_sw);
        sModuleLoaded = false;
        LOG_INFO("CuTe DSL FMHA kernel modules unloaded");
    }
}

bool CuteDslFMHARunner::canImplement(int32_t headSize, int32_t smVersion)
{
    bool const checkSMNumber = (smVersion >= 100);
    bool const checkHeadSize = (headSize == 64 || headSize == 128);
    return checkSMNumber && checkHeadSize;
}

CuteDslFMHARunner::CuteDslFMHARunner(
    int32_t b, int32_t s_q, int32_t kvCacheCapacity, int32_t h_q, int32_t h_k, int32_t d)
    : mBatchSize(b)
    , mSeqLenQ(s_q)
    , mKVCacheCapacity(kvCacheCapacity)
    , mNumHeadsQ(h_q)
    , mNumHeadsK(h_k)
    , mHeadDim(d)
{
}

void CuteDslFMHARunner::run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens,
    cudaStream_t stream, int32_t slidingWindowSize)
{
    if (!sModuleLoaded)
    {
        LOG_ERROR("CuTe DSL FMHA kernel module not loaded. Call loadKernelModule() first.");
        return;
    }

    // Compute scale values
    float const softmaxScale = 1.0f / std::sqrt(static_cast<float>(mHeadDim));
    float const scaleSoftmaxLog2 = softmaxScale * static_cast<float>(M_LOG2E);
    float const scaleOutput = 1.0f;

    int32_t const b = mBatchSize;
    int32_t const s_q = mSeqLenQ;
    int32_t const h_q = mNumHeadsQ;
    int32_t const h_k = mNumHeadsK;
    int32_t const d = mHeadDim;
    int32_t const cap = mKVCacheCapacity; // Physical capacity (for stride computation)

    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);

    // All CuTe DSL generated tensor types share the same struct layout;
    // only the type names and wrapper function symbols differ per variant.
    // KV tensor shape[3] = cap (physical capacity) so compact-dynamic strides are correct.
    // Effective KV length is passed via cum_seqlen_k tensor for per-batch iteration bounds.
    // clang-format off
#define CALL_FMHA_KERNEL(PREFIX, MODULE, WSL)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = b;                                                                                 \
        qTensor.dynamic_shapes[1] = s_q;                                                                               \
        qTensor.dynamic_shapes[2] = h_q;                                                                               \
        qTensor.dynamic_shapes[3] = d;                                                                                 \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(s_q) * h_q * d;                                             \
        qTensor.dynamic_strides[1] = static_cast<int64_t>(h_q) * d;                                                   \
        qTensor.dynamic_strides[2] = static_cast<int64_t>(d);                                                          \
                                                                                                                       \
        PREFIX##_Tensor_kv_cache_t kvTensor{};                                                                          \
        kvTensor.data = const_cast<void*>(kvPtr);                                                                       \
        kvTensor.dynamic_shapes[0] = b;                                                                                 \
        kvTensor.dynamic_shapes[1] = 2;                                                                                 \
        kvTensor.dynamic_shapes[2] = h_k;                                                                               \
        kvTensor.dynamic_shapes[3] = cap;                                                                               \
        kvTensor.dynamic_shapes[4] = d;                                                                                 \
        kvTensor.dynamic_strides[0] = static_cast<int64_t>(2) * h_k * cap * d;                                         \
        kvTensor.dynamic_strides[1] = static_cast<int64_t>(h_k) * cap * d;                                             \
        kvTensor.dynamic_strides[2] = static_cast<int64_t>(cap) * d;                                                   \
        kvTensor.dynamic_strides[3] = static_cast<int64_t>(d);                                                          \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                           \
        oTensor.data = oPtr;                                                                                            \
        oTensor.dynamic_shapes[0] = b;                                                                                  \
        oTensor.dynamic_shapes[1] = s_q;                                                                                \
        oTensor.dynamic_shapes[2] = h_q;                                                                                \
        oTensor.dynamic_shapes[3] = d;                                                                                  \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(s_q) * h_q * d;                                              \
        oTensor.dynamic_strides[1] = static_cast<int64_t>(h_q) * d;                                                    \
        oTensor.dynamic_strides[2] = static_cast<int64_t>(d);                                                           \
                                                                                                                       \
        PREFIX##_Tensor_cum_seqlen_k_t cumSeqlenK{};                                                                     \
        cumSeqlenK.data = const_cast<void*>(static_cast<void const*>(cuKVSeqLens));                                     \
        cumSeqlenK.dynamic_shapes[0] = b + 1;                                                                           \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(                                                                              \
            &(MODULE), &qTensor, &kvTensor, &oTensor, &cumSeqlenK, (WSL),                                              \
            scaleSoftmaxLog2, softmaxScale, scaleOutput, stream);                                                       \
    } while (0)
    // clang-format on

    int32_t ret = -1;
    int32_t constexpr kNoLimit = 1 << 30;
    int32_t const windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;

    if (d == 64)
    {
        if (useSlidingWindow)
        {
            CALL_FMHA_KERNEL(fmha_d64_sw, sKernelModule_d64_sw, windowSizeLeft);
        }
        else
        {
            CALL_FMHA_KERNEL(fmha_d64, sKernelModule_d64, windowSizeLeft);
        }
    }
    else if (d == 128)
    {
        if (useSlidingWindow)
        {
            CALL_FMHA_KERNEL(fmha_d128_sw, sKernelModule_d128_sw, windowSizeLeft);
        }
        else
        {
            CALL_FMHA_KERNEL(fmha_d128, sKernelModule_d128, windowSizeLeft);
        }
    }
    else
    {
        LOG_ERROR("CuTe DSL FMHA: unsupported head_dim=%d (only 64 and 128 supported)", d);
        return;
    }

#undef CALL_FMHA_KERNEL

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL FMHA kernel (d=%d, sw=%s) failed with error code: %d", d,
            useSlidingWindow ? "true" : "false", ret);
    }
}

} // namespace trt_edgellm

#endif // CUTE_DSL_FMHA_ENABLED
