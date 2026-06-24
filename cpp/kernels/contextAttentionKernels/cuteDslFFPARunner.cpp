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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include "cuteDslFFPARunner.h"
#include "common/logger.h"

#include <cmath>
#include <mutex>

namespace trt_edgellm
{

ffpa_d512_causal_Kernel_Module_t CuteDslFFPARunner::sD512CausalModule{};
bool CuteDslFFPARunner::sLoaded{false};
std::mutex CuteDslFFPARunner::sMutex;

bool CuteDslFFPARunner::canImplement(int32_t headDim, int32_t smVersion)
{
    if (headDim != 512)
    {
        return false;
    }

    switch (smVersion)
    {
    case 80:
    case 86:
    case 87:
    case 89:
    case 100:
    case 101:
    case 110:
    case 120:
    case 121: return true;
    default: return false;
    }
}

bool CuteDslFFPARunner::loadKernelModule()
{
    std::lock_guard<std::mutex> lock{sMutex};
    if (sLoaded)
    {
        return true;
    }

    try
    {
        ffpa_d512_causal_Kernel_Module_Load(&sD512CausalModule);
        sLoaded = true;
        LOG_DEBUG("CuTe DSL FFPA d512 causal kernel module(s) loaded");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load FFPA d512 causal CuTe DSL kernel module.");
        return false;
    }
}

void CuteDslFFPARunner::unloadKernelModule()
{
    std::lock_guard<std::mutex> lock{sMutex};
    if (!sLoaded)
    {
        return;
    }

    ffpa_d512_causal_Kernel_Module_Unload(&sD512CausalModule);
    sD512CausalModule = {};
    sLoaded = false;
}

int CuteDslFFPARunner::run(CuteDslFFPAParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("FFPA CuTe DSL kernel module is not loaded.");
        return -1;
    }

    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.o == nullptr)
    {
        LOG_ERROR("FFPA CuTe DSL kernel received a null tensor pointer.");
        return -1;
    }

    if (params.batchSize <= 0 || params.seqlenQ <= 0 || params.seqlenK <= 0 || params.numQHeads <= 0
        || params.numKVHeads <= 0 || params.headDim <= 0)
    {
        LOG_ERROR("FFPA CuTe DSL kernel received invalid tensor dimensions.");
        return -1;
    }

    if (params.headDim != 512)
    {
        LOG_ERROR("FFPA CuTe DSL kernel only supports head dimension 512.");
        return -1;
    }

    // GQA: the kernel takes num_kv_heads as a runtime argument and derives the
    // group size as numQHeads / numKVHeads internally, so a single AOT kernel
    // serves MHA (numKVHeads == numQHeads) and any GQA group size.  The only
    // constraint is that Q heads partition evenly across K/V heads.
    if (params.numQHeads % params.numKVHeads != 0)
    {
        LOG_ERROR("FFPA d512 causal CuTe DSL kernel requires numQHeads (%d) to be divisible by numKVHeads (%d).",
            params.numQHeads, params.numKVHeads);
        return -1;
    }

    // Strides for Q/O (indexed by numQHeads) and K/V (indexed by numKVHeads).
    int64_t const qStrideBatch
        = static_cast<int64_t>(params.seqlenQ) * static_cast<int64_t>(params.numQHeads) * params.headDim;
    int64_t const qStrideSeq = static_cast<int64_t>(params.numQHeads) * params.headDim;
    int64_t const kStrideBatch
        = static_cast<int64_t>(params.seqlenK) * static_cast<int64_t>(params.numKVHeads) * params.headDim;
    int64_t const kStrideSeq = static_cast<int64_t>(params.numKVHeads) * params.headDim;
    float const softmaxScale
        = params.softmaxScale > 0.0F ? params.softmaxScale : 1.0F / std::sqrt(static_cast<float>(params.headDim));

    // The H-stride is statically D and the D-stride is statically 1, so neither
    // is passed across the ABI.
    ffpa_d512_causal_Tensor_mQ_t qTensor{};
    qTensor.data = const_cast<void*>(params.q);
    qTensor.dynamic_shapes[0] = params.batchSize;
    qTensor.dynamic_shapes[1] = params.seqlenQ;
    qTensor.dynamic_shapes[2] = params.numQHeads;
    qTensor.dynamic_strides[0] = qStrideBatch;
    qTensor.dynamic_strides[1] = qStrideSeq;

    ffpa_d512_causal_Tensor_mK_t kTensor{};
    kTensor.data = const_cast<void*>(params.k);
    kTensor.dynamic_shapes[0] = params.batchSize;
    kTensor.dynamic_shapes[1] = params.seqlenK;
    kTensor.dynamic_shapes[2] = params.numKVHeads;
    kTensor.dynamic_strides[0] = kStrideBatch;
    kTensor.dynamic_strides[1] = kStrideSeq;

    ffpa_d512_causal_Tensor_mV_t vTensor{};
    vTensor.data = const_cast<void*>(params.v);
    vTensor.dynamic_shapes[0] = params.batchSize;
    vTensor.dynamic_shapes[1] = params.seqlenK;
    vTensor.dynamic_shapes[2] = params.numKVHeads;
    vTensor.dynamic_strides[0] = kStrideBatch;
    vTensor.dynamic_strides[1] = kStrideSeq;

    ffpa_d512_causal_Tensor_mO_t oTensor{};
    oTensor.data = params.o;
    oTensor.dynamic_shapes[0] = params.batchSize;
    oTensor.dynamic_shapes[1] = params.seqlenQ;
    oTensor.dynamic_shapes[2] = params.numQHeads;
    oTensor.dynamic_strides[0] = qStrideBatch;
    oTensor.dynamic_strides[1] = qStrideSeq;

    return cute_dsl_ffpa_d512_causal_wrapper(
        &sD512CausalModule, &qTensor, &kTensor, &vTensor, &oTensor, softmaxScale, params.numKVHeads, stream);
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FFPA_ENABLED)
