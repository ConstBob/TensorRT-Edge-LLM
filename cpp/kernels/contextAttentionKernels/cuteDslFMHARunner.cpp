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

#ifdef CUTE_DSL_FMHA_ENABLED

#include "cuteDslFMHARunner.h"

#include "attentionScaleUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "cuteDslTensorDescriptors.h"

#include <climits>
#include <cmath>

namespace trt_edgellm
{

namespace
{

bool isSupportedBlackwellFmha(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

} // namespace

// =====================================================================
// Static member initialization
// =====================================================================

// LLM (FP16)
fmha_d64_Kernel_Module_t CuteDslFMHARunner::sLLM_d64 = {};
fmha_d128_Kernel_Module_t CuteDslFMHARunner::sLLM_d128 = {};
fmha_d256_Kernel_Module_t CuteDslFMHARunner::sLLM_d256 = {};
fmha_d64_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw = {};
fmha_d128_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw = {};
fmha_d256_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw = {};
// LLM skip-softmax (BLASST, FP16 causal)
fmha_d64_skipsoftmax_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_skipsoftmax = {};
fmha_d128_skipsoftmax_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_skipsoftmax = {};
// LLM (FP8 input, FP16 output)
fmha_d64_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_fp8 = {};
fmha_d128_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_fp8 = {};
fmha_d256_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_fp8 = {};
fmha_d64_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_fp8 = {};
fmha_d128_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_fp8 = {};
fmha_d256_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_fp8 = {};
// LLM paged KV cache (FP16)
fmha_d64_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_paged = {};
fmha_d128_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_paged = {};
fmha_d256_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_paged = {};
fmha_d256_dense_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_dense_paged = {};
fmha_d512_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_paged = {};
fmha_d512_dense_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_dense_paged = {};
fmha_d64_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_paged = {};
fmha_d128_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_paged = {};
fmha_d256_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_paged = {};
fmha_d512_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_sw_paged = {};
// LLM paged KV cache (FP8 input, FP16 output)
fmha_d64_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_paged_fp8 = {};
fmha_d128_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_paged_fp8 = {};
fmha_d256_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_paged_fp8 = {};
fmha_d256_dense_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_dense_paged_fp8 = {};
fmha_d512_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_paged_fp8 = {};
fmha_d512_dense_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_dense_paged_fp8 = {};
fmha_d64_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_paged_fp8 = {};
fmha_d128_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_paged_fp8 = {};
fmha_d256_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_paged_fp8 = {};
fmha_d512_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d512_sw_paged_fp8 = {};
bool CuteDslFMHARunner::sLLMLoaded = false;
std::mutex CuteDslFMHARunner::sLLMMutex;

// ViT
vit_fmha_d64_Kernel_Module_t CuteDslFMHARunner::sViT_d64 = {};
vit_fmha_d72_Kernel_Module_t CuteDslFMHARunner::sViT_d72 = {};
vit_fmha_d80_Kernel_Module_t CuteDslFMHARunner::sViT_d80 = {};
vit_fmha_d128_Kernel_Module_t CuteDslFMHARunner::sViT_d128 = {};
bool CuteDslFMHARunner::sViTLoaded = false;
std::mutex CuteDslFMHARunner::sViTMutex;

// =====================================================================
// Kernel module loading
// =====================================================================

bool CuteDslFMHARunner::loadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock(sLLMMutex);
    if (sLLMLoaded)
    {
        return true;
    }
    try
    {
        fmha_d64_Kernel_Module_Load(&sLLM_d64);
        fmha_d128_Kernel_Module_Load(&sLLM_d128);
        fmha_d256_Kernel_Module_Load(&sLLM_d256);
        fmha_d64_sw_Kernel_Module_Load(&sLLM_d64_sw);
        fmha_d128_sw_Kernel_Module_Load(&sLLM_d128_sw);
        fmha_d256_sw_Kernel_Module_Load(&sLLM_d256_sw);
        fmha_d64_skipsoftmax_Kernel_Module_Load(&sLLM_d64_skipsoftmax);
        fmha_d128_skipsoftmax_Kernel_Module_Load(&sLLM_d128_skipsoftmax);
        fmha_d64_fp8_Kernel_Module_Load(&sLLM_d64_fp8);
        fmha_d128_fp8_Kernel_Module_Load(&sLLM_d128_fp8);
        fmha_d256_fp8_Kernel_Module_Load(&sLLM_d256_fp8);
        fmha_d64_sw_fp8_Kernel_Module_Load(&sLLM_d64_sw_fp8);
        fmha_d128_sw_fp8_Kernel_Module_Load(&sLLM_d128_sw_fp8);
        fmha_d256_sw_fp8_Kernel_Module_Load(&sLLM_d256_sw_fp8);
        fmha_d64_paged_Kernel_Module_Load(&sLLM_d64_paged);
        fmha_d128_paged_Kernel_Module_Load(&sLLM_d128_paged);
        fmha_d256_paged_Kernel_Module_Load(&sLLM_d256_paged);
        fmha_d256_dense_paged_Kernel_Module_Load(&sLLM_d256_dense_paged);
        fmha_d512_paged_Kernel_Module_Load(&sLLM_d512_paged);
        fmha_d512_dense_paged_Kernel_Module_Load(&sLLM_d512_dense_paged);
        fmha_d64_sw_paged_Kernel_Module_Load(&sLLM_d64_sw_paged);
        fmha_d128_sw_paged_Kernel_Module_Load(&sLLM_d128_sw_paged);
        fmha_d256_sw_paged_Kernel_Module_Load(&sLLM_d256_sw_paged);
        fmha_d512_sw_paged_Kernel_Module_Load(&sLLM_d512_sw_paged);
        fmha_d64_paged_fp8_Kernel_Module_Load(&sLLM_d64_paged_fp8);
        fmha_d128_paged_fp8_Kernel_Module_Load(&sLLM_d128_paged_fp8);
        fmha_d256_paged_fp8_Kernel_Module_Load(&sLLM_d256_paged_fp8);
        fmha_d256_dense_paged_fp8_Kernel_Module_Load(&sLLM_d256_dense_paged_fp8);
        fmha_d512_paged_fp8_Kernel_Module_Load(&sLLM_d512_paged_fp8);
        fmha_d512_dense_paged_fp8_Kernel_Module_Load(&sLLM_d512_dense_paged_fp8);
        fmha_d64_sw_paged_fp8_Kernel_Module_Load(&sLLM_d64_sw_paged_fp8);
        fmha_d128_sw_paged_fp8_Kernel_Module_Load(&sLLM_d128_sw_paged_fp8);
        fmha_d256_sw_paged_fp8_Kernel_Module_Load(&sLLM_d256_sw_paged_fp8);
        fmha_d512_sw_paged_fp8_Kernel_Module_Load(&sLLM_d512_sw_paged_fp8);
        sLLMLoaded = true;
        LOG_DEBUG("CuTe DSL LLM FMHA kernel modules loaded (FP16 + FP8 + paged)");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL LLM FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHARunner::unloadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock(sLLMMutex);
    if (sLLMLoaded)
    {
        fmha_d64_Kernel_Module_Unload(&sLLM_d64);
        fmha_d128_Kernel_Module_Unload(&sLLM_d128);
        fmha_d256_Kernel_Module_Unload(&sLLM_d256);
        fmha_d64_sw_Kernel_Module_Unload(&sLLM_d64_sw);
        fmha_d128_sw_Kernel_Module_Unload(&sLLM_d128_sw);
        fmha_d256_sw_Kernel_Module_Unload(&sLLM_d256_sw);
        fmha_d64_skipsoftmax_Kernel_Module_Unload(&sLLM_d64_skipsoftmax);
        fmha_d128_skipsoftmax_Kernel_Module_Unload(&sLLM_d128_skipsoftmax);
        fmha_d64_fp8_Kernel_Module_Unload(&sLLM_d64_fp8);
        fmha_d128_fp8_Kernel_Module_Unload(&sLLM_d128_fp8);
        fmha_d256_fp8_Kernel_Module_Unload(&sLLM_d256_fp8);
        fmha_d64_sw_fp8_Kernel_Module_Unload(&sLLM_d64_sw_fp8);
        fmha_d128_sw_fp8_Kernel_Module_Unload(&sLLM_d128_sw_fp8);
        fmha_d256_sw_fp8_Kernel_Module_Unload(&sLLM_d256_sw_fp8);
        fmha_d64_paged_Kernel_Module_Unload(&sLLM_d64_paged);
        fmha_d128_paged_Kernel_Module_Unload(&sLLM_d128_paged);
        fmha_d256_paged_Kernel_Module_Unload(&sLLM_d256_paged);
        fmha_d256_dense_paged_Kernel_Module_Unload(&sLLM_d256_dense_paged);
        fmha_d512_paged_Kernel_Module_Unload(&sLLM_d512_paged);
        fmha_d512_dense_paged_Kernel_Module_Unload(&sLLM_d512_dense_paged);
        fmha_d64_sw_paged_Kernel_Module_Unload(&sLLM_d64_sw_paged);
        fmha_d128_sw_paged_Kernel_Module_Unload(&sLLM_d128_sw_paged);
        fmha_d256_sw_paged_Kernel_Module_Unload(&sLLM_d256_sw_paged);
        fmha_d512_sw_paged_Kernel_Module_Unload(&sLLM_d512_sw_paged);
        fmha_d64_paged_fp8_Kernel_Module_Unload(&sLLM_d64_paged_fp8);
        fmha_d128_paged_fp8_Kernel_Module_Unload(&sLLM_d128_paged_fp8);
        fmha_d256_paged_fp8_Kernel_Module_Unload(&sLLM_d256_paged_fp8);
        fmha_d256_dense_paged_fp8_Kernel_Module_Unload(&sLLM_d256_dense_paged_fp8);
        fmha_d512_paged_fp8_Kernel_Module_Unload(&sLLM_d512_paged_fp8);
        fmha_d512_dense_paged_fp8_Kernel_Module_Unload(&sLLM_d512_dense_paged_fp8);
        fmha_d64_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d64_sw_paged_fp8);
        fmha_d128_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d128_sw_paged_fp8);
        fmha_d256_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d256_sw_paged_fp8);
        fmha_d512_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d512_sw_paged_fp8);
        sLLMLoaded = false;
    }
}

bool CuteDslFMHARunner::loadViTKernelModule()
{
    std::lock_guard<std::mutex> lock(sViTMutex);
    if (sViTLoaded)
    {
        return true;
    }
    try
    {
        vit_fmha_d64_Kernel_Module_Load(&sViT_d64);
        vit_fmha_d72_Kernel_Module_Load(&sViT_d72);
        vit_fmha_d80_Kernel_Module_Load(&sViT_d80);
        vit_fmha_d128_Kernel_Module_Load(&sViT_d128);
        sViTLoaded = true;
        LOG_DEBUG("CuTe DSL ViT FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL ViT FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHARunner::unloadViTKernelModule()
{
    std::lock_guard<std::mutex> lock(sViTMutex);
    if (sViTLoaded)
    {
        vit_fmha_d64_Kernel_Module_Unload(&sViT_d64);
        vit_fmha_d72_Kernel_Module_Unload(&sViT_d72);
        vit_fmha_d80_Kernel_Module_Unload(&sViT_d80);
        vit_fmha_d128_Kernel_Module_Unload(&sViT_d128);
        sViTLoaded = false;
    }
}

bool CuteDslFMHARunner::canImplement(int32_t headSize, int32_t smVersion)
{
    bool const supportedHeadSize = headSize == 64 || headSize == 128 || headSize == 256 || headSize == 512;
    return isSupportedBlackwellFmha(smVersion) && supportedHeadSize;
}

bool CuteDslFMHARunner::canImplementViT(int32_t headSize, int32_t smVersion)
{
    return isSupportedBlackwellFmha(smVersion)
        && (headSize == 64 || headSize == 72 || headSize == 80 || headSize == 128);
}

// =====================================================================
// Constructors
// =====================================================================

CuteDslFMHARunner::CuteDslFMHARunner(
    int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize, int32_t seqLenQ, int32_t kvCacheCapacity)
    : mBatchSize(batchSize)
    , mSeqLenQ(seqLenQ)
    , mKVCacheCapacity(kvCacheCapacity)
    , mNumHeadsQ(numQHeads)
    , mNumHeadsK(numKVHeads)
    , mHeadDim(headDim)
{
}

namespace
{

using cutedsl::makeCuSeqLenTensor;
using cutedsl::makePackedTensor;
using cutedsl::makeStridedTensor;
using cutedsl::WrapperArgT;
using cutedsl::WrapperArity;

//! Everything the dense LLM descriptors need, gathered once per run() call.
struct LlmFmhaParams
{
    void const* qPtr{};
    void const* kvPtr{};
    void* oPtr{};
    int32_t const* cuKVSeqLens{};
    int32_t batchSize{};
    int32_t seqLenQ{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t kvCacheCapacity{};
    int32_t windowSizeLeft{};
    float attentionScale{};
    float scaleQ{};
    float scaleK{};
    float scaleV{};
    float invScaleO{};
    cudaStream_t stream{};
};

//! Launch a dense (combined KV cache) LLM FMHA variant. The exported signature is
//!   (module, q_tensor, kv_cache, o_tensor, cum_seqlen_k, window_size_left, attention_scale,
//!    scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream, trailing...)
//! and @p trailing is empty for every variant that does not append extra runtime scalars.
template <auto Wrapper, class... Trailing>
int32_t callLlmFmha(WrapperArgT<0, decltype(Wrapper)>& module, LlmFmhaParams const& params, Trailing... trailing)
{
    static_assert(WrapperArity<decltype(Wrapper)>::value == 13 + sizeof...(Trailing),
        "callLlmFmha: not a dense LLM FMHA wrapper (module, q_tensor, kv_cache, o_tensor, cum_seqlen_k, "
        "window_size_left, attention_scale, scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream).");

    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(Wrapper)>>(
        params.qPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto kvTensor = makePackedTensor<WrapperArgT<2, decltype(Wrapper)>>(
        params.kvPtr, {params.batchSize, 2, params.numKVHeads, params.kvCacheCapacity, params.headDim});
    auto oTensor = makePackedTensor<WrapperArgT<3, decltype(Wrapper)>>(
        params.oPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto cumSeqlenK = makeCuSeqLenTensor<WrapperArgT<4, decltype(Wrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    return Wrapper(&module, &qTensor, &kvTensor, &oTensor, &cumSeqlenK, params.windowSizeLeft, params.attentionScale,
        params.scaleQ, params.scaleK, params.scaleV, params.invScaleO, getDeviceMultiProcessorCount(), params.stream,
        trailing...);
}

//! Everything the paged LLM descriptors need, gathered once per runPaged() call.
struct LlmFmhaPagedParams
{
    void const* qPtr{};
    void const* pagedKVPoolPtr{};
    int32_t const* kvCachePageList{};
    void* oPtr{};
    int32_t const* cuKVSeqLens{};
    int32_t batchSize{};
    int32_t seqLenQ{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t numPages{};
    int32_t maxPagesPerSeq{};
    int32_t tokensPerPage{};
    int32_t windowSizeLeft{};
    float attentionScale{};
    float scaleQ{};
    float scaleK{};
    float scaleV{};
    float invScaleO{};
    cudaStream_t stream{};
};

//! Launch a paged LLM FMHA variant. The exported signature matches callLlmFmha() except that
//! kv_cache is replaced by the (kv_cache_pool, kv_cache_page_list) pair.
template <auto Wrapper, class... Trailing>
int32_t callLlmFmhaPaged(
    WrapperArgT<0, decltype(Wrapper)>& module, LlmFmhaPagedParams const& params, Trailing... trailing)
{
    static_assert(WrapperArity<decltype(Wrapper)>::value == 14 + sizeof...(Trailing),
        "callLlmFmhaPaged: not a paged LLM FMHA wrapper (module, q_tensor, kv_cache_pool, kv_cache_page_list, "
        "o_tensor, cum_seqlen_k, window_size_left, attention_scale, scale_q, scale_k, scale_v, inv_scale_o, sm_count, "
        "stream).");

    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(Wrapper)>>(
        params.qPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});

    // The physical paged KV pool layout is fixed to NHD [numPages, tokensPerPage, H_kv, D]. CuTe DSL still
    // receives logical shape [numPages, H_kv, tokensPerPage, D], and the permutation lives in these strides,
    // so this is the one descriptor here that is not packed row-major.
    auto kvPoolTensor = makeStridedTensor<WrapperArgT<2, decltype(Wrapper)>>(params.pagedKVPoolPtr,
        {params.numPages, params.numKVHeads, params.tokensPerPage, params.headDim},
        {static_cast<int64_t>(params.numKVHeads) * params.tokensPerPage * params.headDim,
            static_cast<int64_t>(params.headDim), static_cast<int64_t>(params.numKVHeads) * params.headDim});

    auto pageListTensor = makePackedTensor<WrapperArgT<3, decltype(Wrapper)>>(
        params.kvCachePageList, {params.batchSize, 2, params.maxPagesPerSeq});
    auto oTensor = makePackedTensor<WrapperArgT<4, decltype(Wrapper)>>(
        params.oPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto cumSeqlenK = makeCuSeqLenTensor<WrapperArgT<5, decltype(Wrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    return Wrapper(&module, &qTensor, &kvPoolTensor, &pageListTensor, &oTensor, &cumSeqlenK, params.windowSizeLeft,
        params.attentionScale, params.scaleQ, params.scaleK, params.scaleV, params.invScaleO,
        getDeviceMultiProcessorCount(), params.stream, trailing...);
}

//! Everything the ViT descriptors need, gathered once per ViT run() call.
struct VitFmhaParams
{
    void const* qPtr{};
    void const* kPtr{};
    void const* vPtr{};
    void* oPtr{};
    int32_t const* cuSeqLens{};
    int32_t totalSeqLen{};
    int32_t numHeads{};
    int32_t headDim{};
    int32_t maxSeqLen{};
    int32_t batchSize{};
    float scaleSoftmaxLog2{};
    float attentionScale{};
    float scaleOutput{};
    cudaStream_t stream{};
};

//! Launch a ViT FMHA variant over packed varlen [total_S, H, D] Q/K/V.
template <auto Wrapper>
int32_t callVitFmha(WrapperArgT<0, decltype(Wrapper)>& module, VitFmhaParams const& params)
{
    static_assert(WrapperArity<decltype(Wrapper)>::value == 12,
        "callVitFmha: not a ViT FMHA wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cu_seqlens, "
        "max_seqlen, scale_softmax_log2, scale_softmax, scale_output, sm_count, stream).");

    int32_t const shape[] = {params.totalSeqLen, params.numHeads, params.headDim};
    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(Wrapper)>>(params.qPtr, shape);
    auto kTensor = makePackedTensor<WrapperArgT<2, decltype(Wrapper)>>(params.kPtr, shape);
    auto vTensor = makePackedTensor<WrapperArgT<3, decltype(Wrapper)>>(params.vPtr, shape);
    auto oTensor = makePackedTensor<WrapperArgT<4, decltype(Wrapper)>>(params.oPtr, shape);
    auto cuSeqlensTensor
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(Wrapper)>>(params.cuSeqLens, params.batchSize + 1);

    return Wrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &cuSeqlensTensor, params.maxSeqLen,
        params.scaleSoftmaxLog2, params.attentionScale, params.scaleOutput, getDeviceMultiProcessorCount(),
        params.stream);
}

} // namespace

// =====================================================================
// LLM run: batched Q + combined KV cache (FP16 / FP8→FP16 / FP8→FP8)
// =====================================================================

void CuteDslFMHARunner::run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens,
    cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input, float qScale, float kScale,
    float vScale, bool enableSkipSoftmax)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel module not loaded.");
        return;
    }

    validateAttentionScale(attentionScale);

    int32_t const headDim = mHeadDim;
    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);
    int32_t constexpr kNoLimit = 1 << 30;

    LlmFmhaParams params{};
    params.qPtr = qPtr;
    params.kvPtr = kvPtr;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsK;
    params.headDim = headDim;
    params.kvCacheCapacity = mKVCacheCapacity;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;
    params.attentionScale = attentionScale;
    params.scaleQ = qScale;
    params.scaleK = kScale;
    params.scaleV = vScale;
    params.invScaleO = 1.0F;
    params.stream = stream;

    int32_t ret = -1;

    if (enableSkipSoftmax)
    {
        if (fp8Input || useSlidingWindow)
        {
            LOG_ERROR("CuTe DSL LLM FMHA: skip-softmax variant is FP16 causal only (fp8Input=%s, sw=%s)",
                fp8Input ? "true" : "false", useSlidingWindow ? "true" : "false");
            return;
        }
        switch (headDim)
        {
        case 64: ret = callLlmFmha<cute_dsl_fmha_d64_skipsoftmax_wrapper>(sLLM_d64_skipsoftmax, params); break;
        case 128: ret = callLlmFmha<cute_dsl_fmha_d128_skipsoftmax_wrapper>(sLLM_d128_skipsoftmax, params); break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return;
        }
    }
    else if (fp8Input)
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d64_sw_fp8_wrapper>(sLLM_d64_sw_fp8, params)
                                   : callLlmFmha<cute_dsl_fmha_d64_fp8_wrapper>(sLLM_d64_fp8, params);
            break;
        case 128:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d128_sw_fp8_wrapper>(sLLM_d128_sw_fp8, params)
                                   : callLlmFmha<cute_dsl_fmha_d128_fp8_wrapper>(sLLM_d128_fp8, params);
            break;
        case 256:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d256_sw_fp8_wrapper>(sLLM_d256_sw_fp8, params)
                                   : callLlmFmha<cute_dsl_fmha_d256_fp8_wrapper>(sLLM_d256_fp8, params);
            break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return;
        }
    }
    else
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d64_sw_wrapper>(sLLM_d64_sw, params)
                                   : callLlmFmha<cute_dsl_fmha_d64_wrapper>(sLLM_d64, params);
            break;
        case 128:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d128_sw_wrapper>(sLLM_d128_sw, params)
                                   : callLlmFmha<cute_dsl_fmha_d128_wrapper>(sLLM_d128, params);
            break;
        case 256:
            ret = useSlidingWindow ? callLlmFmha<cute_dsl_fmha_d256_sw_wrapper>(sLLM_d256_sw, params)
                                   : callLlmFmha<cute_dsl_fmha_d256_wrapper>(sLLM_d256, params);
            break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return;
        }
    }

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel (d=%d, sw=%s, fp8in=%s) failed with error code: %d", headDim,
            useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false", ret);
    }
}

void CuteDslFMHARunner::runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList,
    void* oPtr, int32_t const* cuKVSeqLens, int32_t numPages, int32_t maxPagesPerSeq, int32_t tokensPerPage,
    nvinfer1::DataType kvDataType, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input,
    float qScale, float kScale, float vScale, bool isCausal)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel module not loaded.");
        return;
    }

    check::check(qPtr != nullptr, "CuTe DSL paged FMHA qPtr must not be null.");
    check::check(pagedKVPoolPtr != nullptr, "CuTe DSL paged FMHA KV pool must not be null.");
    check::check(kvCachePageList != nullptr, "CuTe DSL paged FMHA page list must not be null.");
    check::check(oPtr != nullptr, "CuTe DSL paged FMHA oPtr must not be null.");
    check::check(cuKVSeqLens != nullptr, "CuTe DSL paged FMHA cuKVSeqLens must not be null.");
    check::check(numPages > 0 && maxPagesPerSeq > 0 && tokensPerPage > 0,
        "CuTe DSL paged FMHA requires positive numPages/maxPagesPerSeq/tokensPerPage.");
    // Direct paged CuTe DSL keeps the existing TMA load pipeline: each logical K/V tile maps to one physical page.
    // These AOT variants use tile_N=128, so smaller pages would require stitching one tile from multiple pages.
    check::check(tokensPerPage == 128,
        "CuTe DSL direct paged FMHA requires tokensPerPage == 128 because one K/V TMA tile maps to one page.");
    check::check(mKVCacheCapacity == maxPagesPerSeq * tokensPerPage,
        "CuTe DSL paged FMHA runner capacity must equal maxPagesPerSeq * tokensPerPage.");
    check::check(kvDataType == nvinfer1::DataType::kHALF || kvDataType == nvinfer1::DataType::kFP8,
        "CuTe DSL paged FMHA supports FP16 or FP8 KV cache.");
    check::check((kvDataType == nvinfer1::DataType::kFP8) == fp8Input,
        "CuTe DSL paged FMHA requires fp8Input to match the paged KV cache dtype.");
    check::check(isCausal || slidingWindowSize == INT_MAX,
        "CuTe DSL dense non-causal paged FMHA does not support sliding-window masking.");
    check::check(isCausal || mHeadDim == 256 || mHeadDim == 512,
        "CuTe DSL dense non-causal paged FMHA currently supports head_dim=256 or 512 only.");

    int32_t const headDim = mHeadDim;
    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);
    int32_t constexpr kNoLimit = 1 << 30;

    LlmFmhaPagedParams params{};
    params.qPtr = qPtr;
    params.pagedKVPoolPtr = pagedKVPoolPtr;
    params.kvCachePageList = kvCachePageList;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsK;
    params.headDim = headDim;
    params.numPages = numPages;
    params.maxPagesPerSeq = maxPagesPerSeq;
    params.tokensPerPage = tokensPerPage;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;
    params.attentionScale = attentionScale;
    params.scaleQ = qScale;
    params.scaleK = kScale;
    params.scaleV = vScale;
    params.invScaleO = 1.0F;
    params.stream = stream;

    int32_t ret = -1;

    if (!isCausal)
    {
        // The check::check above already narrowed the dense non-causal path to head_dim 256 or 512.
        if (fp8Input)
        {
            ret = headDim == 256
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_dense_paged_fp8_wrapper>(sLLM_d256_dense_paged_fp8, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_dense_paged_fp8_wrapper>(sLLM_d512_dense_paged_fp8, params);
        }
        else
        {
            ret = headDim == 256
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_dense_paged_wrapper>(sLLM_d256_dense_paged, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_dense_paged_wrapper>(sLLM_d512_dense_paged, params);
        }
    }
    else if (fp8Input)
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d64_sw_paged_fp8_wrapper>(sLLM_d64_sw_paged_fp8, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d64_paged_fp8_wrapper>(sLLM_d64_paged_fp8, params);
            break;
        case 128:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d128_sw_paged_fp8_wrapper>(sLLM_d128_sw_paged_fp8, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d128_paged_fp8_wrapper>(sLLM_d128_paged_fp8, params);
            break;
        case 256:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_sw_paged_fp8_wrapper>(sLLM_d256_sw_paged_fp8, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d256_paged_fp8_wrapper>(sLLM_d256_paged_fp8, params);
            break;
        case 512:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d512_sw_paged_fp8_wrapper>(sLLM_d512_sw_paged_fp8, params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_paged_fp8_wrapper>(sLLM_d512_paged_fp8, params);
            break;
        default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim); return;
        }
    }
    else
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow ? callLlmFmhaPaged<cute_dsl_fmha_d64_sw_paged_wrapper>(sLLM_d64_sw_paged, params)
                                   : callLlmFmhaPaged<cute_dsl_fmha_d64_paged_wrapper>(sLLM_d64_paged, params);
            break;
        case 128:
            ret = useSlidingWindow ? callLlmFmhaPaged<cute_dsl_fmha_d128_sw_paged_wrapper>(sLLM_d128_sw_paged, params)
                                   : callLlmFmhaPaged<cute_dsl_fmha_d128_paged_wrapper>(sLLM_d128_paged, params);
            break;
        case 256:
            ret = useSlidingWindow ? callLlmFmhaPaged<cute_dsl_fmha_d256_sw_paged_wrapper>(sLLM_d256_sw_paged, params)
                                   : callLlmFmhaPaged<cute_dsl_fmha_d256_paged_wrapper>(sLLM_d256_paged, params);
            break;
        case 512:
            ret = useSlidingWindow ? callLlmFmhaPaged<cute_dsl_fmha_d512_sw_paged_wrapper>(sLLM_d512_sw_paged, params)
                                   : callLlmFmhaPaged<cute_dsl_fmha_d512_paged_wrapper>(sLLM_d512_paged, params);
            break;
        default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim); return;
        }
    }

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL paged LLM FMHA kernel (d=%d, causal=%s, sw=%s, fp8in=%s) failed with error code: %d",
            headDim, isCausal ? "true" : "false", useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false",
            ret);
    }
}

// =====================================================================
// ViT run: packed varlen separate Q/K/V
// =====================================================================

void CuteDslFMHARunner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
    int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale)
{
    if (!sViTLoaded)
    {
        LOG_ERROR("CuTe DSL ViT FMHA kernel module not loaded.");
        return;
    }

    validateAttentionScale(attentionScale);

    int32_t const headDim = mHeadDim;

    VitFmhaParams params{};
    params.qPtr = qPtr;
    params.kPtr = kPtr;
    params.vPtr = vPtr;
    params.oPtr = oPtr;
    params.cuSeqLens = cuSeqLens;
    params.totalSeqLen = totalSeqLen;
    params.numHeads = mNumHeadsQ;
    params.headDim = headDim;
    params.maxSeqLen = maxSeqLen;
    params.batchSize = batchSize;
    params.scaleSoftmaxLog2 = attentionScale * static_cast<float>(M_LOG2E);
    params.attentionScale = attentionScale;
    params.scaleOutput = 1.0F;
    params.stream = stream;

    int32_t ret = -1;

    switch (headDim)
    {
    case 64: ret = callVitFmha<cute_dsl_vit_fmha_d64_wrapper>(sViT_d64, params); break;
    case 72: ret = callVitFmha<cute_dsl_vit_fmha_d72_wrapper>(sViT_d72, params); break;
    case 80: ret = callVitFmha<cute_dsl_vit_fmha_d80_wrapper>(sViT_d80, params); break;
    case 128: ret = callVitFmha<cute_dsl_vit_fmha_d128_wrapper>(sViT_d128, params); break;
    default: LOG_ERROR("CuTe DSL ViT FMHA: unsupported head_dim=%d", headDim); return;
    }

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL ViT FMHA kernel (d=%d) failed with error code: %d", headDim, ret);
    }
}

} // namespace trt_edgellm

#endif // CUTE_DSL_FMHA_ENABLED
