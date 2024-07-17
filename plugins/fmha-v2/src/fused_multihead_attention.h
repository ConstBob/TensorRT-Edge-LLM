/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuda.h>
#include <vector>
#include <fmha/hopper/tma_types.h>
#include <fmha/attention_mask_type.h>
#include <fmha/alibi_params.h>
#include <fused_multihead_attention_utils.h>

using Attention_mask_type = fmha::Attention_mask_type;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace bert {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if USE_DEMO_BERT_PARAMS

//TODO TRT plugins use a different parameter struct taken from the old XMMA fork.
//     Until all cubins in the plugin are replaced with new kernels, we need to conform to that.
#include <fused_multihead_attention_demo_bert_params.h>

#else
struct Fused_multihead_attention_params_base {
    // The QKV matrices.
    void *qkv_ptr;
    // The O matrix (output).
    void *o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void *p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif  // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void *s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif  // defined(STORE_S)


#if defined(DEBUG_HAS_PRINT_BUFFER)
    void *print_ptr;
#endif

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // true: for int8, instead of doing max reduce, use max value encoded in scale factor
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    fmha::AlibiParams alibi_params;

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int *o_scratch_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v1 : Fused_multihead_attention_params_base {
    // The mask to implement drop-out.
    void *packed_mask_ptr;

    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v2 : Fused_multihead_attention_params_base {
    // array of length b+1 holding prefix sum of actual sequence lenghts.
    int *cu_seqlens;

    // tma descriptors on device
    fmha::cudaTmaDesc tma_desc_q;
    fmha::cudaTmaDesc tma_desc_k;
    fmha::cudaTmaDesc tma_desc_v;

    // when we use TMA we need to know the actual seqlens to set the TMA desc. 
    uint32_t *seqlens;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    int h_kv = 0;

    // Max_past_length for attention.
    // Only pay attention to [max(0, seq_length - max_past_length), seq_length).
    int max_past_s = INT_MAX;

    // is input/output padded
    bool is_s_padded = false;
};

#endif

// flags to control kernel choice
struct Fused_multihead_attention_launch_params {
    // flags to control small batch kernel choice
    // true: never unroll
    bool ignore_b1opt = false;
    // true: always unroll
    bool force_unroll = false;
    // use fp32 accumulation
    bool force_fp32_acc = false;
    // the C/32 format
    bool interleaved = false;
    // by default TMA is not used.
    bool use_tma = false;
    // host seqlens to set tma descriptors
    uint32_t* seqlens = nullptr;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool use_granular_tiling = false;
    // causal masking or limited_length_causal masking or dense(padding) mask.
    Attention_mask_type attention_mask_type = Attention_mask_type::PADDING;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace bert

