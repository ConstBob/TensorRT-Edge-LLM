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

#pragma once

#include <limits.h>
#include <math.h>
#include <stdint.h>

struct AlibiParams
{
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;
    AlibiParams(int h, float scale_after_alibi = 1.f)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }
    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi = 1.f)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    float scale_after_alibi{};
    // Could be simplified to `int rank` derive the others as `num_heads * rank, s * rank` at
    // runtime, but this makes assumptions about the layout downstream
    // (e.g. downstream may only split across the head dimension, so s would be the full sequence)
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};

// TMA desc
typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;

struct KvBlockArray
{
    using PtrType = int32_t;

    // Current number of sequences
    int32_t mMaxSeqs;
    // Max number of blocks per sequence
    int32_t mMaxBlocksPerSeq;
    // Number of tokens. It must be power of 2.
    int32_t mTokensPerBlock;
    // Exponent of number of tokens with base 2.
    // E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
    int32_t mTokensPerBlockLog2;
    // Table maps logical block idx to the data pointer of k/v cache block pool
    // Shape [B, W, 2, M], where 2 is table for K and V,
    // B is current number of sequences
    // W is beam width
    // M is Max number of blocks per sequence

    // Size of KV cache blocks in bytes (H*D*T*sizeof(DataType))
    int32_t mBytesPerBlock;
    // Pointer to beginning of pool.
    void* mPoolPtr;
    // Pointer to block offsets.
    PtrType* mBlockOffsets;

    KvBlockArray() = default;

    KvBlockArray(
        int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t bytesPerBlock, void* poolPtr)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , mBytesPerBlock{bytesPerBlock}
        , mPoolPtr{poolPtr}
        , mBlockOffsets{nullptr}
    {
        float const tokensPerBlockSeqLog2 = std::log2(mTokensPerBlock);
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

enum class ContextAttentionMaskType
{
    // Mask the padded tokens.
    PADDING = 0,
    // Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL,
    // Causal mask + attend to the specific sliding window or chunk.
    SLIDING_OR_CHUNKED_CAUSAL,
    // The custom mask input.
    CUSTOM_MASK,
};

enum class AttentionInputLayout
{
    // QKV are packed into [B, S, 3, H, D] layout.
    PACKED_QKV = 0,
    // Q has contiguous [Compact_S, H, D] layout, while KV has contiguous [Compact_S, 2, H, D] layout.
    CONTIGUOUS_Q_KV,
    // Q has contiguous [B, S, H, D] layout, while paged KV layout are blocks of indices with shape
    // of [B, 2, Blocks_per_Seq], and the indice indicates the block distance to the pool ptr in
    // global memory.
    Q_PAGED_KV,
    // Q has [B, S, H, D] layout,
    // K has [B, S, H_kv, D] layout,
    // V has [B, S, H_kv, Dv] layout,
    SEPARATE_Q_K_V,
};

struct FusedMultiheadAttentionParamsV2
{
    // The packed QKV matrices.
    void* qkv_ptr;
    // The separate Q matrice.
    void* q_ptr;
    // The separate K matrice.
    void* k_ptr;
    // The separate V matrice.
    void* v_ptr;
    // The separate KV matrice (contiguous KV).
    void* kv_ptr;
    // The separate paged kv cache.
    KvBlockArray paged_kv_cache;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The attention sinks (per head).
    float* attention_sinks;
    // The O matrix (output).
    void* o_ptr;
    // The Softmax stats vector of layout [2, B, S, H], including softmax_sum and softmax_max
    void* softmax_stats_ptr;

    // The stride between rows of Q.
    int64_t q_stride_in_bytes;
    // The stride between rows of K.
    int64_t k_stride_in_bytes;
    // The stride between rows of V.
    int64_t v_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;
    // The stride between rows of softmax_stats_ptr
    int64_t softmax_stats_stride_in_bytes;

    // tma descriptors on device.
    // Either q in packed qkv [B, S, 3, H, D] of separate q layout [B, S, H, D].
    cudaTmaDesc tma_desc_q;
    // Tma descriptors for packed/contiguous/paged kv cache.
    // Kv in packed qkv layout: [B, S, 3, H, D]
    // Contiguous kv layout: [B, 2, H, S, D].
    // Paged kv layout: [UINT32_MAX, H, Tokens_per_block, D].
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v;
    // Tma descriptor for o
    cudaTmaDesc tma_desc_o;

    // Tma load of paged kv cache.
    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, h_kv, h_q_per_kv, s, s_kv, d;
    // The dimension of V. If unset, dv = d.
    int dv = 0;
    // The number of grouped heads in the seqlen dimension.
    int num_grouped_heads = 1;
    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;
    // The chunked attention size in log2 (> 0 means that chunked attention is enabled).
    int log2_chunked_attention_size = 0;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, softcapping_scale_bmm1, scale_softmax, scale_bmm2;

    // The scaling factors in the device memory (required by TRT-LLM + FP8 FMHA).
    uint32_t* scale_bmm1_d;
    uint32_t* scale_bmm2_d;

    // array of length b+1 holding prefix sum of actual q sequence lengths.
    int* cu_q_seqlens;
    // array of length b+1 holding prefix sum of actual kv sequence lengths.
    int* cu_kv_seqlens;
    // array of length b+1 holding prefix sum of actual mask sequence lengths.
    // it might not be the same as cu_q_seqlens as the mask seqlens will be padded.
    int* cu_mask_rows;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    AlibiParams alibi_params{};

    // M tile id counter for dynamic scheduling
    uint32_t* tile_id_counter_ptr;
    uint32_t num_tiles;
    uint32_t num_tiles_per_head;
    bool use_balanced_scheduling;

    // is input/output padded
    bool is_s_padded = false;

    struct SageAttention
    {
        struct Scales
        {
            // ceil(max_seqlen / block_size)
            int max_nblock;
            // The scale of each block, layout: (B, H, max_nblock)
            float* scales;
        } q, k, v;
    } sage;
};

// flags to control kernel choice
struct LaunchParams
{
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
    // total number of q tokens to set tma descriptors
    int total_q_seqlen = 0;
    // total number of kv tokens to set tma descriptors
    int total_kv_seqlen = 0;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool use_granular_tiling = false;
    // causal masking or sliding_or_chunked_causal masking or dense(padding) mask.
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    // the attention input layout.
    AttentionInputLayout attention_input_layout = AttentionInputLayout::PACKED_QKV;
    // enable_attn_logit_softcapping (choose kernels with softcapping_scale_bmm1).
    bool enable_attn_logit_softcapping = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
};