#pragma once

#include <limits.h>
#include <stdint.h>

enum class ContextAttentionMaskType
{
    PADDING,
    CAUSAL,
    SLIDING_WINDOW_CAUSAL
};

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

    AlibiParams(int h)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    AlibiParams(int h, int s, int tp_size, int rank)
        : AlibiParams(h * tp_size)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    // Could be simplified to `int rank` derive the others as `num_heads * rank, s * rank` at
    // runtime, but this makes assumptions about the layout downstream
    // (e.g. downstream may only split across the head dimension, so s would be the full sequence)
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};

typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;

struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    const void* qkv_ptr;
    // The mask to implement drop-out.
    const void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use trick to avoid I2F/F2I in the INT8 kernel.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lengths
    const int* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    AlibiParams alibi_params{};

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int* o_scratch_ptr;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    int h_kv;

    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;

    // is input/output padded
    bool is_s_padded = false;

    // tma descriptors
    cudaTmaDesc tma_desc_q{};
    cudaTmaDesc tma_desc_k{};
    cudaTmaDesc tma_desc_v{};

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        counters = nullptr;
        max_barriers = nullptr;
        sum_barriers = nullptr;
        locks = nullptr;
        max_scratch_ptr = nullptr;
        sum_scratch_ptr = nullptr;
        o_scratch_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        use_int8_scale_max = false;

        h_kv = 0;
        sliding_window_size = INT_MAX;
        is_s_padded = false;

        has_alibi = false;
        alibi_params = AlibiParams{};
    }
};

// flags to control kernel choice
struct Launch_params
{
    // seq_length to select the kernel
    int kernel_s = 0;
    // kv_seq_length to set launch strategies.
    int kernel_kv_s = 0;
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
    int* seqlens = nullptr;
    // number of paged kv blocks for context sequence.
    int blocks_per_context_sequence = 0;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool granular_tiling = false;
    // mask type: padding, causal, sliding_window_causal
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;

    void set_default_kernel_selection_params()
    {
        kernel_s = 0;
        kernel_kv_s = 0;
        force_unroll = false;
        use_tma = false;
        flash_attention = false;
        warp_specialization = false;
        granular_tiling = false;
        attention_mask_type = (attention_mask_type == ContextAttentionMaskType::PADDING)
            ? ContextAttentionMaskType::PADDING
            : ContextAttentionMaskType::CAUSAL;
    }
};