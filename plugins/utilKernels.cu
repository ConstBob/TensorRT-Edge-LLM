/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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


#include "utilKernels.h"
#include "pluginUtils.h"
namespace
{

inline __device__ uint32_t float2_to_half2(float2 f)
{
    union
    {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
    return tmp.u32;
}

inline __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}


inline __device__ float2 half2_to_float2(uint32_t v)
{
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

inline __device__ float2 rotary_embedding_coefficient(
    const int zid, const int rot_embed_dim, const float base, const float scale, const int t_step)
{
    const float inv_freq = (t_step * scale) / pow(base, zid / (float) rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
{
    float2 fv = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

inline __device__ void apply_rotary_embedding(
    float2& q, float2& k, int tid, int rot_embed_dim, float base, float scale, int t_step)
{
    if (2 * tid >= rot_embed_dim)
    {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
    q = rotary_embedding_transform(q, coef);
    k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(
    uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, float base, float scale, int t_step)
{
    if (2 * tid >= rot_embed_dim)
    {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
    q = rotary_embedding_transform(q, coef);
    k = rotary_embedding_transform(k, coef);
}

template <typename T>
struct Vec_t
{
    static constexpr int size = 0;
};

template <>
struct Vec_t<float>
{
    using Type = float2;
    static constexpr int size = 2;
};

template <>
struct Vec_t<half>
{
    using Type = uint32_t;
    static constexpr int size = 2;
};

}

template <typename T, bool IsGenerate>
__global__ void applyBiasRopeUpdateKVCache(T* QKV, T* Q, T* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    float rotary_embedding_base, float rotary_embedding_scale)
{
    // The kernel take QKV tensor, apply rotary embedding, and
    //      1. At context phase, write qkv back to original QKV tensor and fill in KVcache
    //      2. At generation phase, write q vector to Q tensor and kv vector to KVcache. 

    //   QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                   ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)

    // We currently only handle case with batch size = 1.
    // TODO: Need a proper way to deal with un-equal batch of sequences.

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t = typename Vec_t<T>::Type;
    const int token_idx = blockIdx.x;

    constexpr int k_BatchIndex = 0;

    // We only use the same tensor data and kv-cache type now.
    // TODO: Explore the need if we want to support different data and kvcache type.
    using T_cache = T;
    using T_dst = T_cache;

    // In generate phase, there is one token to generate per sequence and the token index 
    // in the sequence is actual_seq_len - 1. In generation phase, the value is token_idx.
    const int actual_seq_len = seq_lens[k_BatchIndex];
    const int token_idx_in_seq = IsGenerate ? actual_seq_len - 1 : token_idx;

    const int head_idx = blockIdx.y;

    // Each thread block will handle D elements, each thread is assigned two elements.
    const int tidx = threadIdx.x;

    const int hidden_size = head_num * size_per_head;
    const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
    const int qheads_per_kv_head = head_num / kv_head_num;
    const int kv_head_idx = head_idx / qheads_per_kv_head;
    const int hidden_idx_kv = kv_head_idx * size_per_head + tidx * vec_size;

    // Totoal elements for a token.
    const int n = (head_num + 2 * kv_head_num) * size_per_head;

    // offset the total number of data elements in Q heads.
    const int src_k_offset = hidden_size;
    // offset the totoal number of data elements in K heads.
    const int src_v_offset = hidden_size + kv_head_num * size_per_head;

    // NOTE: q has seq len excluding prefix prompt
    // head_num == kv_head_num:
    //   src QKV: [batch, time, 3, head_num, size_per_head]
    // head_num != kv_head_num:
    //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
    const int src_q_idx = token_idx * n + hidden_idx;
    const int src_k_idx = token_idx * n + src_k_offset + hidden_idx_kv;
    const int src_v_idx = token_idx * n + src_v_offset + hidden_idx_kv;

    Vec_t q, k, v;

    // load q,k,v
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);
    k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
    v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

    // rotary position encoding will apply transformation to pair of data based on token_index in the sequence and
    // position of pair of data in the D dimension. From original paper, theta_i = 10000^(-2(i)/d) where i = tidx.
    apply_rotary_embedding(
        q, k, tidx, size_per_head, rotary_embedding_base, rotary_embedding_scale, token_idx_in_seq);

    // KV-cache is of shape [B, 2, H, S, D] where S is the capacity of the kvcache buffer (max total context length).
    // K shape is [B, 1, H, S, D]
    // V shape is [B, 1, H, S, D]
    int const bytes_per_seq = kv_head_num * kv_cache_capacity * size_per_head;  // max bytes of K or V per (total) sequence
    int const offset_sequence = k_BatchIndex * (2 * bytes_per_seq);            // offset of the currect sequence in linear buffer. 
    int const offset_local_kv = kv_head_idx * kv_cache_capacity * size_per_head + token_idx_in_seq * size_per_head + tidx * vec_size;

    int const cache_offset_k = offset_sequence + offset_local_kv;
    int const cache_offset_v = offset_sequence + bytes_per_seq + offset_local_kv;

    if constexpr (IsGenerate)
    {
        int const offset_q = token_idx * head_num * size_per_head + hidden_idx;
        *reinterpret_cast<Vec_t*>(&Q[offset_q]) = q;
    }
    else
    {
        // In context phase write back to QKV tensor.
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
    }
    // only one of the qheads_per_kv_head instance need to write kv
    if (head_idx == (kv_head_idx * qheads_per_kv_head))
    {
        if constexpr (!IsGenerate)
        {
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }

        *reinterpret_cast<Vec_t*>(&kvCacheBuffer[cache_offset_k]) = k;
        *reinterpret_cast<Vec_t*>(&kvCacheBuffer[cache_offset_v]) = v;
    }
}

template <typename T, bool IsGenerate>
void dispatchApplyRopeUpdateKV(T* QKV, T* Q, T* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    float rotary_embedding_base, float rotary_embedding_scale,
    const int token_to_process, cudaStream_t stream)
{
    check(QKV != nullptr && kvCacheBuffer != nullptr && seq_lens != nullptr, "Data pointers of qkv, kvcache, and sequence length shall be valid");
    check(token_to_process > 0, "Number of tokens to process shall be valid.");
    check(size_per_head % 32 == 0, "Size per head shall be multiple of 32");
    if constexpr (IsGenerate)
    {
        check(Q != nullptr, "In generation phase Q pointer shall be valid.");
    }
    // To implement rotary embeddings, each thread processes two QKV elems:
    dim3 block(size_per_head / Vec_t<T>::size);
    dim3 grid(token_to_process, head_num);

    // Basic rope only involve pair of data next to each other which doesn't need shared memory.
    size_t const smem_size = 0;
    applyBiasRopeUpdateKVCache<T, IsGenerate><<<grid, block, smem_size, stream>>>(
        QKV, Q, kvCacheBuffer, seq_lens, head_num, kv_head_num, size_per_head, kv_cache_capacity,
        rotary_embedding_base, rotary_embedding_scale);
}

void invokeContextApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    float rotary_embedding_base, float rotary_embedding_scale,
    const int token_to_process, cudaStream_t stream)
{
    dispatchApplyRopeUpdateKV<half, false>(
        QKV, Q, kvCacheBuffer, seq_lens, head_num, kv_head_num, size_per_head, kv_cache_capacity,
        rotary_embedding_base, rotary_embedding_scale, token_to_process, stream);
}

void invokeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    float rotary_embedding_base, float rotary_embedding_scale,
    const int token_to_process, cudaStream_t stream)
{
    dispatchApplyRopeUpdateKV<half, true>(
        QKV, Q, kvCacheBuffer, seq_lens, head_num, kv_head_num, size_per_head, kv_cache_capacity,
        rotary_embedding_base, rotary_embedding_scale, token_to_process, stream);
}