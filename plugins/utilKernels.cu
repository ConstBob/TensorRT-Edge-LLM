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

#include "pluginUtils.h"
#include "utilKernels.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

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

__device__ __inline__ void load_vec_from_smem(uint32_t& vec, half* smem, int base_idx, int smem_pitch)
{
    union
    {
        uint32_t u32;
        half u16[2];
    } tmp;

    tmp.u16[0] = smem[base_idx];
    tmp.u16[1] = smem[base_idx + smem_pitch];

    vec = tmp.u32;
}

__device__ __inline__ void write_vec_to_smem(uint32_t const& vec, half* smem, int base_idx, int smem_pitch)
{
    union
    {
        uint32_t u32;
        half u16[2];
    } tmp;

    tmp.u32 = vec;
    smem[base_idx] = tmp.u16[0];
    smem[base_idx + smem_pitch] = tmp.u16[1];
}

inline __device__ float2 rotary_embedding_coefficient_default(
    int const zid, int const rot_embed_dim, float const base, float const scale, int const t_step)
{
    float const inv_freq = (t_step * scale) / pow(base, zid / (float) rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

// The implementation of the llama3 rope coefficient computation refers to HuggingFace's
// modeling_rope::_compute_llama3_parameters. The implementation is not presented by original paper.
inline __device__ float2 rotary_embedding_coefficient_llama3(
    int const zid, int const rot_embed_dim, float const base, float const scale, int const t_step)
{
    constexpr float pi = 3.14159265358;
    constexpr float scaling_factor = 8.0f;
    constexpr float low_freq_factor = 1.0f;
    constexpr float high_freq_factor = 4.0f;
    constexpr float old_context_len = 8192.0f;

    float inv_freq = scale / pow(base, zid / (float) rot_embed_dim);

    float const wavelen = 2.0f * pi / inv_freq;
    float const low_freq_wavelen = old_context_len / low_freq_factor;
    float const high_freq_wavelen = old_context_len / high_freq_factor;

    if (wavelen > low_freq_wavelen)
    {
        inv_freq = inv_freq / scaling_factor;
    }
    else if (wavelen > high_freq_wavelen && wavelen < low_freq_wavelen)
    {
        float const smooth_factor
            = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        inv_freq = (1 - smooth_factor) * inv_freq / scaling_factor + smooth_factor * inv_freq;
    }

    // Apply t_step factor (idx in sequence_length)
    inv_freq = inv_freq * t_step;
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

inline __device__ void apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, float base,
    float scale, int t_step, RopeInitType ropeInitType)
{
    if (2 * tid >= rot_embed_dim)
    {
        return;
    }
    float2 coef;
    switch (ropeInitType)
    {
    case RopeInitType::kDEFAULT:
        coef = rotary_embedding_coefficient_default(2 * tid, rot_embed_dim, base, scale, t_step);
        break;
    case RopeInitType::kLLAMA3:
        coef = rotary_embedding_coefficient_llama3(2 * tid, rot_embed_dim, base, scale, t_step);
        break;
    }

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

} // namespace

template <typename T, bool IsGenerate>
__global__ void applyBiasRopeUpdateKVCache(T* QKV, T* Q, T* kvCacheBuffer, int const* seq_lens, int const head_num,
    int const kv_head_num, int const size_per_head, int const kv_cache_capacity, int padded_seqlen,
    PositionEmbeddingType positionEmbedType, float rotary_embedding_freq, float rotary_embedding_scale,
    RopeInitType ropeInitType)
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

    extern __shared__ __align__(sizeof(float2)) char smem_[];

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t = typename Vec_t<T>::Type;

    // The padded token_idx indicates the index within a multi-batch padded tensor.
    int const padded_token_idx = blockIdx.x;
    int batch_index;
    int token_idx_in_seq;

    if (IsGenerate)
    {
        // In generate phase, we support exactly one token to generate per sequence.
        // Therefore the padded_token_idx also indicates the batch index.
        batch_index = padded_token_idx;
        token_idx_in_seq = seq_lens[batch_index] - 1;
    }
    else
    {
        // In context phase, we support the padded sequence length.
        batch_index = padded_token_idx / padded_seqlen;
        token_idx_in_seq = padded_token_idx % padded_seqlen;
    }

    // We only use the same tensor data and kv-cache type now.
    // TODO: Explore the need if we want to support different data and kvcache type.
    using T_cache = T;
    using T_dst = T_cache;

    int const head_idx = blockIdx.y;

    // Each thread block will handle D elements, each thread is assigned two elements.
    int const tidx = threadIdx.x;

    int const hidden_size = head_num * size_per_head;
    int const hidden_idx = head_idx * size_per_head + tidx * vec_size;
    int const qheads_per_kv_head = head_num / kv_head_num;
    int const kv_head_idx = head_idx / qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * size_per_head + tidx * vec_size;

    // Totoal elements for a token.
    int const n = (head_num + 2 * kv_head_num) * size_per_head;

    // offset the total number of data elements in Q heads.
    int const src_k_offset = hidden_size;
    // offset the totoal number of data elements in K heads.
    int const src_v_offset = hidden_size + kv_head_num * size_per_head;

    // NOTE: q has seq len excluding prefix prompt
    // head_num == kv_head_num:
    //   src QKV: [batch, time, 3, head_num, size_per_head]
    // head_num != kv_head_num:
    //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
    int const src_q_idx = padded_token_idx * n + hidden_idx;
    int const src_k_idx = padded_token_idx * n + src_k_offset + hidden_idx_kv;
    int const src_v_idx = padded_token_idx * n + src_v_offset + hidden_idx_kv;

    Vec_t q, k, v;

    // load q,k,v
    q = *reinterpret_cast<Vec_t const*>(&QKV[src_q_idx]);
    k = *reinterpret_cast<Vec_t const*>(&QKV[src_k_idx]);
    v = *reinterpret_cast<Vec_t const*>(&QKV[src_v_idx]);

    switch (positionEmbedType)
    {
    case PositionEmbeddingType::kROPE_ORIGINAL:
    {
        // Original rotary position encoding will apply transformation to adjacent pair of data.
        apply_rotary_embedding(
            q, k, tidx, size_per_head, rotary_embedding_freq, rotary_embedding_scale, token_idx_in_seq, ropeInitType);
        break;
    }
    case PositionEmbeddingType::kROPE_ROTATE_HALF:
    {
        // With Rotate Half RoPE position embeddeding, the transformation will apply to pair of data
        // with D dimension [tIDX, tIDX + size_per_head / 2]. We first store the adjacent q/k data pair
        // into shared memory and read the two data from tIDX and tIDX + size_per_head / 2
        T* q_smem = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem + size_per_head;

        *reinterpret_cast<Vec_t*>(q_smem + tidx * vec_size) = q;
        *reinterpret_cast<Vec_t*>(k_smem + tidx * vec_size) = k;
        __syncthreads();

        int32_t const half_head_dim = size_per_head / 2;
        load_vec_from_smem(q, q_smem, tidx, half_head_dim);
        load_vec_from_smem(k, k_smem, tidx, half_head_dim);

        apply_rotary_embedding(
            q, k, tidx, size_per_head, rotary_embedding_freq, rotary_embedding_scale, token_idx_in_seq, ropeInitType);

        write_vec_to_smem(q, q_smem, tidx, half_head_dim);
        write_vec_to_smem(k, k_smem, tidx, half_head_dim);
        __syncthreads();

        // Load adjacent pair of data after rope transformation from shared memory.
        q = *reinterpret_cast<Vec_t*>(q_smem + tidx * vec_size);
        k = *reinterpret_cast<Vec_t*>(k_smem + tidx * vec_size);

        break;
    }
    }

    // KV-cache is of shape [B, 2, H, S, D] where S is the capacity of the kvcache buffer (max total context length).
    // K shape is [B, 1, H, S, D]
    // V shape is [B, 1, H, S, D]
    int const bytes_per_seq
        = kv_head_num * kv_cache_capacity * size_per_head;          // max bytes of K or V per (total) sequence
    int const offset_sequence = batch_index * (2 * bytes_per_seq); // offset of the currect sequence in linear buffer.
    int const offset_local_kv
        = kv_head_idx * kv_cache_capacity * size_per_head + token_idx_in_seq * size_per_head + tidx * vec_size;

    int const cache_offset_k = offset_sequence + offset_local_kv;
    int const cache_offset_v = offset_sequence + bytes_per_seq + offset_local_kv;

    if constexpr (IsGenerate)
    {
        int const offset_q = batch_index * head_num * size_per_head + hidden_idx;
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
void dispatchApplyRopeUpdateKV(T* QKV, T* Q, T* kvCacheBuffer, int const* seq_lens, int const head_num,
    int const kv_head_num, int const size_per_head, int const kv_cache_capacity, int const padded_seqlen,
    PositionEmbeddingType positionEmbedType, float rotary_embedding_freq, float rotary_embedding_scale,
    RopeInitType ropeInitType, int const token_to_process, cudaStream_t stream)
{
    check(QKV != nullptr && kvCacheBuffer != nullptr && seq_lens != nullptr,
        "Data pointers of qkv, kvcache, and sequence length shall be valid");
    check(token_to_process > 0, "Number of tokens to process shall be valid.");
    check(size_per_head % 32 == 0, "Size per head shall be multiple of 32");
    if constexpr (IsGenerate)
    {
        check(Q != nullptr, "In generation phase Q pointer shall be valid.");
    }
    // To implement rotary embeddings, each thread processes two QKV elems:
    dim3 block(size_per_head / Vec_t<T>::size);
    dim3 grid(token_to_process, head_num);

    // Determine required shared memory size by type of rope.
    size_t smem_size{0};
    if (positionEmbedType == PositionEmbeddingType::kROPE_ROTATE_HALF)
    {
        // The shared memory should be large enough to contain the data of single head q + k vector.
        smem_size = 2 * size_per_head * sizeof(T);
    }
    applyBiasRopeUpdateKVCache<T, IsGenerate><<<grid, block, smem_size, stream>>>(QKV, Q, kvCacheBuffer, seq_lens,
        head_num, kv_head_num, size_per_head, kv_cache_capacity, padded_seqlen,
        positionEmbedType, rotary_embedding_freq, rotary_embedding_scale, ropeInitType);
}

void invokeContextApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity, int const padded_seqlen,
    PositionEmbeddingType positionEmbedType, float rotary_embedding_freq, float rotary_embedding_scale,
    RopeInitType ropeInitType, int const token_to_process, cudaStream_t stream)
{
    dispatchApplyRopeUpdateKV<half, false>(QKV, Q, kvCacheBuffer, seq_lens, head_num, kv_head_num, size_per_head,
        kv_cache_capacity, padded_seqlen, positionEmbedType, rotary_embedding_freq,
        rotary_embedding_scale, ropeInitType, token_to_process, stream);
}

void invokeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity, int const padded_seqlen,
    PositionEmbeddingType positionEmbedType, float rotary_embedding_freq, float rotary_embedding_scale,
    RopeInitType ropeInitType, int const token_to_process, cudaStream_t stream)
{
    dispatchApplyRopeUpdateKV<half, true>(QKV, Q, kvCacheBuffer, seq_lens, head_num, kv_head_num, size_per_head,
        kv_cache_capacity, padded_seqlen, positionEmbedType, rotary_embedding_freq,
        rotary_embedding_scale, ropeInitType, token_to_process, stream);
}

void invokePrefixSum(int32_t const* in_d, int32_t* out_d, int32_t numSeq, cudaStream_t stream)
{
    thrust::device_ptr<const int32_t> thrust_in_ptr = thrust::device_pointer_cast(in_d);
    thrust::device_ptr<int32_t> thrust_out_ptr(out_d);

    // Fill out_d with zeros and then copy ctxLen contents to &thrust_out_ptr[1] (left leading zero).
    thrust::fill(thrust::cuda::par.on(stream), thrust_out_ptr, thrust_out_ptr + numSeq + 1, 0);
    thrust::copy(thrust::cuda::par.on(stream), thrust_in_ptr, thrust_in_ptr + numSeq, thrust_out_ptr + 1);
    // Apply in-place inclusive scan to compute the result.
    thrust::inclusive_scan(thrust::cuda::par.on(stream), thrust_out_ptr, thrust_out_ptr + numSeq + 1, thrust_out_ptr);
}