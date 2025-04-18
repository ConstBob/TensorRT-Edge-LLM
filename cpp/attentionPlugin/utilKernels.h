/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_fp16.h>

enum class PositionEmbeddingType : int32_t
{
    kNone = 0,
    kROPE_ROTATE_GPTJ = 1,
    kROPE_ROTATE_NEOX = 2,
    kMROPE = 3,
};

constexpr int32_t k_MAX_POSITION_EMBED_TYPE_VAL{3};

enum class RopeInitType : int32_t
{
    // Theta = 1 / (pow(rotary_embedding_freq, 2 * zid / headSize))
    kDEFAULT = 1,
    // On the basis of RopeInit, apply factors and smoothing factors based on wave_length
    kLLAMA3 = 2,
};

void invokeContextApplyRopeUpdateKVFP16(half* QKV, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity,
    int const padded_seq_len, PositionEmbeddingType positionEmbedType, float rotary_embedding_freq,
    float rotary_embedding_scale, RopeInitType rope_init_type, int const token_to_process,
    int const rotary_embedding_max_position, float2 const* mrope_rotary_cos_sin,
    cudaStream_t stream);

void invokeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity,
    int const padded_seq_len, PositionEmbeddingType positionEmbedType, float rotary_embedding_freq,
    float rotary_embedding_scale, RopeInitType rope_init_type, int const token_to_process,
    int const rotary_embedding_max_position, int64_t const* mrope_position_deltas,
    cudaStream_t stream);

void invokeSpecDecodeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const* custom_seq_index, int const head_num, int const kv_head_num, int const size_per_head,
    int const kv_cache_capacity, int const padded_seqlen, PositionEmbeddingType positionEmbedType,
    float rotary_embedding_freq, float rotary_embedding_scale, RopeInitType ropeInitType, int const token_to_process,
    int const rotary_embedding_max_position, int64_t const* mrope_position_deltas,
    cudaStream_t stream);