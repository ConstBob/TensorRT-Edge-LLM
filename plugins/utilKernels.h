/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_fp16.h>

enum class PositionEmbeddingType : int32_t
{
    kNone = 0,
    kROPE_ROTATE_GPTJ = 1,
    kROPE_ROTATE_NEOX = 2,
};

constexpr int32_t k_MAX_POSITION_EMBED_TYPE_VAL{2};

enum class RopeInitType : int32_t
{
    // Theta = 1 / (pow(rotary_embedding_freq, 2 * zid / headSize))
    kDEFAULT = 1,
    // On the basis of RopeInit, apply factors and smoothing factors based on wave_length
    kLLAMA3 = 2,
};

void invokeContextApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity,
    int const padded_seq_len, PositionEmbeddingType positionEmbedType, float rotary_embedding_freq,
    float rotary_embedding_scale, RopeInitType rope_init_type, int const token_to_process, cudaStream_t stream);

void invokeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, int const* seq_lens,
    int const head_num, int const kv_head_num, int const size_per_head, int const kv_cache_capacity,
    int const padded_seq_len, PositionEmbeddingType positionEmbedType, float rotary_embedding_freq,
    float rotary_embedding_scale, RopeInitType rope_init_type, int const token_to_process, cudaStream_t stream);

void invokePrefixSum(int32_t const* in_d, int32_t* out_d, int32_t numSeq, cudaStream_t stream);