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

void invokeContextApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    const int rotary_embedding_dim, float rotary_embedding_base, float rotary_embedding_scale,
    const int token_to_process, cudaStream_t stream);

void invokeGenerationApplyRopeUpdateKVFP16(half* QKV, half* Q, half* kvCacheBuffer, const int* seq_lens,
    const int head_num, const int kv_head_num, const int size_per_head, const int kv_cache_capacity,
    const int rotary_embedding_dim, float rotary_embedding_base, float rotary_embedding_scale,
    const int token_to_process, cudaStream_t stream);

