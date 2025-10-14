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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace drivellm
{
namespace kernel
{

/*!
 * @brief Apply RoPE and write KV cache (general version)
 *
 * Applies rotary position encoding to Q/K and writes K/V to cache with custom position IDs.
 *
 * @param qkv Input QKV tensor
 * @param kvCache KV cache buffer
 * @param qOut Output Q tensor with RoPE applied
 * @param cosSinCache Precomputed cos/sin cache
 * @param kvCacheEndLens KV cache end lengths per batch
 * @param tokenPosIds Token position IDs
 * @param qSeqLen Query sequence length
 * @param totalNumTokens Total number of tokens
 * @param kvCacheCapacity KV cache capacity
 * @param numQHead Number of query heads
 * @param numKVHead Number of KV heads
 * @param headDim Head dimension
 * @param rotaryDim Rotary dimension
 * @param cosSinCacheBatchSize Cos/sin cache batch size
 * @param cosSinCacheSeqLen Cos/sin cache sequence length
 * @param stream CUDA stream
 */
void launchApplyRopeWriteKV(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

/*!
 * @brief Apply RoPE and write KV cache (context/prefill phase)
 *
 * Optimized for context/prefill phase where QKV are contiguous.
 *
 * @param qkv Input QKV tensor
 * @param kvCache KV cache buffer
 * @param cosSinCache Precomputed cos/sin cache
 * @param qSeqLen Query sequence length
 * @param totalNumTokens Total number of tokens
 * @param kvCacheCapacity KV cache capacity
 * @param numQHead Number of query heads
 * @param numKVHead Number of KV heads
 * @param headDim Head dimension
 * @param rotaryDim Rotary dimension
 * @param cosSinCacheBatchSize Cos/sin cache batch size
 * @param cosSinCacheSeqLen Cos/sin cache sequence length
 * @param stream CUDA stream
 */
void launchApplyRopeWriteKVContext(half* qkv, half* kvCache, float const* cosSinCache, int32_t qSeqLen,
    int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

/*!
 * @brief Apply RoPE and write continuous Q and KV cache
 *
 * Writes both Q (with RoPE) and KV to contiguous output tensors.
 *
 * @param qkv Input QKV tensor
 * @param kvCache KV cache buffer
 * @param cosSinCache Precomputed cos/sin cache
 * @param qOut Output Q tensor
 * @param kvCacheEndLens KV cache end lengths
 * @param qSeqLen Query sequence length
 * @param totalNumTokens Total number of tokens
 * @param kvCacheCapacity KV cache capacity
 * @param numQHead Number of query heads
 * @param numKVHead Number of KV heads
 * @param headDim Head dimension
 * @param rotaryDim Rotary dimension
 * @param cosSinCacheBatchSize Cos/sin cache batch size
 * @param cosSinCacheSeqLen Cos/sin cache sequence length
 * @param stream CUDA stream
 */
void launchApplyRopeWriteContinuousQAndKVCache(half* qkv, half* kvCache, float const* cosSinCache, half* qOut,
    int32_t const* kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    cudaStream_t stream);

/*!
 * @brief Apply RoPE and write KV cache (decode phase)
 *
 * Optimized for decode phase where query length is typically 1.
 *
 * @param qkv Input QKV tensor
 * @param kvCache KV cache buffer
 * @param qOut Output Q tensor
 * @param cosSinCache Precomputed cos/sin cache
 * @param kvCacheEndLens KV cache end lengths
 * @param qSeqLen Query sequence length
 * @param totalNumTokens Total number of tokens
 * @param kvCacheCapacity KV cache capacity
 * @param numQHead Number of query heads
 * @param numKVHead Number of KV heads
 * @param headDim Head dimension
 * @param rotaryDim Rotary dimension
 * @param cosSinCacheBatchSize Cos/sin cache batch size
 * @param cosSinCacheSeqLen Cos/sin cache sequence length
 * @param stream CUDA stream
 */
void launchApplyRopeWriteKVDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    cudaStream_t stream);

/*!
 * @brief Apply RoPE and write KV cache (tree decode for speculative)
 *
 * Specialized for tree attention in speculative decoding (Eagle).
 *
 * @param qkv Input QKV tensor
 * @param kvCache KV cache buffer
 * @param qOut Output Q tensor
 * @param cosSinCache Precomputed cos/sin cache
 * @param kvCacheEndLens KV cache end lengths
 * @param tokenPosIds Token position IDs for tree nodes
 * @param qSeqLen Query sequence length
 * @param totalNumTokens Total number of tokens
 * @param kvCacheCapacity KV cache capacity
 * @param numQHead Number of query heads
 * @param numKVHead Number of KV heads
 * @param headDim Head dimension
 * @param rotaryDim Rotary dimension
 * @param cosSinCacheBatchSize Cos/sin cache batch size
 * @param cosSinCacheSeqLen Cos/sin cache sequence length
 * @param stream CUDA stream
 */
void launchApplyRopeWriteKVTreeDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm