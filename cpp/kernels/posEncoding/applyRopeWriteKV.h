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

void launchApplyRopeWriteKV(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

void launchApplyRopeWriteKVContext(half* qkv, half* kvCache, float const* cosSinCache, int32_t qSeqLen,
    int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

void launchApplyRopeWriteContinuousQAndKVCache(half* qkv, half* kvCache, float const* cosSinCache, half* qOut,
    int32_t const* kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    cudaStream_t stream);

void launchApplyRopeWriteKVDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    cudaStream_t stream);

void launchApplyRopeWriteKVTreeDecode(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm