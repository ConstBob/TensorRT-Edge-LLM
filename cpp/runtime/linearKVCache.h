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

#include <common/tensor.h>
#include <cstdint>
#include <cuda_fp16.h>

namespace drivellm
{
namespace rt
{

//! Static Linear KVCache that holds the KVCache for all decoder layers up to maxSequenceLength.
//! The KVCache implement the design of:
//! 1. Allocates memory for max supported batch size.
//! 2. Memory Layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! 3. Synchronous execution of batch requests, all the sequences in the batch will run prefill
//!    or decode at the same time.
class LinearKVCache
{
public:
    struct CacheConfig
    {
        int32_t numDecoderLayers{};
        int32_t maxBatchSize{};
        int32_t maxSequenceLength{};
        int32_t numKVHeads{};
        int32_t headDim{};
    };

    // Only support half precision for now.
    using KVCacheType = half;
    static constexpr nvinfer1::DataType KVCacheTypeTRT{nvinfer1::DataType::kHALF};

    //! Initialize the KVCache with the given config.
    //! @param config The config for the KVCache instance. Once allocated, the device memory won't be reallocated.
    LinearKVCache() = default;
    LinearKVCache(CacheConfig const& config, cudaStream_t stream);
    ~LinearKVCache();

    // Delete copy construction and assignments to avoid accidental large data copy.
    LinearKVCache(LinearKVCache const&) = delete;
    LinearKVCache& operator=(LinearKVCache const&) = delete;
    LinearKVCache(LinearKVCache&&) noexcept;
    LinearKVCache& operator=(LinearKVCache&&) noexcept;

    //! Get the KVCache for the given decoder layer.
    //! @param decoderLayerIdx The index of the decoder layer.
    //! @return A non-owned tensor object that points to the KVCache memory with shape information.
    rt::Tensor getKVCacheForDecoderLayer(int32_t decoderLayerIdx);

    //! Get the full KVCache buffer as a non-owned tensor.
    rt::Tensor getKVCacheBuffer();

    //! Asynchronously reset the KVCache buffer state for a new setup of input context.
    //! @param hostReuseKVCacheLengths The lengths of the KVCache to be reused from precomputed KVCache content.
    //! @param stream The stream is used to perform GPU memory operations.
    void resetForNewSequences(rt::Tensor const& hostReuseKVCacheLengths, cudaStream_t stream);

    //! Asynchronously commit the KVCache buffer for a prefill request, record stored KVCache lengths.
    //! @param newContextLengths [GPU, Int32]: The context length to commit for the KVCache.
    //! @param stream The stream is used to perform GPU memory operations.
    void commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream);

    //! Commit the KVCache buffer for a decode request, increment the KVCache lengths by 1 for active sequences.
    //! @param stream The stream is used to perform GPU memory operations.
    void commitSequenceLength(int32_t increment, cudaStream_t stream);

    //! Get the KVCache lengths for active sequences.
    rt::Tensor& getKVCacheLengths();

    //! Get the KVCache config.
    CacheConfig getConfig() const;

    //! Get the active batch size.
    int32_t getActiveBatchSize() const;

private:
    // Config parameters of the KVCache instance.
    CacheConfig mConfig{};

    // Runtime parameters of the KVCache instance.
    int32_t mActiveBatchSize{};
    rt::Tensor mDeviceKVCacheLengths{};

    // KVCache memory buffer.
    KVCacheType* mDeviceKVCache{nullptr};
};

} // namespace rt
} // namespace drivellm