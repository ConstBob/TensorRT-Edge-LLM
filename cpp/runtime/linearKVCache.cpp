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

#include "runtime/linearKVCache.h"

#include "common/cudaUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include <cuda_bf16.h>
#include <type_traits>

using namespace nvinfer1;

namespace drivellm
{
namespace rt
{

LinearKVCache::LinearKVCache(CacheConfig const& config)
    : mConfig(config)
{
    int32_t const kvCacheVolume = mConfig.numDecoderLayers * mConfig.maxBatchSize * 2 * mConfig.numKVHeads
        * mConfig.maxSequenceLength * mConfig.headDim;
    CUDA_CHECK(cudaMalloc(&mDeviceKVCache, kvCacheVolume * sizeof(KVCacheType)));
    mDeviceKVCacheLengths = rt::Tensor({mConfig.maxBatchSize}, DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemset(mDeviceKVCacheLengths.rawPointer(), 0, mDeviceKVCacheLengths.getMemoryCapacity()));
}

LinearKVCache::~LinearKVCache()
{
    CUDA_CHECK(cudaFree(mDeviceKVCache));
    mDeviceKVCache = nullptr;
}

LinearKVCache::LinearKVCache(LinearKVCache&& other) noexcept
{
    mConfig = other.mConfig;
    mActiveBatchSize = other.mActiveBatchSize;
    mDeviceKVCache = other.mDeviceKVCache;
    mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);

    other.mConfig = CacheConfig{};
    other.mActiveBatchSize = 0;
    other.mDeviceKVCache = nullptr;
}

LinearKVCache& LinearKVCache::operator=(LinearKVCache&& other) noexcept
{
    if (this != &other)
    {
        // Release current KVCache memory.
        CUDA_CHECK(cudaFree(mDeviceKVCache));
        mConfig = other.mConfig;
        mActiveBatchSize = other.mActiveBatchSize;
        mDeviceKVCache = other.mDeviceKVCache;
        mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);

        other.mConfig = CacheConfig{};
        other.mActiveBatchSize = 0;
        other.mDeviceKVCache = nullptr;
    }
    return *this;
}

rt::Tensor LinearKVCache::getKVCacheForDecoderLayer(int32_t decoderLayerIdx)
{
    int64_t kvCacheOffset
        = decoderLayerIdx * mConfig.maxBatchSize * 2 * mConfig.numKVHeads * mConfig.maxSequenceLength * mConfig.headDim;
    KVCacheType* kvCachePtr = mDeviceKVCache + kvCacheOffset;
    return rt::Tensor(kvCachePtr,
        {mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength, mConfig.headDim}, DeviceType::kGPU,
        KVCacheTypeTRT);
}

rt::Tensor LinearKVCache::getKVCacheBuffer()
{
    return rt::Tensor(mDeviceKVCache,
        {mConfig.numDecoderLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength,
            mConfig.headDim},
        DeviceType::kGPU, KVCacheTypeTRT);
}

void LinearKVCache::resetForNewSequences(int32_t batchSize, cudaStream_t stream)
{
    if (batchSize > mConfig.maxBatchSize)
    {
        throw std::runtime_error(
            "BatchSize of this batch of sequences exceeds the maximum batchSize supported by the KVCache");
    }
    mActiveBatchSize = batchSize;
    CUDA_CHECK(
        cudaMemsetAsync(mDeviceKVCacheLengths.rawPointer(), 0, mDeviceKVCacheLengths.getMemoryCapacity(), stream));
    assert(mDeviceKVCacheLengths.reshape({mActiveBatchSize}));
}

void LinearKVCache::commitPrefillRequest(rt::Tensor const& prefillLengths, cudaStream_t stream)
{
    assert(prefillLengths.getDataType() == DataType::kINT32);
    assert(prefillLengths.getDeviceType() == DeviceType::kCPU);
    assert(prefillLengths.getShape().getNumDims() == 1);
    assert(prefillLengths.getShape()[0] == mActiveBatchSize);

    CUDA_CHECK(cudaMemcpyAsync(mDeviceKVCacheLengths.rawPointer(), prefillLengths.rawPointer(),
        prefillLengths.getShape()[0] * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

void LinearKVCache::commitDecodeRequest(cudaStream_t stream)
{
    kernel::incrementKVCacheLengths(mDeviceKVCacheLengths, mActiveBatchSize, stream);
    CUDA_CHECK(cudaGetLastError());
}

rt::Tensor& LinearKVCache::getKVCacheLengths()
{
    return mDeviceKVCacheLengths;
}

LinearKVCache::CacheConfig LinearKVCache::getConfig() const
{
    return mConfig;
}

int32_t LinearKVCache::getActiveBatchSize() const
{
    return mActiveBatchSize;
}

} // namespace rt
} // namespace drivellm