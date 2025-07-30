#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace drivellm
{
namespace kernel
{

void launchApplyRopeWriteKV(half* qkv, half* kvCache, half* qOut, float const* cosSinCache,
    int32_t const* kvCacheStartIds, int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens,
    int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim,
    int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

void launchApplyRopeWriteKVContext(half* qkv, half* kvCache, float const* cosSinCache, int32_t qSeqLen,
    int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, cudaStream_t stream);

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