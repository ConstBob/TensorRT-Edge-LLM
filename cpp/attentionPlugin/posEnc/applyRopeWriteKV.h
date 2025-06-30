#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launchApplyRopeWriteKV(half* qkv, half* kvCache, half* qOut, float* cosSinCache, int64_t* kvCacheStartIds,
    int64_t* tokenPosIds, int64_t qSeqLen, int64_t totalNumTokens, int64_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t rotaryDim, bool useInterleaveRope, cudaStream_t stream);