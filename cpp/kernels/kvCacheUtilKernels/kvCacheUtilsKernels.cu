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

#include "kvCacheUtilsKernels.h"

namespace drivellm
{
namespace kernel
{

__global__ void incrementKVCacheLengthsKernel(int32_t* kvCacheLengths, int32_t activeBatchSize)
{
    int32_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t gridSize = blockDim.x * gridDim.x;
    for (int32_t i = tIdx; i < activeBatchSize; i += gridSize)
    {
        kvCacheLengths[i] += 1;
    }
}
void incrementKVCacheLengths(rt::Tensor& kvCacheLengths, int32_t activeBatchSize, cudaStream_t stream)
{
    constexpr int32_t kBLOCK_SIZE = 32;
    constexpr int32_t kGRID_SIZE = 1;
    incrementKVCacheLengthsKernel<<<kGRID_SIZE, kBLOCK_SIZE, 0, stream>>>(
        kvCacheLengths.dataPointer<int32_t>(), activeBatchSize);
}
} // namespace kernel
} // namespace drivellm