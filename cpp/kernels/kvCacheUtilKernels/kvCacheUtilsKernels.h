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

#include "common/tensor.h"

namespace drivellm
{
namespace kernel
{

void incrementKVCacheLengths(rt::Tensor& kvCacheLengths, int32_t activeBatchSize, cudaStream_t stream);

//! Helper function to instantiate the KVCache from a pre-computed KVCache tensor. Used to support KVCache reuse across
//! multiple inference requests to speedup prefill step.
//! Inputs:
//! - KVCacheBuffer: The KVCache buffer to be instantiated layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads,
//! maxSequenceLength, headDim].
//! - srcKVCacheTensor: The pre-computed KVCache tensor. layout: [numDecoderLayers, 2, numKVHeads, sequenceLength,
//! headDim].
//! - batchIdx: The batch index of the KVCache to be instantiated.
//! - stream: The CUDA stream to be used.
void instantiateKVCacheFromTensor(
    rt::Tensor& dstKVCacheBuffer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream);

//! Helper function to save the KVCache into a tensor. Used to support KVCache reuse across multiple inference requests
//! to speedup prefill step. SequenceLength of dstKVCacheTensor must be saved from the srcKVCacheBuffer.
//! Inputs:
//! - dstKVCacheTensor: The KVCache tensor to be saved. layout: [numDecoderLayers, 2, numKVHeads, sequenceLength,
//! headDim].
//! - srcKVCacheBuffer: The KVCache buffer to be saved. layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads,
//! maxSequenceLength, headDim].
//! headDim].
//! - batchIdx: The batch index of the KVCache to be saved.
//! - stream: The CUDA stream to be used.
void saveKVCacheIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheBuffer, int32_t batchIdx, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm