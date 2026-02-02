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

/*
 * Edge-LLM MoE Marlin W4A16 GEMM Kernel Wrapper
 *
 * This file provides the rt::Tensor based interface for MoE W4A16 GEMM
 * using Marlin optimized kernels with AWQ (kU4) quantization format.
 *
 * AWQ format (kU4):
 * - Uses unsigned 4-bit integers with zero point = 8 baked in
 * - Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 */

#include <cuda_runtime.h>

#if CUDA_VERSION >= 11080

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "marlin/scalar_type.hpp"
#include "moeMarlin.h"

#include <algorithm>
#include <cstdint>

// Forward declare the raw marlin_mm function from ops.cu
// Note: The namespace matches MARLIN_NAMESPACE_NAME defined in ops.cu
namespace marlin_moe_wna16
{

// Constants from kernel.h
constexpr int min_thread_n = 64;
constexpr int max_thread_n = 256;

void marlin_mm(void const* A, void const* B, void* C, void* C_tmp, void* b_bias, void* a_s, void* b_s, void* g_s,
    void* zp, void* g_idx, void* perm, void* a_tmp, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_past_padded, void* topk_weights, int moe_block_size, int num_experts, int top_k,
    bool mul_topk_weights, int prob_m, int prob_n, int prob_k, void* workspace,
    trt_edgellm::marlin_dtypes::ScalarType const& a_type, trt_edgellm::marlin_dtypes::ScalarType const& b_type,
    trt_edgellm::marlin_dtypes::ScalarType const& c_type, trt_edgellm::marlin_dtypes::ScalarType const& s_type,
    bool has_bias, bool has_act_order, bool is_k_full, bool has_zp, int num_groups, int group_size, int dev,
    cudaStream_t stream, int thread_k, int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float);

} // namespace marlin_moe_wna16

namespace trt_edgellm
{
namespace kernel
{

using namespace trt_edgellm::format;

// Internal constants for FP32 reduction (always enabled for accuracy)
constexpr bool kUseFp32Reduce = true;
constexpr bool kUseAtomicAdd = false;

void moeAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& sortedTokenIds, rt::Tensor const& expertIds,
    rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights, rt::Tensor& workspace, int64_t moeBlockSize,
    int64_t topK, bool mulTopkWeights, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = scales.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D tensor [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D tensor [numTokens*topK, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D tensor [numExperts, K/tile, N*tile/pack]");
    check::check(scalesShape.getNumDims() == 3, "Scales must be 3D tensor [numExperts, numGroups, outDim]");

    int64_t numTokens = inputShape[0];
    int64_t hiddenDim = inputShape[1];
    int64_t numExperts = weightsShape[0];
    int64_t numGroups = scalesShape[1];
    int64_t outDim = scalesShape[2];

    // Validate output shape
    check::check(outputShape[0] == numTokens * topK,
        fmtstr("Output shape[0] %ld != numTokens*topK %ld", outputShape[0], numTokens * topK));
    check::check(outputShape[1] == outDim, fmtstr("Output shape[1] %ld != outDim %ld", outputShape[1], outDim));

    // Validate data types
    check::check(input.getDataType() == nvinfer1::DataType::kHALF, "Input must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output must be FP16");
    check::check(weights.getDataType() == nvinfer1::DataType::kINT32, "Weights must be INT32 (Marlin packed)");
    check::check(scales.getDataType() == nvinfer1::DataType::kHALF, "Scales must be FP16");
    check::check(sortedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "sortedTokenIds must be INT32");
    check::check(expertIds.getDataType() == nvinfer1::DataType::kINT32, "expertIds must be INT32");
    check::check(numTokensPostPadded.getDataType() == nvinfer1::DataType::kINT32, "numTokensPostPadded must be INT32");
    check::check(topkWeights.getDataType() == nvinfer1::DataType::kFLOAT, "topkWeights must be FP32");
    check::check(workspace.getDataType() == nvinfer1::DataType::kINT32, "workspace must be INT32");

    // Validate moe_block_size
    if (moeBlockSize != 8)
    {
        check::check(moeBlockSize % 16 == 0, fmtstr("moeBlockSize %ld must be divisible by 16", moeBlockSize));
        check::check(
            moeBlockSize >= 16 && moeBlockSize <= 64, fmtstr("moeBlockSize %ld must be in [16, 64]", moeBlockSize));
    }

    // Calculate group size
    int groupSize = (numGroups > 1) ? static_cast<int>(hiddenDim / numGroups) : -1;

    // Get device info
    int dev;
    int sms;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    // AWQ W4A16 fixed scalar types
    trt_edgellm::marlin_dtypes::ScalarType aType = trt_edgellm::marlin_dtypes::kFloat16; // A16 activation
    trt_edgellm::marlin_dtypes::ScalarType bType
        = trt_edgellm::marlin_dtypes::kU4; // AWQ INT4 with zero point = 8 baked in
    trt_edgellm::marlin_dtypes::ScalarType cType = trt_edgellm::marlin_dtypes::kFloat16; // Output FP16
    trt_edgellm::marlin_dtypes::ScalarType sType = trt_edgellm::marlin_dtypes::kFloat16; // Scales FP16

    // Extract raw pointers
    void* inputPtr = const_cast<void*>(input.rawPointer());
    void* outputPtr = output.rawPointer();
    void* weightsPtr = const_cast<void*>(weights.rawPointer());
    void* scalesPtr = const_cast<void*>(scales.rawPointer());
    void* sortedTokenIdsPtr = const_cast<void*>(sortedTokenIds.rawPointer());
    void* expertIdsPtr = const_cast<void*>(expertIds.rawPointer());
    void* numTokensPostPaddedPtr = const_cast<void*>(numTokensPostPadded.rawPointer());
    void* topkWeightsPtr = const_cast<void*>(topkWeights.rawPointer());

    // Workspace layout: [locks (INT32)] [c_tmp (FP32)]
    int64_t numTokensPadded = sortedTokenIds.getShape()[0];
    int64_t locksSize
        = std::min((outDim / marlin_moe_wna16::min_thread_n) * ((numTokensPadded + moeBlockSize - 1) / moeBlockSize),
            static_cast<int64_t>(sms * 4));

    int32_t* workspaceBasePtr = static_cast<int32_t*>(workspace.rawPointer());
    void* locksPtr = workspaceBasePtr;

    // c_tmp buffer starts after locks (FP32 reduction buffer)
    void* cTmpPtr = workspaceBasePtr + locksSize;

    // Call the raw marlin_mm function with AWQ-specific parameters
    marlin_moe_wna16::marlin_mm(inputPtr, // A
        weightsPtr,                       // B
        outputPtr,                        // C
        cTmpPtr,                          // C_tmp (FP32 reduction buffer)
        nullptr,                          // b_bias (not used)
        nullptr,                          // a_scales (not used for A16)
        scalesPtr,                        // b_scales
        nullptr,                          // global_scale (not used)
        nullptr,                          // zp (not used - zero point = 8 baked in)
        nullptr,                          // g_idx (not used - no act_order)
        nullptr,                          // perm (not used - no act_order)
        nullptr,                          // a_tmp (not used - no act_order)
        sortedTokenIdsPtr, expertIdsPtr, numTokensPostPaddedPtr, topkWeightsPtr, static_cast<int>(moeBlockSize),
        static_cast<int>(numExperts), static_cast<int>(topK), mulTopkWeights,
        static_cast<int>(numTokens), // prob_m
        static_cast<int>(outDim),    // prob_n
        static_cast<int>(hiddenDim), // prob_k
        locksPtr, aType, bType, cType, sType,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full (always true for standard AWQ)
        false, // has_zp (zero point = 8 baked into dequant)
        static_cast<int>(numGroups), groupSize, dev, stream,
        -1, // thread_k (auto)
        -1, // thread_n (auto)
        sms,
        -1, // blocks_per_sm (auto)
        kUseAtomicAdd, kUseFp32Reduce,
        false // is_zp_float
    );
}

int64_t getMoeMarlinWorkspaceSize(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize, int64_t numSMs)
{
    // Locks size (INT32)
    int64_t maxNTiles = outDim / marlin_moe_wna16::min_thread_n;
    int64_t numBlocks = (numTokensPadded + moeBlockSize - 1) / moeBlockSize;
    int64_t locksSize = std::min(maxNTiles * numBlocks, numSMs * 4);

    // C_tmp size for FP32 reduction (same element size as INT32)
    int64_t cTmpSize = std::min(
        outDim * numTokensPadded, numSMs * 4 * moeBlockSize * static_cast<int64_t>(marlin_moe_wna16::max_thread_n));
    if (moeBlockSize == 8)
    {
        cTmpSize *= 2;
    }

    // Total workspace = locks + c_tmp (both are 4 bytes per element)
    return locksSize + cTmpSize;
}

} // namespace kernel
} // namespace trt_edgellm

#endif // CUDA_VERSION >= 11080