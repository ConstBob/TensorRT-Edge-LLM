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

#include "fmhaParams_v2.h"

#include <NvInferRuntime.h>
namespace drivellm
{

class ContextFMHARunner
{
public:
    ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t paddedSeqLen, int32_t numQHeads,
        int32_t numKvHeads, int32_t headSize, int32_t smVersion);

    ContextFMHARunner() = delete;

    ~ContextFMHARunner() = default;

    size_t getWorkspaceSize();

    // The function will setup kernel parameters except device pointers.
    // Device pointers shall be set by caller of FMHA runner.
    void setupParams(Fused_multihead_attention_params_v2& params);

    // Dispatch FMHA kernel.
    void dispatchFMHAKernel(Fused_multihead_attention_params_v2& params, cudaStream_t const& stream);

    // Static methods to check kernel availability and load cubins into device.
    static bool canImplement(int32_t headSize, int32_t sm, nvinfer1::DataType dataType);
    static bool loadContextFMHAKernels(int32_t sm, nvinfer1::DataType dataType);

private:
    nvinfer1::DataType mDataType;
    int32_t mBatchSize;
    int32_t mPaddedSequenceLen;
    int32_t mNumHeads;
    int32_t mNumKVHeads;
    int32_t mHeadSize;

    int32_t mSmVersion;
    Launch_params mLaunchParams;
};

} // namespace drivellm