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

#include "fmhaParams_v2.h"

#include <NvInferRuntime.h>
namespace drivellm
{

class ContextFMHARunner
{
public:
    ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t paddedSeqLen, int32_t numQHeads,
        int32_t numKvHeads, int32_t headSize, int32_t smVersion, AttentionInputLayout inputLayout);

    ContextFMHARunner() = delete;

    ~ContextFMHARunner() = default;

    size_t getWorkspaceSize();

    // The function will setup kernel parameters except device pointers.
    // Device pointers shall be set by caller of FMHA runner.
    void setupParams(FusedMultiheadAttentionParamsV2& params);

    // Dispatch FMHA kernel.
    void dispatchFMHAKernel(FusedMultiheadAttentionParamsV2& params, cudaStream_t const& stream);

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
    LaunchParams mLaunchParams;
};

} // namespace drivellm