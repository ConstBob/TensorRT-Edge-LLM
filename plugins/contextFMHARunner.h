/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t numQHeads, int32_t seqLen,
        int32_t numKvHeads, int32_t headSize, int32_t smVersion);

    ~ContextFMHARunner() = default;

    size_t getWorkspaceSize();

    // The function will setup an ampty FMHA_v2 parameter. Device pointers shall be setup via caller.
    void setupParams(Fused_multihead_attention_params_v2& params);

    // Dispatch XQA kernel and compute the attention result.
    void dispatchFMHAKernel(Fused_multihead_attention_params_v2 & params, cudaStream_t const& stream);

    // The call load and prepare kernel to dispatch. After the call, the CUmodule will be loaded to device
    // and kernel functions are prepared to launch.
    int32_t prepareToRun();

private:
    Launch_params mLaunchParams;
    int32_t mBatchSize;
    int32_t mSequenceLen;
    int32_t mNumHeads;
    int32_t mNumKVHeads;
    int32_t mHeadSize;

    nvinfer1::DataType mDataType;
    int32_t mSmVersion;
};

}