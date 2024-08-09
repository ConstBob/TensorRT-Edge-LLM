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

#include <NvInferRuntime.h>

namespace drivellm
{

struct XQALaunchParams
{
    struct KVCache
    {
        void* data = nullptr;
        int32_t const* sequence_lengths = nullptr;
        uint32_t capacity = 0;
    };
    
    void* output = nullptr;
    void const* qInputPtr = nullptr;
    KVCache kvCache;
    float const* kvScale = nullptr;
    int32_t* semaphores = nullptr;
    void* scratch = nullptr;

    // MHA parameter to locate a kernel to launch.
    int32_t numQheads = 0;
    int32_t numKVheads = 0;
    int32_t headSize = 0;
    int32_t batchSize = 0;

    // I/O type of the kernel
    nvinfer1::DataType dataType;

    // Hardware specific config.
    int32_t mSmVersion;
};

class DecoderXQARunner
{
public:
    DecoderXQARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion);

    ~DecoderXQARunner() = default;

    size_t getWorkspaceSize(int max_num_tokens);

    // The call load and prepare kernel to dispatch. After the call, the CUmodule will be loaded to device
    // and kernel functions are prepared to launch.
    int32_t prepareToRun(XQALaunchParams const& params);

    // Dispatch XQA kernel and compute the attention result.
    void dispatchXQAKernel(XQALaunchParams & params, cudaStream_t const& stream);

    // Initialize a XQA parameter with MHA and hardware configuration to query. The XQA parameter can be used by
    // prepareToRun() to query kernel to dispatch. Device pointer shall be setup by caller to dispatch XQA kernel.
    XQALaunchParams initXQAParams();

private:
    nvinfer1::DataType mDataType;
    int32_t mBatchSize;
    int32_t mNumHeads;
    int32_t mNumKVHeads;
    int32_t mHeadSize;

    int32_t mSmVersion;
};

} // namespace drivellm

