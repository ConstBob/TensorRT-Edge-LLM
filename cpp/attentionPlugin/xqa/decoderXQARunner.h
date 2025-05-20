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

    // Device memory pointers to launch XQA kernel.
    void* output = nullptr;
    void const* qInputPtr = nullptr;
    KVCache kvCache;
    float const* kvScale = nullptr;
    int32_t* semaphores = nullptr;
    void* scratch = nullptr;

    // Unique device memory pointer for spec-decode tree attention.
    void* treeAttnMask = nullptr;
    int32_t* qCuSeqLen = nullptr;

    // MHA parameter to locate a kernel to launch.
    int32_t numQheads = 0;
    int32_t numKVheads = 0;
    int32_t headSize = 0;
    int32_t batchSize = 0;

    // Parameters for spec-decode tree attention
    int32_t qSeqLen = 0;
    float qScale = 1.0F;
    int32_t headGroupSize = 0;

    // I/O type of the kernel
    nvinfer1::DataType dataType;
};

class DecoderXQARunner
{
public:
    DecoderXQARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t numQHeads, int32_t numKvHeads,
        int32_t headSize, int32_t smVersion);

    DecoderXQARunner() = default;

    ~DecoderXQARunner() = default;

    // Dispatch XQA kernel and compute the attention result.
    void dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream);
    void dispatchSpecDecodeXQAKernel(XQALaunchParams& params, cudaStream_t const& stream);

    // Initialize a XQA parameter with MHA and hardware configuration to query. The XQA parameter can be used by
    // prepareToRun() to query kernel to dispatch. Device pointer shall be setup by caller to dispatch XQA kernel.
    XQALaunchParams initXQAParams();

    static bool canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t smVersion, nvinfer1::DataType dataType);
    static bool loadDecodeXQAKernels(int32_t smVersion, nvinfer1::DataType dataType, bool useSpecDecodeKernels);

private:
    nvinfer1::DataType mDataType;
    uint32_t mBatchSize;
    uint32_t mNumHeads;
    uint32_t mNumKVHeads;
    uint32_t mHeadSize;

    int32_t mSmVersion;
};

} // namespace drivellm