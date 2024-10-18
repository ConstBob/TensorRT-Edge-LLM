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

#include "decoderXQARunner.h"
#include "pluginUtils.h"
#include "xqa/cubin/xqa_kernel_cubin.h"

#include <memory>
#include <mutex>
#include <unordered_map>

using namespace nvinfer1;
using namespace drivellm;

using Data_type = xqa::kernels::Data_type;

namespace
{

Data_type trtToXqaDataType(nvinfer1::DataType type)
{
    Data_type xqaType{Data_type::DATA_TYPE_FP16};
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: xqaType = Data_type::DATA_TYPE_FP32; break;
    case nvinfer1::DataType::kHALF: xqaType = Data_type::DATA_TYPE_FP16; break;
    case nvinfer1::DataType::kBF16: xqaType = Data_type::DATA_TYPE_BF16; break;
    case nvinfer1::DataType::kFP8: xqaType = Data_type::DATA_TYPE_E4M3; break;
    default: throw std::runtime_error("Unsupported datatype for XQA.");
    }
    return xqaType;
}
struct XQAKernelLoadHashKey
{
    Data_type data_type;
    int32_t sm;

    bool operator==(XQAKernelLoadHashKey const& other) const
    {
        return data_type == other.data_type && sm == other.sm;
    }
};

struct XQAKernelLoadHasher
{
    size_t operator()(XQAKernelLoadHashKey const& s) const
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.sm;
        return key;
    }
};

struct XQAKernelRuntimeHashKey
{
    Data_type kv_data_type;
    int32_t head_size;
    int32_t num_q_heads_per_kv;
    int32_t beam_size;

    bool operator==(XQAKernelRuntimeHashKey const& other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size;
    }
};

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQALaunchParams const& xqaParams)
{
    constexpr int32_t kBEAM_SIZE{1}; // Hardcode beam_size for now
    int32_t numQHeadPerKV = xqaParams.numQheads / xqaParams.numKVheads;
    return {trtToXqaDataType(xqaParams.dataType), xqaParams.headSize, numQHeadPerKV, kBEAM_SIZE};
}

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const
    {
        size_t key = s.kv_data_type;
        key <<= 16;
        key ^= s.head_size;
        key <<= 8;
        key ^= s.num_q_heads_per_kv;
        key <<= 8;
        key ^= s.beam_size;
        return key;
    }
};

struct XQAKernelFuncInfo
{
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
};

class XQAKernelList
{
    using TKernelMetaInfo = xqa::kernels::XQAKernelMetaInfo;

public:
    XQAKernelList(Data_type type, int32_t sm)
        : mDataType(type)
        , mSMVersion(sm)
    {
        mKernelMeta = &(xqa::kernels::sXqaKernelMetaInfo[0]);
        mKernelMetaCount = sizeof(xqa::kernels::sXqaKernelMetaInfo) / sizeof(xqa::kernels::sXqaKernelMetaInfo[0]);
    }

    void loadXQAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mDataType != mDataType || kernelMeta.mSM != mSMVersion || kernelMeta.mCubin == nullptr)
            {
                continue;
            }
            // Filter out kernel that irrelevant to this project.
            if (kernelMeta.mPagedKVCache == true || kernelMeta.mMultiQueryTokens == true || kernelMeta.mBeamWidth != 1
                || kernelMeta.mDataType != kernelMeta.mKVDataType)
            {
                continue;
            }
            // load CUmodule
            CUmodule hModule;
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hModule = findModuleIter->second;
            }
            else
            {
                checkCu(cuModuleLoadData(&hModule, kernelMeta.mCubin));
                mModules.insert(std::make_pair(kernelMeta.mCubin, hModule));
            }

            XQAKernelFuncInfo funcInfo{};
            checkCu(cuModuleGetFunction(&funcInfo.mDeviceFunction, hModule, kernelMeta.mFuncName));

            uint32_t* deviceSmemSize{nullptr};
            size_t dataSize{0};
            checkCu(cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&deviceSmemSize), &dataSize, hModule, "smemSize"));
            checkCuda(cudaMemcpy(&funcInfo.mSharedMemBytes, deviceSmemSize, dataSize, cudaMemcpyDeviceToHost));

            // Set 46KB threshold here because we have to take static/driver shared memory into consideration.
            // Default value for shared memory is 48KB, copy the logic from TRT-LLM
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                checkCu(cuFuncSetAttribute(funcInfo.mDeviceFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    funcInfo.mSharedMemBytes));
            }
            XQAKernelRuntimeHashKey hashKey{
                kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mNumQHeadsOverKV, kernelMeta.mBeamWidth};
            mFunctions.insert(std::make_pair(hashKey, funcInfo));
        }
    }

    XQAKernelFuncInfo findKernelFunction(XQAKernelRuntimeHashKey const& key) const
    {
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end())
        {
            // Return empty function info.
            return XQAKernelFuncInfo{};
        }

        return findIter->second;
    }

protected:
    TKernelMetaInfo const* mKernelMeta;
    int32_t mKernelMetaCount;
    int32_t mSMVersion;
    Data_type mDataType;
    std::unordered_map<unsigned long long const*, CUmodule> mModules;

    std::unordered_map<XQAKernelRuntimeHashKey, XQAKernelFuncInfo, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{

public:
    XQAKernelList const* getXQAKernelList(Data_type type, int32_t sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{type, sm};

        auto const findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            XQAKernelList* newKernel = new XQAKernelList{type, sm};
            newKernel->loadXQAKernels();
            mKernels.insert(std::make_pair(hash_key, std::unique_ptr<XQAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static XQAKernelLoader& Get()
    {
        static std::unique_ptr<XQAKernelLoader> kernelLoader = nullptr;
        if (kernelLoader == nullptr)
        {
            kernelLoader = std::make_unique<XQAKernelLoader>(XQAKernelLoader());
        }

        return *kernelLoader;
    }

private:
    XQAKernelLoader() = default;

    std::unordered_map<XQAKernelLoadHashKey, const std::unique_ptr<XQAKernelList>, XQAKernelLoadHasher> mKernels;
};

inline XQAKernelList const* getXQAKernels(Data_type type, int32_t sm)
{
    return XQAKernelLoader::Get().getXQAKernelList(type, sm);
}

} // namespace

DecoderXQARunner::DecoderXQARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t numQHeads,
    int32_t numKvHeads, int32_t headSize, int32_t smVersion)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
{
}

size_t DecoderXQARunner::getWorkspaceSize(int max_num_tokens)
{
    // Right now we don't enable multiple block launch for XQA kernel, so it doesn't need additional
    return 0;
}

bool DecoderXQARunner::prepareToRun()
{
    // Load CUmodules to device and collect device functions.
    XQAKernelList const* xqaKernelList = getXQAKernels(trtToXqaDataType(mDataType), mSmVersion);
    XQAKernelRuntimeHashKey hashKey{trtToXqaDataType(mDataType), mHeadSize, mNumHeads / mNumKVHeads, 1};
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);

    // check if there is a valid kernel corresponding to the requested config.
    bool status = kernelInfo.mSharedMemBytes != 0;
    return status;
}

XQALaunchParams DecoderXQARunner::initXQAParams()
{
    XQALaunchParams params{};
    params.numQheads = mNumHeads;
    params.numKVheads = mNumKVHeads;
    params.headSize = mHeadSize;
    params.batchSize = mBatchSize;
    params.dataType = mDataType;

    return params;
}

void DecoderXQARunner::dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr,
        "Invalid device pointer passed to kernel dispatch function");

    auto hashKey = getRuntimeHashKeyFromXQAParams(params);
    XQAKernelList const* xqaKernelList = getXQAKernels(trtToXqaDataType(mDataType), mSmVersion);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    check(kernelInfo.mSharedMemBytes != 0, "No available kernel available for the GQA");

    void* kernelParams[] = {&params.numKVheads, &params.output, &params.qInputPtr, &params.kvCache, &params.batchSize,
        &params.kvScale, &params.semaphores, &params.scratch, nullptr};

    // The multi-block kernel launch is mainly for long sequence.
    // TODO: Add multiple block launch logic. The launch configuration highly depends on usecase and performance
    // context. The blockDims are hardcoded in both XQA project and TensorRT-LLM
    dim3 const dimGrid{1, mNumKVHeads, mBatchSize};
    dim3 const dimCta{128, 1, 2};
    checkCu(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y, dimCta.z,
        kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}