/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "attentionPlugin.h"
#include "pluginUtils.h"
#include "utilKernels.h"

#include <cassert>
#include <vector>

using namespace nvinfer1;
using namespace drivellm;

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};

constexpr float kROPE_BASE_FREQUENCY = 10000.f;
constexpr float kROPE_SCALE = 1.0f;

constexpr int32_t kDEVICE_ALIGNMENT{128};       // Make sure all device pointers are aligned by 128.


// TODO: Use a CUDA kernel to do the predix sum.
void getSeqLenPrefixLength(int32_t* device_ptr_predix_sum, int32_t const* device_ptr_seqlen, int32_t nbSeq)
{
    std::vector<int32_t> seqlenVec(nbSeq);
    checkCuda(cudaMemcpy(seqlenVec.data(), device_ptr_seqlen, sizeof(int32_t) * nbSeq, cudaMemcpyDeviceToHost));
    std::vector<int32_t> prefixSumVec(nbSeq + 1, 0);
    for (int32_t i = 0; i < nbSeq; ++i)
    {
        prefixSumVec[i+1] = seqlenVec[i] + prefixSumVec[i];
    }
    checkCuda(cudaMemcpy(device_ptr_predix_sum, prefixSumVec.data(), sizeof(int32_t) * (nbSeq + 1), cudaMemcpyHostToDevice));
}

void* alignDevicePtr(void* ptr) {
    // Convert the pointer to an integer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + kDEVICE_ALIGNMENT) & ~static_cast<uintptr_t>(kDEVICE_ALIGNMENT);

    // Convert the aligned address back to a pointer
    return reinterpret_cast<void*>(aligned_addr);
}

} // namespace

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name)
{
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));
    int32_t smVersion = prop.major + prop.minor;

    mFMHARunner = ContextFMHARunner(mDataType, mBatchSize, mInputContextLen,
        mNumHeadQ, mNumHeadK, mNumElemPerHead, smVersion);
    mGQARunner = DecoderXQARunner(mDataType, mBatchSize, mNumHeadQ,
        mNumHeadK, mNumElemPerHead, smVersion);
}

AttentionPlugin::~AttentionPlugin()
{
    // Do nothing now, The plugin class only contains basic data structures and CUDA modules are
    // are not managed by the plugin itself.
}

nvinfer1::IPluginCapability* AttentionPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        assert(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e) {}
    return nullptr;
}

IPluginV3* AttentionPlugin::clone() noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* AttentionPlugin::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output atention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      GEMM-QKV tensor (FP16) with shape [B, S, Hq+Hk+Hv,D]
    //      KV-cache tensor (FP16) with shape [B, 2, Hkv, Smax, D], here Smax is the max capacity of the linear kvcache buffer.
    //      Real context length: [B] (a vector of scalars) with type int32_t, the tensor should reside on host.
    // Support context/generation phase outputs:
    //      attention result (FP16) with shape [B, S. Hq, D]
    //      KV-cache tensor, same as the above.
    // In above context, S can be 1 (generation) or supported input context length.
    auto checkGemmQKV = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mBatchSize;
            status &= tensorDim.d[1] == 1 || tensorDim.d[1] == mInputContextLen;
            status &= tensorDim.d[2] == (mNumHeadQ + mNumHeadK + mNumHeadV) * mNumElemPerHead;
        }
        // std::cout << "Dims: " <<tensorDim.d[0] << " "<< tensorDim.d[1] << " " << tensorDim.d[2] << std::endl;
        return status;
    };

    auto checkKVCache = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 5;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[0] == mBatchSize;
            status &= tensorDim.d[1] == 2;      // Specify K and V
            status &= tensorDim.d[2] == mNumHeadK;
            status &= tensorDim.d[3] == mTotalContextLen || tensorDim.d[3] == 0;
            status &= tensorDim.d[4] == mNumElemPerHead;
        }
        return status;
    };


    auto checkSequenceLen = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kINT64;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[0] == mBatchSize;  // single scalar
        }
        return status;
    };

    auto checkAttentionOutput = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 4;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[0] == mBatchSize;
            status &= tensorDim.d[1] == 1 || tensorDim.d[1] == mInputContextLen;
            status &= tensorDim.d[2] == mNumHeadQ;
            status &= tensorDim.d[3] == mNumElemPerHead;
        }
        return status;
    };

    try
    {
        assert(nbInputs == 3 && nbOutputs == 2);
        assert(pos < (nbInputs + nbOutputs));
        bool result{false};
        switch (pos)
        {
        case 0: result = checkGemmQKV(inOut[0]); break;
        case 1: result = checkKVCache(inOut[1]); break;
        case 2: result = checkSequenceLen(inOut[2]); break;
        case 3: result = checkAttentionOutput(inOut[3]); break;
        case 4: result = checkKVCache(inOut[4]); break;
        default:
            break;
        }
        return result;
    }
    catch (std::exception const& e) {}
    return false;
}

int32_t AttentionPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept 
{
    try
    {
        assert(inputs != nullptr);
        assert(nbInputs == 3);
        assert(nbOutputs == getNbOutputs());

        // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = exprBuilder.constant(mNumHeadQ);
        outputs[0].d[3] = exprBuilder.constant(mNumElemPerHead);

        // Output[1] is KVCache, identical input[1]
        outputs[1] = inputs[1];
        return 0;
    }
    catch (std::exception const& e) {}
    return 1;
}

int32_t AttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept
{
    // Here we may want to switch different MHA runner.
    return 0;
}

size_t AttentionPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // We may want to reserve workspace here, need to determine more details after implementing the runners.
    // For FMHA kernel we need a buffer to store prefix sum of context lengths.
    // For GQA kernel we need to reserve a buffer space to store the Q tensor after rope transformation.
    constexpr int32_t nbBytesPerData{2};
    int32_t const nbBytesQTensor = nbBytesPerData * mBatchSize * mNumHeadQ * mNumElemPerHead;

    // Add alignment to ensure we have enough device space at worst scenrio.
    return nbBytesQTensor + kDEVICE_ALIGNMENT;
}

int32_t AttentionPlugin::getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == getNbOutputs());
        assert(nbInputs == 3);
        outputTypes[0] = DataType::kHALF;
        outputTypes[1] = DataType::kHALF;
        return 0;
    }
    catch(const std::exception& e) {}

    // non-zero return value treated as error code.
    return 1;
}

int32_t AttentionPlugin::onShapeChange(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // We may need switch MHA runner, but it seems not necessary since we will receive shapes in enqueue as well.
    return 0;
}

nvinfer1::IPluginV3* AttentionPlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

PluginFieldCollection const* AttentionPlugin::getFieldsToSerialize() noexcept
{
    return nullptr;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    constexpr int32_t kQKV_INPUT_IDX{0};
    constexpr int32_t kKV_CACHE_INPUT_OUTPUT_IDX{1};
    constexpr int32_t kINPUT_LENGTH_INPUT_IDX{2};
    constexpr int32_t kINPUT_CACHE_SEQUENCE_DIM_IDX{3};
    constexpr int32_t kATTENTION_OUTPUT_IDX{0};

    // Check whether the plugin is running context or generation. Determine by whether input kvCache has zero length.
    PluginTensorDesc const& kvInputDesc = inputDesc[kKV_CACHE_INPUT_OUTPUT_IDX];
    bool const isContextPhase = kvInputDesc.dims.d[kINPUT_CACHE_SEQUENCE_DIM_IDX] == 0;

    half* qkvDevicePtr = reinterpret_cast<half*>(const_cast<void*>(inputs[kQKV_INPUT_IDX]));
    int32_t const* seqLengthDevicePtr = reinterpret_cast<int32_t const*>(inputs[kINPUT_LENGTH_INPUT_IDX]);

    half* attentionResultDevicePtr = reinterpret_cast<half*>(outputs[kATTENTION_OUTPUT_IDX]);
    half* kvCacheDevicePtr = reinterpret_cast<half*>(outputs[kKV_CACHE_INPUT_OUTPUT_IDX]);

    // Align workspace to be minimal aligned.
    void* alignedWorkspacePtr = alignDevicePtr(workspace);

    if (isContextPhase)
    {
        // At Context phase. Do 1. Apply rope and write KVCache. 2. Dispatch FMHA runner.
        invokeContextApplyRopeUpdateKVFP16(qkvDevicePtr, nullptr, kvCacheDevicePtr, seqLengthDevicePtr,
            mNumHeadQ, mNumHeadK, mNumElemPerHead, mTotalContextLen, kROPE_BASE_FREQUENCY, kROPE_SCALE, mInputContextLen, stream);
        
        // Prepare FMHA_v2 params to launch FMHA kernel
        Fused_multihead_attention_params_v2 params{};
        params.clear();
        mFMHARunner.setupParams(params);

        // Compute the prefix sum of sequence length.
        getSeqLenPrefixLength(reinterpret_cast<int32_t*>(alignedWorkspacePtr), seqLengthDevicePtr, mBatchSize);
        
        // Set device ptr for FMHA kernel.
        params.qkv_ptr = qkvDevicePtr;
        params.cu_seqlens = reinterpret_cast<int32_t*>(alignedWorkspacePtr);
        params.o_ptr = attentionResultDevicePtr;

        // Dispatch FMHA kernel
        mFMHARunner.dispatchFMHAKernel(params, stream);
    }
    else
    {
        // Generation phase we first prepare Q vector and update KVCache.
        half* qVecDevicePtr = reinterpret_cast<half*>(alignedWorkspacePtr);
        invokeGenerationApplyRopeUpdateKVFP16(qkvDevicePtr, qVecDevicePtr, kvCacheDevicePtr, seqLengthDevicePtr,
            mNumHeadQ, mNumHeadK, mNumElemPerHead, mTotalContextLen, kROPE_BASE_FREQUENCY, kROPE_SCALE, 1, stream);

        // Prepare GQA runner parameter to dispatch kernel
        XQALaunchParams params = mGQARunner.initXQAParams();
        params.output = attentionResultDevicePtr;
        params.qInputPtr = qVecDevicePtr;
        params.kvCache.data = kvCacheDevicePtr;
        params.kvCache.sequence_lengths = seqLengthDevicePtr;
        params.kvCache.capacity = mTotalContextLen;

        // dispatch GQA runner.
        mGQARunner.dispatchXQAKernel(params, stream);
    }
    return 0;
}

AttentionPluginCreator::AttentionPluginCreator()
{
    mFieldCollection.nbFields = 0;
    mFieldCollection.fields = nullptr;
}

char const* AttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return "";
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV3* AttentionPluginCreator::createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(std::string(name));
    return plugin;
}  