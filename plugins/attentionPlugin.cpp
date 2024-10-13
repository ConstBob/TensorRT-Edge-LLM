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

// Align with Meta's implementation for rotary embedding.
// Use a different set of configuration could harm MMLU score noticably.
constexpr float kROPE_BASE_FREQUENCY = 500000.f;
constexpr float kROPE_SCALE = 1.0f;
constexpr PositionEmbeddingType kROPE_TYPE = PositionEmbeddingType::kROPE_ROTATE_HALF;
constexpr RopeInitType kROPE_INIT_TYPE = RopeInitType::kDEFAULT;

constexpr int32_t kDEVICE_ALIGNMENT{128}; // Make sure all device pointers are aligned by 128.

int8_t* alignDevicePtr(void* ptr)
{
    // Convert the pointer to an integer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + kDEVICE_ALIGNMENT) & ~static_cast<uintptr_t>(kDEVICE_ALIGNMENT);

    // Convert the aligned address back to a pointer
    return reinterpret_cast<int8_t*>(aligned_addr);
}

} // namespace

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name)
{
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));
    mSMVersion = prop.major * 10 + prop.minor;

    // Initialize the attention kernel runner and load the cubinModule / kernel function.
    // We will construct new runner at enqueue time with execution time batch / SequenceLen.
    constexpr int32_t kDEFAULT_BATCH{1};
    constexpr int32_t kDEFAULT_CONTEXT{128};
    auto fmhaRunner = ContextFMHARunner(
        mDataType, kDEFAULT_BATCH, kDEFAULT_CONTEXT, mNumHeadQ, mNumHeadK, mNumElemPerHead, mSMVersion);
    auto xqaRunner = DecoderXQARunner(mDataType, kDEFAULT_BATCH, mNumHeadQ, mNumHeadK, mNumElemPerHead, mSMVersion);

    fmhaRunner.prepareToRun();
    xqaRunner.prepareToRun();
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
    catch (std::exception const& e)
    {
    }
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
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      GEMM-QKV tensor (FP16) with shape [B, S, Hq+Hk+Hv,D]
    //      KV-cache tensor (FP16) with shape [B, 2, Hkv, Smax, D], here Smax is the max capacity of the linear kvcache
    //      buffer. Real context length: [B] (a vector of scalars) with type int32_t, the tensor should reside on host.
    // Support context/generation phase outputs:
    //      attention result (FP16) with shape [B, S, Hq, D]
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
            status &= tensorDim.d[2] == (mNumHeadQ + mNumHeadK + mNumHeadV) * mNumElemPerHead;
        }
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
            status &= tensorDim.d[1] == 2; // Specify K and V
            status &= tensorDim.d[2] == mNumHeadK;
            status &= tensorDim.d[3] == mTotalContextLen || tensorDim.d[3] == 0;
            status &= tensorDim.d[4] == mNumElemPerHead;
        }
        return status;
    };

    auto checkSequenceLen = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
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
        default: break;
        }
        return result;
    }
    catch (std::exception const& e)
    {
    }
    return false;
}

int32_t AttentionPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
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
    catch (std::exception const& e)
    {
    }
    return 1;
}

int32_t AttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // Here we may want to switch different MHA runner.
    return 0;
}

// TODO: extend the worksapce calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // We may want to reserve workspace here, need to determine more details after implementing the runners.
    // For FMHA kernel we need a buffer to store prefix sum of context lengths.
    // For GQA kernel we need to reserve a buffer space to store the Q tensor after rope transformation.
    constexpr int32_t nbBytesPerData{2};
    int32_t const nbBytesQTensor = nbBytesPerData * mMaxBatchSize * mNumHeadQ * mNumElemPerHead;

    // Add alignment to ensure we have enough device space at worst scenrio.
    return nbBytesQTensor + kDEVICE_ALIGNMENT;
}

int32_t AttentionPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == getNbOutputs());
        assert(nbInputs == 3);
        outputTypes[0] = DataType::kHALF;
        outputTypes[1] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
    }

    // non-zero return value treated as error code.
    return 1;
}

int32_t AttentionPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
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

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    constexpr int32_t kQKV_INPUT_IDX{0};
    constexpr int32_t kKV_CACHE_INPUT_OUTPUT_IDX{1};
    constexpr int32_t kINPUT_LENGTH_INPUT_IDX{2};
    constexpr int32_t kATTENTION_OUTPUT_IDX{0};

    // Obtain execution time batch size and input context length
    constexpr int32_t kQKV_INPUT_BATCH_DIM_IDX{0};
    constexpr int32_t kQKV_INPUT_SEQLEN_DIM_IDX{1};
    PluginTensorDesc const& qkvInputDesc = inputDesc[kQKV_INPUT_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputDesc.dims.d[kQKV_INPUT_BATCH_DIM_IDX]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputDesc.dims.d[kQKV_INPUT_SEQLEN_DIM_IDX]);

    check(runtimeBatchSize < mMaxBatchSize,
        "Runtime batchsize exceed max batch size. This will overflow device data buffer");
    check(runtimeSeqLen < mTotalContextLen,
        "Runtime sequence length exceed max total context lengths. This will overflow KVCache buffer");

    // Check whether the plugin is running context or generation. Determine by whether input kvCache has zero length.
    constexpr int32_t kINPUT_CACHE_SEQUENCE_DIM_IDX{3};
    PluginTensorDesc const& kvInputDesc = inputDesc[kKV_CACHE_INPUT_OUTPUT_IDX];
    bool const isContextPhase = kvInputDesc.dims.d[kINPUT_CACHE_SEQUENCE_DIM_IDX] == 0;

    half* qkvDevicePtr = reinterpret_cast<half*>(const_cast<void*>(inputs[kQKV_INPUT_IDX]));
    int32_t const* seqLengthDevicePtr = reinterpret_cast<int32_t const*>(inputs[kINPUT_LENGTH_INPUT_IDX]);

    half* attentionResultDevicePtr = reinterpret_cast<half*>(outputs[kATTENTION_OUTPUT_IDX]);
    half* kvCacheDevicePtr = reinterpret_cast<half*>(outputs[kKV_CACHE_INPUT_OUTPUT_IDX]);

    // Align workspace to be minimal aligned.
    int8_t* alignedWorkspacePtr = alignDevicePtr(workspace);

    if (isContextPhase)
    {
        // At Context phase. Do 1. Apply rope and write KVCache. 2. Dispatch FMHA runner.
        // RoPE kernel now only handle padded input sequence, we treat all "tokens" in the
        // padded input as processing targets.
        // TODO: Explore non-padded input format.
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;
        invokeContextApplyRopeUpdateKVFP16(qkvDevicePtr, nullptr, kvCacheDevicePtr, seqLengthDevicePtr, mNumHeadQ,
            mNumHeadK, mNumElemPerHead, mTotalContextLen, runtimeSeqLen, kROPE_TYPE, kROPE_BASE_FREQUENCY, kROPE_SCALE,
            kROPE_INIT_TYPE, totalProcessToken, stream);

        // Prepare FMHA_v2 params to launch FMHA kernel
        auto fmhaRunner = ContextFMHARunner(
            mDataType, runtimeBatchSize, runtimeSeqLen, mNumHeadQ, mNumHeadK, mNumElemPerHead, mSMVersion);
        Fused_multihead_attention_params_v2 params{};
        params.clear();
        fmhaRunner.setupParams(params);

        // Compute the prefix sum of sequence length.
        int32_t* prefixSumDevicePtr = reinterpret_cast<int32_t*>(alignedWorkspacePtr);
        invokePrefixSum(seqLengthDevicePtr, prefixSumDevicePtr, runtimeBatchSize, stream);

        // Set device ptr for FMHA kernel.
        params.qkv_ptr = qkvDevicePtr;
        params.cu_q_seqlens = prefixSumDevicePtr;
        params.o_ptr = attentionResultDevicePtr;

        // Dispatch FMHA kernel
        fmhaRunner.dispatchFMHAKernel(params, stream);
    }
    else
    {
        // Generation phase we first prepare Q vector and update KVCache.
        // Currently we only supports generating one token per sequence.
        half* qVecDevicePtr = reinterpret_cast<half*>(alignedWorkspacePtr);
        int32_t const totalProcessToken = runtimeBatchSize;
        invokeGenerationApplyRopeUpdateKVFP16(qkvDevicePtr, qVecDevicePtr, kvCacheDevicePtr, seqLengthDevicePtr,
            mNumHeadQ, mNumHeadK, mNumElemPerHead, mTotalContextLen, runtimeSeqLen, kROPE_TYPE, kROPE_BASE_FREQUENCY,
            kROPE_SCALE, kROPE_INIT_TYPE, totalProcessToken, stream);

        // Prepare GQA runner parameter to dispatch kernel
        auto xqaRunner
            = DecoderXQARunner(mDataType, runtimeBatchSize, mNumHeadQ, mNumHeadK, mNumElemPerHead, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.output = attentionResultDevicePtr;
        params.qInputPtr = qVecDevicePtr;
        params.kvCache.data = kvCacheDevicePtr;
        params.kvCache.sequence_lengths = seqLengthDevicePtr;
        params.kvCache.capacity = mTotalContextLen;

        // dispatch GQA runner.
        xqaRunner.dispatchXQAKernel(params, stream);
    }
    return 0;
}

AttentionPluginCreator::AttentionPluginCreator() {}

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

nvinfer1::IPluginV3* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(std::string(name));
    return plugin;
}