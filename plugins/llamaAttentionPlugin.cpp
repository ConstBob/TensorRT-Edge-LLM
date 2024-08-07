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

#include "llamaAttentionPlugin.h"
#include <cassert>

using namespace nvinfer1;
using namespace drivellm;

namespace
{
constexpr char const* kLLAMA_ATTENTION_PLUGIN_VERSION{"v1"};
constexpr char const* kLLAMA_ATTENTION_PLUGIN_NAME{"LlamaAttentionPlugin"};
} // namespace

LlamaAttentionPlugin::LlamaAttentionPlugin(std::string const& name)
    : mLayerName(name)
{
}

LlamaAttentionPlugin::~LlamaAttentionPlugin()
{
    // Do nothing now, but we may want to release the cudaModule with real implementation.
}

nvinfer1::IPluginCapability* LlamaAttentionPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
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

IPluginV3* LlamaAttentionPlugin::clone() noexcept
{
    LlamaAttentionPlugin* plugin = new LlamaAttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* LlamaAttentionPlugin::getPluginName() const noexcept
{
    return kLLAMA_ATTENTION_PLUGIN_NAME;
}

char const* LlamaAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LlamaAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* LlamaAttentionPlugin::getPluginVersion() const noexcept
{
    return kLLAMA_ATTENTION_PLUGIN_VERSION;
}

int32_t LlamaAttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output atention result and kv-cache.
    return 2;
}

bool LlamaAttentionPlugin::supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      GEMM-QKV tensor (FP16) with shape [B, S, Hq+Hk+Hv,D]
    //      KV-cache tensor (FP16) with shape [B, 2, Hkv, Smax, D], here Smax is the max capacity of the linear kvcache buffer.
    //      Rotary transformation coefficient (FP16) with shape [S, 2, D], S can be 1 (generation) or supported input context length.
    //      Real context length: [1] (a scalar) with type int32_t, the tensor should reside on host.
    // Support context/generation phase outputs:
    //      attention result (FP16) with shape [B, S. Hq, D]
    //      KV-cache tensor, same as the above.
    // In above context, S can be 1 (generation) or supported input context length.
    auto checkGemmQKV = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
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
            status &= tensorDim.d[2] == (mNumHeadQ + mNumHeadK + mNumHeadV);
            status &= tensorDim.d[3] == mNumElemPerHead;
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
            status &= tensorDim.d[0] == mBatchSize;
            status &= tensorDim.d[1] == 2;      // Specify K and V
            status &= tensorDim.d[2] == mNumHeadK;
            status &= tensorDim.d[3] == mTotalContextLen;
            status &= tensorDim.d[4] == mNumElemPerHead;
        }
        return status;
    };

    auto checkRotaryMatrix = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[0] == 1 || tensorDim.d[0] == mInputContextLen;
            status &= tensorDim.d[1] == 2;  // From rotary formula
            status &= tensorDim.d[2] == mNumElemPerHead;
        }
        return status;
    };

    auto checkSequenceLen = [this](nvinfer1::DynamicPluginTensorDesc const& dynamicDesc) {
        bool status{true};
        auto const& tensorDesc = dynamicDesc.desc;
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[0] == 1;  // single scalar
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
        assert(nbInputs == 4 && nbOutputs == 2);
        assert(pos < (nbInputs + nbOutputs));
        bool result{false};
        switch (pos)
        {
        case 0: result = checkGemmQKV(inOut[0]); break;
        case 1: result = checkKVCache(inOut[1]); break;
        case 2: result = checkRotaryMatrix(inOut[2]); break;
        case 3: result = checkSequenceLen(inOut[3]); break;
        case 4: result = checkAttentionOutput(inOut[4]); break;
        case 5: result = checkKVCache(inOut[5]); break;
        default:
            break;
        }
        return result;
    }
    catch (std::exception const& e) {}
    return false;
}

int32_t LlamaAttentionPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept 
{
    try
    {
        assert(inputs != nullptr);
        assert(nbInputs == 4);
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

int32_t LlamaAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept
{
    // Here we may want to switch different MHA runner.
    return 0;
}

size_t LlamaAttentionPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // We may want to reserve workspace here, need to determine more details after implementing the runners.
    return 0;
}

int32_t LlamaAttentionPlugin::getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == getNbOutputs());
        assert(nbInputs == 4);
        assert(inputTypes[0] == DataType::kHALF);
        outputTypes[0] = DataType::kHALF;
        outputTypes[1] = DataType::kHALF;
        return 0;
    }
    catch(const std::exception& e) {}

    // non-zero return value treated as error code.
    return 1;
}

int32_t LlamaAttentionPlugin::onShapeChange(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // We may need switch MHA runner, but it seems not necessary since we will receive shapes in enqueue as well.
    return 0;
}

nvinfer1::IPluginV3* LlamaAttentionPlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    LlamaAttentionPlugin* plugin = new LlamaAttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

PluginFieldCollection const* LlamaAttentionPlugin::getFieldsToSerialize() noexcept
{
    return nullptr;
}

int32_t LlamaAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    return 0;
}