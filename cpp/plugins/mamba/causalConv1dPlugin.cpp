/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "causalConv1dPlugin.h"

#include "common/logger.h"
#include "kernels/mamba/causalConv1d.h"
#include "plugins/utils/pluginUtils.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <mutex>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kCAUSAL_CONV_PLUGIN_VERSION{"1"};
constexpr char const* kCAUSAL_CONV_PLUGIN_NAME{"causal_conv1d"};

constexpr int32_t kIN_X_IDX{0};
constexpr int32_t kIN_WEIGHT_IDX{1};
constexpr int32_t kIN_BIAS_IDX{2};
constexpr int32_t kIN_CONV_STATE_IDX{3};
constexpr int32_t kOUT_IDX{0};
constexpr int32_t kOUT_CONV_STATE_IDX{1};
constexpr int32_t kNUM_INPUTS{4};
constexpr int32_t kNUM_OUTPUTS{2};

std::optional<int32_t> parsePluginIntField(std::string const& fieldName, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& pluginField = fc->fields[i];
        if (fieldName != pluginField.name || pluginField.length != 1 || pluginField.data == nullptr)
        {
            continue;
        }
        if (pluginField.type == PluginFieldType::kINT32)
        {
            return *static_cast<int32_t const*>(pluginField.data);
        }
        if (pluginField.type == PluginFieldType::kINT64)
        {
            return static_cast<int32_t>(*static_cast<int64_t const*>(pluginField.data));
        }
    }
    return std::nullopt;
}

} // namespace

PluginFieldCollection CausalConv1dPluginCreator::mFieldCollection{};
std::vector<PluginField> CausalConv1dPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CausalConv1dPluginCreator);

CausalConv1dPlugin::CausalConv1dPlugin(
    std::string const& name, int32_t stride, int32_t padding, int32_t dilation, int32_t groups)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mGroups(groups)
{
}

CausalConv1dPlugin::~CausalConv1dPlugin() {}

IPluginCapability* CausalConv1dPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* CausalConv1dPlugin::clone() noexcept
{
    auto* plugin = new CausalConv1dPlugin(mLayerName, mStride, mPadding, mDilation, mGroups);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t CausalConv1dPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t CausalConv1dPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[kOUT_IDX] = inputTypes[kIN_X_IDX];
    outputTypes[kOUT_CONV_STATE_IDX] = inputTypes[kIN_X_IDX];
    return 0;
}

int32_t CausalConv1dPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /* exprBuilder */) noexcept
{
    // Output: same shape as x [batch, seq_len, dim].
    outputs[kOUT_IDX].nbDims = inputs[kIN_X_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_IDX].nbDims; ++i)
    {
        outputs[kOUT_IDX].d[i] = inputs[kIN_X_IDX].d[i];
    }
    // Conv state output: same shape as conv_state input [batch, dim, kernel].
    outputs[kOUT_CONV_STATE_IDX].nbDims = inputs[kIN_CONV_STATE_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_CONV_STATE_IDX].nbDims; ++i)
    {
        outputs[kOUT_CONV_STATE_IDX].d[i] = inputs[kIN_CONV_STATE_IDX].d[i];
    }
    return 0;
}

bool CausalConv1dPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return false;
    }
    bool const isLinear = inOut[pos].desc.format == TensorFormat::kLINEAR;
    bool const isSupportedType = inOut[pos].desc.type == DataType::kHALF;
    if (!isLinear || !isSupportedType)
    {
        return false;
    }
    if (pos > 0)
    {
        return inOut[pos].desc.type == inOut[kIN_X_IDX].desc.type;
    }
    return true;
}

int32_t CausalConv1dPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return -1;
    }
    if (in[kIN_X_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR(
            "causal_conv1d: only FP16 input is supported; got type %d", static_cast<int32_t>(in[kIN_X_IDX].desc.type));
        return -1;
    }
    return 0;
}

size_t CausalConv1dPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t CausalConv1dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    auto const& xDesc = inputDesc[kIN_X_IDX];
    auto const& wDesc = inputDesc[kIN_WEIGHT_IDX];
    auto const& outDesc = outputDesc[kOUT_IDX];

    if (xDesc.dims.nbDims != 3 || wDesc.dims.nbDims != 3 || outDesc.dims.nbDims != 3)
    {
        LOG_ERROR("causal_conv1d expects 3D tensors for x/weight/output.");
        return 1;
    }

    int32_t const batch = static_cast<int32_t>(xDesc.dims.d[0]);
    int32_t const seqLen = static_cast<int32_t>(xDesc.dims.d[1]);
    int32_t const dim = static_cast<int32_t>(xDesc.dims.d[2]);
    int32_t const width = static_cast<int32_t>(wDesc.dims.d[2]);

    int32_t const groups = mGroups == 0 ? dim : mGroups;
    if (groups != dim)
    {
        LOG_ERROR("causal_conv1d currently supports depthwise conv only: groups=%d, dim=%d", groups, dim);
        return 1;
    }

    void* convStateOut = outputs[kOUT_CONV_STATE_IDX];

    namespace rt = trt_edgellm::rt;

    if (seqLen > 1)
    {
        // PREFILL path
        int32_t const outSeqLen = static_cast<int32_t>(outDesc.dims.d[1]);

        // Non-owning tensor views; const_cast is safe (read-only in kernel).
        auto xTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto weightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto biasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto outTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, outSeqLen, dim}, rt::DeviceType::kGPU, xDesc.type};

        trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasTensor));
        mamba_ssm::invokeCausalConv1d(xTensor, weightTensor, biasOpt, outTensor, mStride, mPadding, mDilation, stream);
        auto captureXTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto captureStateTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        mamba_ssm::invokeCaptureConvState(captureXTensor, captureStateTensor, stream);
    }
    else
    {
        // DECODE path (seqLen == 1): copy conv_state to output, then shift+insert and compute dot product.
        if (convStateOut != inputs[kIN_CONV_STATE_IDX])
        {
            size_t const stateBytes = static_cast<size_t>(batch) * dim * width * sizeof(half);
            cudaMemcpyAsync(convStateOut, inputs[kIN_CONV_STATE_IDX], stateBytes, cudaMemcpyDeviceToDevice, stream);
        }

        auto decodeStateTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeNewColTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        mamba_ssm::invokeConvStateShiftInsert(decodeStateTensor, decodeNewColTensor, stream);

        auto decodeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        trt_edgellm::rt::OptionalInputTensor decodeBiasOpt = std::optional(std::cref(decodeBiasTensor));
        mamba_ssm::invokeCausalConv1dDecode(
            decodeStateTensor, decodeWeightTensor, decodeBiasOpt, decodeOutTensor, stream);
    }

    return 0;
}

int32_t CausalConv1dPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* CausalConv1dPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* CausalConv1dPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("stride", &mStride, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("padding", &mPadding, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("dilation", &mDilation, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("groups", &mGroups, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

char const* CausalConv1dPlugin::getPluginName() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

char const* CausalConv1dPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CausalConv1dPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* CausalConv1dPlugin::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

CausalConv1dPluginCreator::CausalConv1dPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* CausalConv1dPluginCreator::getPluginName() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

PluginFieldCollection const* CausalConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void CausalConv1dPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* CausalConv1dPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* CausalConv1dPluginCreator::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

IPluginV3* CausalConv1dPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        int32_t const stride = parsePluginIntField("stride", fc).value_or(1);
        int32_t const padding = parsePluginIntField("padding", fc).value_or(0);
        int32_t const dilation = parsePluginIntField("dilation", fc).value_or(1);
        int32_t const groups = parsePluginIntField("groups", fc).value_or(0);
        auto* plugin = new CausalConv1dPlugin(name, stride, padding, dilation, groups);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create CausalConv1dPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
