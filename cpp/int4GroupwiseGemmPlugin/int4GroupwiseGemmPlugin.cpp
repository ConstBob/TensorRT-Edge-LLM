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

#include "int4GroupwiseGemmPlugin.h"
#include "int4GroupwiseGemmPlugin/kernel/int4GroupwiseGemm.h"

#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

#include <iostream>

using namespace nvinfer1;
using namespace drivellm;

namespace
{
constexpr char const* kINT4_GEMM_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_GEMM_PLUGIN_NAME{"Int4GroupwiseGemmPlugin"};

constexpr int32_t kGROUP_SIZE{128};

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    [[maybe_unused]] nvinfer1::PluginFieldType toFieldType<T>()                                                                         \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
#undef SPECIALIZE_TO_FIELD_TYPE

template <typename T>
std::optional<T> parsePluginScalarField(std::string const& fieldName, nvinfer1::PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& pluginField = fc->fields[i];
        if (fieldName.compare(pluginField.name) == 0)
        {
            assert(toFieldType<T>() == pluginField.type);
            assert(pluginField.length == 1 && pluginField.data != nullptr);
            return std::optional{*static_cast<T const*>(pluginField.data)};
        }
    }

    return std::nullopt;
}

template <typename T, class Enable = void>
struct Serializer
{
};

template <typename T>
struct Serializer<T, typename std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
{
    static void serialize(void** buffer, T const& value)
    {
        ::memcpy(*buffer, &value, sizeof(T));
        reinterpret_cast<char*&>(*buffer) += sizeof(T);
    }
    static void deserialize(void const** buffer, size_t* buffer_size, T* value)
    {
        assert(*buffer_size >= sizeof(T));
        ::memcpy(value, *buffer, sizeof(T));
        reinterpret_cast<char const*&>(*buffer) += sizeof(T);
        *buffer_size -= sizeof(T);
    }
};

template <typename T>
inline void serializeValue(void** buffer, T const& value)
{
    return Serializer<T>::serialize(buffer, value);
}

template <typename T>
inline void deserializeValue(void const** buffer, size_t* buffer_size, T* value)
{
    return Serializer<T>::deserialize(buffer, buffer_size, value);
}

} // namespace

// Static class fields initialization
PluginFieldCollection Int4GroupwsieGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Int4GroupwsieGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4GroupwsieGemmPluginCreator);

Int4GroupwsieGemmPlugin::Int4GroupwsieGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

Int4GroupwsieGemmPlugin::Int4GroupwsieGemmPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mGemmN);
    deserializeValue(&data, &length, &mGemmK);
    deserializeValue(&data, &length, &mGroupSize);
}

Int4GroupwsieGemmPlugin::~Int4GroupwsieGemmPlugin() {}

IPluginV2DynamicExt* Int4GroupwsieGemmPlugin::clone() const noexcept
{
    Int4GroupwsieGemmPlugin* plugin = new Int4GroupwsieGemmPlugin(mLayerName, mGemmN, mGemmK, mGroupSize);
    return plugin;
}

char const* Int4GroupwsieGemmPlugin::getPluginType() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

char const* Int4GroupwsieGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4GroupwsieGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* Int4GroupwsieGemmPlugin::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

int32_t Int4GroupwsieGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

bool Int4GroupwsieGemmPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // input 0: Fp16 activation tensor, input 1: packed int4 weights in type int8, input2: Fp16 scale values.
    // output 0: Fp16 computed result of the int4-woq gemm
    try
    {
        assert(nbInputs == 3 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos];
        bool status{true};

        switch (pos)
        {
        case 0:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmK;
            break;
        }
        case 1:
        {
            // The int4 weights are packed and swizzled into a special layout with int16 [N/4, K].
            // Since TensorRT doesn't have Int16 datatype, we use int8 datatype to store the weights.
            // Therefore the type should be [N/2, K] in int8.
            status &= tensorDesc.type == DataType::kINT8;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmN / 2;
            status &= tensorDesc.dims.d[1] == mGemmK;
            break;
        }
        case 2:
        {
            // The accepted scale for the kernel should be fp16 with [K/group_size,N]
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        }
        case 3:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmN;
            break;
        }
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
    }
    return false;
}

// IPluginV2Ext Methods
DataType Int4GroupwsieGemmPlugin::getOutputDataType(
    [[maybe_unused]] int32_t index, [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    return DataType::kHALF;
}

DimsExprs Int4GroupwsieGemmPlugin::getOutputDimensions([[maybe_unused]] int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
    DimsExprs output;

    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(mGemmN);
    return output;
}

void Int4GroupwsieGemmPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in, [[maybe_unused]] int32_t nbInputs,
    [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
}

// TODO: extend the worksapce calculation to a more generalized form.
size_t Int4GroupwsieGemmPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs, [[maybe_unused]] int32_t nbInputs,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t Int4GroupwsieGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    auto const& inputDesc0 = inputDesc[0];
    int32_t const M = inputDesc0.dims.d[0] * inputDesc0.dims.d[1];

    half* gemmInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[0]));
    int8_t* weightsInPtr = reinterpret_cast<int8_t*>(const_cast<void*>(inputs[1]));
    half* ScaleInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[2]));
    half* gemmOutDevicePtr = reinterpret_cast<half*>(outputs[0]);

    if (M <= 6)
    {
        gemv_forward_cuda_new(
            gemmInPtr, weightsInPtr, ScaleInPtr, gemmOutDevicePtr, M, mGemmN, mGemmK, mGroupSize, stream);
    }
    else
    {
        gemm_forward_cuda_new(
            gemmInPtr, weightsInPtr, ScaleInPtr, gemmOutDevicePtr, M, mGemmN, mGemmK, mGroupSize, stream);
    }
    return 0;
}

size_t Int4GroupwsieGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mGemmN) + sizeof(mGemmK) + sizeof(mGroupSize);
}

void Int4GroupwsieGemmPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mGemmN);
    serializeValue(&buffer, mGemmK);
    serializeValue(&buffer, mGroupSize);
}

int32_t Int4GroupwsieGemmPlugin::initialize() noexcept
{
    return 0;
}

void Int4GroupwsieGemmPlugin::terminate() noexcept {}

void Int4GroupwsieGemmPlugin::destroy() noexcept
{
    delete this;
}

Int4GroupwsieGemmPluginCreator::Int4GroupwsieGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4GroupwsieGemmPluginCreator::getPluginName() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4GroupwsieGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4GroupwsieGemmPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4GroupwsieGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4GroupwsieGemmPluginCreator::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* Int4GroupwsieGemmPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        // Read N, K attributes for the plugin.
        std::optional<int32_t> gemmN = parsePluginScalarField<int32_t>("gemm_n", fc);
        std::optional<int32_t> gemmK = parsePluginScalarField<int32_t>("gemm_k", fc);
        std::optional<int32_t> groupSize = parsePluginScalarField<int32_t>("group_size", fc);

        bool checkRequiredFields = gemmN.has_value() && gemmK.has_value() && groupSize.has_value();
        if (!checkRequiredFields)
        {
            return nullptr;
        }

        Int4GroupwsieGemmPlugin* plugin
            = new Int4GroupwsieGemmPlugin(std::string(name), gemmN.value(), gemmK.value(), groupSize.value());
        return plugin;
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}

nvinfer1::IPluginV2* Int4GroupwsieGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new Int4GroupwsieGemmPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}