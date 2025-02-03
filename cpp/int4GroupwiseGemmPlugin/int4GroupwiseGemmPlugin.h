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
#include <string>
#include <vector>

namespace drivellm
{
class Int4GroupwsieGemmPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    // Plugin constructor and attention specific utility methods
    Int4GroupwsieGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize);

    Int4GroupwsieGemmPlugin(std::string const& name, void const* data, size_t length);
    
    // Force to distinguish different instances of the plugin.
    Int4GroupwsieGemmPlugin() = delete;

    Int4GroupwsieGemmPlugin(Int4GroupwsieGemmPlugin const&) = delete;

    ~Int4GroupwsieGemmPlugin() override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginVersion() const noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

protected:
    std::string mLayerName;
    std::string mNamespace;
    
    int32_t mGemmN;
    int32_t mGemmK;
    int32_t mGroupSize;
    int32_t mSMVersion;
};

class Int4GroupwsieGemmPluginCreator : public nvinfer1::IPluginCreator
{
public:
    Int4GroupwsieGemmPluginCreator();

    ~Int4GroupwsieGemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace drivellm