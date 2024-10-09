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

#include "contextFMHARunner.h"
#include "decoderXQARunner.h"

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace drivellm
{
class AttentionPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime
{
public:
    AttentionPlugin(std::string const& name);

    // Force to distinguish different instances of the plugin.
    AttentionPlugin() = delete;

    AttentionPlugin(AttentionPlugin const&) = delete;

    ~AttentionPlugin() override;

    // IPluginV3 Methods
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;
    //

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginVersion() const noexcept override;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    // end IPluginV3Runtime Methods

protected:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mSMVersion;

    nvinfer1::DataType mDataType{nvinfer1::DataType::kHALF};

    // Number of heads and head dimension are specified by model and are runtime constant.
    // TODO: Make number of heads become plugin attribute.
    // TODO: The attention kernel
    int32_t const mNumHeadQ{32};
    int32_t const mNumHeadK{8};
    int32_t const mNumHeadV{8};
    int32_t const mNumElemPerHead{128};

    // Temporary variable we hardcode for now which should be later expanded as user
    // configurable field.
    // TODO: Generalize the plugin usage and mark them as plugin attribute.
    int32_t const mMaxBatchSize{16};
    int32_t const mTotalContextLen{2050};
};

class AttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    AttentionPluginCreator();

    ~AttentionPluginCreator() = default;

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFieldCollection;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace drivellm