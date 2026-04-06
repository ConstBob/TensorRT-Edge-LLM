/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for Gated Delta Net.
//!
//! Registered as "gated_delta_net". Dispatches to decode (seq_len==1) or
//! prefill (seq_len>1) CuTe DSL kernels. Requires SM80+ and K=V=128.
//!
//! \par Dimension notation
//!   n   = batch size
//!   h   = number of Q/K heads
//!   hv  = number of V heads
//!   k   = head dimension K (must be 128)
//!   v   = head dimension V (must be 128)
//!
//! \par Inputs
//!   [0]  q               [n, seq_len, h,  k]   FP16  query
//!   [1]  k               [n, seq_len, h,  k]   FP16  key
//!   [2]  v               [n, seq_len, hv, v]   FP16  value
//!   [3]  a               [n, seq_len, hv]      FP16  input gate
//!   [4]  b               [n, seq_len, hv]      FP16  output gate
//!   [5]  A_log           [hv]                  FP32  log decay
//!   [6]  dt_bias         [hv]                  FP16  delta-time bias
//!   [7]  h0_source       [n, hv, k, v]         FP32  recurrent state in (batch-dense)
//!   [8]  context_lengths [n]                   INT32 valid token count per batch row
//!
//! \par Outputs
//!   [0]  o               [n, seq_len, hv, v]   FP16  output
//!   [1]  h0_out          [n, hv, k, v]         FP32  recurrent state out
class GatedDeltaNetPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    //! \param name         Plugin instance name
    //! \param kDim         Head dimension K (must be 128 for CuTe DSL kernel)
    //! \param vDim         Head dimension V (must be 128 for CuTe DSL kernel)
    //!
    //! The plugin currently only supports CuTe DSL GDN (when CUTE_DSL_GDN_ENABLED). The constructor loads AOT modules
    //! and throws std::runtime_error if:
    //!   - kDim != 128 or vDim != 128 (kernels are built for fixed K/V=128 only), or
    //!   - SM < 80 (Ampere+), or
    //!   - gdn_decode / gdn_prefill AOT .o load fails (missing or mismatched artifacts vs. headers).
    GatedDeltaNetPlugin(std::string const& name, int32_t kDim = 128, int32_t vDim = 128);

    GatedDeltaNetPlugin() = delete;
    GatedDeltaNetPlugin(GatedDeltaNetPlugin const&) = delete;
    ~GatedDeltaNetPlugin() override;

    //! \brief Return capability interface for the given type
    //! \param[in] type Requested capability type (kBUILD or kRUNTIME)
    //! \return Interface pointer or nullptr
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    //! \brief Clone the plugin instance
    //! \return New plugin instance
    nvinfer1::IPluginV3* clone() noexcept override;

    //! \brief Get plugin name
    //! \return "gated_delta_net"
    char const* getPluginName() const noexcept override;

    //! \brief Get plugin version
    //! \return Version string
    char const* getPluginVersion() const noexcept override;

    //! \brief Get plugin namespace
    //! \return Namespace string
    char const* getPluginNamespace() const noexcept override;

    //! \brief Get number of outputs (2: o and h0_out)
    //! \return 2
    int32_t getNbOutputs() const noexcept override;

    //! \brief Get output data types from input types
    //! \param[out] outputTypes Output type array
    //! \param[in] nbOutputs Number of outputs
    //! \param[in] inputTypes Input type array
    //! \param[in] nbInputs Number of inputs
    //! \return 0 on success, -1 on invalid counts
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    //! \brief Get output shapes from input shapes
    //! \param[in] inputs Input shape expressions
    //! \param[in] nbInputs Number of inputs
    //! \param[in] shapeInputs Shape input expressions (unused)
    //! \param[in] nbShapeInputs Number of shape inputs (unused)
    //! \param[out] outputs Output shape expressions
    //! \param[in] nbOutputs Number of outputs
    //! \param[in] exprBuilder Expression builder (unused)
    //! \return 0 on success, -1 on invalid counts
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    //! \brief Check if format combination is supported for the given position
    //! \param[in] pos Index in combined input/output list
    //! \param[in] inOut Input/output descriptors
    //! \param[in] nbInputs Number of inputs
    //! \param[in] nbOutputs Number of outputs
    //! \return true if supported
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    //! \brief Configure plugin from dynamic input descriptors
    //! \param[in] in Input descriptors
    //! \param[in] nbInputs Number of inputs
    //! \param[in] out Output descriptors (unused)
    //! \param[in] nbOutputs Number of outputs (unused)
    //! \return 0 on success, -1 on invalid config
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    //! \brief Get workspace size (0)
    //! \return 0
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    //! \brief Execute the plugin (decode or prefill)
    //! \param[in] inputDesc Input tensor descriptors
    //! \param[in] outputDesc Output tensor descriptors
    //! \param[in] inputs Input data pointers
    //! \param[out] outputs Output data pointers
    //! \param[in] workspace Workspace (unused)
    //! \param[in] stream CUDA stream
    //! \return 0 on success, -1 on failure
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    //! \brief Called when tensor shapes change (no-op)
    //! \return 0
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    //! \brief Attach to resource context (returns clone)
    //! \param[in] context Resource context (unused)
    //! \return Cloned plugin
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    //! \brief Get fields to serialize
    //! \return Field collection for k_dim, v_dim
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    //! \brief Set plugin namespace
    //! \param[in] pluginNamespace Namespace string
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mKDim{128};    //!< Head dimension K (kernel supports 128 only)
    int32_t mVDim{128};    //!< Head dimension V (kernel supports 128 only)
    int32_t mSMVersion{0}; //!< Captured device SM version used for build-time capability checks
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class GatedDeltaNetPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    GatedDeltaNetPluginCreator();
    ~GatedDeltaNetPluginCreator() override = default;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    //! \brief Get plugin name
    char const* getPluginName() const noexcept override;
    //! \brief Get plugin version
    char const* getPluginVersion() const noexcept override;
    //! \brief Get field names for createPlugin
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    //! \brief Get plugin namespace
    char const* getPluginNamespace() const noexcept override;
    //! \brief Set plugin namespace
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    //! \brief Create plugin from field collection
    //! \param[in] name Layer name
    //! \param[in] fc Fields (k_dim, v_dim)
    //! \param[in] phase Build or runtime phase
    //! \return New GatedDeltaNetPlugin or nullptr if GDN disabled
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
