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

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <string>
#include <vector>

#include <cstdint>

namespace trt_edgellm
{
namespace plugins
{
//! Router selection kernel chosen by the \c routing_mode plugin attribute.
enum class Nvfp4MoeRoutingMode : int32_t
{
    kSOFTMAX_TOPK = 0,       //!< \c moeTopkSoftmax: softmax over experts + flat top-k + renormalize (default).
    kSIGMOID_GROUP_TOPK = 1, //!< \c moeSigmoidGroupTopk: sigmoid + grouped top-k + renormalize + scale (NemotronH).
};

/*!
 * @brief TensorRT plugin: Nemotron-style MoE MLP W4A4 NVFP4 decode (no separate gate projection).
 *
 * Per expert the math is \c down_proj( act( up_proj(x) ) ): one up-projection and one down-projection
 * around a nonlinearity (\c act), matching the Nemotron MoE decode math (split W4A16 up/down GEMV kernels).
 * The up pass writes FP16 intermediate activations \c z; the down pass loads FP16 and evaluates \c act in FP32.
 * Router input is pre-activation router logits; \c enqueue() (via private \c enqueueDecoding) dispatches to one of
 * two routing kernels before the decode GEMVs, selected by the \c routing_mode attribute:
 *   - \c 0 (\c kSOFTMAX_TOPK, default): \c moeTopkSoftmax (softmax + flat top-k + renormalize).
 *   - \c 1 (\c kSIGMOID_GROUP_TOPK): \c moeSigmoidGroupTopk (sigmoid + grouped top-k + renormalize + scale).
 * There is no separate gate-projection weight tensor or gate GEMV.
 *
 * Layout: router logits are \c [batch * seq_len, num_experts] (2D; leading dim is \c num_tokens). Hidden
 * activations are either FP16 \c [batch, seq_len, hidden_size] (W4A16 path) or INT8 NVFP4-packed
 * \c [batch, seq_len, hidden_size/2] plus \c hidden_block_scale / \c hidden_global_scale (W4A4 path). \c seq_len may
 * be greater than 1. The CUDA path passes top-k expert indices and weights \c [num_tokens, top_k] to
 * \c launchNemotronMoeW4A16DecodeGemvCuda (FP16 hidden) or \c launchNemotronMoeW4A4DecodeUpGemvCuda /
 * \c launchNemotronMoeW4A4DecodeDownGemvCuda (packed NVFP4 hidden). \c hidden_size and \c moe_inter_size must be
 * multiples of 64 (decode GEMV / Marlin tile chunks).
 * Expert up quantized weights are INT8 \c [E, hidden_size/2, moe_inter_size] (two NVFP4 values per byte along
 * \c hidden_size); up block scales INT8 \c [E, hidden_size/16, moe_inter_size]. Down quantized weights are INT8
 * \c [E, moe_inter_size, hidden_size/2]; down block scales INT8 \c [E, moe_inter_size, hidden_size/16] (group size 16).
 * Per-expert FP32 global scales \c [E] for up and down.
 * \c e_score_correction_bias \c [E] FP32 is input [10]: used as optional bias by \c moeTopkSoftmax (mode 0) and as
 * the expert load-balancing bias by \c moeSigmoidGroupTopk (mode 1). Pass zeros when no bias is desired.
 */
class Nvfp4MoePlugin : public nvinfer1::IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize,
        nvinfer1::ActivationType activationType = static_cast<nvinfer1::ActivationType>(0), int32_t nGroup = 1,
        int32_t topkGroup = 1, int32_t normTopkProb = 1, float routedScalingFactor = 1.0f,
        int32_t routingMode = static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK));

    Nvfp4MoePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4MoePlugin() = delete;
    Nvfp4MoePlugin(Nvfp4MoePlugin const&) = delete;

    ~Nvfp4MoePlugin() noexcept override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    //! Sigmoid group top-k routing then W4A16 or W4A4 decode GEMVs (FP16 or NVFP4-packed hidden; NVFP4 expert weights).
    int32_t enqueueDecoding(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    std::string mLayerName;
    //! Empty like \c Int4MoePlugin; ONNX still uses domain ``trt`` (``trt::Nvfp4MoePlugin`` in PyTorch).
    std::string mNamespace;
    int32_t mNumExperts{};
    int32_t mTopK{};
    int32_t mHiddenSize{};
    int32_t mMoeInterSize{};
    nvinfer1::ActivationType mActivationType{};
    //! Marlin NVFP4 block scales along \c K; only \c 16 is supported (serialized for parity with \c Int4MoePlugin).
    int32_t mQuantizationGroupSize{};
    //! NemotronH sigmoid group top-k routing parameters (used when \c mRoutingMode == \c kSIGMOID_GROUP_TOPK).
    int32_t mNGroup{1};
    int32_t mTopkGroup{1};
    int32_t mNormTopkProb{1}; //!< Stored as int32 for serialization; nonzero = true.
    float mRoutedScalingFactor{1.0f};
    //! Router selection kernel: see \c Nvfp4MoeRoutingMode.
    int32_t mRoutingMode{static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK)};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class Nvfp4MoePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4MoePluginCreator();
    ~Nvfp4MoePluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    //! Empty like \c Int4MoePluginCreator (ONNX domain ``trt`` is separate from creator namespace).
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
