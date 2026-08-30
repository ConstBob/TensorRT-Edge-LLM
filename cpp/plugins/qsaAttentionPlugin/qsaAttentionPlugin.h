/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvInferPlugin.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! QSA (Qwen Sparse Attention) prefill plugin — Qwen3.8-Flash-Next sparse attention layers.
//!
//! Fuses, in one enqueue:
//!   1. packed-QKV split + per-head qk-norm (gammas pre-folded to 1+w) + partial RoPE +
//!      paged-KV write + dense K/V scratch mirrors (kernel::launchApplyRopeFromPackedToSplit);
//!   2. the QSA indexer (kernel::runQsaIndexerPrefill): weight-free block-compressed scoring,
//!      per-row top-512 blocks, expand + causal tail -> int32 index lists [B, S, 2051];
//!   3. sparse GQA attention over the dense mirrors (CuteDslQsaSparsePrefillRunner, CuTe DSL
//!      AOT group "qsa").
//!
//! v1 contract: PREFILL ONLY (kvcache_start_index must have runtime shape [0]); padded
//! activations; FP16 surface; all 11 inputs required (no optional slots). The attention
//! output gate (out * sigmoid(gate)) and every projection stay in the graph.
class QsaAttentionPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuildV2,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    QsaAttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
        int32_t indexerNumHeads, int32_t indexerHeadDim, int32_t indexerBudget, int32_t indexerCompressRatio,
        float attentionScale, float rmsNormEps);

    QsaAttentionPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    QsaAttentionPlugin() = delete;
    ~QsaAttentionPlugin() override = default;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    // IPluginV3OneBuild(V2)
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
    int32_t getAliasedInput(int32_t outputIndex) noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int32_t enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    //! Validate the (numQHeads, numKVHeads, headSize, indexer*) configuration against the
    //! specialized v1 kernels; throws on an unimplementable configuration.
    void validateConfiguration() const;

    std::string mLayerName;
    std::string mNamespace;

    int32_t mNumQHeads{};
    int32_t mNumKVHeads{};
    int32_t mHeadSize{};
    int32_t mIndexerNumHeads{};
    int32_t mIndexerHeadDim{};
    int32_t mIndexerBudget{};
    int32_t mIndexerCompressRatio{};
    float mAttentionScale{};
    float mRmsNormEps{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class QsaAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    QsaAttentionPluginCreator();
    ~QsaAttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
