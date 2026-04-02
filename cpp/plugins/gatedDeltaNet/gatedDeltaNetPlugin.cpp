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

#include "gatedDeltaNetPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#include "kernels/gdnKernels/gdnKernelUtils.cuh"
#endif

#include <cstdint>
#include <cstring>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGDN_PLUGIN_VERSION{"1"};
constexpr char const* kGDN_PLUGIN_NAME{"gated_delta_net"};

constexpr int32_t kIN_Q{0};
constexpr int32_t kIN_K{1};
constexpr int32_t kIN_V{2};
constexpr int32_t kIN_A{3};
constexpr int32_t kIN_B{4};
constexpr int32_t kIN_A_LOG{5};
constexpr int32_t kIN_DT_BIAS{6};
constexpr int32_t kIN_H0_SOURCE{7};
constexpr int32_t kIN_CONTEXT_LENGTHS{8};
constexpr int32_t kOUT_O{0};
constexpr int32_t kOUT_H0_SOURCE{1};
constexpr int32_t kNUM_INPUTS{9};
constexpr int32_t kNUM_OUTPUTS{2};
} // namespace

PluginFieldCollection GatedDeltaNetPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaNetPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaNetPluginCreator);

// ---------------------------------------------------------------------------
// Plugin constructor — only this block is compilation-guarded.
// When CUTE_DSL_GDN_ENABLED is not set the constructor throws immediately so
// the object can never be constructed; all other methods are shared.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mSMVersion(getSMVersion())
{
    if (!CuteDslGDNRunner::canImplement(mKDim, mVDim, mSMVersion))
    {
        LOG_ERROR(
            "Cannot implement GatedDeltaNetPlugin (CuTe DSL): k_dim=%d v_dim=%d SM=%d. "
            "CuTe DSL GDN is only built for k=v=128 and requires SM>=80 (Ampere+). "
            "Use k_dim=v_dim=128 on a supported GPU, or rebuild without CuTe DSL GDN if applicable.",
            mKDim, mVDim, mSMVersion);
        throw std::runtime_error("Cannot implement the GatedDeltaNetPlugin configuration (CuTe DSL GDN).");
    }

    if (!CuteDslGDNRunner::loadKernelModules())
    {
        LOG_ERROR(
            "Failed to load CuTe DSL GDN kernel modules (gdn_decode / gdn_prefill AOT). "
            "Check that the engine was built with ENABLE_CUTE_DSL=gdn (or ALL), AOT .o/.h are present and match the "
            "exported API, and the CUDA driver is compatible.");
        throw std::runtime_error("Cannot load CuTe DSL GDN kernel modules for GatedDeltaNetPlugin.");
    }
}
#else
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
{
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
}
#endif // CUTE_DSL_GDN_ENABLED

GatedDeltaNetPlugin::~GatedDeltaNetPlugin() = default;

IPluginCapability* GatedDeltaNetPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
        return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME)
        return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* GatedDeltaNetPlugin::clone() noexcept
{
    try
    {
        auto* p = new GatedDeltaNetPlugin(mLayerName, mKDim, mVDim);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* GatedDeltaNetPlugin::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPlugin::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

char const* GatedDeltaNetPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t GatedDeltaNetPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t GatedDeltaNetPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
        return -1;
    outputTypes[kOUT_O] = inputTypes[kIN_Q];
    outputTypes[kOUT_H0_SOURCE] = inputTypes[kIN_H0_SOURCE];
    return 0;
}

int32_t GatedDeltaNetPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /* exprBuilder */) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
        return -1;
    outputs[kOUT_O].nbDims = inputs[kIN_V].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_O].nbDims; ++i)
        outputs[kOUT_O].d[i] = inputs[kIN_V].d[i];
    outputs[kOUT_H0_SOURCE].nbDims = inputs[kIN_H0_SOURCE].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_H0_SOURCE].nbDims; ++i)
        outputs[kOUT_H0_SOURCE].d[i] = inputs[kIN_H0_SOURCE].d[i];
    return 0;
}

bool GatedDeltaNetPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
        return false;
    if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        return false;
    if (pos == kIN_A_LOG || pos == kIN_H0_SOURCE)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (pos == kIN_CONTEXT_LENGTHS)
        return inOut[pos].desc.type == DataType::kINT32;
    if (pos == kNUM_INPUTS + kOUT_H0_SOURCE)
        return inOut[pos].desc.type == DataType::kFLOAT;
    return inOut[pos].desc.type == DataType::kHALF;
}

int32_t GatedDeltaNetPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    if (nbInputs != kNUM_INPUTS)
        return -1;
    if (in[kIN_Q].desc.type != DataType::kHALF || in[kIN_V].desc.type != DataType::kHALF)
        return -1;
    if (in[kIN_Q].desc.dims.nbDims != 4 || in[kIN_V].desc.dims.nbDims != 4)
        return -1;
    if (in[kIN_CONTEXT_LENGTHS].desc.type != DataType::kINT32 || in[kIN_CONTEXT_LENGTHS].desc.dims.nbDims != 1)
        return -1;
    int32_t const n_batch = static_cast<int32_t>(in[kIN_Q].desc.dims.d[0]);
    int32_t const ctx_len_dim = static_cast<int32_t>(in[kIN_CONTEXT_LENGTHS].desc.dims.d[0]);
    if (ctx_len_dim != n_batch && ctx_len_dim != -1)
        return -1;
    int32_t const k_dim = static_cast<int32_t>(in[kIN_Q].desc.dims.d[3]);
    int32_t const v_dim = static_cast<int32_t>(in[kIN_V].desc.dims.d[3]);
    if (k_dim != mKDim || v_dim != mVDim)
        return -1;
#ifdef CUTE_DSL_GDN_ENABLED
    if (!CuteDslGDNRunner::canImplement(k_dim, v_dim, mSMVersion))
        return -1; // Unsupported on this device or k/v config; kernel requires k=v=128 and SM>=80
#endif
    return 0;
}

size_t GatedDeltaNetPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    // cu_seqlens [N+1] int32 workspace: prefix-sum of context_lengths for Blackwell prefill padding masking.
    int32_t const maxBatchSize = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS].max.d[0]);
    return static_cast<size_t>(maxBatchSize + 1) * sizeof(int32_t);
#else
    (void) inputs;
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// enqueue — only this block is compilation-guarded.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    CuteDslGDNRunner::loadKernelModules();

    int64_t const* qDims = inputDesc[kIN_Q].dims.d;
    int32_t const n = static_cast<int32_t>(qDims[0]);
    int32_t const seq_len = static_cast<int32_t>(qDims[1]);
    int32_t const h = static_cast<int32_t>(qDims[2]);
    int32_t const k_dim = static_cast<int32_t>(qDims[3]);

    int64_t const* vDims = inputDesc[kIN_V].dims.d;
    int32_t const hv = static_cast<int32_t>(vDims[2]);
    int32_t const v_dim = static_cast<int32_t>(vDims[3]);

    // h0 is batch-dense [n, hv, k, v]
    size_t const h0Bytes = static_cast<size_t>(n) * hv * static_cast<size_t>(k_dim) * v_dim * sizeof(float);
    void* h0Out = outputs[kOUT_H0_SOURCE];
    if (h0Out != inputs[kIN_H0_SOURCE])
    {
        cudaMemcpyAsync(h0Out, inputs[kIN_H0_SOURCE], h0Bytes, cudaMemcpyDeviceToDevice, stream);
    }

    GDNParams params{};
    params.q = const_cast<void*>(inputs[kIN_Q]);
    params.k = const_cast<void*>(inputs[kIN_K]);
    params.v = const_cast<void*>(inputs[kIN_V]);
    params.a = const_cast<void*>(inputs[kIN_A]);
    params.b = const_cast<void*>(inputs[kIN_B]);
    params.A_log = const_cast<void*>(inputs[kIN_A_LOG]);
    params.dt_bias = const_cast<void*>(inputs[kIN_DT_BIAS]);
    params.h0_source = h0Out;
    params.context_lengths = const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS]);
    params.o = outputs[kOUT_O];
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k_dim;
    params.v_dim = v_dim;
    params.smVersion = mSMVersion;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    // Blackwell prefill: convert context_lengths [N] → cu_seqlens [N+1] in workspace.
    if (seq_len > 1 && mSMVersion >= 100)
    {
        launchGdnCalCuSeqLens(inputs[kIN_CONTEXT_LENGTHS], workspace, n, stream);
        params.cu_seqlens = workspace;
    }
#endif

    CuteDslGDNRunner runner;
    int ret = runner.run(params, stream);

    return (ret == 0) ? 0 : -1;
}
#else
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* /* inputDesc */, PluginTensorDesc const* /* outputDesc */,
    void const* const* /* inputs */, void* const* /* outputs */, void* /* workspace */,
    cudaStream_t /* stream */) noexcept
{
    // Constructor already threw; this path should be unreachable.
    return -1;
}
#endif // CUTE_DSL_GDN_ENABLED

int32_t GatedDeltaNetPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* GatedDeltaNetPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* GatedDeltaNetPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("k_dim", &mKDim, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("v_dim", &mVDim, nvinfer1::PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void GatedDeltaNetPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// Creator
GatedDeltaNetPluginCreator::GatedDeltaNetPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("k_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("v_dim", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetPluginCreator::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPluginCreator::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaNetPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* GatedDeltaNetPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

IPluginV3* GatedDeltaNetPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        int32_t kDim = parsePluginScalarField<int32_t>("k_dim", fc).value_or(128);
        int32_t vDim = parsePluginScalarField<int32_t>("v_dim", fc).value_or(128);
        return new GatedDeltaNetPlugin(name, kDim, vDim);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedDeltaNetPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
