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

#include "nvfp4MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "kernels/moe/nvf4_w4an/kernels.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

// Nemotron-style fused MoE MLP (W4A16: FP16 activation, NVFP4 weights): for each expert e,
//   y_e = down_proj_e( act( up_proj_e(x) ) ),
// then outputs are combined with Top-K router weights. There is no separate gate projection.
//
// Gating: input [0] is **pre-softmax** router logits FP32 \c [batch * seq_len, num_experts] (2D row-major;
// leading dim is \c num_tokens). Like Int4MoePlugin, enqueue() runs kernel::moeTopkSoftmax()
// (moeTopkSoftmaxKernels) **before** decode GEMVs to produce per-token top-k weights and indices; those are
// passed to launchNemotronMoeW4A16DecodeUpGemvCuda then launchNemotronMoeW4A16DecodeDownGemvCuda (no dense gate).
//
// Shapes: router \c d[0] must equal \c batch * seq_len from hidden_states [1]; hidden states are FP16
// \c [batch, seq_len, hidden_size] row-major (same linear layout as \c [num_tokens, hidden_size]).
// Output is FP16 with the same shape as hidden states \c [batch, seq_len, hidden_size].
//
// Inputs (see supportsFormatCombination / enqueue):
//   [0] router logits FP32 [batch * seq_len, num_experts] (pre-softmax; softmax+top-k inside plugin)
//   [1] hidden activations: FP16 \c [batch, seq_len, hidden_size] (W4A16), or INT8 NVFP4 packed payload \c
//   [batch, seq_len, hidden_size/2] (W4A4; two FP4 nibbles per byte along hidden)
//   [2] hidden_block_scale: INT8 — W4A4: \c [batch, seq_len, hidden_size/16] (Marlin tile scale bytes). W4A16: unused
//   (any INT8 tensor may be bound as a placeholder; shape is ignored).
//   [3] hidden_global_scale: FP32 length 1 — W4A4: device scalar for \c activation.global_scale[0]; W4A16: unused dummy
//   [4][5][6] up_proj: INT8 NVFP4 payload \c [E, K/2, moe_inter_size], INT8 block scales \c [E, K/16, moe_inter_size],
//   FP32 per-expert global scale \c [E]
//   [7][8][9] down_proj: INT8 NVFP4 payload \c [E, moe_inter_size, K/2]; INT8 block scales \c [E, moe_inter_size,
//   K/16]; FP32 per-expert global scale \c [E]

namespace
{
// ONNX custom-op import uses version "1" when the node has no plugin_version (same as Int4MoePlugin).
constexpr char const* kNVFP4_MOE_PLUGIN_VERSION{"1"};
constexpr char const* kNVFP4_MOE_PLUGIN_NAME{"Nvfp4MoePlugin"};
constexpr int32_t kNbPluginInputs{10};
//! NVFP4 Marlin tile scale stride along \c hidden_size; other values are rejected at build time.
constexpr int32_t kNvfp4MoeQuantizationGroupSize{16};

#if SUPPORTS_FP4
//! Map serialized \c activation_type (stored as \c nvinfer1::ActivationType-sized int32, same as \c Int4MoePlugin)
//! to Nemotron W4A16 decode nonlinearity. Integer values follow \c MoEActivationKind: 0 = ReLU^2, 1 = SiLU.
MoEActivationKind nvfp4StoredActivationToKernelKind(ActivationType const t) noexcept
{
    int32_t const v = static_cast<int32_t>(t);
    if (v == static_cast<int32_t>(MoEActivationKind::kSiLU))
    {
        return MoEActivationKind::kSiLU;
    }
    return MoEActivationKind::kReLU2;
}
#endif // SUPPORTS_FP4

// Workspace for top-k softmax temporaries. Sized by max num_tokens over the dynamic range.
// W4A4 decode GEMV kernels take explicit batch and seq_len (same contract as W4A16 decode); num_tokens = batch *
// seq_len. If a separate prefill GEMM kernel is added for performance, extend this helper (or add a sibling) for any
// extra temporaries that path requires.
size_t computeNvfp4MoeDecodeWorkspaceSize(
    int32_t numTokens, int32_t numExperts, int32_t topK, int32_t moeInterSize, int32_t hiddenSize) noexcept
{
    (void) hiddenSize;
    try
    {
        // Optional temp storage when numExperts is not a power of two (same contract as Int4MoePlugin).
        size_t softmaxWorkspaceSizeBytes = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, numExperts);
        size_t size = 0;
        // moeTopkSoftmax outputs: selected weights and expert indices [numTokens, topK]
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kINT32);
        if (softmaxWorkspaceSizeBytes > 0)
        {
            size = accumulateWorkspaceSize(
                size, rt::Coords{static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8);
        }
        int64_t const interElems = trt_edgellm::nemotronMoeW4A16InterBufferNumElems(numTokens, topK, moeInterSize);
        size = accumulateWorkspaceSize(size, rt::Coords{interElems}, DataType::kHALF);
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4MoePlugin workspace size: %s", e.what());
        return 0;
    }
}
} // namespace

PluginFieldCollection Nvfp4MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4MoePluginCreator);

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, int32_t const numExperts, int32_t const topK,
    int32_t const hiddenSize, int32_t const moeInterSize, ActivationType const activationType)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mQuantizationGroupSize(kNvfp4MoeQuantizationGroupSize)
{
}

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "num_experts")
        {
            mNumExperts = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "top_k")
        {
            mTopK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "moe_inter_size")
        {
            mMoeInterSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "activation_type")
        {
            // Serialized as INT32 (PluginFieldType::kINT32); read as int32 — same storage as MoE activation kind.
            int32_t const v = *static_cast<int32_t const*>(fc->fields[i].data);
            mActivationType = static_cast<ActivationType>(v);
        }
        else if (fieldName == "quantization_group_size")
        {
            mQuantizationGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
    if (mQuantizationGroupSize <= 0)
    {
        mQuantizationGroupSize = kNvfp4MoeQuantizationGroupSize;
    }
    if (mQuantizationGroupSize != kNvfp4MoeQuantizationGroupSize)
    {
        throw std::invalid_argument(format::fmtstr("Nvfp4MoePlugin: quantization_group_size must be %d, got %d",
            static_cast<int>(kNvfp4MoeQuantizationGroupSize), static_cast<int>(mQuantizationGroupSize)));
    }
}

Nvfp4MoePlugin::~Nvfp4MoePlugin() noexcept = default;

IPluginCapability* Nvfp4MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Nvfp4MoePlugin(mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4MoePlugin::getPluginName() const noexcept
{
    return kNVFP4_MOE_PLUGIN_NAME;
}

char const* Nvfp4MoePlugin::getPluginVersion() const noexcept
{
    return kNVFP4_MOE_PLUGIN_VERSION;
}

char const* Nvfp4MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Nvfp4MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4MoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    assert(nbOutputs == 1);
    (void) nbOutputs;
    (void) nbInputs;
    (void) inputTypes;
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Nvfp4MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == kNbPluginInputs);
    assert(nbOutputs == 1);
    (void) nbInputs;
    (void) nbOutputs;
    (void) shapeInputs;
    (void) nbShapeInputs;
    outputs[0].nbDims = 3;
    // Batch/seq from inputs[1]; last dim is always model hidden_size (FP16 or NVFP4-packed hidden uses d[2] = H/2).
    outputs[0].d[0] = inputs[1].d[0];
    outputs[0].d[1] = inputs[1].d[1];
    outputs[0].d[2] = exprBuilder.constant(static_cast<int64_t>(mHiddenSize));
    return 0;
}

bool Nvfp4MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == kNbPluginInputs && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    (void) nbInputs;
    (void) nbOutputs;

    auto const& td = inOut[pos].desc;
    bool ok{true};
    ok &= td.format == TensorFormat::kLINEAR;

    // Row-major [batch * seq_len, num_experts]; d[0] must match hidden_states batch*seq_len at runtime.
    auto const checkRouter = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kFLOAT;
        s &= t.dims.nbDims == 2;
        if (s)
        {
            s &= t.dims.d[1] == mNumExperts;
        }
        return s;
    };

    auto const checkHiddenFp16 = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kHALF;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            s &= t.dims.d[2] == mHiddenSize;
        }
        return s;
    };

    //! W4A4 packed hidden: INT8 row-major \c [batch, seq_len, hidden_size/2] (two NVFP4 values per byte).
    auto const checkHiddenNvfp4Packed = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kINT8;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            s &= t.dims.d[2] == mHiddenSize / 2;
        }
        return s;
    };

    auto const checkOutputFp16 = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kHALF;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            s &= t.dims.d[2] == mHiddenSize;
        }
        return s;
    };

    // Marlin NVFP4 tile layout as INT8: \c [E, K/2, inter] with \c K = hidden_size (two FP4 per byte).
    // Same underlying bytes as the legacy INT32 view \c [E, K/64, inter, 8].
    auto const checkUpPayload = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kINT8;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            s &= t.dims.d[0] == mNumExperts;
            s &= t.dims.d[1] == mHiddenSize / 2;
            s &= t.dims.d[2] == mMoeInterSize;
        }
        return s;
    };

    // Atom-layout block scales for up-proj: [E, padded_H, I/16] INT8 (M=hidden padded to 128, K_sf=inter/16 padded to
    // 4).
    auto const checkUpBlockScale = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kINT8;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            int32_t const paddedM = ((mHiddenSize + 127) / 128) * 128;
            int32_t const numSfCols = mMoeInterSize / kNvfp4MoeQuantizationGroupSize;
            int32_t const paddedSfCols = ((numSfCols + 3) / 4) * 4;
            s &= t.dims.d[0] == mNumExperts;
            s &= t.dims.d[1] == paddedM;
            s &= t.dims.d[2] == paddedSfCols;
        }
        return s;
    };

    auto const checkUpGlobalScale = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kFLOAT;
        s &= t.dims.nbDims == 1;
        if (s)
        {
            s &= t.dims.d[0] == mNumExperts;
        }
        return s;
    };

    auto const checkDownGlobalScale = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kFLOAT;
        s &= t.dims.nbDims == 1;
        if (s)
        {
            s &= t.dims.d[0] == mNumExperts;
        }
        return s;
    };

    auto const checkDownPayload = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kINT8;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            s &= t.dims.d[0] == mNumExperts;
            s &= t.dims.d[1] == mMoeInterSize;
            s &= t.dims.d[2] == mHiddenSize / 2;
        }
        return s;
    };

    // Atom-layout block scales for down-proj: [E, padded_I, H/16] INT8 (M=inter padded to 128, K_sf=hidden/16 padded to
    // 4).
    auto const checkDownBlockScale = [this](PluginTensorDesc const& t) {
        bool s{true};
        s &= t.type == DataType::kINT8;
        s &= t.dims.nbDims == 3;
        if (s)
        {
            int32_t const paddedM = ((mMoeInterSize + 127) / 128) * 128;
            int32_t const numSfCols = mHiddenSize / kNvfp4MoeQuantizationGroupSize;
            int32_t const paddedSfCols = ((numSfCols + 3) / 4) * 4;
            s &= t.dims.d[0] == mNumExperts;
            s &= t.dims.d[1] == paddedM;
            s &= t.dims.d[2] == paddedSfCols;
        }
        return s;
    };

    switch (pos)
    {
    case 0: return ok && checkRouter(td);
    case 1: return ok && (checkHiddenFp16(td) || checkHiddenNvfp4Packed(td));
    case 2:
    {
        PluginTensorDesc const& h = inOut[1].desc;
        if (h.type == DataType::kHALF)
        {
            // W4A16: activation block scales are not read; accept any INT8 placeholder (shape ignored).
            return ok && td.type == DataType::kINT8 && td.dims.nbDims >= 1;
        }
        if (h.type == DataType::kINT8)
        {
            return ok && td.type == DataType::kINT8 && td.dims.nbDims == 3 && td.dims.d[0] == h.dims.d[0]
                && td.dims.d[1] == h.dims.d[1] && td.dims.d[2] == mHiddenSize / mQuantizationGroupSize;
        }
        return false;
    }
    case 3: return ok && td.type == DataType::kFLOAT && td.dims.nbDims == 1 && td.dims.d[0] == 1;
    case 4: return ok && checkUpPayload(td);
    case 5: return ok && checkUpBlockScale(td);
    case 6: return ok && checkUpGlobalScale(td);
    case 7: return ok && checkDownPayload(td);
    case 8: return ok && checkDownBlockScale(td);
    case 9: return ok && checkDownGlobalScale(td);
    case 10: return ok && checkOutputFp16(td);
    default: return false;
    }
}

int32_t Nvfp4MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void) out;
    (void) nbOutputs;
    if (nbInputs != kNbPluginInputs)
    {
        return -1;
    }
    if (mHiddenSize % 64 != 0)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_size (%d) must be a multiple of 64 (Marlin NVFP4 tile chunks)", mHiddenSize);
        return -1;
    }
    if (mMoeInterSize % 64 != 0)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: moe_inter_size (%d) must be a multiple of 64 (decode GEMV strip tiling)", mMoeInterSize);
        return -1;
    }
    if (mQuantizationGroupSize != kNvfp4MoeQuantizationGroupSize)
    {
        LOG_ERROR("Nvfp4MoePlugin: quantization_group_size (%d) must be %d", static_cast<int>(mQuantizationGroupSize),
            static_cast<int>(kNvfp4MoeQuantizationGroupSize));
        return -1;
    }
    if (mHiddenSize % kNvfp4MoeQuantizationGroupSize != 0)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_size (%d) must be a multiple of %d", mHiddenSize,
            static_cast<int>(kNvfp4MoeQuantizationGroupSize));
        return -1;
    }
    bool const hiddenIsNvfp4Packed = (in[1].desc.type == DataType::kINT8);
    if (!hiddenIsNvfp4Packed)
    {
        if (static_cast<int32_t>(in[1].max.d[2]) != mHiddenSize)
        {
            LOG_ERROR("Nvfp4MoePlugin: hidden_states last dim (%d) must equal hidden_size (%d)",
                static_cast<int>(in[1].max.d[2]), mHiddenSize);
            return -1;
        }
    }
    else
    {
        if (static_cast<int32_t>(in[1].max.d[2]) != mHiddenSize / 2)
        {
            LOG_ERROR("Nvfp4MoePlugin: packed hidden last dim (%d) must equal hidden_size/2 (%d)",
                static_cast<int>(in[1].max.d[2]), static_cast<int>(mHiddenSize / 2));
            return -1;
        }
    }
    if (static_cast<int32_t>(in[4].max.d[1]) != mHiddenSize / 2
        || static_cast<int32_t>(in[4].max.d[2]) != mMoeInterSize)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: up_proj qweight shape mismatch (expected [E, hidden_size/2, moe_inter] = "
            "[%d, %d, %d])",
            static_cast<int>(mNumExperts), static_cast<int>(mHiddenSize / 2), static_cast<int>(mMoeInterSize));
        return -1;
    }
    if (static_cast<int32_t>(in[7].max.d[1]) != mMoeInterSize
        || static_cast<int32_t>(in[7].max.d[2]) != mHiddenSize / 2)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: down_proj qweight shape mismatch (expected [E, moe_inter, hidden_size/2] = "
            "[%d, %d, %d])",
            static_cast<int>(mNumExperts), static_cast<int>(mMoeInterSize), static_cast<int>(mHiddenSize / 2));
        return -1;
    }
    // Atom-layout block scales: [E, padded_M, padded_sf_cols] where
    //   up:   M=hidden (padded to 128), sf_cols=inter/16 (padded to 4)
    //   down: M=inter  (padded to 128), sf_cols=hidden/16 (padded to 4)
    int32_t const upPaddedM = ((mHiddenSize + 127) / 128) * 128;
    int32_t const upNumSfCols = mMoeInterSize / kNvfp4MoeQuantizationGroupSize;
    int32_t const upPaddedSfCols = ((upNumSfCols + 3) / 4) * 4;
    if (static_cast<int32_t>(in[5].max.d[1]) != upPaddedM || static_cast<int32_t>(in[5].max.d[2]) != upPaddedSfCols)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: up_proj block_scale shape mismatch (expected [E, padded_hidden, padded_inter/16] = "
            "[%d, %d, %d])",
            static_cast<int>(mNumExperts), static_cast<int>(upPaddedM), static_cast<int>(upPaddedSfCols));
        return -1;
    }
    int32_t const dnPaddedM = ((mMoeInterSize + 127) / 128) * 128;
    int32_t const dnNumSfCols = mHiddenSize / kNvfp4MoeQuantizationGroupSize;
    int32_t const dnPaddedSfCols = ((dnNumSfCols + 3) / 4) * 4;
    if (static_cast<int32_t>(in[8].max.d[1]) != dnPaddedM || static_cast<int32_t>(in[8].max.d[2]) != dnPaddedSfCols)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: down_proj block_scale shape mismatch (expected [E, padded_inter, padded_hidden/16] = "
            "[%d, %d, %d])",
            static_cast<int>(mNumExperts), static_cast<int>(dnPaddedM), static_cast<int>(dnPaddedSfCols));
        return -1;
    }
    // When bounds are static, router token count must match hidden_states batch × seq_len (supports seq_len > 1).
    {
        int64_t const routerTokens = static_cast<int64_t>(in[0].max.d[0]);
        int64_t const maxB = static_cast<int64_t>(in[1].max.d[0]);
        int64_t const maxS = static_cast<int64_t>(in[1].max.d[1]);
        if (routerTokens > 0 && maxB > 0 && maxS > 0 && routerTokens != maxB * maxS)
        {
            LOG_ERROR(
                "Nvfp4MoePlugin: router_logits max d[0] (%lld) must equal hidden_states max d[0]*d[1] (%lld*%lld)",
                static_cast<long long>(routerTokens), static_cast<long long>(maxB), static_cast<long long>(maxS));
            return -1;
        }
    }
    return 0;
}

size_t Nvfp4MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    assert(nbInputs == kNbPluginInputs);
    (void) outputs;
    (void) nbOutputs;
    // Upper bound on num_tokens; router [0] should equal batch*seq_len of hidden_states [1].
    int64_t const maxHiddenTokens = static_cast<int64_t>(inputs[1].max.d[0]) * static_cast<int64_t>(inputs[1].max.d[1]);
    int64_t const maxTokens = std::max(static_cast<int64_t>(inputs[0].max.d[0]), maxHiddenTokens);
    int32_t const numTokens
        = static_cast<int32_t>(std::min(maxTokens, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
    return computeNvfp4MoeDecodeWorkspaceSize(numTokens, mNumExperts, mTopK, mMoeInterSize, mHiddenSize);
}

int32_t Nvfp4MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        return enqueueDecoding(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Nvfp4MoePlugin::enqueueDecoding(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) outputDesc;
    if (inputDesc[1].dims.nbDims != 3)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states must be 3D [batch, seq_len, hidden_size], got nbDims=%d",
            static_cast<int>(inputDesc[1].dims.nbDims));
        return -1;
    }
    int32_t const batch = inputDesc[1].dims.d[0];
    int32_t const seqLen = inputDesc[1].dims.d[1];
    if (batch < 1 || seqLen < 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states batch and seq_len must be >= 1, got batch=%d seq_len=%d",
            static_cast<int>(batch), static_cast<int>(seqLen));
        return -1;
    }
    int64_t const numTokens64 = static_cast<int64_t>(batch) * static_cast<int64_t>(seqLen);
    if (numTokens64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        LOG_ERROR("Nvfp4MoePlugin: batch*seq_len (%lld) exceeds int32 max", static_cast<long long>(numTokens64));
        return -1;
    }
    int32_t const numTokens = static_cast<int32_t>(numTokens64);
    if (inputDesc[0].dims.d[0] != numTokens)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: token dimension mismatch — router d[0]=%d, hidden_states batch*seq_len=%d*%d (must equal "
            "num_tokens)",
            static_cast<int>(inputDesc[0].dims.d[0]), static_cast<int>(batch), static_cast<int>(seqLen));
        return -1;
    }

    bool const w4a4Hidden = (inputDesc[1].type == DataType::kINT8);
    if (w4a4Hidden)
    {
        if (inputDesc[1].dims.d[2] != mHiddenSize / 2)
        {
            LOG_ERROR("Nvfp4MoePlugin: packed hidden last dim (%d) must equal hidden_size/2 (%d)",
                static_cast<int>(inputDesc[1].dims.d[2]), static_cast<int>(mHiddenSize / 2));
            return -1;
        }
    }
    else if (inputDesc[1].type == DataType::kHALF)
    {
        if (inputDesc[1].dims.d[2] != mHiddenSize)
        {
            LOG_ERROR("Nvfp4MoePlugin: hidden_states last dim (%d) must equal hidden_size (%d)",
                static_cast<int>(inputDesc[1].dims.d[2]), mHiddenSize);
            return -1;
        }
    }
    else
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states[1] must be FP16 (W4A16) or INT8 packed NVFP4 (W4A4), got type=%d",
            static_cast<int>(inputDesc[1].type));
        return -1;
    }

    size_t const softmaxWsBytes = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);

    // ==================== Workspace allocation (order matches computeNvfp4MoeDecodeWorkspaceSize) ====================
    std::byte* ws = static_cast<std::byte*>(workspace);
    // Buffers for kernel::moeTopkSoftmax: FP32 weights and INT32 expert indices per token-slot.
    float* topkWeightsPtr
        = static_cast<float*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
    int32_t* topkIndicesPtr
        = static_cast<int32_t*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kINT32).rawPointer());
    void* softmaxWsPtr = (softmaxWsBytes > 0)
        ? assignTensorFromWorkspace(ws, {static_cast<int64_t>(softmaxWsBytes)}, DataType::kINT8).rawPointer()
        : nullptr;
    rt::Tensor routerLogitsTensor(
        const_cast<void*>(inputs[0]), rt::Coords{inputDesc[0].dims}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor topkWeightsTensor(topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor topkIndicesTensor(topkIndicesPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kINT32);

    // ==================== Step 1: TopK softmax on router logits (before any W4A4 decode GEMV) ====================
    // Uses moeTopkSoftmaxKernels::moeTopkSoftmax (same entry point as Int4MoePlugin): softmax over experts,
    // then top-k selection; renormalize=true so selected weights sum to 1 per token.
    trt_edgellm::kernel::moeTopkSoftmax(routerLogitsTensor, topkWeightsTensor, topkIndicesTensor, mTopK, softmaxWsPtr,
        softmaxWsBytes, stream, true, 0.0F);
    CUDA_CHECK(cudaGetLastError());

#if SUPPORTS_FP4
    // ==================== Step 2: Bind NVFP4 tensor views (up_proj, down_proj) ====================
    // NVFP4Tensor::strides are in int4 elements (tile grid x,y,z); each tile uses kNvfp4Int4PerTilePayload
    // contiguous int4 vectors — see nvfp4_tensor.cuh.
    int64_t const int4PerNvfp4Tile = trt_edgellm::kNvfp4Int4PerTilePayload;
    int32_t const numHiddenChunks = mHiddenSize / 64;
    int32_t const numInterChunks = mMoeInterSize / 64;
    int64_t const interFp16Elems = trt_edgellm::nemotronMoeW4A16InterBufferNumElems(numTokens, mTopK, mMoeInterSize);
    __half* interFp16Scratch
        = static_cast<__half*>(assignTensorFromWorkspace(ws, rt::Coords{interFp16Elems}, DataType::kHALF).rawPointer());

    NVFP4Tensor up{};
    // INT8 inputs [E, hidden/2, inter]: same byte stream as Marlin \c int4 tiles (two FP4 nibbles per byte).
    up.quantized_data = reinterpret_cast<int4*>(const_cast<void*>(inputs[4]));
    // Atom-layout block scales: INT8 [E, atom_sf_bytes] → reinterpret as int* (int32 word index).
    up.block_scale = reinterpret_cast<int*>(const_cast<void*>(inputs[5]));
    up.global_scale = reinterpret_cast<float*>(const_cast<void*>(inputs[6]));
    {
        // Tile grid matches \c NemotronHMoEW4A4Plugin::populate_marlin_plugin_buffers: \c Dim3(expert, jj, c) with
        // \c jj in \c [0, hidden_size) (one NVFP4 tile per hidden matrix row) and \c c in \c [0, moe_inter/64).
        // Strides are in \c int4 elements (\ref NVFP4Tensor); do not use \c jj/64 here — that is a different layout.
        int64_t const h = static_cast<int64_t>(mHiddenSize);
        int64_t const nic = static_cast<int64_t>(numInterChunks);
        up.strides[0] = h * nic * int4PerNvfp4Tile;
        up.strides[1] = nic * int4PerNvfp4Tile;
        up.strides[2] = int4PerNvfp4Tile;
        // Atom-layout scale metadata: Dim3(expert=x, hidden_row=y, inter_chunk=z)
        up.scaleMDimIdx = 1; // y = hidden_row (M)
        up.scaleKDimIdx = 2; // z = inter_chunk (K), each chunk = 64 elements = 4 SF columns
        int32_t const numSfColsUp = mMoeInterSize / kNvfp4MoeQuantizationGroupSize;
        up.scaleNumKTiles = (numSfColsUp + 3) / 4;
        int32_t const numMTilesUp = (mHiddenSize + 127) / 128;
        up.scaleExpertStride = static_cast<int64_t>(numMTilesUp) * up.scaleNumKTiles * 128;
    }

    NVFP4Tensor dn{};
    // INT8 row-major \c [E, inter, hidden/2]: innermost int4 tiles step along \c hidden/2. Tile \c Dim3 is
    // \c (expert, inter, hidden_chunk) = \c (x,y,z). Block scales use atom-layout 128×4 swizzle.
    dn.quantized_data = reinterpret_cast<int4*>(const_cast<void*>(inputs[7]));
    dn.block_scale = reinterpret_cast<int*>(const_cast<void*>(inputs[8]));
    dn.global_scale = reinterpret_cast<float*>(const_cast<void*>(inputs[9]));
    {
        int64_t const i = static_cast<int64_t>(mMoeInterSize);
        int64_t const h32 = static_cast<int64_t>(mHiddenSize) / 32;
        dn.strides[0] = i * h32;
        dn.strides[1] = h32;
        dn.strides[2] = int4PerNvfp4Tile;
        // Atom-layout scale metadata: Dim3(expert=x, inter_row=y, hidden_chunk=z)
        dn.scaleMDimIdx = 1; // y = inter_row (M)
        dn.scaleKDimIdx = 2; // z = hidden_chunk (K), each chunk = 64 elements = 4 SF columns
        int32_t const numSfColsDn = mHiddenSize / kNvfp4MoeQuantizationGroupSize;
        dn.scaleNumKTiles = (numSfColsDn + 3) / 4;
        int32_t const numMTilesDn = (mMoeInterSize + 127) / 128;
        dn.scaleExpertStride = static_cast<int64_t>(numMTilesDn) * dn.scaleNumKTiles * 128;
    }

    MoEActivationKind const activationKind = nvfp4StoredActivationToKernelKind(mActivationType);
    if (w4a4Hidden)
    {
        // W4A4: row-major [batch, seq_len, hidden_size/2]; kernels index flattened tokens t in [0, batch*seq_len).
        // strides[0] is int4 offset between consecutive tokens (one row); valid for any seq_len >= 1.
        int32_t const w4a4Tb = trt_edgellm::nemotronMoeW4A4DecodeThreadBlockSizeForDims(mHiddenSize, mMoeInterSize);
        if (w4a4Tb == 0)
        {
            LOG_ERROR(
                "Nvfp4MoePlugin: W4A4 decode could not pick a thread block size for hidden_size=%d moe_inter_size=%d",
                mHiddenSize, mMoeInterSize);
            return -1;
        }
        int32_t const numHiddenTiles = mHiddenSize / 64;
        NVFP4Tensor actNvfp4{};
        actNvfp4.quantized_data = reinterpret_cast<int4*>(const_cast<void*>(inputs[1]));
        actNvfp4.block_scale = reinterpret_cast<int*>(const_cast<void*>(inputs[2]));
        actNvfp4.global_scale = reinterpret_cast<float*>(const_cast<void*>(inputs[3]));
        int64_t const strideTokenInt4 = static_cast<int64_t>(numHiddenTiles) * int4PerNvfp4Tile;
        actNvfp4.strides[0] = strideTokenInt4;
        actNvfp4.strides[1] = int4PerNvfp4Tile;
        actNvfp4.strides[2] = 0;
        // Activation uses plain linear scale layout (readBlockScaleWordLinear) — no atom-swizzle needed.
        // Atom-layout fields are unused for activation; zero-init from NVFP4Tensor{} is sufficient.

        trt_edgellm::launchNemotronMoeW4A4DecodeUpGemvCuda(batch, seqLen, mHiddenSize, mMoeInterSize, mNumExperts,
            mTopK, topkIndicesPtr, actNvfp4, up, interFp16Scratch, stream, w4a4Tb);
        CUDA_CHECK(cudaGetLastError());

        trt_edgellm::launchNemotronMoeW4A4DecodeDownGemvCuda(batch, seqLen, mHiddenSize, mMoeInterSize, numHiddenChunks,
            mNumExperts, mTopK, topkIndicesPtr, topkWeightsPtr, interFp16Scratch, dn, static_cast<__half*>(outputs[0]),
            stream, w4a4Tb, activationKind);
        CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        __half const* actFp16 = static_cast<__half const*>(inputs[1]);
        // ==================== Step 3a: Up-proj NVFP4 GEMV → FP16 z [num_tokens, top_k, moe_inter] ====================
        // Zeros interFp16Scratch inside the launch; topk_weights unused here (applied in the down pass).
        trt_edgellm::launchNemotronMoeW4A16DecodeUpGemvCuda(batch, seqLen, mHiddenSize, mMoeInterSize, numInterChunks,
            mNumExperts, mTopK, topkIndicesPtr, topkWeightsPtr, actFp16, up, interFp16Scratch, stream);
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 3b: Down-proj (act(z)×router) NVFP4 GEMV → FP16 out (atomic half2 CAS)
        // ====================
        trt_edgellm::launchNemotronMoeW4A16DecodeDownGemvCuda(batch, seqLen, mHiddenSize, mMoeInterSize,
            numHiddenChunks, mNumExperts, mTopK, topkIndicesPtr, topkWeightsPtr, interFp16Scratch, dn,
            static_cast<__half*>(outputs[0]), stream, activationKind);
        CUDA_CHECK(cudaGetLastError());
    }
#else
    LOG_ERROR("Nvfp4MoePlugin: NVFP4 MoE decode requires CUDA >= 12.8 (FP4 support)");
    return -1;
#endif // SUPPORTS_FP4

    return 0;
}

int32_t Nvfp4MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void) in;
    (void) nbInputs;
    (void) out;
    (void) nbOutputs;
    return 0;
}

IPluginV3* Nvfp4MoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Nvfp4MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("quantization_group_size", &mQuantizationGroupSize, PluginFieldType::kINT32, 1);

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

Nvfp4MoePluginCreator::Nvfp4MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quantization_group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4MoePluginCreator::getPluginName() const noexcept
{
    return kNVFP4_MOE_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Nvfp4MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Nvfp4MoePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Nvfp4MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Nvfp4MoePluginCreator::getPluginVersion() const noexcept
{
    return kNVFP4_MOE_PLUGIN_VERSION;
}

IPluginV3* Nvfp4MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        Nvfp4MoePlugin* plugin = new Nvfp4MoePlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4MoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
