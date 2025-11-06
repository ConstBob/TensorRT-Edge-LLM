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

#include "attentionPlugin.h"

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};

// Workaround for CUDA12/13 Thor re-numbering. The kernels themselves have version compatibility.
void applyThorSMRenumberWAR(int32_t& smVersion)
{
    if (smVersion == 110)
    {
        smVersion = 101;
    }
}

// Define the mapping of input and output indices of the AttentionPlugin.
constexpr int32_t kIN_QKV_IDX{0};
constexpr int32_t kIN_KV_CACHE_IDX{1};
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{2};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{3};
constexpr int32_t kADDITIONAL_INPUT_START_IDX{4};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t maxBatchSize, int32_t kvCacheCapacity, int32_t enableTreeAttention, int32_t enableReuseKVCache)
    : mLayerName(name)
    , mNumHeadQ(numQHeads)
    , mNumHeadKV(numKVHeads)
    , mNumElemPerHead(headSize)
    , mMaxBatchSize(maxBatchSize)
    , mKVCacheCapacity(kvCacheCapacity)
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableReuseKVCache(enableReuseKVCache)
{
    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    bool canImplement = ContextFMHARunner::canImplement(mNumElemPerHead, mSMVersion, mDataType)
        && DecoderXQARunner::canImplement(mNumHeadQ, mNumHeadKV, mSMVersion, mDataType);
    if (!canImplement)
    {
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }

    // Load FMHA and XQA kernels to device. The kernel code will only be loaded once if
    // multiple AttentionPlugin instances exist in the model.
    // TODO: Fix me too pass spec-deocde support through plugin attributes.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType, useSpecDecode);
}

AttentionPlugin::AttentionPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mMaxBatchSize);
    deserializeValue(&data, &length, &mKVCacheCapacity);
    deserializeValue(&data, &length, &mNumHeadQ);
    deserializeValue(&data, &length, &mNumHeadKV);
    deserializeValue(&data, &length, &mNumElemPerHead);
    deserializeValue(&data, &length, &mEnableTreeAttention);
    deserializeValue(&data, &length, &mEnableReuseKVCache);

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    // TODO: Fix me too pass spec-deocde support through plugin attributes.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType, useSpecDecode);
}

AttentionPlugin::~AttentionPlugin() {}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(mLayerName, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mMaxBatchSize,
        mKVCacheCapacity, mEnableTreeAttention, mEnableReuseKVCache);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* AttentionPlugin::getPluginType() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      GEMM-QKV tensor (linear FP16) with shape [B, S, Hq+Hk+Hv, D]
    //      KV-cache tensor (linear FP16) with shape [B, 2, Hkv, Smax, D], here Smax is the kvcache capacity
    //      buffer.
    //      Real context length: [B] (a vector of scalars) with type int32_t.
    //      RoPE cos/sin cache: [B or 1, Smax, D] (a tensor of scalars) with type float.
    //            Rope CosSin can be ND vector depending on rope type.
    //      Persistent KV cache length [B] (a vector of scalars) with type int32_t.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [B, S, Hq, D]
    //      KV-cache tensor, same as the above.
    auto checkGemmQKV = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[2] == (mNumHeadQ + mNumHeadKV + mNumHeadKV) * mNumElemPerHead;
        }
        return status;
    };

    auto checkKVCache = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 5;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[1] == 2; // Specify K and V
            status &= tensorDim.d[2] == mNumHeadKV;
            status &= tensorDim.d[3] == 0 || tensorDim.d[3] == mKVCacheCapacity;
            status &= tensorDim.d[4] == mNumElemPerHead;
        }
        return status;
    };

    auto checkSequenceLen = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkPosEncodingCosSin = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        status &= tensorDesc.dims.d[2] <= mNumElemPerHead;
        return status;
    };
    auto checkAttentionMask = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        return status;
    };
    auto checkAttentionPosId = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };
    auto checkKVCacheStartIdx = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    // Output tensor checks
    auto checkAttentionOutput = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 4;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[2] == mNumHeadQ;
            status &= tensorDim.d[3] == mNumElemPerHead;
        }
        return status;
    };

    try
    {
        assert(nbOutputs == 2);

        bool result{true};

        if (pos < kADDITIONAL_INPUT_START_IDX)
        {
            switch (pos)
            {
            case kIN_QKV_IDX: result = checkGemmQKV(inOut[0]); break;
            case kIN_KV_CACHE_IDX: result = checkKVCache(inOut[1]); break;
            case kIN_CONTEXT_LENGTH_IDX: result = checkSequenceLen(inOut[2]); break;
            case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[3]); break;
            default: break;
            }
        }
        else if (pos >= nbInputs)
        {
            int32_t outPos = pos - nbInputs;
            switch (outPos)
            {
            case 0: result = checkAttentionOutput(inOut[pos]); break;
            case 1: result = checkKVCache(inOut[pos]); break;
            default: break;
            }
        }

        // The indices for optional inputs are dynamic, depending on which features are enabled.
        // We start checking after the 4 base inputs.
        int32_t currentOptionalInputIdx = kADDITIONAL_INPUT_START_IDX;
        if (mEnableReuseKVCache)
        {
            if (pos == currentOptionalInputIdx)
            {
                result = checkKVCacheStartIdx(inOut[pos]);
            }
            currentOptionalInputIdx++;
        }
        if (mEnableTreeAttention)
        {
            if (pos == currentOptionalInputIdx)
            {
                result = checkAttentionMask(inOut[pos]);
            }
            currentOptionalInputIdx++;

            if (pos == currentOptionalInputIdx)
            {
                result = checkAttentionPosId(inOut[pos]);
            }
            currentOptionalInputIdx++;
        }

        assert(nbInputs == currentOptionalInputIdx);

        return result;
    }
    catch (std::exception const& e)
    {
    }
    return false;
}

// IPluginV2Ext Methods
DataType AttentionPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    return DataType::kHALF;
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
    DimsExprs output;
    if (outputIndex == kOUT_ATTENTION_IDX)
    {
        output.nbDims = 4;
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = exprBuilder.constant(mNumHeadQ);
        output.d[3] = exprBuilder.constant(mNumElemPerHead);
    }
    else if (outputIndex == kOUT_KV_CACHE_IDX)
    {
        // Output[1] is KVCache
        output.nbDims = 5;
        output.d[0] = inputs[1].d[0];
        output.d[1] = inputs[1].d[1];
        output.d[2] = exprBuilder.constant(mNumHeadKV);
        output.d[3] = exprBuilder.constant(mKVCacheCapacity);
        output.d[4] = exprBuilder.constant(mNumElemPerHead);
    }
    return output;
}

void AttentionPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return; // No need to configure anything.
}

// TODO: extend the workspace calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // TensorRT will supply max profile shape for each input/output tensor across all optimization profiles.
    // We will request workspace to keep intermediate tensors under prefill/decode phase executions.
    // Obtain max supported batch size from the input tensor shapes. The QKV input tensor will be in shape
    // [B, S, Hq+Hk+Hv, D] where S is padded the max length of the input sequence within this batch..
    PluginTensorDesc const& qkvInputDesc = inputs[kIN_QKV_IDX];
    int64_t const maxBatchSize = qkvInputDesc.dims.d[0];
    int64_t const maxInputSeqLen = qkvInputDesc.dims.d[1];

    // We use half precisions for now.
    int32_t const nbBytesPerData{2};
    int32_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {maxBatchSize + 1}, DataType::kINT32);
    // The workspace will be used to store the Q tensor. For eagle mode, maxDecodingTokens means the number of Q tensor.
    // Set maxDecodingTokens to 128, which means the max value supported is 128. This value should be sufficient.
    constexpr int64_t kMAX_EAGLE_DECODING_TOKENS = 128;
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, {maxBatchSize, kMAX_EAGLE_DECODING_TOKENS, mNumHeadQ, mNumElemPerHead}, DataType::kHALF);

    if (mEnableReuseKVCache)
    {
        // CuTotalKvCacheLens to describe the cumulative length of KV tensors.
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
        // KVCache ends that denote the end index of each KVCache lane after adding current contents.
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize}, DataType::kINT32);
        // Separate Q Tensor space to keep roped Q results.
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{maxBatchSize, maxInputSeqLen, mNumHeadQ, mNumElemPerHead}, DataType::kHALF);
        // KV Tensor to store concated KV that include pre-cached KV and current KV.
        workspaceSize = accumulateWorkspaceSize(workspaceSize,
            rt::Coords{maxBatchSize, 2, mNumHeadKV, mKVCacheCapacity, mNumElemPerHead},
            DataType::kHALF); // KVCacheCompact for FMHA.
    }

    // Request another alignment size to align the workspace pointer.
    workspaceSize += kDEVICE_ALIGNMENT;
    return workspaceSize;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    // Determine the indices of the additional inputs based on plugin configuration.
    // We will further optimize the logic to reduce the complexity.
    int32_t kvCacheStartInputIdx{};
    int32_t attentionMaskInputIdx{};
    int32_t attentionPosIdInputIdx{};

    int32_t currentInputIdx = kADDITIONAL_INPUT_START_IDX;
    if (mEnableReuseKVCache)
    {
        kvCacheStartInputIdx = currentInputIdx;
        currentInputIdx += 1;
    }
    if (mEnableTreeAttention)
    {
        attentionMaskInputIdx = currentInputIdx;
        currentInputIdx += 1;
        attentionPosIdInputIdx = currentInputIdx;
        currentInputIdx += 1;
    }

    // Construct non-owned tensor objects from I/O data pointers and shapes.
    PluginTensorDesc const& qkvInputDesc = inputDesc[kIN_QKV_IDX];
    rt::Tensor qkvInputTensor(
        const_cast<void*>(inputs[kIN_QKV_IDX]), rt::Coords{qkvInputDesc.dims}, rt::DeviceType::kGPU, qkvInputDesc.type);
    int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputTensor.getShape()[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputTensor.getShape()[1]);

    PluginTensorDesc const& contextLengthInputDesc = inputDesc[kIN_CONTEXT_LENGTH_IDX];
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{contextLengthInputDesc.dims}, rt::DeviceType::kGPU, contextLengthInputDesc.type);

    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{posEncodingCosSinDesc.dims}, rt::DeviceType::kGPU, posEncodingCosSinDesc.type);
    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(ropeCosSinTensor.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(ropeCosSinTensor.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(ropeCosSinTensor.getShape()[2]);

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    // Construct the KVCache tensor from the output shape since we have workaround implementation from KVCache input.
    PluginTensorDesc const& kvCacheOutputDesc = outputDesc[kOUT_KV_CACHE_IDX];
    rt::Tensor kvCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheOutputDesc.dims}, rt::DeviceType::kGPU, kvCacheOutputDesc.type);

    // Optional Inputs that are not used under all scenarios.
    rt::Tensor kvCacheStartIdxTensor{};
    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};

    if (mEnableReuseKVCache)
    {
        PluginTensorDesc const& kvCacheStartInputDesc = inputDesc[kvCacheStartInputIdx];
        kvCacheStartIdxTensor = rt::Tensor(const_cast<void*>(inputs[kvCacheStartInputIdx]),
            rt::Coords{kvCacheStartInputDesc.dims}, rt::DeviceType::kGPU, kvCacheStartInputDesc.type);
    }
    if (mEnableTreeAttention)
    {
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[attentionMaskInputIdx];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[attentionPosIdInputIdx];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[attentionMaskInputIdx]),
            rt::Coords{attentionMaskInputDesc.dims}, rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[attentionPosIdInputIdx]),
            rt::Coords{attentionPosIdInputDesc.dims}, rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }

    // This is a workaround implementation that use zero length in the kvCache to indicate prefill phase.
    // We will adjust this logic to properly define the semantics.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    int32_t const kvCacheInputLength = kvCacheInputDesc.dims.d[3];
    bool const isPrefillPhase = kvCacheInputLength == 0;

    // Align the workspace pointer so that each tensor assigned from the workspace will align to the device alignment
    // granularity.
    void* alignedWorkspacePtr = alignDevicePtr(workspace);

    if (isPrefillPhase)
    {
        // At Context phase. Do 1. Apply rope and write KVCache. 2. Dispatch FMHA runner.
        // RoPE kernel now only handle padded input sequence, we treat all "tokens" in the
        // padded input as processing targets.
        // TODO: Explore non-padded input format.
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;
        AttentionInputLayout attentionInputLayout
            = mEnableReuseKVCache ? AttentionInputLayout::CONTIGUOUS_Q_KV : AttentionInputLayout::PACKED_QKV;

        rt::Tensor cuQSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);

        rt::Tensor cuTotalKvCacheLensTensor{};
        rt::Tensor kvCacheEndIdxsTensor{};
        if (mEnableReuseKVCache)
        {
            cuTotalKvCacheLensTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
            kvCacheEndIdxsTensor = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);
        }

        kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor.dataPointer<int32_t>(),
            cuQSeqLensTensor.dataPointer<int32_t>(), kvCacheStartIdxTensor.dataPointer<int32_t>(),
            cuTotalKvCacheLensTensor.dataPointer<int32_t>(), kvCacheEndIdxsTensor.dataPointer<int32_t>(), runtimeSeqLen,
            runtimeBatchSize, stream);

        auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumHeadQ, mNumHeadKV,
            mNumElemPerHead, mSMVersion, attentionInputLayout);

        // Prepare FMHA_v2 params to launch FMHA kernel
        FusedMultiheadAttentionParamsV2 params{};
        memset(&params, 0, sizeof(params));
        fmhaRunner.setupParams(params);
        params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

        if (attentionInputLayout == AttentionInputLayout::CONTIGUOUS_Q_KV)
        {
            // Assign Q tensor to keep roped Q results with layout [B, Sq, Hq, D]
            rt::Tensor qVecTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumHeadQ, mNumElemPerHead}, DataType::kHALF);
            // q: [b, s, hq+hk+hv, d] -> [b, s, hq, d]
            kernel::launchApplyRopeWriteContinuousQAndKVCache(qkvInputTensor.dataPointer<half>(),
                kvCacheTensor.dataPointer<half>(), ropeCosSinTensor.dataPointer<float>(),
                qVecTensor.dataPointer<half>(), kvCacheEndIdxsTensor.dataPointer<int32_t>(), runtimeSeqLen,
                totalProcessToken, mKVCacheCapacity, mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim,
                cosSinCacheBatchSize, cosSinCacheSeqLen, stream);

            rt::Tensor kvCacheFMHATensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                {runtimeBatchSize, runtimeSeqLen, 2, mNumHeadKV, mKVCacheCapacity, mNumElemPerHead}, DataType::kHALF);
            // kvCache: [b, 2, hkv, s, d] -> [b, s, 2, hkv, d]
            kernel::cvtKVCachelayoutXQAToFMHA<half>(kvCacheTensor.dataPointer<half>(),
                kvCacheFMHATensor.dataPointer<half>(), runtimeBatchSize, mKVCacheCapacity, mNumHeadKV, mNumElemPerHead,
                stream);

            // Set device ptr for FMHA kernel.
            params.s_kv = mKVCacheCapacity;
            params.q_ptr = qVecTensor.dataPointer<half>();
            params.kv_ptr = kvCacheFMHATensor.dataPointer<half>();
            params.cu_kv_seqlens = cuTotalKvCacheLensTensor.dataPointer<int32_t>();
            params.o_ptr = attentionOutputTensor.dataPointer<half>();
        }
        else
        { // PACKED_QKV
            kernel::launchApplyRopeWriteKVContext(qkvInputTensor.dataPointer<half>(), kvCacheTensor.dataPointer<half>(),
                ropeCosSinTensor.dataPointer<float>(), runtimeSeqLen, totalProcessToken, mKVCacheCapacity, mNumHeadQ,
                mNumHeadKV, mNumElemPerHead, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, stream);
            params.qkv_ptr = qkvInputTensor.dataPointer<half>();
            params.cu_kv_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();
            params.o_ptr = attentionOutputTensor.dataPointer<half>();
        }

        // Dispatch FMHA kernel
        fmhaRunner.dispatchFMHAKernel(params, stream);
    }
    else
    {
        // Generation phase we first prepare Q vector and update KVCache.
        // Currently we only supports generating one token per sequence.
        rt::Tensor qVecTensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumHeadQ, mNumElemPerHead}, DataType::kHALF);
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;
        if (mEnableTreeAttention)
        {

            kernel::launchApplyRopeWriteKVTreeDecode(qkvInputTensor.dataPointer<half>(),
                kvCacheTensor.dataPointer<half>(), qVecTensor.dataPointer<half>(),
                ropeCosSinTensor.dataPointer<float>(), contextLengthTensor.dataPointer<int32_t>(),
                attentionPosIdTensor.dataPointer<int32_t>(), runtimeSeqLen, totalProcessToken, mKVCacheCapacity,
                mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, stream);
        }
        else
        {
            kernel::launchApplyRopeWriteKVDecode(qkvInputTensor.dataPointer<half>(), kvCacheTensor.dataPointer<half>(),
                qVecTensor.dataPointer<half>(), ropeCosSinTensor.dataPointer<float>(),
                contextLengthTensor.dataPointer<int32_t>(), runtimeSeqLen, totalProcessToken, mKVCacheCapacity,
                mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, stream);
        }
        // Prepare GQA runner parameter to dispatch kernel
        auto xqaRunner
            = DecoderXQARunner(mDataType, runtimeBatchSize, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.output = attentionOutputTensor.dataPointer<half>();
        params.qInputPtr = qVecTensor.dataPointer<half>();
        params.kvCache.data = kvCacheTensor.dataPointer<half>();
        params.kvCache.sequence_lengths = contextLengthTensor.dataPointer<int32_t>();
        params.kvCache.capacity = mKVCacheCapacity;
        if (mEnableTreeAttention && runtimeSeqLen > 1)
        {
            params.treeAttnMask = attentionMaskTensor.dataPointer<int32_t>();
            params.qSeqLen = runtimeSeqLen;
            xqaRunner.dispatchSpecDecodeXQAKernel(params, stream);
        }
        else if (runtimeSeqLen == 1)
        {
            // dispatch GQA runner.
            xqaRunner.dispatchXQAKernel(params, stream);
        }
        else
        {
            assert(false);
        }
    }
    return 0;
}

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mMaxBatchSize) + sizeof(mKVCacheCapacity) + sizeof(mNumHeadQ) + sizeof(mNumHeadKV)
        + sizeof(mNumElemPerHead) + sizeof(mEnableTreeAttention) + sizeof(mEnableReuseKVCache);
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mMaxBatchSize);
    serializeValue(&buffer, mKVCacheCapacity);
    serializeValue(&buffer, mNumHeadQ);
    serializeValue(&buffer, mNumHeadKV);
    serializeValue(&buffer, mNumElemPerHead);
    serializeValue(&buffer, mEnableTreeAttention);
    serializeValue(&buffer, mEnableReuseKVCache);
}

int32_t AttentionPlugin::initialize() noexcept
{
    return 0;
}

void AttentionPlugin::terminate() noexcept {}

void AttentionPlugin::destroy() noexcept
{
    delete this;
}

AttentionPluginCreator::AttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_batch_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kv_cache_capacity", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("enable_tree_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_reuse_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> maxBatchSize = parsePluginScalarField<int32_t>("max_batch_size", fc);
        std::optional<int32_t> kvCacheCapacity = parsePluginScalarField<int32_t>("kv_cache_capacity", fc);
        std::optional<int32_t> numQHeads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> numKVHeads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        // Make enable_tree_attention optional with default value 0 (disable by default)
        std::optional<int32_t> enableTreeAttention = parsePluginScalarField<int32_t>("enable_tree_attention", fc);
        int32_t enableTreeAttentionValue = enableTreeAttention.value_or(0);
        // Make enable_reuse_kv_cache optional with default value 0 (disable by default)
        std::optional<int32_t> enableReuseKVCache = parsePluginScalarField<int32_t>("enable_reuse_kv_cache", fc);
        int32_t enableReuseKVCacheValue = enableReuseKVCache.value_or(0);

        // Enforce Core parameters are specified.
        bool checkRequiredFields = maxBatchSize.has_value() && kvCacheCapacity.has_value() && numQHeads.has_value()
            && headSize.has_value() && numKVHeads.has_value();
        if (!checkRequiredFields)
        {
            return nullptr;
        }

        AttentionPlugin* plugin
            = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(), headSize.value(),
                maxBatchSize.value(), kvCacheCapacity.value(), enableTreeAttentionValue, enableReuseKVCacheValue);

        return plugin;
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}

nvinfer1::IPluginV2* AttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new AttentionPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
