/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "attentionPlugin/fmha-v2/contextFMHARunner.h"
#include "attentionPlugin/xqa/decoderXQARunner.h"
#include "pluginUtils.h"

#include <cassert>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;
using namespace drivellm;

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};

// TODO: Remove this type for long-context optimization since we don't need it now.
constexpr RopeInitType kROPE_INIT_TYPE = RopeInitType::kDEFAULT;

constexpr int32_t kDEVICE_ALIGNMENT{128}; // Make sure all device pointers are aligned by 128.

int8_t* alignDevicePtr(void* ptr)
{
    // Convert the pointer to an integer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + kDEVICE_ALIGNMENT) & ~static_cast<uintptr_t>(kDEVICE_ALIGNMENT);

    // Convert the aligned address back to a pointer
    return reinterpret_cast<int8_t*>(aligned_addr);
}

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    nvinfer1::PluginFieldType toFieldType<T>()                                                                         \
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
            check(toFieldType<T>() == pluginField.type, "Mismatch datatype of plugin field");
            check(pluginField.length == 1 && pluginField.data != nullptr, "Invalid plugin field");
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
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t maxBatchSize, int32_t kvCacheCapacity, PositionEmbeddingType posEmbedType, int32_t halfRotaryDim,
    int32_t rotaryEmbeddingMaxPositions)
    : mLayerName(name)
    , mNumHeadQ(numQHeads)
    , mNumHeadKV(numKVHeads)
    , mNumElemPerHead(headSize)
    , mMaxBatchSize(maxBatchSize)
    , mKVCacheCapacity(kvCacheCapacity)
    , mPosEmbedType(posEmbedType)
    , mHalfRotaryDim(halfRotaryDim)
    , mRotaryEmbeddingMaxPositions(rotaryEmbeddingMaxPositions)
{
    mSMVersion = getSMVersion();

    bool canImplement = ContextFMHARunner::canImplement(mNumElemPerHead, mSMVersion, mDataType)
        && DecoderXQARunner::canImplement(mNumHeadQ, mNumHeadKV, mSMVersion, mDataType);
    if (!canImplement)
    {
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }

    // Load FMHA and XQA kernels to device. The kernel code will only be loaded once if
    // multiple AttentionPlugin instances exist in the model.
    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType);
}

AttentionPlugin::AttentionPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mMaxBatchSize);
    deserializeValue(&data, &length, &mKVCacheCapacity);
    deserializeValue(&data, &length, &mNumHeadQ);
    deserializeValue(&data, &length, &mNumHeadKV);
    deserializeValue(&data, &length, &mNumElemPerHead);
    deserializeValue(&data, &length, &mPosEmbedType);
    deserializeValue(&data, &length, &mRotaryScale);
    deserializeValue(&data, &length, &mRotaryBaseFrequency);

    mSMVersion = getSMVersion();
    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType);
}

AttentionPlugin::~AttentionPlugin() {}

void AttentionPlugin::setRotaryConfig(float ropeScale, float ropeBaseFrequency)
{
    mRotaryScale = ropeScale;
    mRotaryBaseFrequency = ropeBaseFrequency;
}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(mLayerName, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mMaxBatchSize,
        mKVCacheCapacity, mPosEmbedType, mHalfRotaryDim, mRotaryEmbeddingMaxPositions);
    plugin->setRotaryConfig(mRotaryScale, mRotaryBaseFrequency);
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
    //      buffer. Real context length: [B] (a vector of scalars) with type int32_t.
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

    auto checkMropeRotaryCosSin = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    auto checkMropePositionDeltas = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT64;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    try
    {
        if (mPosEmbedType == PositionEmbeddingType::kMROPE)
        {
            assert(nbInputs == 5 && nbOutputs == 2);
        }
        else
        {
            assert(nbInputs == 3 && nbOutputs == 2);
        }
        assert(pos < (nbInputs + nbOutputs));
        bool result{false};
        switch (pos)
        {
        case 0: result = checkGemmQKV(inOut[0]); break;
        case 1: result = checkKVCache(inOut[1]); break;
        case 2: result = checkSequenceLen(inOut[2]); break;
        case 3:
            if (mPosEmbedType == PositionEmbeddingType::kMROPE)
            {
                result = checkMropeRotaryCosSin(inOut[3]);
            }
            else
            {
                result = checkAttentionOutput(inOut[3]);
            }
            break;

        case 4:
            if (mPosEmbedType == PositionEmbeddingType::kMROPE)
            {
                result = checkMropePositionDeltas(inOut[4]);
            }
            else
            {
                result = checkKVCache(inOut[4]);
            }
            break;
        case 5:
            if (mPosEmbedType == PositionEmbeddingType::kMROPE)
            {
                result = checkAttentionOutput(inOut[5]);
            }
            break;
        case 6:
            if (mPosEmbedType == PositionEmbeddingType::kMROPE)
            {
                result = checkKVCache(inOut[6]);
            }
            break;
        default: break;
        }
        return result;
    }
    catch (std::exception const& e)
    {
    }
    return false;
}

// IPluginV2Ext Methods
DataType AttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return DataType::kHALF;
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
    DimsExprs output;
    if (outputIndex == 0)
    {
        output.nbDims = 4;
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = exprBuilder.constant(mNumHeadQ);
        output.d[3] = exprBuilder.constant(mNumElemPerHead);
    }
    else
    {
        // Output[1] is KVCache, identical input[1]
        output = inputs[1];
    }
    return output;
}

void AttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

// TODO: extend the worksapce calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // We may want to reserve workspace here, need to determine more details after implementing the runners.
    // For FMHA kernel we need a buffer to store prefix sum of context lengths.
    // For GQA kernel we need to reserve a buffer space to store the Q tensor after rope transformation.
    constexpr int32_t nbBytesPerData{2};
    int32_t const nbBytesQTensor = nbBytesPerData * mMaxBatchSize * mNumHeadQ * mNumElemPerHead;

    // Add alignment to ensure we have enough device space at worst scenrio.
    return nbBytesQTensor + kDEVICE_ALIGNMENT;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    constexpr int32_t kQKV_INPUT_IDX{0};
    constexpr int32_t kKV_CACHE_INPUT_OUTPUT_IDX{1};
    constexpr int32_t kINPUT_LENGTH_INPUT_IDX{2};
    constexpr int32_t kATTENTION_OUTPUT_IDX{0};
    constexpr int32_t kMrope_Rotary_Cos_Sin_IDX{3};
    constexpr int32_t kMrope_Position_Deltas_IDX{4};

    // Obtain execution time batch size, input context length, and KV-cache capacity per sequence.
    constexpr int32_t kQKV_INPUT_BATCH_DIM_IDX{0};
    constexpr int32_t kQKV_INPUT_SEQLEN_DIM_IDX{1};
    PluginTensorDesc const& qkvInputDesc = inputDesc[kQKV_INPUT_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputDesc.dims.d[kQKV_INPUT_BATCH_DIM_IDX]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputDesc.dims.d[kQKV_INPUT_SEQLEN_DIM_IDX]);

    constexpr int32_t kKV_CACHE_SEQUENCE_LENGTH_DIM_IDX{3};
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kKV_CACHE_INPUT_OUTPUT_IDX];

    // The input kv-cache length is zero at context phase. We use it to distinguish context
    // and decoding phase.
    int32_t const kvCacheInputLength = kvCacheInputDesc.dims.d[kKV_CACHE_SEQUENCE_LENGTH_DIM_IDX];
    bool const isContextPhase = kvCacheInputLength == 0;

    // Check the runtime batch size and input context length are valid for execution.
    check(runtimeBatchSize < mMaxBatchSize,
        "Runtime batchsize exceed max batch size. This will overflow device data buffer");
    check(runtimeSeqLen < mKVCacheCapacity,
        "Runtime sequence length exceed max total context lengths. This will overflow KVCache buffer");

    half* qkvDevicePtr = reinterpret_cast<half*>(const_cast<void*>(inputs[kQKV_INPUT_IDX]));
    int32_t const* seqLengthDevicePtr = reinterpret_cast<int32_t const*>(inputs[kINPUT_LENGTH_INPUT_IDX]);
    half* attentionResultDevicePtr = reinterpret_cast<half*>(outputs[kATTENTION_OUTPUT_IDX]);
    half* kvCacheDevicePtr = reinterpret_cast<half*>(outputs[kKV_CACHE_INPUT_OUTPUT_IDX]);
    float2 const* mrope_rotary_cos_sin = (mPosEmbedType == PositionEmbeddingType::kMROPE)
        ? reinterpret_cast<float2 const*>(inputs[kMrope_Rotary_Cos_Sin_IDX])
        : nullptr;
    int64_t const* mrope_position_deltas = (mPosEmbedType == PositionEmbeddingType::kMROPE)
        ? reinterpret_cast<int64_t const*>(inputs[kMrope_Position_Deltas_IDX])
        : nullptr;

    // Align workspace to be minimal aligned.
    int8_t* alignedWorkspacePtr = alignDevicePtr(workspace);

    if (isContextPhase)
    {
        // At Context phase. Do 1. Apply rope and write KVCache. 2. Dispatch FMHA runner.
        // RoPE kernel now only handle padded input sequence, we treat all "tokens" in the
        // padded input as processing targets.
        // TODO: Explore non-padded input format.
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;
        invokeContextApplyRopeUpdateKVFP16(qkvDevicePtr, nullptr, kvCacheDevicePtr, seqLengthDevicePtr, mNumHeadQ,
            mNumHeadKV, mNumElemPerHead, mKVCacheCapacity, runtimeSeqLen, mPosEmbedType, mRotaryBaseFrequency,
            mRotaryScale, kROPE_INIT_TYPE, totalProcessToken, mHalfRotaryDim, mRotaryEmbeddingMaxPositions,
            mrope_rotary_cos_sin, stream);

        // Prepare FMHA_v2 params to launch FMHA kernel
        auto fmhaRunner = ContextFMHARunner(
            mDataType, runtimeBatchSize, runtimeSeqLen, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mSMVersion);
        Fused_multihead_attention_params_v2 params{};
        params.clear();
        fmhaRunner.setupParams(params);

        // Set device ptr for FMHA kernel.
        params.qkv_ptr = qkvDevicePtr;
        params.cu_q_seqlens = seqLengthDevicePtr;
        params.o_ptr = attentionResultDevicePtr;

        // Dispatch FMHA kernel
        fmhaRunner.dispatchFMHAKernel(params, stream);
    }
    else
    {
        // Generation phase we first prepare Q vector and update KVCache.
        // Currently we only supports generating one token per sequence.
        half* qVecDevicePtr = reinterpret_cast<half*>(alignedWorkspacePtr);
        int32_t const totalProcessToken = runtimeBatchSize;
        invokeGenerationApplyRopeUpdateKVFP16(qkvDevicePtr, qVecDevicePtr, kvCacheDevicePtr, seqLengthDevicePtr,
            mNumHeadQ, mNumHeadKV, mNumElemPerHead, mKVCacheCapacity, runtimeSeqLen, mPosEmbedType,
            mRotaryBaseFrequency, mRotaryScale, kROPE_INIT_TYPE, totalProcessToken, mHalfRotaryDim,
            mRotaryEmbeddingMaxPositions, mrope_position_deltas, stream);

        // Prepare GQA runner parameter to dispatch kernel
        auto xqaRunner
            = DecoderXQARunner(mDataType, runtimeBatchSize, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.output = attentionResultDevicePtr;
        params.qInputPtr = qVecDevicePtr;
        params.kvCache.data = kvCacheDevicePtr;
        params.kvCache.sequence_lengths = seqLengthDevicePtr;
        params.kvCache.capacity = mKVCacheCapacity;

        // dispatch GQA runner.
        xqaRunner.dispatchXQAKernel(params, stream);
    }
    return 0;
}

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mMaxBatchSize) + sizeof(mKVCacheCapacity) + sizeof(mNumHeadQ) + sizeof(mNumHeadKV)
        + sizeof(mNumElemPerHead) + sizeof(mPosEmbedType) + sizeof(mRotaryScale) + sizeof(mRotaryBaseFrequency);
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mMaxBatchSize);
    serializeValue(&buffer, mKVCacheCapacity);
    serializeValue(&buffer, mNumHeadQ);
    serializeValue(&buffer, mNumHeadKV);
    serializeValue(&buffer, mNumElemPerHead);
    serializeValue(&buffer, mPosEmbedType);
    serializeValue(&buffer, mRotaryScale);
    serializeValue(&buffer, mRotaryBaseFrequency);
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
    mPluginAttributes.emplace_back(PluginField("position_embedding_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rotary_scaling", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rotary_base_frequency", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("half_rotary_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_max_positions", nullptr, PluginFieldType::kINT32, 1));

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

AttentionPlugin* createDefaultAttentionPlugin(char const* name)
{
    constexpr int32_t numQHeads{32};
    constexpr int32_t numKVHeads{8};
    constexpr int32_t headSize{128};
    constexpr int32_t maxBatchSize{16};
    constexpr int32_t kvCacheCapacity{4096};
    constexpr PositionEmbeddingType posEmbedType{PositionEmbeddingType::kROPE_ROTATE_NEOX};
    constexpr int32_t halfRotaryDim{64};
    constexpr int32_t rotaryEmbeddingMaxPositions{32768};

    // Align with Meta's implementation for rotary embedding.
    constexpr float rotaryScale{1.0F};
    constexpr float rotaryFrequency{500000.f};

    AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads, numKVHeads, headSize, maxBatchSize,
        kvCacheCapacity, posEmbedType, halfRotaryDim, rotaryEmbeddingMaxPositions);
    plugin->setRotaryConfig(rotaryScale, rotaryFrequency);
    return plugin;
}

nvinfer1::IPluginV2* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        // If no plugin attribute is available, construct an AttentionPlugin used by llama3-8B model by
        // default. Otherwise, all plugin attributes shall be specified.
        if (fc->nbFields == 0)
        {
            return createDefaultAttentionPlugin(name);
        }

        std::optional<int32_t> maxBatchSize = parsePluginScalarField<int32_t>("max_batch_size", fc);
        std::optional<int32_t> kvCacheCapacity = parsePluginScalarField<int32_t>("kv_cache_capacity", fc);
        std::optional<int32_t> numQHeads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> numKVHeads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        std::optional<int32_t> posEmbedVal = parsePluginScalarField<int32_t>("position_embedding_type", fc);
        std::optional<int32_t> halfRotaryDim = parsePluginScalarField<int32_t>("half_rotary_dim", fc);
        std::optional<int32_t> rotaryEmbeddingMaxPositions
            = parsePluginScalarField<int32_t>("rotary_embedding_max_positions", fc);

        bool checkRequiredFields = maxBatchSize.has_value() && kvCacheCapacity.has_value() && numQHeads.has_value()
            && headSize.has_value() && numKVHeads.has_value() && posEmbedVal.has_value();
        if (!checkRequiredFields)
        {
            return nullptr;
        }
        if (posEmbedVal.value() > k_MAX_POSITION_EMBED_TYPE_VAL)
        {
            return nullptr;
        }

        PositionEmbeddingType const posEmbedType = static_cast<PositionEmbeddingType>(posEmbedVal.value());
        bool const useRotaryEmbed = posEmbedType == PositionEmbeddingType::kROPE_ROTATE_GPTJ
            || posEmbedType == PositionEmbeddingType::kROPE_ROTATE_NEOX
            || posEmbedType == PositionEmbeddingType::kMROPE;

        std::optional<float> rotaryScale{std::nullopt};
        std::optional<float> rotaryFrequency{std::nullopt};
        if (useRotaryEmbed)
        {
            rotaryScale = parsePluginScalarField<float>("rotary_scaling", fc);
            rotaryFrequency = parsePluginScalarField<float>("rotary_base_frequency", fc);
            bool checkRotaryFields = rotaryScale.has_value() && rotaryFrequency.has_value();
            if (!checkRotaryFields)
            {
                return nullptr;
            }
        }

        AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(),
            headSize.value(), maxBatchSize.value(), kvCacheCapacity.value(), posEmbedType, halfRotaryDim.value(),
            rotaryEmbeddingMaxPositions.value());
        if (useRotaryEmbed)
        {
            plugin->setRotaryConfig(rotaryScale.value(), rotaryFrequency.value());
        }

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