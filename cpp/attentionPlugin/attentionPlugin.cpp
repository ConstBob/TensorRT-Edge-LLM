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

#include "attentionPlugin.h"
#include "common/common.h"
#include "common/cudaUtils.h"

#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"

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
    int32_t maxBatchSize, int32_t kvCacheCapacity, int32_t enableTreeAttention)
    : mLayerName(name)
    , mNumHeadQ(numQHeads)
    , mNumHeadKV(numKVHeads)
    , mNumElemPerHead(headSize)
    , mMaxBatchSize(maxBatchSize)
    , mKVCacheCapacity(kvCacheCapacity)
    , mEnableTreeAttention(enableTreeAttention)
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

    mSMVersion = getSMVersion();
    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    // TODO: Fix me too pass spec-deocde support through plugin attributes.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType, useSpecDecode);
}

AttentionPlugin::~AttentionPlugin() {}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(
        mLayerName, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mMaxBatchSize, mKVCacheCapacity, mEnableTreeAttention);
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
        status &= tensorDesc.dims.d[2] == mNumElemPerHead;
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

        if (mEnableTreeAttention)
        {
            assert(nbInputs == 6 && nbOutputs == 2);
        }
        else
        {
            assert(nbInputs == 4 && nbOutputs == 2);
        }

        bool result{true};
        if (pos < nbInputs)
        {
            switch (pos)
            {
            case 0: result = checkGemmQKV(inOut[0]); break;
            case 1: result = checkKVCache(inOut[1]); break;
            case 2: result = checkSequenceLen(inOut[2]); break;
            case 3: result = checkPosEncodingCosSin(inOut[3]); break;
            case 4: result = checkAttentionMask(inOut[4]); break;
            case 5: result = checkAttentionPosId(inOut[5]); break;
            default: break;
            }
        }
        else
        {
            int32_t outPos = pos - nbInputs;
            switch (outPos)
            {
            case 0: result = checkAttentionOutput(inOut[pos]); break;
            case 1: result = checkKVCache(inOut[pos]); break;
            default: break;
            }
        }

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

void AttentionPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
}

// TODO: extend the worksapce calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // We may want to reserve workspace here, need to determine more details after implementing the runners.
    // For FMHA kernel we need a buffer to store prefix sum of context lengths.
    // For GQA kernel we need to reserve a buffer space to store the Q tensor after rope transformation.

    constexpr int32_t nbBytesPerData{2};
    // The worksapce will be used to store the Q tensor. For eagle mode, maxDecodingTokens means the number of Q tensor.
    // Set maxDecodingTokens to 128, which means the max value supported is 128. Please change it if need more.
    constexpr int32_t maxDecodingTokens = 128;
    int32_t const nbBytesQTensor = nbBytesPerData * mMaxBatchSize * mNumHeadQ * mNumElemPerHead * maxDecodingTokens;
    // Add alignment to ensure we have enough device space at worst scenrio.
    return nbBytesQTensor + kDEVICE_ALIGNMENT;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    constexpr int32_t kQKV_INPUT_IDX{0};
    constexpr int32_t kKV_CACHE_INPUT_OUTPUT_IDX{1};
    constexpr int32_t kINPUT_LENGTH_INPUT_IDX{2};
    constexpr int32_t kATTENTION_OUTPUT_IDX{0};
    constexpr int32_t kPOS_ENCODING_COS_SIN_IDX{3};

    // Optional Inputs that only used with spec decoding tree attention.
    constexpr int32_t kATTENTION_MASK_INPUT_IDX{4};
    constexpr int32_t kATTENTION_POS_ID_INPUT_IDX{5};

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

    // Obtain rotaryDim
    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kPOS_ENCODING_COS_SIN_IDX];
    constexpr int32_t kCOS_SIN_ROTARY_DIM_IDX{2};
    int32_t const rotaryDim = static_cast<int32_t>(posEncodingCosSinDesc.dims.d[kCOS_SIN_ROTARY_DIM_IDX]);

    // Check the runtime batch size and input context length are valid for execution.
    check(runtimeBatchSize < mMaxBatchSize,
        "Runtime batchsize exceed max batch size. This will overflow device data buffer");
    check(runtimeSeqLen < mKVCacheCapacity,
        "Runtime sequence length exceed max total context lengths. This will overflow KVCache buffer");
    check(rotaryDim <= mNumElemPerHead, "Rotary dimension exceed head size");

    half* qkvDevicePtr = reinterpret_cast<half*>(const_cast<void*>(inputs[kQKV_INPUT_IDX]));
    int32_t const* seqLengthDevicePtr = reinterpret_cast<int32_t const*>(inputs[kINPUT_LENGTH_INPUT_IDX]);
    half* attentionResultDevicePtr = reinterpret_cast<half*>(outputs[kATTENTION_OUTPUT_IDX]);
    half* kvCacheDevicePtr = reinterpret_cast<half*>(outputs[kKV_CACHE_INPUT_OUTPUT_IDX]);
    float const* posEncodingCosSinDevicePtr = reinterpret_cast<float const*>(inputs[kPOS_ENCODING_COS_SIN_IDX]);

    int32_t* attention_mask = nullptr;
    int32_t* customSeqIndex = nullptr;
    if (mEnableTreeAttention)
    {
        attention_mask = reinterpret_cast<int32_t*>(const_cast<void*>(inputs[kATTENTION_MASK_INPUT_IDX]));
        customSeqIndex = reinterpret_cast<int32_t*>(const_cast<void*>(inputs[kATTENTION_POS_ID_INPUT_IDX]));
    }

    // Align workspace to be minimal aligned.
    int8_t* alignedWorkspacePtr = alignDevicePtr(workspace);

    if (isContextPhase)
    {
        // At Context phase. Do 1. Apply rope and write KVCache. 2. Dispatch FMHA runner.
        // RoPE kernel now only handle padded input sequence, we treat all "tokens" in the
        // padded input as processing targets.
        // TODO: Explore non-padded input format.
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;

        drivellm::kernel::launchApplyRopeWriteKVContext(qkvDevicePtr, kvCacheDevicePtr, posEncodingCosSinDevicePtr,
            runtimeSeqLen, totalProcessToken, mKVCacheCapacity, mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim,
            stream);

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
        int32_t const totalProcessToken = runtimeBatchSize * runtimeSeqLen;
        if (mEnableTreeAttention)
        {

            drivellm::kernel::launchApplyRopeWriteKVTreeDecode(qkvDevicePtr, kvCacheDevicePtr, qVecDevicePtr,
                posEncodingCosSinDevicePtr, seqLengthDevicePtr, customSeqIndex, runtimeSeqLen, totalProcessToken,
                mKVCacheCapacity, mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim, stream);
        }
        else
        {
            drivellm::kernel::launchApplyRopeWriteKVDecode(qkvDevicePtr, kvCacheDevicePtr, qVecDevicePtr,
                posEncodingCosSinDevicePtr, seqLengthDevicePtr, runtimeSeqLen, totalProcessToken, mKVCacheCapacity,
                mNumHeadQ, mNumHeadKV, mNumElemPerHead, rotaryDim, stream);
        }
        // Prepare GQA runner parameter to dispatch kernel
        auto xqaRunner
            = DecoderXQARunner(mDataType, runtimeBatchSize, mNumHeadQ, mNumHeadKV, mNumElemPerHead, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.output = attentionResultDevicePtr;
        params.qInputPtr = qVecDevicePtr;
        params.kvCache.data = kvCacheDevicePtr;
        params.kvCache.sequence_lengths = seqLengthDevicePtr;
        params.kvCache.capacity = mKVCacheCapacity;
        if (mEnableTreeAttention && runtimeSeqLen > 1)
        {
            params.treeAttnMask = attention_mask;
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
        + sizeof(mNumElemPerHead) + sizeof(mEnableTreeAttention);
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mMaxBatchSize);
    serializeValue(&buffer, mKVCacheCapacity);
    serializeValue(&buffer, mNumHeadQ);
    serializeValue(&buffer, mNumHeadKV);
    serializeValue(&buffer, mNumElemPerHead);
    serializeValue(&buffer, mEnableTreeAttention);
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
        // Make enable_tree_attention optional with default value 0
        std::optional<int32_t> enableTreeAttention = parsePluginScalarField<int32_t>("enable_tree_attention", fc);
        int32_t enableTreeAttentionValue = enableTreeAttention.value_or(0);

        // Enforce Core parameters are specified.
        bool checkRequiredFields = maxBatchSize.has_value() && kvCacheCapacity.has_value() && numQHeads.has_value()
            && headSize.has_value() && numKVHeads.has_value();
        if (!checkRequiredFields)
        {
            return nullptr;
        }

        AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(),
            headSize.value(), maxBatchSize.value(), kvCacheCapacity.value(), enableTreeAttentionValue);

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