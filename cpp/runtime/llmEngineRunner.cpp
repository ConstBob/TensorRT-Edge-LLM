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

#include "runtime/llmEngineRunner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/newEagleUtilKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace drivellm;
using namespace nvinfer1;

namespace
{
std::string formatEngineConfig(drivellm::rt::LLMEngineRunnerConfig const& config)
{
    std::stringstream ss;

    ss << std::boolalpha;
    ss << "LLMEngineRunnerConfig:" << "  enableReuseKVCache: " << config.enableReuseKVCache
       << "  numDecoderLayers: " << config.numDecoderLayers << "  numKVHeads: " << config.numKVHeads
       << "  headDim: " << config.headDim << "  rotaryDim: " << config.rotaryDim
       << "  maxSupportedBatchSize: " << config.maxSupportedBatchSize
       << "  minSupportedInputLength: " << config.minSupportedInputLength
       << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  maxSequenceLength: " << config.maxSequenceLength
       << "  maxSupportedLoraRank: " << config.maxSupportedLoraRank;
    return ss.str();
}

// Compute a unique hash value that can distinguish the various decoding steps.
// Extend this function when we need to capture more information.
size_t hashDecodingInput(rt::Tensor const& inputIds, rt::Tensor const& outputLogits, std::string const& loraWeightsName)
{
    // For vanilla decoding step, the shape can be distingusihed by active batch size.
    // Also capture the pointer address to ensure we are read/write correct locations.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    uintptr_t const inputIdsAddr = reinterpret_cast<uintptr_t>(inputIds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());

    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, activeBatchSize);
    hash_utils::hashCombine(hashValue, inputIdsAddr);
    hash_utils::hashCombine(hashValue, outputLogitsAddr);
    hash_utils::hashCombine(hashValue, loraWeightsName);
    return hashValue;
}
} // namespace

namespace drivellm
{
namespace rt
{

//! Current implementation limits to two optimization profiles per LLM engine.
static constexpr int32_t kCONTEXT_PROFILE_INDEX{0};
static constexpr int32_t kGENERATION_PROFILE_INDEX{1};

// Define the input/output tensor names so that we can modify them uniformly later.
std::string const inputIdsName{"input_ids"};
std::string const contextLengthsName{"context_lengths"};
std::string const lastTokenIdsName{"last_token_ids"};
std::string const logitsName{"logits"};
std::string const ropeCosSinName{"rope_rotary_cos_sin"};
std::string const multimodalEmbeddingsName{"image_embeds"};
std::string const kvCacheStartIndexName{"kvcache_start_index"};
std::string const attentionMaskName{"attention_mask"};
std::string const attentionPosIdName{"attention_pos_id"};
std::string const outputHiddenStatesName{"hidden_states"};

LLMEngineRunner::LLMEngineRunner(std::filesystem::path const& enginePath, std::filesystem::path const& configPath,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    LOG_INFO("Initializing LLMEngineRunner from engine file: %s", enginePath.string().c_str());
    LOG_INFO("Using config file %s", configPath.string().c_str());

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("LLMEngineRunner(): Failed to use MMap to read engine from file path: %s", enginePath.string());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mContextExecutionContext->setOptimizationProfileAsync(kCONTEXT_PROFILE_INDEX, stream);
    setOptimizationProfileStatus
        &= mGenerationExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    // Initialize persistent Rope CosSinCache buffers if they are not dependent on input context.
    Json configJson;
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file with error: %s", e.what());
        throw std::runtime_error("Failed to parse config file: " + configPath.string());
    }

    if (!this->initializeConfigFromJson(configJson))
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner from config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to initialize LLMEngineRunner from config file: " + configPath.string());
    }

    if (!this->validateConfigFromEngine())
    {
        LOG_ERROR("Failed to match config file %s with engine file: %s", configPath.string().c_str(),
            enginePath.string().c_str());
        throw std::runtime_error(
            "Failed to match config file " + configPath.string() + " with engine file: " + enginePath.string());
    }

    auto ropeConfig = collectBaseRopeConfig(configJson);
    if (ropeConfig.type != RopeType::kMRope)
    {
        LOG_DEBUG("LLMEngineRunner(): Initialize persistent Rope CosSinCache.");
        this->mPosEncCosSinCache
            = rt::Tensor({1, mConfig.maxSequenceLength, mConfig.rotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
        bool const initRopeStatus = initializeRopeCosSinCache(mPosEncCosSinCache, ropeConfig, configJson, stream);
        if (!initRopeStatus)
        {
            LOG_ERROR("LLMEngineRunner(): Failed to initialize persistent Rope CosSinCache.");
            throw std::runtime_error("Failed to initialize persistent Rope CosSinCache.");
        }
    }
    else
    {
        this->mPosEncCosSinCache
            = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxSequenceLength, mConfig.rotaryDim},
                rt::DeviceType::kGPU, DataType::kFLOAT);
        CUDA_CHECK(cudaMemsetAsync(mPosEncCosSinCache.rawPointer(), 0, mPosEncCosSinCache.getMemoryCapacity(), stream));
    }
    bool setRopeCosSinCacheStatus{true};
    setRopeCosSinCacheStatus
        &= mContextExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    setRopeCosSinCacheStatus
        &= mGenerationExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    if (!setRopeCosSinCacheStatus)
    {
        LOG_ERROR("Failed to set rope cos sin cache to the engine");
        throw std::runtime_error("Failed to set rope cos sin cache to the engine");
    }

    // Instantiate the KVCache instance of the EngineRunner.
    this->mKVCache
        = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers, mConfig.maxSupportedBatchSize,
                                mConfig.maxSequenceLength, mConfig.numKVHeads, mConfig.headDim},
            stream);

    // Instantiate other GPU memory input that needed by the Engine execution.
    this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT64);
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, mSelectTokenIndices.getMemoryCapacity(), stream));
    this->mSequenceContextLengths = rt::Tensor({mConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(
        cudaMemsetAsync(mSequenceContextLengths.rawPointer(), 0, mSequenceContextLengths.getMemoryCapacity(), stream));

    // Add the LoRA weights to the engine.
    if (isLoraWeightsSupported())
    {
        for (auto const& [loraWeightsName, loraWeightsPath] : loraWeightsMap)
        {
            if (loraWeightsPath.empty())
            {
                continue;
            }
            if (!this->addLoraWeights(loraWeightsName, loraWeightsPath, stream))
            {
                LOG_ERROR("Failed to add LoRA weights: %s", loraWeightsName.c_str());
                throw std::runtime_error("Failed to add LoRA weights: " + loraWeightsName);
            }
        }
    }

    // Initialize the dummy LoRA weights tensor as TensorRT does not support nullptr for binding, even when the LoRA
    // rank is 0.
    mDummyLoraWeightsTensor = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Reset the LoRA weights to zero tensors.
    if (!this->resetLoraWeights(stream))
    {
        LOG_ERROR("Failed to initialize LoRA weights to zero tensors");
        throw std::runtime_error("Failed to initialize LoRA weights to zero tensors");
    }

    // Synchronize the stream to ensure all the operations have completed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool LLMEngineRunner::initializeConfigFromJson(Json const& configJson)
{
    try
    {
        // Define required fields for main config
        std::vector<std::string> const requiredConfigFields = {"num_hidden_layers", "num_key_value_heads", "head_dim",
            "vocab_size", "partial_rotary_factor", "builder_config", "enable_reuse_kv_cache"};

        // Validate required fields exist in main config
        for (auto const& field : requiredConfigFields)
        {
            if (!configJson.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in config", field.c_str());
                return false;
            }
        }

        auto const& builderConfig = configJson["builder_config"];

        // Define required fields for builder_config
        std::vector<std::string> const requiredBuilderConfigFields
            = {"max_batch_size", "max_input_len", "max_seq_len", "max_lora_rank"};

        // Validate required fields exist in builder_config
        for (auto const& field : requiredBuilderConfigFields)
        {
            if (!builderConfig.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in builder_config", field.c_str());
                return false;
            }
        }

        // Extract values with proper type checking
        mConfig.numDecoderLayers = configJson["num_hidden_layers"].get<int32_t>();
        mConfig.numKVHeads = configJson["num_key_value_heads"].get<int32_t>();
        mConfig.headDim = configJson["head_dim"].get<int32_t>();

        auto const partialRotaryFactor = configJson["partial_rotary_factor"].get<float>();
        mConfig.rotaryDim = static_cast<int32_t>(partialRotaryFactor * static_cast<float>(mConfig.headDim));

        mConfig.vocabSize = configJson["vocab_size"].get<int32_t>();

        mConfig.enableReuseKVCache = configJson["enable_reuse_kv_cache"].get<bool>();

        mConfig.maxSupportedBatchSize = builderConfig["max_batch_size"].get<int32_t>();
        mConfig.minSupportedInputLength = 1; // TODO: Change this to min input length
        mConfig.maxSupportedInputLength = builderConfig["max_input_len"].get<int32_t>();
        mConfig.maxSequenceLength = builderConfig["max_seq_len"].get<int32_t>();
        mConfig.maxSupportedLoraRank = builderConfig["max_lora_rank"].get<int32_t>();

        // Validate configuration values - all must be positive except max_lora_rank
        std::vector<std::pair<std::string, int32_t>> positiveFields
            = {{"num_decoder_layers", mConfig.numDecoderLayers}, {"num_key_value_heads", mConfig.numKVHeads},
                {"head_dim", mConfig.headDim}, {"rotary_dim", mConfig.rotaryDim}, {"vocab_size", mConfig.vocabSize},
                {"max_batch_size", mConfig.maxSupportedBatchSize}, {"max_input_len", mConfig.maxSupportedInputLength},
                {"max_seq_len", mConfig.maxSequenceLength}};

        for (auto const& [fieldName, value] : positiveFields)
        {
            if (value <= 0)
            {
                LOG_ERROR("initializeConfigFromJson(): Invalid %s: %d (must be positive)", fieldName.c_str(), value);
                return false;
            }
        }

        // Validate max_lora_rank separately (must be non-negative)
        if (mConfig.maxSupportedLoraRank < 0)
        {
            LOG_ERROR("initializeConfigFromJson(): Invalid max_lora_rank: %d (must be non-negative)",
                mConfig.maxSupportedLoraRank);
            return false;
        }
        if (mConfig.maxSupportedInputLength > mConfig.maxSequenceLength)
        {
            LOG_ERROR(
                "initializeConfigFromJson(): Invalid configuration: max_input_len (%d) cannot be greater than "
                "max_seq_len (%d)",
                mConfig.maxSupportedInputLength, mConfig.maxSequenceLength);
            return false;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("initializeConfigFromJson(): Unexpected error while parsing config: %s", e.what());
        return false;
    }

    LOG_INFO("initializeConfigFromJson(): Loaded LLMEngineRunner with config: %s", formatEngineConfig(mConfig).c_str());
    return true;
}

bool LLMEngineRunner::validateConfigFromEngine()
{
    auto identifyKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 5 && bindingName.find("present_key_values") != std::string::npos;
    };

    // If the engine comes with "kvcache_start_index" binding, it means the engine enables reuse KVCache.
    auto identifyReuseKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 1 && bindingName.find("kvcache_start_index") != std::string::npos;
    };

    int32_t nbKVCacheInputs{0};
    int32_t numIOBindings = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numIOBindings; ++i)
    {
        std::string const bindingName = mEngine->getIOTensorName(i);
        Dims const tensorDim = mEngine->getTensorShape(bindingName.c_str());

        if (identifyKVCacheBinding(bindingName, tensorDim))
        {
            if (mConfig.numKVHeads != tensorDim.d[2])
            {
                LOG_ERROR("numKVHeads is not consistent. From engine: %d, from config: %d", tensorDim.d[2],
                    mConfig.numKVHeads);
                return false;
            }
            if (mConfig.maxSequenceLength != tensorDim.d[3])
            {
                LOG_ERROR("maxSequenceLength is not consistent. From engine: %d, from config: %d", tensorDim.d[3],
                    mConfig.maxSequenceLength);
                return false;
            }
            if (mConfig.headDim != tensorDim.d[4])
            {
                LOG_ERROR(
                    "headDim is not consistent. From engine: %d, from config: %d", tensorDim.d[4], mConfig.headDim);
                return false;
            }
            ++nbKVCacheInputs;
        }
        if (identifyReuseKVCacheBinding(bindingName, tensorDim))
        {
            if (!mConfig.enableReuseKVCache)
            {
                LOG_ERROR(
                    "Enable reuse KVCache is not consistent. Identified reuse KVCache binding, but enableReuseKVCache "
                    "is false from config");
                return false;
            }
        }
    }
    if (nbKVCacheInputs != mConfig.numDecoderLayers)
    {
        LOG_ERROR("numDecoderLayers is not consistent. From engine: %d, from config: %d", nbKVCacheInputs,
            mConfig.numDecoderLayers);
        return false;
    }
    Dims const minInputCtxShape
        = mEngine->getProfileShape(inputIdsName.c_str(), kCONTEXT_PROFILE_INDEX, OptProfileSelector::kMIN);
    Dims const maxInputCtxShape
        = mEngine->getProfileShape(inputIdsName.c_str(), kCONTEXT_PROFILE_INDEX, OptProfileSelector::kMAX);
    if (mConfig.minSupportedInputLength != minInputCtxShape.d[1])
    {
        LOG_ERROR("minSupportedInputLength is not consistent. From engine: %d, from config: %d", minInputCtxShape.d[1],
            mConfig.minSupportedInputLength);
        return false;
    }
    if (mConfig.maxSupportedInputLength != maxInputCtxShape.d[1])
    {
        LOG_ERROR("maxSupportedInputLength is not consistent. From engine: %d, from config: %d", maxInputCtxShape.d[1],
            mConfig.maxSupportedInputLength);
        return false;
    }
    if (mConfig.maxSupportedBatchSize != maxInputCtxShape.d[0])
    {
        LOG_ERROR("maxSupportedBatchSize is not consistent. From engine: %d, from config: %d", maxInputCtxShape.d[0],
            mConfig.maxSupportedBatchSize);
        return false;
    }

    // Obtain vocab size from the engine.
    Dims const logitsDim = mEngine->getTensorShape(logitsName.c_str());
    if (mConfig.vocabSize != logitsDim.d[1])
    {
        LOG_ERROR("vocabSize is not consistent. From engine: %d, from config: %d", logitsDim.d[1], mConfig.vocabSize);
        return false;
    }

    // Obtain rotary dim from the engine.
    Dims const ropeCosSinCacheDim = mEngine->getTensorShape(ropeCosSinName.c_str());
    if (mConfig.rotaryDim != ropeCosSinCacheDim.d[2])
    {
        LOG_ERROR("rotaryDim is not consistent. From engine: %d, from config: %d", ropeCosSinCacheDim.d[2],
            mConfig.rotaryDim);
        return false;
    }

    return true;
}

LLMEngineRunner::~LLMEngineRunner()
{
    for (auto& [hashValue, graphPair] : mCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
}

bool LLMEngineRunner::bindKVCacheToEngine(int32_t activeBatchSize)
{
    // Prepare special input binding shape for prefill stage KVCache input.
    // TODO: Unify the semantics to always pass full KVCache shape.
    Dims const kvCacheDimPrefillIn = {5, {activeBatchSize, 2, mConfig.numKVHeads, 0, mConfig.headDim}};
    Dims const kvCacheDimDecodeIn
        = {5, {activeBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength, mConfig.headDim}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string const pastKeyValuesName = format::fmtstr("past_key_values.%d", i);
        std::string const presentKeyValuesName = format::fmtstr("present_key_values.%d", i);

        rt::Tensor kvCacheBlock = mKVCache.getKVCacheForDecoderLayer(i);
        status &= mContextExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mContextExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status
            &= mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());

        status &= mContextExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDimPrefillIn);
        status &= mGenerationExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDimDecodeIn);
    }
    return status;
}

rt::Tensor& LLMEngineRunner::getRopeCosSinCacheTensor()
{
    return mPosEncCosSinCache;
}

LLMEngineRunnerConfig LLMEngineRunner::getEngineConfig() const
{
    return mConfig;
}

rt::LinearKVCache& LLMEngineRunner::getLinearKVCache()
{
    return mKVCache;
}

bool LLMEngineRunner::prefillStepInputValidation(
    rt::Tensor const& inputIds, rt::Tensor const& contextLengths, rt::Tensor const& outputLogits)
{
    int32_t activeBatchSize = inputIds.getShape()[0];
    int32_t prefillSequenceLength = inputIds.getShape()[1];

    bool const checkInputsGPUTensor = inputIds.getDeviceType() == rt::DeviceType::kGPU
        && contextLengths.getDeviceType() == rt::DeviceType::kCPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "executePrefill(): Invalid device type of I/O tensors. ContextLengths input should reside on CPU and "
            "the rest should reside on GPU.");
        return false;
    }
    bool const isBatchValid = activeBatchSize <= mConfig.maxSupportedBatchSize
        && contextLengths.getShape()[0] == activeBatchSize && outputLogits.getShape()[0] == activeBatchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "executePrefill(): Invalid batchSize of the input tensors. Either batchSize is larger than "
            "maxSupportedBatchSize or batchSize is not consistent among the input tensors. "
            "Current inputIds shape: %s, contextLengths shape: %s, logits shape: %s",
            inputIds.getShape().formatString().c_str(), contextLengths.getShape().formatString().c_str(),
            outputLogits.getShape().formatString().c_str());
        return false;
    }
    if (prefillSequenceLength > mConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "executePrefill(): Invalid sequence length of the input tensors. Input sequence length (%d) is larger "
            "than maxSupportedInputLength (%d). Current inputIds shape: %s.",
            prefillSequenceLength, mConfig.maxSupportedInputLength, inputIds.getShape().formatString().c_str());
        return false;
    }
    bool const isLogitsShapeValid
        = outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.vocabSize;
    if (!isLogitsShapeValid)
    {
        LOG_ERROR(
            "executePrefill(): Invalid shape of the output logits tensor. The output logits tensor should have shape "
            "[activeBatchSize, VocabSize]. Current logits shape is %s.",
            outputLogits.getShape().formatString().c_str());
        return false;
    }
    return true;
}

bool LLMEngineRunner::executePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& hostContextLengths,
    rt::Tensor const& multimodalEmbeddings, rt::Tensor& outputLogits, cudaStream_t stream)
{
    bool const validateInputStatus = this->prefillStepInputValidation(inputIds, hostContextLengths, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("executePrefill(): Prefill request not performed due to invalid input tensors.");
        return false;
    }

    // Verirify input tensorShape is valid.
    int32_t activeBatchSize = inputIds.getShape()[0];

    // conduct preparation work for the engine execution.
    mSelectTokenIndices.reshape({activeBatchSize, 1});
    mSequenceContextLengths.reshape({activeBatchSize});

    std::vector<int64_t> selectTokenIndicesHost(activeBatchSize, 0);
    int32_t const* contextLengthsData = hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectTokenIndicesHost[i] = contextLengthsData[i] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(mSelectTokenIndices.rawPointer(), selectTokenIndicesHost.data(),
        activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), hostContextLengths.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    bool setEngineIOStatus{true};
    // Engine input tensors.
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
    if (!multimodalEmbeddings.isEmpty())
    {
        setEngineIOStatus &= mContextExecutionContext->setTensorAddress(
            multimodalEmbeddingsName.c_str(), const_cast<void*>(multimodalEmbeddings.rawPointer()));
        setEngineIOStatus &= mContextExecutionContext->setInputShape(
            multimodalEmbeddingsName.c_str(), multimodalEmbeddings.getShape().getTRTDims());
    }
    if (mConfig.enableReuseKVCache)
    {
        setEngineIOStatus &= mContextExecutionContext->setTensorAddress(
            kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mContextExecutionContext->setInputShape(
            kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().getShape().getTRTDims());
    }

    // Engine output tensors.
    setEngineIOStatus &= mContextExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());
    // Bind the KVCache IO to the engine.
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executePrefill(): Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mContextExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("executePrefill(): Failed on TensorRT prefill stage enqueueV3() call.");
        return false;
    }
    // Prefill operation has completed, commit the new contents with KVCache.
    mKVCache.commitPrefillRequest(mSequenceContextLengths, stream);

    LOG_DEBUG("executePrefill(): Prefill stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool LLMEngineRunner::vanlliaDecodingStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& outputLogits)
{
    int32_t activeBatchSize = inputIds.getShape()[0];
    bool const checkInputsGPUTensor
        = inputIds.getDeviceType() == rt::DeviceType::kGPU && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid device type of the input tensors. inputIds and outputLogits should reside on "
            "GPU.");
        return false;
    }
    bool const isBatchValid = activeBatchSize == mKVCache.getActiveBatchSize();
    if (!isBatchValid)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid batchSize of the input tensors. batchSize shall be equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }
    bool checkInputShapeValid = inputIds.getShape().getNumDims() == 2 && inputIds.getShape()[1] == 1
        && outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.vocabSize;
    if (!checkInputShapeValid)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid shape of the input tensors. The input tensor should have shape "
            "[activeBatchSize, 1] and the output tensor should have shape [activeBatchSize, VocabSize].");
        return false;
    }

    return true;
}

bool LLMEngineRunner::executeVanillaDecodingStep(
    rt::Tensor const& inputIds, rt::Tensor const& multimodalEmbeddings, rt::Tensor& outputLogits, cudaStream_t stream)
{
    bool const validateInputStatus = this->vanlliaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("executeGeneration(): Generation request not performed due to invalid input tensors.");
        return false;
    }

    int32_t activeBatchSize = inputIds.getShape()[0];
    // For vanllia decode stage, the selected token indices are always 0.
    // Also setup the sequence length of each sequence for this run based on committed KVCache length.
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), mKVCache.getKVCacheLengths().rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    // Increment the sequence length due to the implementation constraint of AttentionPlugin.
    constexpr int32_t kDECODE_INCREMENT{1};
    kernel::incrementLengthTensor(mSequenceContextLengths, kDECODE_INCREMENT, stream);

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    size_t const graphHash = hashDecodingInput(inputIds, outputLogits, mActiveLoraWeightsName);
    if (mCudaGraphs.find(graphHash) != mCudaGraphs.end())
    {
        LOG_DEBUG("executeVanillaDecodingStep(): Use pre-captured CUDA graph for this decoding step.");
        cudaGraphExec_t graphExec = mCudaGraphs[graphHash].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setEngineIOStatus{true};
        // Engine input tensors.
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
        setEngineIOStatus
            &= mGenerationExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
        if (!multimodalEmbeddings.isEmpty())
        {
            setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
                multimodalEmbeddingsName.c_str(), const_cast<void*>(multimodalEmbeddings.rawPointer()));
            auto multimodalEmbeddingsDim = multimodalEmbeddings.getShape()[1];
            setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
                multimodalEmbeddingsName.c_str(), {2, {1, multimodalEmbeddingsDim}});
        }
        if (mConfig.enableReuseKVCache)
        {
            setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
                kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().rawPointer());
            setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
                kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().getShape().getTRTDims());
        }
        // Engine output tensors.
        setEngineIOStatus
            &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("executeVanillaDecodingStep(): Failed to set engine input tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("executeVanillaDecodingStep(): Failed on TensorRT decode stage enqueueV3() call.");
            return false;
        }
    }

    // Completed decoding step, commit the KVCache length of this run.
    mKVCache.commitDecodeRequest(stream);
    LOG_DEBUG("executeVanillaDecodingStep(): Decoding stage execution completed for request with batch size %d.",
        activeBatchSize);
    return true;
}

bool LLMEngineRunner::eagleBaseTreeDecodingStepInputValidation(rt::Tensor const& baseTreeDecodingInputIds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = baseTreeDecodingInputIds.getDeviceType() == rt::DeviceType::kGPU
        && baseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid device type of I/O tensors. All inputs and outputs "
            "shall "
            "reside on GPU.");
        return false;
    }
    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = baseTreeDecodingInputIds.getDataType() == DataType::kINT32
        && baseTreeDecodingMask.getDataType() == DataType::kINT8 && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Input token ids shall be INT32, hidden states I/O shall be "
            "FLOAT16, "
            "base tree decoding mask shall be INT8, output logits shall be FLOAT32.");
        return false;
    }
    // Validate shapes of the input tensors.
    bool const isBatchValid = baseTreeDecodingInputIds.getShape()[0] == mKVCache.getActiveBatchSize()
        && baseTreeDecodingMask.getShape()[0] == mKVCache.getActiveBatchSize();
    if (!isBatchValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid batchSize of the input tensors. batchSize shall be "
            "equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }

    int64_t const baseTreeDecodingSize = baseTreeDecodingInputIds.getShape()[1];
    bool const isBaseTreeDecodingSizeValid = baseTreeDecodingMask.getShape()[1] == baseTreeDecodingSize
        && baseTreeDecodingMask.getShape()[2] == baseTreeDecodingSize;
    if (!isBaseTreeDecodingSizeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid base tree decoding size of the input tensors. "
            "Base tree decoding size %d, current base tree decoding mask shape: %s",
            baseTreeDecodingSize, baseTreeDecodingMask.getShape().formatString().c_str());
        return false;
    }

    Dims const outputHiddenStatesDim = mEngine->getTensorShape(outputHiddenStatesName.c_str());
    int32_t const baseModelHiddenDim = outputHiddenStatesDim.d[2];
    bool const isOutputShapeValid = outputLogits.getShape()[0] == outputHiddenStates.getShape()[0]
        && outputLogits.getShape()[1] == mConfig.vocabSize && outputHiddenStates.getShape()[1] == baseModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid shape of the output tensors. Logits shape shall be "
            "[select-token-size, %d], hidden states shape shall be [select-token-size, %d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            mConfig.vocabSize, baseModelHiddenDim, outputLogits.getShape().formatString().c_str(),
            outputHiddenStates.getShape().formatString().c_str());
        return false;
    }

    return true;
}

bool LLMEngineRunner::executeEagleBaseTreeDecodingStep(rt::Tensor const& baseTreeDecodingInputIds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& multimodalEmbeddings, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->eagleBaseTreeDecodingStepInputValidation(
        baseTreeDecodingInputIds, baseTreeDecodingMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR(
            "executeEagleBaseTreeDecodingStep(): Eagle base tree decoding request not performed due to invalid input "
            "tensors.");
        return false;
    }

    int32_t const activeBatchSize = baseTreeDecodingInputIds.getShape()[0];
    int32_t const baseTreeDecodingSize = static_cast<int32_t>(baseTreeDecodingInputIds.getShape()[1]);
    int32_t const packedBaseTreeDecodingMaskLen = static_cast<int32_t>(divUp(baseTreeDecodingSize, 32));

    // Prepare extra input for engine execution. Assemble packed base tree decoding mask, position indices, select token
    // indices, sequence context lengths.
    mSelectTokenIndices.reshape({activeBatchSize, baseTreeDecodingSize});
    mSequenceContextLengths.reshape({activeBatchSize});
    mEagleBasePositionIds.reshape({activeBatchSize, baseTreeDecodingSize});
    mEagleBasePackedMask.reshape({activeBatchSize, baseTreeDecodingSize, packedBaseTreeDecodingMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndices = mKVCache.getKVCacheLengths();
    kernel::prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMask, sequenceStartIndices, mEagleBasePackedMask,
        mEagleBasePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        inputIdsName.c_str(), const_cast<void*>(baseTreeDecodingInputIds.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        inputIdsName.c_str(), baseTreeDecodingInputIds.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(attentionMaskName.c_str(), mEagleBasePackedMask.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionMaskName.c_str(), mEagleBasePackedMask.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        attentionPosIdName.c_str(), mEagleBasePositionIds.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionPosIdName.c_str(), mEagleBasePositionIds.getShape().getTRTDims());
    if (!multimodalEmbeddings.isEmpty())
    {
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            multimodalEmbeddingsName.c_str(), const_cast<void*>(multimodalEmbeddings.rawPointer()));
        auto multimodalEmbeddingsDim = multimodalEmbeddings.getShape()[1];
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            multimodalEmbeddingsName.c_str(), {2, {1, multimodalEmbeddingsDim}});
    }
    // Bind the output tensor into the engine.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        outputHiddenStatesName.c_str(), outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executeEagleBaseTreeDecodingStep(): Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR(
            "executeEagleBaseTreeDecodingStep(): Failed on TensorRT eagle base tree decoding stage enqueueV3() call.");
        return false;
    }

    // Note in the base tree decoding step we explicitly don't commit the KVCache since we process the "whole tree" in
    // these steps.
    LOG_DEBUG(
        "executeEagleBaseTreeDecodingStep(): Eagle base tree decoding stage execution completed for request with batch "
        "size %d.",
        activeBatchSize);
    return true;
}

bool LLMEngineRunner::captureVanillaDecodingCudaGraph(
    rt::Tensor const& inputIds, rt::Tensor& outputLogits, std::string const& loraWeightsPath, cudaStream_t stream)
{
    size_t const hashValue = hashDecodingInput(inputIds, outputLogits, loraWeightsPath);
    if (mCudaGraphs.find(hashValue) != mCudaGraphs.end())
    {
        LOG_INFO(
            "captureVanillaDecodingCudaGraph(): CUDA graph already captured for the input tensors with LoRA weights "
            "%s.",
            loraWeightsPath.c_str());
        return true;
    }

    if (isLoraWeightsSupported() && !this->switchLoraWeights(loraWeightsPath, stream))
    {
        LOG_ERROR(
            "captureVanillaDecodingCudaGraph(): Failed to switch LoRA weights to '%s', unable to capture CUDA graph.",
            loraWeightsPath.c_str());
        return false;
    }

    // To avoid CUDA graph error from TensorRT engine, we need to enqueueV3() once prior to graph capture.
    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(
        reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);

    mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    // Validate the condition here after the simulate prefill step.
    bool const validateInputStatus = this->vanlliaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("captureVanillaDecodingCudaGraph(): Generation request is invalid, unable to capture CUDA graph.");
        return false;
    }

    // Set shape of mSelectTokenIndices and set value to all zero.
    // Set sequence context length input for decoding step.
    mSelectTokenIndices.reshape({activeBatchSize, 1});
    mSequenceContextLengths.reshape({activeBatchSize});
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), mKVCache.getKVCacheLengths().rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));

    // Set engine I/O using the same logic as executeVanillaDecodingStep().
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mGenerationExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
    if (mConfig.enableReuseKVCache)
    {
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            kvCacheStartIndexName.c_str(), mKVCache.getKVCacheLengths().getShape().getTRTDims());
    }

    // Engine output tensors.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());

    // Bind the KVCache since we haven't executed the real prefill step.
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);
    if (!setEngineIOStatus)
    {
        LOG_ERROR(
            "captureVanillaDecodingCudaGraph(): Failed to set engine input tensors, unable to capture CUDA graph.");
        return false;
    }

    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("captureVanillaDecodingCudaGraph(): Failed on TensorRT engine enqueueV3() call.");
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    mCudaGraphs[hashValue] = std::make_pair(graph, graphExec);

    if (!executeStatus)
    {
        LOG_WARNING(
            "captureVanillaDecodingCudaGraph(): Failed on TensorRT engine enqueueV3() call during CUDA graph capture.");
        return false;
    }
    else
    {
        LOG_DEBUG(
            "captureVanillaDecodingCudaGraph(): CUDA graph captured successfully for input shape %s with LoRA weights "
            "'%s' (Empty string if no LoRA weights).",
            inputIds.getShape().formatString().c_str(), loraWeightsPath.c_str());
    }

    return true;
}

bool LLMEngineRunner::resetLoraWeights(cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        return true;
    }
    mActiveLoraWeightsName = "";
    bool resetStatus{true};
    for (auto const& loraWeightsTensorName : getLoraWeightsTensorNames())
    {
        nvinfer1::Dims zeroShape
            = mEngine->getProfileShape(loraWeightsTensorName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        resetStatus &= mContextExecutionContext->setTensorAddress(
            loraWeightsTensorName.c_str(), mDummyLoraWeightsTensor.rawPointer());
        resetStatus &= mGenerationExecutionContext->setTensorAddress(
            loraWeightsTensorName.c_str(), mDummyLoraWeightsTensor.rawPointer());
        if (loraWeightsTensorName.find("lora_A") != std::string::npos)
        {
            zeroShape.d[1] = 0;
        }
        else if (loraWeightsTensorName.find("lora_B") != std::string::npos)
        {
            zeroShape.d[0] = 0;
        }
        resetStatus &= mContextExecutionContext->setInputShape(loraWeightsTensorName.c_str(), zeroShape);
        resetStatus &= mGenerationExecutionContext->setInputShape(loraWeightsTensorName.c_str(), zeroShape);
        if (!resetStatus)
        {
            LOG_ERROR("Failed to reset LoRA weights: %s", loraWeightsTensorName.c_str());
            return false;
        }
    }
    return resetStatus;
}

bool LLMEngineRunner::addLoraWeights(
    std::string const& loraWeightsName, std::string const& loraWeightsPath, cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        LOG_ERROR("addLoraWeights(): Engine does not support LoRA weights.");
    }

    if (mLoraWeights.find(loraWeightsName) != mLoraWeights.end())
    {
        LOG_ERROR("addLoraWeights(): LoRA weights %s already added", loraWeightsName.c_str());
        return false;
    }

    // Load tensors using the new unified interface
    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(loraWeightsPath, tensors, stream))
    {
        LOG_ERROR("addLoraWeights(): Failed to load LoRA weights %s from: %s", loraWeightsName.c_str(),
            loraWeightsPath.c_str());
        return false;
    }

    // Validate the LoRA weights do not exceed the max LoRA rank
    for (auto const& tensor : tensors)
    {
        if (tensor.getName().find("lora_A") != std::string::npos)
        {
            if (tensor.getShape()[1] > mConfig.maxSupportedLoraRank)
            {
                LOG_ERROR("addLoraWeights(): LoRA A (%s) tensor's rank (%d) exceeds the max LoRA rank (%d)",
                    tensor.getName().c_str(), tensor.getShape()[1], mConfig.maxSupportedLoraRank);
                return false;
            }
        }
        else if (tensor.getName().find("lora_B") != std::string::npos)
        {
            if (tensor.getShape()[0] > mConfig.maxSupportedLoraRank)
            {
                LOG_ERROR("addLoraWeights(): LoRA B (%s) tensor's rank (%d) exceeds the max LoRA rank (%d)",
                    tensor.getName().c_str(), tensor.getShape()[0], mConfig.maxSupportedLoraRank);
                return false;
            }
        }
    }

    // Store the tensors in our map
    mLoraWeights[loraWeightsName] = std::move(tensors);
    LOG_INFO("addLoraWeights(): Added LoRA weights %s from: %s", loraWeightsName.c_str(), loraWeightsPath.c_str());
    return true;
}

std::vector<std::string> LLMEngineRunner::getLoraWeightsTensorNames() const
{
    std::vector<std::string> loraWeightsTensorNames;
    // Get the number of bindings in the engine
    int32_t numBindings = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numBindings; ++i)
    {
        char const* bindingName = mEngine->getIOTensorName(i);
        std::string bindingNameStr(bindingName);
        if (bindingNameStr.find("lora_") != std::string::npos)
        {
            loraWeightsTensorNames.push_back(bindingNameStr);
        }
    }
    return loraWeightsTensorNames;
}

bool LLMEngineRunner::switchLoraWeights(std::string const& loraWeightsName, cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        LOG_ERROR("switchLoraWeights(): API call is invalid. LLM engine does not support LoRA weights.");
        return false;
    }
    if (loraWeightsName.empty())
    {
        this->resetLoraWeights(stream);
        LOG_DEBUG("switchLoraWeights(): Switched to no LoRA weights.");
        return true;
    }

    // Check if the requested LoRA exists
    auto it = mLoraWeights.find(loraWeightsName);
    if (it == mLoraWeights.end())
    {
        LOG_ERROR("switchLoraWeights(): LoRA weights with name '%s' not found", loraWeightsName.c_str());
        return false;
    }

    auto& loraTensors = it->second;

    // Iterate through all LoRA weights bindings
    for (auto const& loraWeightsTensorName : this->getLoraWeightsTensorNames())
    {
        // Try to find the tensor in the LoRA weights
        auto loraTensorIt = std::find_if(loraTensors.begin(), loraTensors.end(),
            [loraWeightsTensorName](rt::Tensor const& tensor) { return tensor.getName() == loraWeightsTensorName; });

        bool setLoraWeightsStatus{true};

        if (loraTensorIt != loraTensors.end())
        {
            // Found matching tensor, use its data
            setLoraWeightsStatus
                &= mContextExecutionContext->setInputShape(loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus &= mGenerationExecutionContext->setInputShape(
                loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus &= mContextExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
            setLoraWeightsStatus &= mGenerationExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
            LOG_DEBUG("switchLoraWeights(): LoRA weights tensor with name '%s' found. Set shape to %s.",
                loraWeightsTensorName.c_str(), loraTensorIt->getShape().formatString().c_str());
        }
        else
        {
            nvinfer1::Dims shape
                = mEngine->getProfileShape(loraWeightsTensorName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            if (loraWeightsTensorName.find("lora_A") != std::string::npos)
            {
                shape.d[1] = 0;
            }
            else if (loraWeightsTensorName.find("lora_B") != std::string::npos)
            {
                shape.d[0] = 0;
            }
            setLoraWeightsStatus &= mContextExecutionContext->setInputShape(loraWeightsTensorName.c_str(), shape);
            setLoraWeightsStatus &= mGenerationExecutionContext->setInputShape(loraWeightsTensorName.c_str(), shape);
            mContextExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), mDummyLoraWeightsTensor.rawPointer());
            setLoraWeightsStatus &= mGenerationExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), mDummyLoraWeightsTensor.rawPointer());
            LOG_DEBUG("switchLoraWeights(): LoRA weights tensor with name '%s' not found. Set shape to rank 0.",
                loraWeightsTensorName.c_str());
        }
        if (!setLoraWeightsStatus)
        {
            LOG_ERROR("Failed to set LoRA weights: %s", loraWeightsTensorName.c_str());
            return false;
        }
    }
    // Set the active LoRA weights name
    mActiveLoraWeightsName = loraWeightsName;
    LOG_DEBUG("switchLoraWeights(): Switched to LoRA weights with name '%s'.", loraWeightsName.c_str());
    return true;
}

std::string LLMEngineRunner::getActiveLoraWeightsName() const
{
    return mActiveLoraWeightsName;
}

std::vector<std::string> LLMEngineRunner::getAvailableLoraWeights() const
{
    std::vector<std::string> loraWeightsNames;
    for (auto const& [loraWeightsName, _] : mLoraWeights)
    {
        loraWeightsNames.push_back(loraWeightsName);
    }
    return loraWeightsNames;
}

bool LLMEngineRunner::isLoraWeightsSupported() const
{
    return mConfig.maxSupportedLoraRank > 0;
}

} // namespace rt
} // namespace drivellm
