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

#include "llmEngineRunner.h"

#include "common/logger.h"
#include <sstream>
#include <string>

using namespace nvinfer1;
using Json = nlohmann::json;

namespace
{
std::string formatEngineConfig(drivellm::rt::LLMEngineRunnerConfig const& config)
{
    std::stringstream ss;

    ss << std::boolalpha;
    ss << "LLMEngineRunnerConfig:"
       << "  enableDynamicShape: " << config.enableDynamicShape << "  numDecoderLayers: " << config.numDecoderLayers
       << "  numKVHeads: " << config.numKVHeads << "  headDim: " << config.headDim
       << "  rotaryDim: " << config.rotaryDim << "  maxSupportedBatchSize: " << config.maxSupportedBatchSize
       << "  minSupportedInputLength: " << config.minSupportedInputLength
       << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  maxSequenceLength: " << config.maxSequenceLength;

    return ss.str();
}

// Identity case where we need to have external Rope CosSinCache.
bool identifyContextDependentRopeConfig(Json const& configJson)
{
    if (configJson.contains("rope_scaling"))
    {
        return configJson["rope_scaling"].contains("type")
            && configJson["rope_scaling"]["type"].get<std::string>() == "mrope";
    }
    return false;
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

LLMEngineRunner::LLMEngineRunner(
    std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream)
{
    LOG_INFO("Initializing LLMEngineRunner from engine file: {}", enginePath.string());
    LOG_INFO("Using config file {}", configPath.string());

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    auto mmapReader = std::make_unique<MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("LLMEngineRunner(): Failed to use MMap to read engine from file path: %s", enginePath.string());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    mContextExecutionContext->setOptimizationProfileAsync(kCONTEXT_PROFILE_INDEX, stream);
    mGenerationExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);

    // Obtain engine config from TensorRT engine file.
    // TODO: Obtain the config from json config file and validate When TensorRT engine can implement the config.
    this->initializeConfigFromEngine();

    // Initialize persistent Rope CosSinCache buffers if they are not dependent on input context.
    Json configJson;
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string());
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

    mConfig.useContextDependentRope = identifyContextDependentRopeConfig(configJson);
    if (!mConfig.useContextDependentRope)
    {
        LOG_DEBUG("LLMEngineRunner(): Initialize persistent Rope CosSinCache.");
        this->mPosEncCosSinCache
            = rt::Tensor({1, mConfig.maxSequenceLength, mConfig.rotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mContextExecutionContext->setInputShape(ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
        mGenerationExecutionContext->setInputShape(ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
        // TODO: Unify the move the logic from legacy decoder.cpp to here.
    }
    else
    {
        this->mPosEncCosSinCache
            = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxSequenceLength, mConfig.rotaryDim},
                rt::DeviceType::kGPU, DataType::kFLOAT);
        CUDA_CHECK(cudaMemsetAsync(mPosEncCosSinCache.rawPointer(), 0, mPosEncCosSinCache.getMemoryCapacity(), stream));
        // Shape has to be set during the engine execution time since it depends on the active batch size.
    }
    mContextExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    mGenerationExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());

    // Instantiate the KVCache instance of the EngineRunner.
    this->mKVCache = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers,
        mConfig.maxSupportedBatchSize, mConfig.maxSequenceLength, mConfig.numKVHeads, mConfig.headDim});
    bool const bindKVCacheStatus = this->bindKVCacheToEngine();
    if (!bindKVCacheStatus)
    {
        LOG_ERROR("LLMEngineRunner(): Failed to bind KVCache to the Engine.");
        throw std::runtime_error("Failed to bind KVCache to the Engine.");
    }

    // Instantiate other GPU memory input that needed by the Engine execution.
    this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT64);
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, mSelectTokenIndices.getMemoryCapacity(), stream));
}

LLMEngineRunner::~LLMEngineRunner() {}

void LLMEngineRunner::initializeConfigFromEngine()
{
    // Obtain number of decoder layers from the engine.
    // Each decoder layer has one input and one output KVCache binding
    // with layout [B, 2, Hkv, Smax, D]
    auto identifyKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 5 && bindingName.find("present_key_values") != std::string::npos;
    };
    int32_t nbKVCacheInputs{0};
    int32_t numIOBindings = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numIOBindings; ++i)
    {
        std::string const bindingName = mEngine->getIOTensorName(i);
        Dims const tensorDim = mEngine->getTensorShape(bindingName.c_str());
        if (identifyKVCacheBinding(bindingName, tensorDim))
        {
            if (nbKVCacheInputs == 0)
            {
                mConfig.numKVHeads = tensorDim.d[2];
                mConfig.maxSequenceLength = tensorDim.d[3];
                mConfig.headDim = tensorDim.d[4];
            }
            ++nbKVCacheInputs;
        }
    }
    mConfig.numDecoderLayers = nbKVCacheInputs;

    // Obtain inference configs from supported length.
    // TODO: Obtain the config from json file and compare with TensorRT engine.
    Dims const minInputCtxShape
        = mEngine->getProfileShape(inputIdsName.c_str(), kCONTEXT_PROFILE_INDEX, OptProfileSelector::kMIN);
    Dims const maxInputCtxShape
        = mEngine->getProfileShape(inputIdsName.c_str(), kCONTEXT_PROFILE_INDEX, OptProfileSelector::kMAX);
    mConfig.minSupportedInputLength = minInputCtxShape.d[1];
    mConfig.maxSupportedInputLength = maxInputCtxShape.d[1];
    mConfig.maxSupportedBatchSize = maxInputCtxShape.d[0];
    mConfig.enableDynamicShape = mConfig.minSupportedInputLength != mConfig.maxSupportedInputLength;

    // Obtain vocab size from the engine.
    Dims const logitsDim = mEngine->getTensorShape(logitsName.c_str());
    mConfig.vocabSize = logitsDim.d[1];

    // Obtain rotary dim from the engine.
    Dims const ropeCosSinCacheDim = mEngine->getTensorShape(ropeCosSinName.c_str());
    mConfig.rotaryDim = ropeCosSinCacheDim.d[2];

    LOG_INFO("Loaded LLMEngineRunner with config: %s", formatEngineConfig(mConfig));
}

bool LLMEngineRunner::bindKVCacheToEngine()
{
    // Prepare special input binding shape for prefill stage KVCache input.
    // TODO: Unify the semantics to always pass full KVCache shape.
    Dims const kvCacheDimPrefillIn = {5, {mConfig.maxSupportedBatchSize, 2, mConfig.numKVHeads, 0, mConfig.headDim}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string const pastKeyValuesName = fmtstr("past_key_values.%d", i);
        std::string const presentKeyValuesName = fmtstr("present_key_values.%d", i);

        rt::Tensor kvCacheBlock = mKVCache.getKVCacheForDecoderLayer(i);
        status &= mContextExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mContextExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status
            &= mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());

        status &= mContextExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDimPrefillIn);
        status &= mGenerationExecutionContext->setInputShape(
            pastKeyValuesName.c_str(), kvCacheBlock.getShape().getTRTDims());
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
        && contextLengths.getShape()[0] == activeBatchSize && contextLengths.getShape()[1] == activeBatchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "executePrefill(): Invalid batchSize of the input tensors. Either batchSize is larger than "
            "maxSupportedBatchSize or batchSize is not consistent among the input tensors. "
            "Current inputIds shape: %s, contextLengths shape: %s, logits shape: %s",
            inputIds.getShape().formatString(), contextLengths.getShape().formatString(),
            outputLogits.getShape().formatString());
        return false;
    }
    bool const isSequenceLengthValid = mConfig.enableDynamicShape
        ? prefillSequenceLength <= mConfig.maxSupportedInputLength
        : prefillSequenceLength == mConfig.maxSupportedInputLength;
    if (!isSequenceLengthValid)
    {
        LOG_ERROR(
            "executePrefill(): Invalid sequence length of the input tensors. Either input sequence length is larger "
            "than maxSupportedInputLength or input sequence length is not equal to maxSupportedInputLength with static "
            "shape LLM Engine. Current inputIds shape: %s.",
            inputIds.getShape().formatString());
        return false;
    }
    bool const isLogitsShapeValid
        = outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.vocabSize;
    if (!isLogitsShapeValid)
    {
        LOG_ERROR(
            "executePrefill(): Invalid shape of the output logits tensor. The output logits tensor should have shape "
            "[activeBatchSize, VocabSize]. Current logits shape is %s.",
            outputLogits.getShape().formatString());
        return false;
    }
    return true;
}

bool LLMEngineRunner::executePrefillStep(
    rt::Tensor const& inputIds, rt::Tensor const& hostContextLengths, rt::Tensor& outputLogits, cudaStream_t stream)
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
    mKVCache.resetForNewSequences(activeBatchSize, stream);
    mSelectTokenIndices.reshape({activeBatchSize, 1});

    std::vector<int64_t> selectTokenIndicesHost(activeBatchSize, 0);
    int32_t const* contextLengthsData = hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectTokenIndicesHost[i] = contextLengthsData[i] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(mSelectTokenIndices.rawPointer(), selectTokenIndicesHost.data(),
        activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Commit prefill request sequence will help propagate the context length data to GPU.
    // We need to supply the contextLength because of AttentionPlugin design.
    // TODO: Update this logic to always pass the KVCache start indices prior to execution.
    mKVCache.commitPrefillRequest(hostContextLengths, stream);
    rt::Tensor& contextLengthDevice = mKVCache.getKVCacheLengths();

    bool setEngineIOStatus{true};
    // Engine input tensors.
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(contextLengthsName.c_str(), contextLengthDevice.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        contextLengthsName.c_str(), contextLengthDevice.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());

    // Engine output tensors.
    setEngineIOStatus &= mContextExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executePrefill(): Failed to set engine input tensors.");
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
    rt::Tensor const& inputIds, rt::Tensor& outputLogits, cudaStream_t stream)
{
    bool const validateInputStatus = this->vanlliaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("executeGeneration(): Generation request not performed due to invalid input tensors.");
        return false;
    }

    int32_t activeBatchSize = inputIds.getShape()[0];
    // conduct preparation work for the engine execution.
    // TODO: Adjust the logic to pass in KVCache start indices prior to execution.
    mKVCache.commitDecodeRequest(stream);
    rt::Tensor& contextLengthDevice = mKVCache.getKVCacheLengths();
    // For vanllia decode stage, the selected token indices are always 0.
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));

    bool setEngineIOStatus{true};
    // Engine input tensors.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mGenerationExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(contextLengthsName.c_str(), contextLengthDevice.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        contextLengthsName.c_str(), contextLengthDevice.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(lastTokenIdsName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        lastTokenIdsName.c_str(), mSelectTokenIndices.getShape().getTRTDims());

    // Engine output tensors.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executeGeneration(): Failed to set engine input tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("executeGeneration(): Failed on TensorRT decode stage enqueueV3() call.");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace drivellm