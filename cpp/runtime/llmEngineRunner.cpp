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

#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/stringUtils.h"
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

using namespace drivellm;
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

template <typename T>
void hashCombine(size_t& seed, T const& value)
{
    constexpr size_t kDELTA = 0x9e3779b9;
    seed ^= std::hash<T>()(value) + kDELTA + (seed << 6) + (seed >> 2);
}

// Compute a unique hash value that can distinguish the various decoding steps.
// Extend this function when we need to capture more information.
size_t hashDecodingInput(rt::Tensor const& inputIds, rt::Tensor const& outputLogits)
{
    // For vanilla decoding step, the shape can be distingusihed by active batch size.
    // Also capture the pointer address to ensure we are read/write correct locations.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    uintptr_t const inputIdsAddr = reinterpret_cast<uintptr_t>(inputIds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());

    size_t hashValue = 0;
    hashCombine(hashValue, activeBatchSize);
    hashCombine(hashValue, inputIdsAddr);
    hashCombine(hashValue, outputLogitsAddr);

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

LLMEngineRunner::LLMEngineRunner(
    std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream)
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
    mContextExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    mGenerationExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());

    // Instantiate the KVCache instance of the EngineRunner.
    this->mKVCache = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers,
        mConfig.maxSupportedBatchSize, mConfig.maxSequenceLength, mConfig.numKVHeads, mConfig.headDim});

    // Instantiate other GPU memory input that needed by the Engine execution.
    this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT64);
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, mSelectTokenIndices.getMemoryCapacity(), stream));

    // Synchronize the stream to ensure all the operations have completed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

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

    LOG_INFO("Loaded LLMEngineRunner with config: %s", formatEngineConfig(mConfig).c_str());
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
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());

    if (!multimodalEmbeddings.isEmpty())
    {
        setEngineIOStatus &= mContextExecutionContext->setTensorAddress(
            multimodalEmbeddingsName.c_str(), const_cast<void*>(multimodalEmbeddings.rawPointer()));
        setEngineIOStatus &= mContextExecutionContext->setInputShape(
            multimodalEmbeddingsName.c_str(), multimodalEmbeddings.getShape().getTRTDims());
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
    // conduct preparation work for the engine execution.
    // TODO: Adjust the logic to pass in KVCache start indices prior to execution.
    mKVCache.commitDecodeRequest(stream);
    rt::Tensor& contextLengthDevice = mKVCache.getKVCacheLengths();
    // For vanllia decode stage, the selected token indices are always 0.
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    size_t const graphHash = hashDecodingInput(inputIds, outputLogits);
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
            contextLengthsName.c_str(), contextLengthDevice.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            contextLengthsName.c_str(), contextLengthDevice.getShape().getTRTDims());
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

    LOG_DEBUG("executeVanillaDecodingStep(): Decoding stage execution completed for request with batch size %d.",
        activeBatchSize);
    return true;
}

bool LLMEngineRunner::captureVanillaDecodingCudaGraph(
    rt::Tensor const& inputIds, rt::Tensor& outputLogits, cudaStream_t stream)
{
    size_t const hashValue = hashDecodingInput(inputIds, outputLogits);
    if (mCudaGraphs.find(hashValue) != mCudaGraphs.end())
    {
        LOG_INFO("captureVanillaDecodingCudaGraph(): CUDA graph already captured for the input tensors.");
        return true;
    }

    // To avoid CUDA graph error from TensorRT engine, we need to enqueueV3() once prior to graph capture.
    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> simulateCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const prefillLengthsTensor(
        simulateCacheLengths.data(), {activeBatchSize}, DeviceType::kCPU, DataType::kINT32);
    mKVCache.resetForNewSequences(activeBatchSize, stream);
    mKVCache.commitPrefillRequest(prefillLengthsTensor, stream);

    // Validate the condition here after the simulate prefill step.
    bool const validateInputStatus = this->vanlliaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("captureVanillaDecodingCudaGraph(): Generation request is invalid, unable to capture CUDA graph.");
        return false;
    }

    // Set shape of mSelectTokenIndices and set value to all zero..
    mSelectTokenIndices.reshape({activeBatchSize, 1});
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));

    // Set engine I/O using the same logic as executeVanillaDecodingStep().
    rt::Tensor& contextLengthDevice = mKVCache.getKVCacheLengths();
    bool setEngineIOStatus{true};
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
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());

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

    LOG_INFO("captureVanillaDecodingCudaGraph(): CUDA graph captured successfully for input shape %s.",
        inputIds.getShape().formatString().c_str());
    return true;
}

} // namespace rt
} // namespace drivellm
