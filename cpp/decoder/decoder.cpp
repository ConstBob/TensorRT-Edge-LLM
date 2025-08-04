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

#include "decoder.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "sampler/sampling.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <utility>

using Json = nlohmann::json;

bool Decoder::setup(
    std::filesystem::path const& fp, cudaStream_t& stream, bool useCudaGraph, int64_t batchSize, bool isEagle)
{
    try
    {
        mStream = stream;
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        char const* disableMmapLoad = std::getenv("DISABLE_MMAP_LOAD");
        if (disableMmapLoad != nullptr)
        {
            StreamReader _sr(fp);
            mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(_sr));
        }
        else
        {
            auto mmapReader = std::make_unique<MmapReader>(fp);
            if (mmapReader->getData() == nullptr)
            {
                LOG_ERROR("Failed to use MMap to read engine from file path: %s", fp.string().c_str());
                return false;
            }
            mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
                mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
        }

        mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        assert(mEngine->getNbOptimizationProfiles() == 2 && "The engine requires 2 optimization profiles");
        mContextExecutionContext->setOptimizationProfileAsync(0, mStream);
        mGenerationExecutionContext->setOptimizationProfileAsync(1, mStream);
        // before allocateBuffer()
        mIsEagle = isEagle;
        mEnginePath = fp;
        validateAndFillConfig(batchSize);
        allocateBuffer();
        mUseCudaGraph = useCudaGraph;
        mCudaGraphCaptured = false;
        // Set all LoRA weights to rank 0, if any
        switchLora("None");
        isSetup = true;
    }
    catch (std::exception const& e)
    {
        isSetup = false;
        LOG_ERROR(e.what());
        return false;
    }
    return true;
}

void Decoder::setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs)
{
    for (size_t i = 0; i < extraInputs.size(); ++i)
    {
        char const* inputName = extraInputs[i].name.c_str();
        void* inputDeviceForContext = extraInputs[i].deviceBufferForContext;
        void* inputDeviceForDecode = extraInputs[i].deviceBufferForDecode;
        nvinfer1::Dims contextDims = extraInputs[i].contextDims;
        nvinfer1::Dims generationDims = extraInputs[i].generationDims;
        if (inputDeviceForContext != nullptr)
        {
            mContextExecutionContext->setTensorAddress(inputName, inputDeviceForContext);
            mContextExecutionContext->setInputShape(inputName, contextDims);
        }
        if (inputDeviceForDecode != nullptr)
        {
            mGenerationExecutionContext->setTensorAddress(inputName, inputDeviceForDecode);
            mGenerationExecutionContext->setInputShape(inputName, generationDims);
            // Need to reset cudaGraph because the setInputShape and setTensorAddress requires cudaGraph to be
            // recaptured
            if (mUseCudaGraph)
            {
                mCudaGraphCaptured = false;
            }
        }
    }
}

void Decoder::setupRopeCosSin(std::string const& configPath)
{
    Json jsonConfig;

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("setupRopeCosSin(): Failed to open config file: %s", configPath.c_str());
        throw std::runtime_error("setupRopeCosSin(): Failed to open config file: " + configPath);
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("setupRopeCosSin(): Failed to parse config file with error: %s", e.what());
        throw std::runtime_error("setupRopeCosSin(): Failed to parse config file: " + configPath);
    }

    std::string ropeType = "default";
    float rotaryScale = 1.0f;
    float rotaryTheta = 100000.0f;
    int32_t maxPositionEmbeddings = 32768;

    if (jsonConfig.contains("rope_scaling"))
    {
        if (jsonConfig["rope_scaling"].contains("type"))
        {
            ropeType = jsonConfig["rope_scaling"]["type"].get<std::string>();
        }
    }

    if (jsonConfig.contains("rope_theta"))
    {
        rotaryTheta = jsonConfig["rope_theta"].get<float>();
    }

    if (jsonConfig.contains("max_position_embeddings"))
    {
        maxPositionEmbeddings = jsonConfig["max_position_embeddings"].get<int32_t>();
    }

    LOG_DEBUG("Setup Rope Cos Sin with type: %s, scale: %f, theta: %f, maxPositionEmbeddings: %d", ropeType.c_str(),
        rotaryScale, rotaryTheta, maxPositionEmbeddings);

    if (ropeType == "default")
    {
        if (mConfig.maxLength > maxPositionEmbeddings)
        {
            LOG_WARNING(
                "Context length is greater than maxPositionEmbeddings indicated by model config, this could cause "
                "generation results");
        }

        // Allocate buffer
        void* ropeRotaryCosSinDevice;
        CUDA_CHECK(cudaMalloc(&ropeRotaryCosSinDevice, mConfig.maxLength * mConfig.rotaryDim * sizeof(float)));
        mDeviceBuffer["rope_rotary_cos_sin"] = ropeRotaryCosSinDevice;

        // Setup extra inputs
        std::vector<EngineInputDesc> extraInputs;
        extraInputs.emplace_back(EngineInputDesc{"rope_rotary_cos_sin", mDeviceBuffer["rope_rotary_cos_sin"],
            mDeviceBuffer["rope_rotary_cos_sin"], {3, {1, mConfig.maxLength, mConfig.rotaryDim}},
            {3, {1, mConfig.maxLength, mConfig.rotaryDim}}});
        setupExtraInputs(extraInputs);

        // Initialize
        drivellm::kernel::initializeNormalRopeCosSin(reinterpret_cast<float*>(ropeRotaryCosSinDevice), rotaryTheta,
            rotaryScale, mConfig.rotaryDim, mConfig.maxLength, mStream);
    }
    else if (ropeType == "longrope")
    {
        // Allocate device buffer for short and long cos sin
        void* shortCosSinDevice;
        CUDA_CHECK(cudaMalloc(&shortCosSinDevice, mConfig.maxLength * mConfig.rotaryDim * sizeof(float)));
        mDeviceBuffer["short_cos_sin"] = shortCosSinDevice;
        void* longCosSinDevice;
        CUDA_CHECK(cudaMalloc(&longCosSinDevice, mConfig.maxLength * mConfig.rotaryDim * sizeof(float)));
        mDeviceBuffer["long_cos_sin"] = longCosSinDevice;

        // Note: Need to setupExtraInputs according to runtime context length before inference
        //     For context length > originalMaxPositionEmbeddings, use mDeviceBuffer["long_cos_sin"]
        //     For context length <= originalMaxPositionEmbeddings, use mDeviceBuffer["short_cos_sin"]

        // Helper function to read factor data from json
        auto readFactorData = [&](std::string const& factorName) -> float* {
            auto factorValue = jsonConfig["rope_scaling"][factorName];
            assert(factorValue.is_array() && factorValue.size() == mConfig.rotaryDim / 2
                && (std::string(factorName) + " size should be equal to rotaryDim / 2").c_str());

            std::vector<float> factor = factorValue.get<std::vector<float>>();

            float* factorDevice;
            CUDA_CHECK(cudaMalloc(&factorDevice, factor.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(factorDevice, factor.data(), factor.size() * sizeof(float), cudaMemcpyHostToDevice));

            return factorDevice;
        };

        auto shortFactorDevice = readFactorData("short_factor");
        auto longFactorDevice = readFactorData("long_factor");
        int32_t originalMaxPositionEmbeddings = jsonConfig["original_max_position_embeddings"].get<int32_t>();

        drivellm::kernel::initializeLongRopeCosSin(reinterpret_cast<float*>(shortCosSinDevice),
            reinterpret_cast<float*>(longCosSinDevice), shortFactorDevice, longFactorDevice, rotaryTheta,
            mConfig.rotaryDim, mConfig.maxLength, maxPositionEmbeddings, originalMaxPositionEmbeddings, mStream);

        // Free shortFactorDevice and longFactorDevice
        CUDA_CHECK(cudaFree(shortFactorDevice));
        CUDA_CHECK(cudaFree(longFactorDevice));
    }
    else
    {
        LOG_ERROR("Unsupported rope type: %s", ropeType.c_str());
        throw std::runtime_error(
            "setupRopeCosSin(): Unsupported rope type when initializing rope cos sin: " + ropeType);
    }
}

// Helper function to check 2 dims are equal.
bool checkDimsEqual(nvinfer1::Dims& A, nvinfer1::Dims& B)
{
    if (A.nbDims != B.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < A.nbDims; ++i)
    {
        if (A.d[i] != B.d[i])
        {
            return false;
        }
    }
    return true;
}

// Helper function to check a certain input tensor has static shape
bool Decoder::checkStaticShape(std::string& name)
{
    for (int32_t i = 0; i < mEngine->getNbOptimizationProfiles(); ++i)
    {
        nvinfer1::Dims minShape = mEngine->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims optShape = mEngine->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kOPT);
        nvinfer1::Dims maxShape = mEngine->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kMAX);
        if (!checkDimsEqual(minShape, optShape))
        {
            return false;
        }
        if (!checkDimsEqual(optShape, maxShape))
        {
            return false;
        }
    }
    return true;
}

bool Decoder::validateAndFillConfig(int64_t batchSize)
{
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t rotaryDim;
    int64_t minSupportedInputLength;
    int64_t maxSupportedInputLength;
    int64_t maxLength;
    int64_t nbIOs = static_cast<int64_t>(mEngine->getNbIOTensors());
    int64_t numLayers = 0;
    for (int32_t i = 0; i < nbIOs; ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorShape(name).nbDims == 5)
        {
            numLayers++;
        }
    }
    numLayers = numLayers / 2;
    std::string const inputIdsName = "input_ids";
    nvinfer1::Dims inputIdsShapeContextMax
        = mEngine->getProfileShape(inputIdsName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims inputIdsShapeContextMin
        = mEngine->getProfileShape(inputIdsName.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    int64_t minBatchSize = inputIdsShapeContextMin.d[0];
    int64_t maxBatchSize = inputIdsShapeContextMax.d[0];
    // If min = max, this is a static shape engine.
    if (minBatchSize == maxBatchSize)
    {
        batchSize = minBatchSize;
    }
    else
    {
        // This is a dynamic batch engine, but we need to check if the provided batchSize is between min and max
        assert(batchSize >= minBatchSize && batchSize <= maxBatchSize);
    }
    minSupportedInputLength = inputIdsShapeContextMin.d[1];
    maxSupportedInputLength = inputIdsShapeContextMax.d[1];
    if (!mIsEagle)
    {
        nvinfer1::Dims inputIdsShapeGeneration
            = mEngine->getProfileShape(inputIdsName.c_str(), 1, nvinfer1::OptProfileSelector::kMAX);
        assert(inputIdsShapeGeneration.d[1] == 1);
    }

    for (int32_t i = 0; i < numLayers; ++i)
    {
        std::string kvName = fmtstr("past_key_values.%d", i);
        nvinfer1::Dims kvShapeContext = mEngine->getProfileShape(kvName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims kvShapeGeneration
            = mEngine->getProfileShape(kvName.c_str(), 1, nvinfer1::OptProfileSelector::kMAX);
        assert(
            kvShapeContext.nbDims == 5 && kvShapeGeneration.nbDims == 5 && "KV Cache should have [b, 2, h, s, d_kv]");
        if (i == 0)
        {
            assert(kvShapeContext.d[1] == 2);
            numHead = kvShapeContext.d[2];
            assert(kvShapeContext.d[3] == 0);
            hiddenSizePerHead = kvShapeContext.d[4];
            maxLength = kvShapeGeneration.d[3];
        }
        else
        {
            assert((kvShapeContext.d[1] == 2) && (kvShapeContext.d[2] == numHead) && (kvShapeContext.d[3] == 0)
                && (kvShapeContext.d[4] == hiddenSizePerHead));
        }
        assert((kvShapeGeneration.d[1] == 2) && (kvShapeGeneration.d[2] == numHead)
            && (kvShapeGeneration.d[3] == (maxLength)) && (kvShapeGeneration.d[4] == hiddenSizePerHead));
    }

    nvinfer1::Dims logitsShape = mEngine->getTensorShape("logits");
    // Needs the vocab size
    int64_t vocabSize = logitsShape.d[1];

    nvinfer1::Dims rotaryDimShape = mEngine->getTensorShape("rope_rotary_cos_sin");
    rotaryDim = rotaryDimShape.d[2];

    mConfig = {batchSize, numHead, hiddenSizePerHead, rotaryDim, minSupportedInputLength, maxSupportedInputLength,
        maxLength, numLayers, vocabSize};

    return 0;
}

void Decoder::allocateBufferForKVCache()
{
    int32_t sizeOfHalf = 2;
    // Allocate buffers for kv cache and set the shape
    void* kvCacheDevice;
    CUDA_CHECK(cudaMalloc(&kvCacheDevice,
        (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead * mConfig.numLayers
            * sizeOfHalf)));
    CUDA_CHECK(cudaMemsetAsync(kvCacheDevice, 0,
        (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead * mConfig.numLayers
            * sizeOfHalf)));
    size_t const bytesPerLayer
        = mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead * sizeOfHalf;
    mDeviceBuffer["kv_cache"] = kvCacheDevice;
    for (int32_t i = 0; i < mConfig.numLayers; ++i)
    {
        std::string pastKeyValuesName = fmtstr("past_key_values.%d", i);
        std::string presentKeyValuesName = fmtstr("present_key_values.%d", i);
        void* curLayerKVCacheAddr = static_cast<uint8_t*>(kvCacheDevice) + bytesPerLayer * i;
        mContextExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), curLayerKVCacheAddr);
        mContextExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), curLayerKVCacheAddr);
        mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), curLayerKVCacheAddr);
        mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), curLayerKVCacheAddr);
        mContextExecutionContext->setInputShape(
            pastKeyValuesName.c_str(), {5, {mConfig.batchSize, 2, mConfig.numHead, 0, mConfig.hiddenSizePerHead}});
        mGenerationExecutionContext->setInputShape(pastKeyValuesName.c_str(),
            {5, {mConfig.batchSize, 2, mConfig.numHead, mConfig.maxLength, mConfig.hiddenSizePerHead}});
    }
}

void Decoder::allocateCommonBuffers()
{
    allocateBufferForKVCache();
    void* contextLengthDevice;
    CUDA_CHECK(cudaMalloc(&contextLengthDevice, mConfig.batchSize * sizeof(int32_t)));
    mContextExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mContextExecutionContext->setInputShape("context_lengths", {1, {mConfig.batchSize}});
    mGenerationExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mGenerationExecutionContext->setInputShape("context_lengths", {1, {mConfig.batchSize}});
    mDeviceBuffer["context_lengths"] = contextLengthDevice;

    // Allocate buffer for selected indices (used by sampling kernels)
    void* selectedIndicesDevice;
    CUDA_CHECK(cudaMalloc(&selectedIndicesDevice, mConfig.batchSize * sizeof(int64_t)));
    mDeviceBuffer["selected_indices"] = selectedIndicesDevice;

    // Initialize dummy LoRA buffer
    void* dummyLoraBuffer;
    CUDA_CHECK(cudaMalloc(&dummyLoraBuffer, sizeof(LoraWeightType)));
    mDeviceBuffer["dummy_lora"] = dummyLoraBuffer;

    void* samplingWorkspaceBuffer;
    drivellm::SamplingParams samplingParams(mConfig.batchSize, mConfig.vocabSize, 1.0f, 1);
    size_t workspaceSize
        = drivellm::getTopKtopPSamplingWorkspaceSize<LogitsType>(mConfig.batchSize, mConfig.vocabSize, samplingParams);
    CUDA_CHECK(cudaMalloc(&samplingWorkspaceBuffer, workspaceSize));
    mDeviceBuffer["samplingWorkspace"] = samplingWorkspaceBuffer;
}

void Decoder::allocateExtraBufferForEagle()
{
    void* lastTokenIdsDevice;
    CUDA_CHECK(cudaMalloc(&lastTokenIdsDevice, mConfig.batchSize * mConfig.maxSupportedInputLength * sizeof(int64_t)));
    mContextExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
    mContextExecutionContext->setInputShape("last_token_ids", {1, {1}});
    mDeviceBuffer["last_token_ids"] = lastTokenIdsDevice;
}

void Decoder::allocateExtraBufferForVanilla()
{
    int32_t sizeOfHalf = 2;
    void* lastTokenIdsDevice;
    CUDA_CHECK(cudaMalloc(&lastTokenIdsDevice, mConfig.batchSize * 1 * sizeof(int64_t)));
    mContextExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
    mContextExecutionContext->setInputShape("last_token_ids", {2, {mConfig.batchSize, 1}});
    mGenerationExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
    mGenerationExecutionContext->setInputShape("last_token_ids", {2, {mConfig.batchSize, 1}});
    mDeviceBuffer["last_token_ids"] = lastTokenIdsDevice;

    void* inputIdsDevice;
    CUDA_CHECK(cudaMalloc(&inputIdsDevice, (mConfig.batchSize * mConfig.maxLength) * sizeof(int64_t)));
    mDeviceBuffer["input_ids"] = inputIdsDevice;
    mGenerationExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
    mGenerationExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, 1}});

    void* logitsDevice;
    CUDA_CHECK(cudaMalloc(&logitsDevice, (mConfig.batchSize * mConfig.vocabSize) * sizeOfHalf));
    mDeviceBuffer["logits"] = logitsDevice;
    mContextExecutionContext->setTensorAddress("logits", logitsDevice);
    mGenerationExecutionContext->setTensorAddress("logits", logitsDevice);

    mHostBuffer["finished_states"] = malloc(mConfig.batchSize * sizeof(bool));
}

void Decoder::allocateBuffer()
{
    allocateCommonBuffers();
    if (mIsEagle)
    {
        allocateExtraBufferForEagle();
    }
    else
    {
        allocateExtraBufferForVanilla();
    }
}

void Decoder::addNewBuffer(std::string const& name, nvinfer1::Dims const dimsContext, int sizeOfByte)
{

    void* devicePtr;
    if (name == "input_ids")
    {
        // allocate 1 more for eagle
        CUDA_CHECK(cudaMalloc(&devicePtr, (mConfig.maxLength + 1) * mConfig.batchSize * sizeOfByte));
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&devicePtr, volume(dimsContext) * sizeOfByte));
    }
    mDeviceBuffer[name] = devicePtr;
    mContextExecutionContext->setTensorAddress(name.c_str(), devicePtr);
    mGenerationExecutionContext->setTensorAddress(name.c_str(), devicePtr);
    return;
}

void* Decoder::getDeviceBuffer(std::string const& name)
{
    auto it = mDeviceBuffer.find(name);
    if (it != mDeviceBuffer.end())
    {
        return it->second;
    }
    else
    {
        assert(false && "The tensor name provided for device buffer doesn't find.");
    }
    return nullptr;
}

std::string formatFloat16Vector(std::vector<half> const& vec, int64_t batchSize)
{
    size_t batchVecSize = vec.size() / batchSize;
    std::ostringstream oss;
    for (int i = 0; i < batchSize; ++i)
    {
        oss << "Batch " << i << ":\n";
        auto beginIter = vec.begin() + i * batchVecSize;
        auto endIter = vec.begin() + (i + 1) * batchVecSize;

        // Find the maximum value
        auto maxElementIter = std::max_element(beginIter, endIter);
        // Find the minimum value
        auto minElementIter = std::min_element(beginIter, endIter);

        // Calculate the average
        float sum = std::accumulate(
            beginIter, endIter, 0.0f, [](float sum, half val) { return sum + static_cast<float>(val); });
        float average = sum / batchVecSize;

        oss << "Maximum: " << static_cast<float>(*maxElementIter) << " at " << std::distance(beginIter, maxElementIter)
            << ". ";
        oss << "Minimum: " << static_cast<float>(*minElementIter) << " at " << std::distance(beginIter, minElementIter)
            << ". ";
        oss << " Average: " << average << ". ";

        oss << "First 10 elements: [";
        for (size_t j = 0; j < 10; ++j)
        {
            oss << static_cast<float>(*(beginIter + j));
            if (j != 9)
            {
                oss << ",";
            }
        }
        oss << "]" << std::endl;
    }
    return oss.str();
}

// This is a helper function to dump kv cache information
std::string Decoder::printKVCache()
{
    std::ostringstream oss;
    size_t totalKVSize = mConfig.batchSize * 2 * mConfig.hiddenSizePerHead * mConfig.numHead * mConfig.maxLength;
    printf(
        "totalKVSize: %d and mConfig.batchSize: %d, mConfig.numLayers: %d, mConfig.maxLength: %d, "
        "mConfig.hiddenSizePerHead: %d, mConfig.numHead: %d\n",
        totalKVSize, mConfig.batchSize, mConfig.numLayers, mConfig.maxLength, mConfig.hiddenSizePerHead,
        mConfig.numHead);
    std::vector<KVCacheType> kvCache(totalKVSize, 0.0);
    oss << "Context Length is: " << mConfig.maxLength << std::endl;
    auto const sizeOfHalf = 2;
    size_t const bytesPerLayer = totalKVSize * sizeOfHalf;
    for (int i = 0; i < mConfig.numLayers; ++i)
    {
        oss << "Layer = " << i << "\n";
        auto curLayerKVCacheAddr = static_cast<uint8_t*>(mDeviceBuffer["kv_cache"]) + bytesPerLayer * i;
        CUDA_CHECK(cudaMemcpyAsync(kvCache.data(), curLayerKVCacheAddr, bytesPerLayer, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        oss << formatFloat16Vector(kvCache, mConfig.batchSize);
    }
    return oss.str();
}

// This is a helper function to print logits
std::string Decoder::printLogits()
{
    size_t totalLogitSize = mConfig.batchSize * 1 * mConfig.vocabSize;
    std::vector<LogitsType> logits(totalLogitSize, 0.0);
    CUDA_CHECK(cudaMemcpyAsync(
        logits.data(), mDeviceBuffer["logits"], totalLogitSize * sizeof(LogitsType), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    return formatFloat16Vector(logits, mConfig.batchSize);
}

void Decoder::initDecodingPhaseCudaGraph(std::vector<int32_t> const& contextLengths)
{
    // Capture cuda graph only on the first run. This cuda graph will be cached and reused for all the other runs.
    if (mUseCudaGraph && !mCudaGraphCaptured)
    {
        try
        {
            // Destroy the existing graph and execution context if they exist
            if (mGenerationGraph)
            {
                CUDA_CHECK(cudaGraphDestroy(mGenerationGraph));
                mGenerationGraph = nullptr;
            }
            if (mGenerationGraphExec)
            {
                CUDA_CHECK(cudaGraphExecDestroy(mGenerationGraphExec));
                mGenerationGraphExec = nullptr;
            }
            // Set up inputs to valid values to comply with decoding phase enqueueV3() call.
            // This won't have side effect for ongoing request.
            CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
                mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
            CUDA_CHECK(
                cudaMemsetAsync(mDeviceBuffer["last_token_ids"], 0, mConfig.batchSize * sizeof(int64_t), mStream));
            // Call enqueueV3() once prior to cudaGraph capture to execute TRT dynamic shape machine.
            // TRT shape machine could invoke host memory operation on Thor that invalidate cudaGraph capture.
            mGenerationExecutionContext->enqueueV3(mStream);
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
            mGenerationExecutionContext->enqueueV3(mStream);
            CUDA_CHECK(cudaStreamEndCapture(mStream, &mGenerationGraph));
            CUDA_CHECK(cudaGraphInstantiate(&mGenerationGraphExec, mGenerationGraph, 0));
            mCudaGraphCaptured = true;
            LOG_INFO("Cuda graph captured successfully.");
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("Cuda graph cannot be captured due to %s. Fall back to non cuda graph.", e.what());
            mUseCudaGraph = false;
            mCudaGraphCaptured = false;
        }
    }
}

void Decoder::generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
    std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds,
    std::shared_ptr<BenchmarkProfiler> const profiler)
{
    // Initialize decoding phase cuda graph
    initDecodingPhaseCudaGraph(contextLengths);

    int32_t maxInputContextLength = *std::max_element(contextLengths.begin(), contextLengths.end());
    // if Enable dynamic shape, the input contexts are padded to max input lengths within this batch,
    // Otherwise the input contexts are padded to maxSupportedInputLength.
    bool const engineSupportDynamicShape = mConfig.minSupportedInputLength != mConfig.maxSupportedInputLength;
    int32_t const contextLenStride
        = engineSupportDynamicShape ? maxInputContextLength : mConfig.maxSupportedInputLength;
    if (inputIds.size() != contextLenStride * mConfig.batchSize)
    {
        throw std::runtime_error("InputIds for generation are not padded correctly.");
    }

    // The generation step stop at longest sequence reach the maxLength.
    int32_t generationIter = maxInputContextLength;
    memset(mHostBuffer["finished_states"], 0, sizeof(bool) * mConfig.batchSize);
    auto finishedStates = reinterpret_cast<bool*>(mHostBuffer["finished_states"]);
    int64_t unfinishedBatchNum = mConfig.batchSize;

    std::vector<int64_t> lastTokenIds(mConfig.batchSize);
    for (int i = 0; i < mConfig.batchSize; i++)
    {
        assert(mConfig.maxSupportedInputLength >= contextLengths[i]);
        lastTokenIds[i] = contextLengths[i] - 1;
        outputIds[i].clear();
    }

    assert(outputIds.size() == static_cast<size_t>(mConfig.batchSize));
    assert(mConfig.maxLength >= generationConfig.maxLength);
    assert(generationConfig.maxLength >= generationConfig.minLength);

    auto sampleToken = [this, &outputIds, &generationIter, endIds, &generationConfig, &finishedStates, &contextLengths,
                           &unfinishedBatchNum]() {
        // Get device memory for selected indices
        int64_t* deviceSelectedIndices = reinterpret_cast<int64_t*>(mDeviceBuffer["selected_indices"]);

        // Use greedy sampling (top_k=1, temperature=1.0f)
        // TODO: add temperature, top_k and top_p sampling
        drivellm::SamplingParams params(mConfig.batchSize, mConfig.vocabSize, 1.0f, 1, 1.0f);

        drivellm::topKtopPSamplingFromLogits<LogitsType>(
            reinterpret_cast<LogitsType const*>(mDeviceBuffer["logits"]), // logits
            deviceSelectedIndices,                                        // selected_indices
            params,                                                       // params
            mDeviceBuffer["samplingWorkspace"],                           // workspace
            drivellm::getTopKtopPSamplingWorkspaceSize<LogitsType>(
                mConfig.batchSize, mConfig.vocabSize, params), // workspaceSize
            mStream                                            // stream
        );

        // Copy results back to host directly as int64_t
        std::vector<int64_t> generatedToken(mConfig.batchSize);

        CUDA_CHECK(cudaMemcpyAsync(generatedToken.data(), deviceSelectedIndices, mConfig.batchSize * sizeof(int64_t),
            cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        ++generationIter;
        for (int i = 0; i < mConfig.batchSize; i++)
        {
            if (!finishedStates[i])
            {
                outputIds[i].push_back(generatedToken[i]);
                ++contextLengths[i];
                // Reaches eos token and reaches minLength.
                finishedStates[i] = generatedToken[i] == endIds && generationIter > generationConfig.minLength;
                if (finishedStates[i])
                {
                    unfinishedBatchNum--;
                }
            }
        }
        return generatedToken;
    };

    // Context phase
    // Extra model inputs should be set with `setupExtraInputs` before this function
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], inputIds.data(), inputIds.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice, mStream));

    generateForContext(
        mDeviceBuffer["input_ids"], contextLengths, lastTokenIds, {2, {mConfig.batchSize, contextLenStride}});

    auto generatedToken = sampleToken();
    std::fill(lastTokenIds.begin(), lastTokenIds.end(), 0);

    if (profiler)
    {
        profiler->recordHostEnd("first token latency");
        profiler->recordDeviceStart("generation");
    }

    // Generation phase
    while (generationIter < generationConfig.maxLength && unfinishedBatchNum != 0)
    {

        CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], generatedToken.data(),
            mConfig.batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

        generateForDecode(contextLengths, lastTokenIds);

        generatedToken = sampleToken();
    }

    if (profiler)
    {
        profiler->recordDeviceEnd("generation");
    }
}

void Decoder::generateForContext(void* inputIds, std::vector<int32_t>& contextLengths,
    std::vector<int64_t> const& lastTokenIds, nvinfer1::Dims const inputDims)
{
    // check input batch size
    assert(contextLengths.size() == mConfig.batchSize && "Input batch size does not match engine batch size.");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], lastTokenIds.data(),
        lastTokenIds.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    mContextExecutionContext->setTensorAddress("input_ids", inputIds);
    mContextExecutionContext->setInputShape("input_ids", inputDims);

    mContextExecutionContext->enqueueV3(mStream);

    LOG_DEBUG("Invoke context phase enqueue with inputDims: [%d, %d]", inputDims.d[0], inputDims.d[1]);
}

void Decoder::generateForDecode(std::vector<int32_t>& contextLengths, std::vector<int64_t>& lastTokenIds)
{

    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], lastTokenIds.data(),
        lastTokenIds.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    if (mUseCudaGraph && mCudaGraphCaptured)
    {
        CUDA_CHECK(cudaGraphLaunch(mGenerationGraphExec, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }
    else
    {
        mGenerationExecutionContext->enqueueV3(mStream);
    }
}

void Decoder::getLastHostLogits(std::vector<LogitsType>& hostLogits)
{
    size_t totalLogitSize = mConfig.batchSize * 1 * mConfig.vocabSize;
    hostLogits.resize(totalLogitSize);
    CUDA_CHECK(cudaMemcpy(
        hostLogits.data(), mDeviceBuffer["logits"], totalLogitSize * sizeof(LogitsType), cudaMemcpyDeviceToHost));
    return;
}

size_t Decoder::getDeviceMemorySize() const noexcept
{
    return mEngine->getDeviceMemorySizeV2();
}

int64_t Decoder::getModelBatchSize() const noexcept
{
    return mConfig.batchSize;
}

int64_t Decoder::getMinSupportedInputLength() const noexcept
{
    return mConfig.minSupportedInputLength;
}

int64_t Decoder::getMaxSupportedInputLength() const noexcept
{
    return mConfig.maxSupportedInputLength;
}

ModelConfig const Decoder::getModelConfig() const noexcept
{
    return mConfig;
}

bool Decoder::addLora(std::string const& name, std::string const& filePath)
{
    if (name == "None")
    {
        LOG_WARNING("'None' is reserved for no LoRA weights.");
        return false;
    }

    try
    {
        auto loader = std::make_unique<drivellm::SafeTensorsLoader>(filePath);

        // Load all tensors to GPU
        if (!loader->loadFromFileToGPU())
        {
            LOG_WARNING("Failed to load LoRA weights to GPU from: %s", filePath.c_str());
            return false;
        }

        // Store the loader in our map
        mLoraWeights[name] = std::move(loader);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("Failed to add LoRA weights: %s", e.what());
        return false;
    }
}

bool Decoder::switchLora(std::string const& name)
{
    // Get the number of bindings in the engine
    int32_t numBindings = mEngine->getNbIOTensors();

    // If name is "None", set all LoRA weights to rank 0
    if (name == "None")
    {
        for (int32_t i = 0; i < numBindings; ++i)
        {
            char const* bindingName = mEngine->getIOTensorName(i);
            std::string bindingNameStr(bindingName);
            if (bindingNameStr.find("lora_") != std::string::npos)
            {
                // Get the shape from profile
                nvinfer1::Dims shape = mEngine->getProfileShape(bindingName, 0, nvinfer1::OptProfileSelector::kMAX);
                // Set rank dimension to 0 (second dim for A, first dim for B)
                if (bindingNameStr.find("lora_A") != std::string::npos)
                {
                    shape.d[1] = 0; // [gemm_k, rank]
                }
                else if (bindingNameStr.find("lora_B") != std::string::npos)
                {
                    shape.d[0] = 0; // [rank, gemm_n]
                }

                // Set shape and address for both contexts
                mContextExecutionContext->setInputShape(bindingName, shape);
                mGenerationExecutionContext->setInputShape(bindingName, shape);
                mContextExecutionContext->setTensorAddress(bindingName, mDeviceBuffer["dummy_lora"]);
                mGenerationExecutionContext->setTensorAddress(bindingName, mDeviceBuffer["dummy_lora"]);
            }
        }
        return true;
    }

    // Check if the requested LoRA exists
    auto it = mLoraWeights.find(name);
    if (it == mLoraWeights.end())
    {
        LOG_WARNING("LoRA weights with name '%s' not found", name.c_str());
        return false;
    }

    auto& loraLoader = it->second;
    auto const& tensorInfo = loraLoader->getSafeTensorsInfo();

    // Iterate through all bindings
    for (int32_t i = 0; i < numBindings; ++i)
    {
        char const* bindingName = mEngine->getIOTensorName(i);
        std::string bindingNameStr(bindingName);
        if (bindingNameStr.find("lora_") != std::string::npos)
        {
            // Get the shape from profile
            nvinfer1::Dims shape = mEngine->getProfileShape(bindingName, 0, nvinfer1::OptProfileSelector::kMAX);

            // Try to find the tensor in the LoRA weights
            auto tensorIt = tensorInfo.find(bindingName);
            if (tensorIt != tensorInfo.end() && tensorIt->second.gpuPtr != nullptr)
            {
                // Found matching tensor, use its data
                mContextExecutionContext->setInputShape(bindingName, shape);
                mGenerationExecutionContext->setInputShape(bindingName, shape);
                mContextExecutionContext->setTensorAddress(bindingName, tensorIt->second.gpuPtr);
                mGenerationExecutionContext->setTensorAddress(bindingName, tensorIt->second.gpuPtr);
            }
            else
            {
                // No matching tensor found, set rank to 0
                if (bindingNameStr.find("lora_A") != std::string::npos)
                {
                    shape.d[1] = 0; // [gemm_k, rank]
                }
                else if (bindingNameStr.find("lora_B") != std::string::npos)
                {
                    shape.d[0] = 0; // [rank, gemm_n]
                }
                mContextExecutionContext->setInputShape(bindingName, shape);
                mGenerationExecutionContext->setInputShape(bindingName, shape);
                mContextExecutionContext->setTensorAddress(bindingName, mDeviceBuffer["dummy_lora"]);
                mGenerationExecutionContext->setTensorAddress(bindingName, mDeviceBuffer["dummy_lora"]);
            }
        }
    }

    // Need to reset cudaGraph because the setInputShape and setTensorAddress requires cudaGraph to be recaptured
    if (mUseCudaGraph)
    {
        mCudaGraphCaptured = false;
    }
    return true;
}

std::vector<std::string> Decoder::getLoraNames() const
{
    std::vector<std::string> names = {"None"};
    for (auto const& [name, _] : mLoraWeights)
    {
        names.push_back(name);
    }
    return names;
}