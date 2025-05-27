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
#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <sstream>
#include <utility>

template <typename T>
bool Decoder<T>::setup(
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

template <typename T>
void Decoder<T>::setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs)
{
    for (size_t i = 0; i < extraInputs.size(); ++i)
    {
        char const* inputName = extraInputs[i].name.c_str();
        void* inputDevice = extraInputs[i].deviceBuffer;
        nvinfer1::Dims contextDims = extraInputs[i].contextDims;
        nvinfer1::Dims generationDims = extraInputs[i].generationDims;

        mContextExecutionContext->setTensorAddress(inputName, inputDevice);
        mGenerationExecutionContext->setTensorAddress(inputName, inputDevice);
        mContextExecutionContext->setInputShape(inputName, contextDims);
        mGenerationExecutionContext->setInputShape(inputName, generationDims);
    }
    // Need to reset cudaGraph because the setInputShape and setTensorAddress requires cudaGraph to be recaptured
    if (mUseCudaGraph)
    {
        mCudaGraphCaptured = false;
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
template <typename T>
bool Decoder<T>::checkStaticShape(std::string& name)
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

template <typename T>
bool Decoder<T>::validateAndFillConfig(int64_t batchSize)
{
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
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
    std::string inputIdsName = "input_ids";
    nvinfer1::Dims inputIdsShapeContext
        = mEngine->getProfileShape(inputIdsName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims inputIdsShapeContextMin
        = mEngine->getProfileShape(inputIdsName.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    int64_t minBatchSize = inputIdsShapeContextMin.d[0];
    int64_t maxBatchSize = inputIdsShapeContext.d[0];
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
    maxInputLength = inputIdsShapeContext.d[1];
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

    mConfig = {batchSize, numHead, hiddenSizePerHead, maxInputLength, maxLength, numLayers, vocabSize};
    mSampler = std::make_unique<Sampler<T>>(batchSize, vocabSize);

    return 0;
}

template <typename T>
void Decoder<T>::allocateBuffer()
{
    // Allocate buffers for inputs and logits, and set the shape
    void* contextLengthDevice;
    CUDA_CHECK(cudaMalloc(&contextLengthDevice, mConfig.batchSize * sizeof(int32_t)));
    mContextExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mContextExecutionContext->setInputShape("context_lengths", {1, {mConfig.batchSize}});
    mGenerationExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mGenerationExecutionContext->setInputShape("context_lengths", {1, {mConfig.batchSize}});
    mDeviceBuffer["context_lengths"] = contextLengthDevice;

    void* lastTokenIdsDevice;
    int32_t sizeOfHalf = 2;
    if (mIsEagle)
    {
        CUDA_CHECK(cudaMalloc(&lastTokenIdsDevice, mConfig.batchSize * mConfig.maxInputLength * sizeof(int64_t)));
        mContextExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
        mContextExecutionContext->setInputShape("last_token_ids", {1, {1}});
        mGenerationExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
        mContextExecutionContext->setInputShape("last_token_ids", {1, {1}});
        mDeviceBuffer["last_token_ids"] = lastTokenIdsDevice;
        mHostBuffer["last_token_ids"] = malloc(mConfig.batchSize * mConfig.maxInputLength * sizeof(int64_t));
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&lastTokenIdsDevice, mConfig.batchSize * 1 * sizeof(int64_t)));
        mContextExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
        mContextExecutionContext->setInputShape("last_token_ids", {2, {mConfig.batchSize, 1}});
        mGenerationExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
        mGenerationExecutionContext->setInputShape("last_token_ids", {2, {mConfig.batchSize, 1}});
        mDeviceBuffer["last_token_ids"] = lastTokenIdsDevice;
        mHostBuffer["last_token_ids"] = malloc(mConfig.batchSize * sizeof(int64_t));

        void* inputIdsDevice;
        CUDA_CHECK(cudaMalloc(&inputIdsDevice, (mConfig.batchSize * mConfig.maxLength) * sizeof(int64_t)));
        mDeviceBuffer["input_ids"] = inputIdsDevice;
        mContextExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
        mGenerationExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
        mContextExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, mConfig.maxInputLength}});
        mGenerationExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, 1}});

        void* logitsDevice;
        CUDA_CHECK(cudaMalloc(&logitsDevice, (mConfig.batchSize * mConfig.vocabSize) * sizeOfHalf));
        mDeviceBuffer["logits"] = logitsDevice;
        mContextExecutionContext->setTensorAddress("logits", logitsDevice);
        mGenerationExecutionContext->setTensorAddress("logits", logitsDevice);
    }

    mHostBuffer["finished_states"] = malloc(mConfig.batchSize * sizeof(bool));

    // Allocate buffers for kv cache and set the shape
    void* kvCacheDevice;
    CUDA_CHECK(cudaMalloc(&kvCacheDevice,
        (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead * mConfig.numLayers
            * sizeOfHalf)));
    CUDA_CHECK(cudaMemsetAsync(kvCacheDevice, 0,
        (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead * mConfig.numLayers
            * sizeOfHalf)));
    const size_t bytesPerLayer
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

    // Initialize dummy LoRA buffer
    void* dummyLoraBuffer;
    CUDA_CHECK(cudaMalloc(&dummyLoraBuffer, sizeof(T)));
    mDeviceBuffer["dummy_lora"] = dummyLoraBuffer;
}

template <typename T>
void Decoder<T>::addNewBuffer(std::string const& name, const nvinfer1::Dims dimsContext, int sizeOfByte)
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

template <typename T>
void* Decoder<T>::getDeviceBuffer(std::string const& name)
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
template <typename T>
std::string Decoder<T>::printKVCache()
{
    std::ostringstream oss;
    size_t totalKVSize = mConfig.batchSize * 2 * mConfig.hiddenSizePerHead * mConfig.numHead * mConfig.maxLength;
    printf(
        "totalKVSize: %d and mConfig.batchSize: %d, mConfig.numLayers: %d, mConfig.maxLength: %d, "
        "mConfig.hiddenSizePerHead: %d, mConfig.numHead: %d\n",
        totalKVSize, mConfig.batchSize, mConfig.numLayers, mConfig.maxLength, mConfig.hiddenSizePerHead,
        mConfig.numHead);
    std::vector<T> kvCache(totalKVSize, 0.0);
    oss << "Context Length is: " << mConfig.maxLength << std::endl;
    auto const sizeOfHalf = 2;
    const size_t bytesPerLayer = totalKVSize * sizeOfHalf;
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
template <typename T>
std::string Decoder<T>::printLogits()
{
    size_t totalLogitSize = mConfig.batchSize * 1 * mConfig.vocabSize;
    std::vector<T> logits(totalLogitSize, 0.0);
    CUDA_CHECK(
        cudaMemcpyAsync(logits.data(), mDeviceBuffer["logits"], totalLogitSize * sizeof(T), cudaMemcpyDeviceToHost));
    return formatFloat16Vector(logits, mConfig.batchSize);
}

template <typename T>
void Decoder<T>::generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
    std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds,
    std::shared_ptr<BenchmarkProfiler> const profiler)
{
    auto lastTokenIds = reinterpret_cast<int64_t*>(mHostBuffer["last_token_ids"]);
    // The generation step stop at longest sequence reach the maxLength.
    int32_t generationIter = *std::max_element(contextLengths.begin(), contextLengths.end());
    memset(mHostBuffer["finished_states"], 0, sizeof(bool) * mConfig.batchSize);
    auto finishedStates = reinterpret_cast<bool*>(mHostBuffer["finished_states"]);
    int64_t unfinishedBatchNum = mConfig.batchSize;

    for (int i = 0; i < mConfig.batchSize; i++)
    {
        assert(mConfig.maxInputLength >= contextLengths[i]);
        lastTokenIds[i] = contextLengths[i] - 1;
        outputIds[i].clear();
    }

    assert(outputIds.size() == static_cast<size_t>(mConfig.batchSize));
    assert(mConfig.maxLength >= generationConfig.maxLength);
    assert(generationConfig.maxLength >= generationConfig.minLength);

    auto sampleToken = [this, &outputIds, &generationIter, endIds, &generationConfig, &finishedStates, &contextLengths,
                           &unfinishedBatchNum]() {
        std::vector<int64_t> generatedToken = mSampler->greedySample(reinterpret_cast<T*>(mDeviceBuffer["logits"]));
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
    assert(contextLengths.size() == static_cast<size_t>(mConfig.batchSize)
        && "Input batch size does not match engine batch size.");

    // Setup "input_ids",  "context_lengths", "last_token_ids"
    // Extra model inputs should be set with `setupExtraInputs` before this function
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], lastTokenIds, mConfig.batchSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], inputIds.data(),
        mConfig.batchSize * mConfig.maxInputLength * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

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
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
            mGenerationExecutionContext->enqueueV3(mStream);
            CUDA_CHECK(cudaStreamEndCapture(mStream, &mGenerationGraph));
            CUDA_CHECK(cudaGraphInstantiate(&mGenerationGraphExec, mGenerationGraph, 0));
            mCudaGraphCaptured = true;
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("Cuda graph cannot be captured due to %s. Fall back to non cuda graph.", e.what());
            mUseCudaGraph = false;
            mCudaGraphCaptured = false;
        }
    }

    // Context Phase
    mContextExecutionContext->enqueueV3(mStream);
    auto generatedToken = sampleToken();
    LOG_DEBUG("Context phase logits:\n%s", printLogits().c_str());
    CUDA_CHECK(cudaMemset(mDeviceBuffer["last_token_ids"], 0, mConfig.batchSize * sizeof(int64_t)));

    if (profiler)
    {
        profiler->recordHostEnd("first token latency");
        profiler->recordDeviceStart("generation");
    }

    while (generationIter < generationConfig.maxLength && unfinishedBatchNum != 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
            mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], generatedToken.data(),
            mConfig.batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

        if (mUseCudaGraph && mCudaGraphCaptured)
        {
            CUDA_CHECK(cudaGraphLaunch(mGenerationGraphExec, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
        else
        {
            mGenerationExecutionContext->enqueueV3(mStream);
        }

        generatedToken = sampleToken();
        LOG_DEBUG("Generation phase logits:\n%s", printLogits().c_str());
    }

    if (profiler)
    {
        profiler->recordDeviceEnd("generation");
    }
}

template <typename T>
void Decoder<T>::generateForContext(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
    GenerationConfig generationConfig, std::vector<int64_t> const& last_token_ids, int64_t endIds, void* attentionMask,
    void* attentionPosId, const nvinfer1::Dims inputDims, const nvinfer1::Dims attentionMaskDims,
    const nvinfer1::Dims attentionPosIdDims, std::shared_ptr<BenchmarkProfiler> const profiler)
{
    assert(mConfig.maxLength >= generationConfig.maxLength);
    assert(generationConfig.maxLength >= generationConfig.minLength);
    assert(contextLengths.size() == mConfig.batchSize && "Input batch size does not match engine batch size.");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], inputIds.data(),
        mConfig.batchSize * contextLengths[0] * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], last_token_ids.data(),
        last_token_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    mContextExecutionContext->setInputShape("input_ids", inputDims);
    mContextExecutionContext->setTensorAddress("attention_mask", attentionMask);
    mContextExecutionContext->setInputShape("attention_mask", attentionMaskDims);
    mContextExecutionContext->setTensorAddress("attention_pos_id", attentionPosId);
    mContextExecutionContext->setInputShape("attention_pos_id", attentionPosIdDims);
    mContextExecutionContext->enqueueV3(mStream);
    LOG_DEBUG("Context phase logits:\n%s", printLogits().c_str());
}

template <typename T>
void Decoder<T>::generateForDecode(void* inputIds, std::vector<int32_t> ContextLengths,
    std::vector<int64_t>& last_token_ids, int32_t maxDecodingTokens, void* attentionMask, void* attentionPosId)
{

    mGenerationExecutionContext->setTensorAddress("input_ids", inputIds);
    const nvinfer1::Dims inputDims = {2, {mConfig.batchSize, maxDecodingTokens}};
    mGenerationExecutionContext->setInputShape("input_ids", inputDims);
    std::vector<int32_t> tempContextLengths = ContextLengths;
    for (size_t i = 0; i < tempContextLengths.size(); ++i)
    {
        tempContextLengths[i] += 1;
        ContextLengths[i] += maxDecodingTokens;
    }
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], ContextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], last_token_ids.data(),
        last_token_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    mGenerationExecutionContext->setTensorAddress("attention_mask", attentionMask);
    const nvinfer1::Dims attentionMaskDims = {3, {mConfig.batchSize, maxDecodingTokens, divUp(maxDecodingTokens, 32)}};
    mGenerationExecutionContext->setInputShape("attention_mask", attentionMaskDims);

    mGenerationExecutionContext->setTensorAddress("attention_pos_id", attentionPosId);
    const nvinfer1::Dims attentionPosIdDims = {2, {mConfig.batchSize, maxDecodingTokens}};
    mGenerationExecutionContext->setInputShape("attention_pos_id", attentionPosIdDims);

    mGenerationExecutionContext->setTensorAddress("last_token_ids", mDeviceBuffer["last_token_ids"]);
    const nvinfer1::Dims lastTokenIdsDims = {1, {maxDecodingTokens}};
    mGenerationExecutionContext->setInputShape("last_token_ids", lastTokenIdsDims);
    mGenerationExecutionContext->enqueueV3(mStream);
    // reset the context_lengths
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], tempContextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
}

template <typename T>
void Decoder<T>::generateForDraftContext(void* inputIds, void* hiddenStates, void* attentionMask, void* attentionPosId,
    std::vector<int32_t> contextLengths, std::vector<int64_t>& last_token_ids, bool isEagle3,
    void* hiddenStatesFromDraftZero, const nvinfer1::Dims inputDims, const nvinfer1::Dims hiddenStatesDims,
    const nvinfer1::Dims attentionMaskDims, const nvinfer1::Dims attentionPosIdDims)
{

    mContextExecutionContext->setTensorAddress("input_ids", inputIds);
    mContextExecutionContext->setInputShape("input_ids", inputDims);
    mContextExecutionContext->setTensorAddress("hidden_states_input", hiddenStates);
    mContextExecutionContext->setInputShape("hidden_states_input", hiddenStatesDims);
    if (isEagle3)
    {
        nvinfer1::Dims hsDimsFromDraft = hiddenStatesDims;
        hsDimsFromDraft.d[2] = hiddenStatesDims.d[2] / 3;
        mContextExecutionContext->setTensorAddress("hidden_states_from_draft", hiddenStatesFromDraftZero);
        mContextExecutionContext->setInputShape("hidden_states_from_draft", hsDimsFromDraft); // set 0
    }

    assert(contextLengths.size() == mConfig.batchSize && "Input batch size does not match engine batch size.");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));

    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], reinterpret_cast<void*>(last_token_ids.data()),
        last_token_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    // no need for context phase
    mContextExecutionContext->setTensorAddress("attention_mask", attentionMask);
    mContextExecutionContext->setInputShape("attention_mask", attentionMaskDims);
    mContextExecutionContext->setTensorAddress("attention_pos_id", attentionPosId);
    mContextExecutionContext->setInputShape("attention_pos_id", attentionPosIdDims);
    mContextExecutionContext->enqueueV3(mStream);

    auto logits_last_token_draft = getDeviceBuffer("logits");
    LOG_DEBUG("Context phase logits:\n%s", printLogits().c_str());
    LOG_DEBUG("Context phase kv cache:\n%s", printKVCache().c_str());
}

template <typename T>
void Decoder<T>::generateForDraftDecode(void* inputIds, void* hiddenStates, void* attention_mask,
    void* attention_pos_id, std::vector<int32_t> contextLengths, std::vector<int64_t>& last_token_ids,
    const nvinfer1::Dims inputDims, const nvinfer1::Dims hiddenStatesDims, const nvinfer1::Dims attentionMaskDims,
    const nvinfer1::Dims attentionPosIdDims, const nvinfer1::Dims lastTokenIdsDims, int layerIdx, bool isEagle3,
    void* hiddenStatesFromDraftZero, void* hiddenStatesFromTargetZero)
{

    nvinfer1::Dims hsDimsFromTarget = hiddenStatesDims;
    hsDimsFromTarget.d[2] = hiddenStatesDims.d[2] * 3;
    if (isEagle3)
    {

        if (layerIdx == 0)
        {

            mGenerationExecutionContext->setTensorAddress("hidden_states_input", hiddenStates);
            mGenerationExecutionContext->setInputShape("hidden_states_input", hsDimsFromTarget);
            mGenerationExecutionContext->setTensorAddress("hidden_states_from_draft", hiddenStatesFromDraftZero);
            mGenerationExecutionContext->setInputShape("hidden_states_from_draft", hiddenStatesDims);
        }
        else
        {
            mGenerationExecutionContext->setTensorAddress("hidden_states_input", hiddenStatesFromTargetZero);
            mGenerationExecutionContext->setInputShape("hidden_states_input", hsDimsFromTarget);
            mGenerationExecutionContext->setTensorAddress("hidden_states_from_draft", hiddenStates);
            mGenerationExecutionContext->setInputShape("hidden_states_from_draft", hiddenStatesDims);
        }
    }
    else
    {
        mGenerationExecutionContext->setTensorAddress("hidden_states_input", hiddenStates);
        mGenerationExecutionContext->setInputShape("hidden_states_input", hiddenStatesDims);
    }

    mGenerationExecutionContext->setTensorAddress("input_ids", inputIds);
    mGenerationExecutionContext->setInputShape("input_ids", inputDims);

    mGenerationExecutionContext->setTensorAddress("attention_mask", attention_mask);
    mGenerationExecutionContext->setInputShape("attention_mask", attentionMaskDims);
    mGenerationExecutionContext->setTensorAddress("attention_pos_id", attention_pos_id);
    mGenerationExecutionContext->setInputShape("attention_pos_id", attentionPosIdDims);

    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_lengths"], contextLengths.data(),
        mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["last_token_ids"], reinterpret_cast<void*>(last_token_ids.data()),
        last_token_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    mGenerationExecutionContext->setInputShape("last_token_ids", lastTokenIdsDims);
    mGenerationExecutionContext->enqueueV3(mStream);
}

template <typename T>
std::vector<T> const& Decoder<T>::getLastHostLogits()
{
    size_t totalLogitSize = mConfig.batchSize * 1 * mConfig.vocabSize;
    static std::vector<T> hostLogits(totalLogitSize);
    CUDA_CHECK(
        cudaMemcpy(hostLogits.data(), mDeviceBuffer["logits"], totalLogitSize * sizeof(T), cudaMemcpyDeviceToHost));
    return hostLogits;
}

template <typename T>
size_t Decoder<T>::getDeviceMemorySize() const noexcept
{
    return mEngine->getDeviceMemorySizeV2();
}

template <typename T>
int64_t Decoder<T>::getModelBatchSize() const noexcept
{
    return mConfig.batchSize;
}

template <typename T>
int64_t Decoder<T>::getMaxContextLength() const noexcept
{
    return mConfig.maxInputLength;
}

template <typename T>
const ModelConfig Decoder<T>::getModelConfig() const noexcept
{
    return mConfig;
}

template <typename T>
bool Decoder<T>::addLora(std::string const& name, std::string const& filePath)
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

template <typename T>
bool Decoder<T>::switchLora(std::string const& name)
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

template <typename T>
std::vector<std::string> Decoder<T>::getLoraNames() const
{
    std::vector<std::string> names = {"None"};
    for (auto const& [name, _] : mLoraWeights)
    {
        names.push_back(name);
    }
    return names;
}

template class Decoder<half>;