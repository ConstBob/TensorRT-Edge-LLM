#include "decoder.h"
#include "common.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <utility>
using namespace nvinfer1;
using namespace std;

bool Decoder::setup(std::filesystem::path const& fp, cudaStream_t& stream)
{
    try
    {
        mStream = stream;
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        StreamReader* _sr = new StreamReader(fp);
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(*_sr));
        mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        assert(mEngine->getNbOptimizationProfiles() == 2 && "The engine requires 2 optimization profiles");
        mContextExecutionContext->setOptimizationProfileAsync(0, mStream);
        mGenerationExecutionContext->setOptimizationProfileAsync(1, mStream);
        validateAndFillConfig();
        allocateBuffer();
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

// Helper function to check 2 dims are equal.
bool checkDimsEqual(Dims& A, Dims& B)
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
        Dims minShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kMIN);
        Dims optShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kOPT);
        Dims maxShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kMAX);
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

bool Decoder::validateAndFillConfig()
{
    int64_t batchSize;
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
    int64_t maxLength;
    int64_t nbIOs = static_cast<int64_t>(mEngine->getNbIOTensors());
    // input_ids, context_length and logits
    int64_t numLayers = (nbIOs - 3) / 2;
    // Check input_ids
    std::string inputIdsName = "input_ids";
    check(checkStaticShape(inputIdsName), fmtstr("%s should be static", inputIdsName.c_str()));
    Dims inputIdsShapeContext = mEngine->getProfileShape(inputIdsName.c_str(), 0, OptProfileSelector::kMIN);
    batchSize = inputIdsShapeContext.d[0];
    maxInputLength = inputIdsShapeContext.d[1];
    Dims inputIdsShapeGeneration = mEngine->getProfileShape(inputIdsName.c_str(), 1, OptProfileSelector::kMIN);
    assert(inputIdsShapeGeneration.d[0] == batchSize && inputIdsShapeGeneration.d[1] == 1);

    for (int32_t i = 0; i < numLayers; ++i)
    {
        std::string kvName = fmtstr("past_key_values.%d", i);
        check(checkStaticShape(kvName), fmtstr("%s should be static", kvName.c_str()));
        Dims kvShapeContext = mEngine->getProfileShape(kvName.c_str(), 0, OptProfileSelector::kMIN);
        Dims kvShapeGeneration = mEngine->getProfileShape(kvName.c_str(), 1, OptProfileSelector::kMIN);
        assert(
            kvShapeContext.nbDims == 5 && kvShapeGeneration.nbDims == 5 && "KV Cache should have [b, 2, h, s, d_kv]");
        if (i == 0)
        {
            assert(kvShapeContext.d[0] == batchSize);
            assert(kvShapeContext.d[1] == 2);
            numHead = kvShapeContext.d[2];
            assert(kvShapeContext.d[3] == 0);
            hiddenSizePerHead = kvShapeContext.d[4];
            maxLength = kvShapeGeneration.d[3];
        }
        else
        {
            assert((kvShapeContext.d[0] == batchSize) && (kvShapeContext.d[1] == 2) && (kvShapeContext.d[2] == numHead)
                && (kvShapeContext.d[3] == 0) && (kvShapeContext.d[4] == hiddenSizePerHead));
        }
        assert((kvShapeGeneration.d[0] == batchSize) && (kvShapeGeneration.d[1] == 2)
            && (kvShapeGeneration.d[2] == numHead) && (kvShapeGeneration.d[3] == (maxLength))
            && (kvShapeGeneration.d[4] == hiddenSizePerHead));
    }

    char const* logitsName = "logits";
    Dims logitsShape = mEngine->getTensorShape("logits");
    // Needs the vocab size
    int64_t vocabSize = logitsShape.d[1];

    mConfig = {batchSize, numHead, hiddenSizePerHead, maxInputLength, maxLength, numLayers, vocabSize};
    mSampler = new Sampler<half>(batchSize, vocabSize);

    return 0;
}

void Decoder::allocateBuffer()
{
    // Allocate buffers for inputs and logits, and set the shape
    void* contextLengthDevice;
    CUDA_CHECK(cudaMalloc(&contextLengthDevice, mConfig.batchSize * sizeof(int32_t)));
    mContextExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mGenerationExecutionContext->setTensorAddress("context_lengths", contextLengthDevice);
    mDeviceBuffer["context_lengths"] = contextLengthDevice;
    void* lastTokenIdsDevice;
    CUDA_CHECK(cudaMalloc(&lastTokenIdsDevice, mConfig.batchSize * 1 * sizeof(int64_t)));
    mContextExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
    mGenerationExecutionContext->setTensorAddress("last_token_ids", lastTokenIdsDevice);
    mDeviceBuffer["last_token_ids"] = lastTokenIdsDevice;
    void* inputIdsDevice;
    CUDA_CHECK(cudaMalloc(&inputIdsDevice, (mConfig.batchSize * mConfig.maxLength) * sizeof(int64_t)));
    mDeviceBuffer["input_ids"] = inputIdsDevice;
    mContextExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
    mGenerationExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
    mContextExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, mConfig.maxInputLength}});
    mGenerationExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, 1}});
    void* logitsDevice;
    int32_t sizeOfFloat = 2;
    CUDA_CHECK(cudaMalloc(&logitsDevice, (mConfig.batchSize * mConfig.vocabSize) * sizeOfFloat));
    mDeviceBuffer["logits"] = logitsDevice;
    mContextExecutionContext->setTensorAddress("logits", logitsDevice);
    mGenerationExecutionContext->setTensorAddress("logits", logitsDevice);
    // Allocate buffers for kv cache and set the shape
    for (int32_t i = 0; i < mConfig.numLayers; ++i)
    {
        void* kvCacheDevice;
        CUDA_CHECK(cudaMalloc(&kvCacheDevice,
            (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead) * sizeOfFloat));
        std::string pastKeyValuesName = fmtstr("past_key_values.%d", i);
        std::string presentKeyValuesName = fmtstr("present_key_values.%d", i);
        mDeviceBuffer[pastKeyValuesName] = kvCacheDevice;
        mContextExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheDevice);
        mContextExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheDevice);
        mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheDevice);
        mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheDevice);
        mContextExecutionContext->setInputShape(
            pastKeyValuesName.c_str(), {5, {mConfig.batchSize, 2, mConfig.numHead, 0, mConfig.hiddenSizePerHead}});
        mGenerationExecutionContext->setInputShape(pastKeyValuesName.c_str(),
            {5, {mConfig.batchSize, 2, mConfig.numHead, mConfig.maxLength, mConfig.hiddenSizePerHead}});
    }
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
        float sum = std::accumulate(beginIter, endIter, 0.0f, [](float sum, half val)
            {
                return sum + static_cast<float>(val);
            }
        );
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
        oss << "]" << endl;
    }
    return oss.str();
}

// This is a helper function to dump kv cache information
std::string Decoder::printKVCache(int64_t contextLength)
{
    ostringstream oss;
    size_t totalKVSize = mConfig.batchSize * 2 * mConfig.hiddenSizePerHead * mConfig.numHead * contextLength;
    std::vector<half> kvCache(totalKVSize, 0.0);
    oss << "Context Length is: " << contextLength << std::endl;
    for (int i = 0; i < mConfig.numLayers; ++i)
    {
        oss << "Layer = " << i << "\n";
        CUDA_CHECK(cudaMemcpyAsync(kvCache.data(), mDeviceBuffer[fmtstr("past_key_values.%d", i)],
            totalKVSize * sizeof(half), cudaMemcpyDeviceToHost));
        oss << formatFloat16Vector(kvCache, mConfig.batchSize);
    }
    return oss.str();
}

// This is a helper function to print logits
std::string Decoder::printLogits()
{
    size_t totalLogitSize = mConfig.batchSize * 1 * mConfig.vocabSize;
    std::vector<half> logits(totalLogitSize, 0.0);
    CUDA_CHECK(
        cudaMemcpyAsync(logits.data(), mDeviceBuffer["logits"], totalLogitSize * sizeof(half), cudaMemcpyDeviceToHost));
    return formatFloat16Vector(logits, mConfig.batchSize);
}

void Decoder::generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths, std::vector<int64_t> lastTokenIds,
    std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int32_t maxContextLength, int64_t endIds, 
    std::shared_ptr<BenchmarkProfiler> const profiler)
{
    assert(contextLengths.size() == mConfig.batchSize), "Input batch size does not match engine batch size.";
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceBuffer["context_lengths"], contextLengths.data(), mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceBuffer["last_token_ids"], lastTokenIds.data(), mConfig.batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceBuffer["input_ids"], inputIds.data(), mConfig.batchSize * maxContextLength * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    // Context Phase
    mContextExecutionContext->enqueueV3(mStream);
    LOG_DEBUG("Context phase logits:\n%s", printLogits().c_str());

    bool contextStep = true;

    // The generation step stop at longest sequence reach the maxLength.
    int32_t generationInter = *std::max_element(contextLengths.begin(), contextLengths.end());
    while ((generationInter < generationConfig.maxLength))
    {
        std::vector<int64_t> const& generatedToken
            = mSampler->greedySample(reinterpret_cast<half*>(mDeviceBuffer["logits"]));
        for (int i = 0; i < mConfig.batchSize; ++i)
        {
            outputIds[i].push_back(generatedToken[i]);
            ++contextLengths[i];
            lastTokenIds[i] = 0;
        }
        if (contextStep && profiler)
        {
            profiler->recordHostEnd("first token latency");
            profiler->recordDeviceStart("generation");
            contextStep = false;
        }
        ++generationInter;
        // Reaches eos token and reaches minLength.
        for (int i = 0; i < mConfig.batchSize; ++i)
        {
            if (generatedToken[i] == endIds && (contextLengths[i] > generationConfig.minLength))
            {
                if (profiler)
                {
                    profiler->recordDeviceEnd("generation");
                }
                return;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceBuffer["context_lengths"], contextLengths.data(), mConfig.batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceBuffer["last_token_ids"], lastTokenIds.data(), mConfig.batchSize * sizeof(int64_t), cudaMemcpyHostToHost, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceBuffer["input_ids"], generatedToken.data(), mConfig.batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

        mGenerationExecutionContext->enqueueV3(mStream);
        LOG_DEBUG("Generation phase logits:\n%s", printLogits().c_str());
    }

    if (profiler)
    {
        profiler->recordDeviceEnd("generation");
    }
}

size_t Decoder::getDeviceMemorySize() const noexcept
{
    return mEngine->getDeviceMemorySizeV2();
}

int64_t Decoder::getModelBatchSize() const noexcept
{
    return mConfig.batchSize;
}