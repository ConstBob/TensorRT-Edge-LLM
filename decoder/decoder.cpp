#include "decoder.h"
#include "common.h"
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cassert>
#include <utility>
#include <filesystem>
using namespace nvinfer1;
using namespace std;

bool Decoder::setup(std::filesystem::path& fp, cudaStream_t& stream){
    try{
        mStream = stream;
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*mLogger));
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
        mLogger->error(e.what());
        return false;
    }
    return true;
}

// Helper function to check 2 dims are equal.
bool checkDimsEqual(Dims& A, Dims& B){
    if (A.nbDims != B.nbDims){
        return false;
    }
    for (int32_t i = 0; i < A.nbDims; ++i){
        if (A.d[i] != B.d[i]){
            return false;
        }
    }
    return true;
}

// Helper function to check a certain input tensor has static shape
bool Decoder::checkStaticShape(std::string& name){
    for (int32_t i = 0; i < mEngine->getNbOptimizationProfiles(); ++i){
        Dims minShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kMIN);
        Dims optShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kOPT);
        Dims maxShape = mEngine->getProfileShape(name.c_str(), i, OptProfileSelector::kMAX);
        if (!checkDimsEqual(minShape, optShape)){
            return false;
        }
        if (!checkDimsEqual(optShape, maxShape)){
            return false;
        }
    }
    return true;
}

bool Decoder::validateAndFillConfig(){
    int64_t batchSize;
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
    int64_t maxLength;
    int64_t nbIOs = static_cast<int64_t>(mEngine-> getNbIOTensors());
    // input_ids, context_length and logits
    int64_t numLayers = (nbIOs - 3) / 2;
    // Check input_ids
    std::string inputIdsName = "input_ids";
    check(checkStaticShape(inputIdsName), fmtstr("%s should be static", inputIdsName));
    Dims inputIdsShapeContext = mEngine->getProfileShape(inputIdsName.c_str(), 0, OptProfileSelector::kMIN);
    batchSize = inputIdsShapeContext.d[0];
    maxInputLength = inputIdsShapeContext.d[1];
    Dims inputIdsShapeGeneration = mEngine->getProfileShape(inputIdsName.c_str(), 1, OptProfileSelector::kMIN);
    assert(inputIdsShapeGeneration.d[0] == batchSize && inputIdsShapeGeneration.d[1] == 1);

    for (int32_t i = 0; i < numLayers; ++i){
        std::string kvName = fmtstr("past_key_values.%d", i);
        check(checkStaticShape(kvName), fmtstr("%s should be static", kvName));
        Dims kvShapeContext = mEngine->getProfileShape(kvName.c_str(), 0, OptProfileSelector::kMIN);
        Dims kvShapeGeneration = mEngine->getProfileShape(kvName.c_str(), 1, OptProfileSelector::kMIN);
        assert(kvShapeContext.nbDims == 5 && kvShapeGeneration.nbDims == 5 && "KV Cache should have [b, 2, h, s, d_kv]");
        if (i == 0){
            assert(kvShapeContext.d[0] == batchSize);
            assert(kvShapeContext.d[1] == 2);
            numHead = kvShapeContext.d[2];
            assert(kvShapeContext.d[3] == 0);
            hiddenSizePerHead = kvShapeContext.d[4];
            maxLength = kvShapeGeneration.d[3];
        }
        else{
            assert((kvShapeContext.d[0] == batchSize) && (kvShapeContext.d[1] == 2) && (kvShapeContext.d[2] == numHead) && (kvShapeContext.d[3] == 0) && (kvShapeContext.d[4] == hiddenSizePerHead));
        }
        assert((kvShapeGeneration.d[0] == batchSize) && (kvShapeGeneration.d[1] == 2) && (kvShapeGeneration.d[2] == numHead) && (kvShapeGeneration.d[3] == (maxLength)) && (kvShapeGeneration.d[4] == hiddenSizePerHead));
    }

    char const* logitsName = "logits";
    Dims logitsShape = mEngine->getTensorShape("logits");
    // Needs the vocab size
    int64_t vocabSize = logitsShape.d[2];

    mConfig = {batchSize, numHead, hiddenSizePerHead, maxInputLength, maxLength, numLayers, vocabSize};
    mSampler = new Sampler<half>(batchSize, vocabSize);

    return 0;
}

void Decoder::allocateBuffer(){
    // Allocate buffers for inputs and logits, and set the shape
    void* contextLengthDevice;
    CUDA_CHECK(cudaMalloc(&contextLengthDevice, sizeof(int64_t)));
    mContextExecutionContext->setTensorAddress("context_length", contextLengthDevice);
    mGenerationExecutionContext->setTensorAddress("context_length", contextLengthDevice);
    mDeviceBuffer["context_length"] = contextLengthDevice;
    void* inputIdsDevice;
    CUDA_CHECK(cudaMalloc(&inputIdsDevice, (mConfig.batchSize * mConfig.maxLength) * sizeof(int64_t)));
    mDeviceBuffer["input_ids"] = inputIdsDevice;
    mContextExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
    mGenerationExecutionContext->setTensorAddress("input_ids", inputIdsDevice);
    mContextExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, mConfig.maxInputLength}});
    mGenerationExecutionContext->setInputShape("input_ids", {2, {mConfig.batchSize, 1}});
    void* logitsDevice;
    int32_t sizeOfFloat = 2;
    CUDA_CHECK(cudaMalloc(&logitsDevice, (mConfig.batchSize * 1 * mConfig.vocabSize) * sizeOfFloat));
    mDeviceBuffer["logits"] = logitsDevice;
    mContextExecutionContext->setTensorAddress("logits", logitsDevice);
    mGenerationExecutionContext->setTensorAddress("logits", logitsDevice);
    // Allocate buffers for kv cache and set the shape
    for (int32_t i = 0; i < mConfig.numLayers; ++i){
        void* kvCacheDevice;
        CUDA_CHECK(cudaMalloc(&kvCacheDevice, (mConfig.batchSize * 2 * mConfig.numHead * mConfig.maxLength * mConfig.hiddenSizePerHead) * sizeOfFloat));
        std::string pastKeyValuesName = fmtstr("past_key_values.%d", i);
        std::string presentKeyValuesName = fmtstr("present_key_values.%d", i);
        mDeviceBuffer[pastKeyValuesName] = kvCacheDevice;
        mContextExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheDevice);
        mContextExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheDevice);
        mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheDevice);
        mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheDevice);
        mContextExecutionContext->setInputShape(pastKeyValuesName.c_str(), {5, {mConfig.batchSize, 2, mConfig.numHead, 0, mConfig.hiddenSizePerHead}});
        mGenerationExecutionContext->setInputShape(pastKeyValuesName.c_str(), {5, {mConfig.batchSize, 2, mConfig.numHead, mConfig.maxLength, mConfig.hiddenSizePerHead}});
    }
}

void Decoder::generate(const std::vector<int64_t>& inputIds, std::vector<int64_t>& outputIds, GenerationConfig generationConfig){
    // We assume bs = 1 for this `generate` function for now. Copy input_ids and context_length
    assert(outputIds.size() == 0);
    int64_t contextLength = inputIds.size();
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_length"], &contextLength, sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], inputIds.data(), contextLength * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    // Context Phase
    mContextExecutionContext->enqueueV3(mStream);
    while ((contextLength < generationConfig.maxLength)){
        const std::vector<int64_t>& generatedToken = mSampler->greedySample(reinterpret_cast<half*>(mDeviceBuffer["logits"]));
        outputIds.push_back(generatedToken[0]);
        std::cout << "Generated token is " << generatedToken[0] << std::endl;
        ++contextLength;
        // Reaches eos token and reaches minLength.
        if ((generatedToken[0] == 128001) && (contextLength > generationConfig.minLength)){
            break;
        }
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["context_length"], &contextLength, sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["input_ids"], generatedToken.data(), 1 * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        mGenerationExecutionContext->enqueueV3(mStream);
    }
}
