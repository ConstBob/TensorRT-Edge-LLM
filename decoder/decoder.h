#pragma once
#ifndef DECODER_H
#define DECODER_H
#include <NvInferRuntime.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cfloat>
#include <cuda_runtime_api.h>
#include "common.h"
#include "sampler.h"


struct ModelConfig
// This is the model config inferred from optimization profiles
{
    int64_t batchSize;
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
    int64_t maxLength; // Equivalent to maxOutputLength;
    int64_t numLayers;
    int64_t vocabSize;
};


struct GenerationConfig
{
    int64_t maxLength; // Equivalent to maxNewTokens + Length of input
    int64_t minLength;
    float topP;
    int64_t topK;
};

class Decoder
{

public:
    Decoder():
        mStream{nullptr},
        mEngine{nullptr},
        mContextExecutionContext{nullptr},
        mGenerationExecutionContext{nullptr},
        isSetup{false},
        mConfig{0,0,0,0,0,0,0},
        mDeviceBuffer{},
        mSampler{nullptr}
    {}
    bool setup(std::filesystem::path& fp, cudaStream_t& stream);
    void generate(const std::vector<int64_t>& inputIds, std::vector<int64_t>& outputIds, GenerationConfig generationConfig);
    ~Decoder(){
        for (auto deviceMem: mDeviceBuffer){
            cudaFree(deviceMem.second);
        }
        mDeviceBuffer.clear();
        isSetup = false;
    };

private:
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    cudaStream_t mStream;
    bool isSetup;
    ModelConfig mConfig;
    std::map<std::string, void*> mDeviceBuffer;
    bool validateAndFillConfig();
    bool checkStaticShape(std::string& name);
    void allocateBuffer();
    Sampler<half>* mSampler;
    void printKVCache(int64_t contextLength);
};

#endif