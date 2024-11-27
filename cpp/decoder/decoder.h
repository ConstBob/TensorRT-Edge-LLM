#pragma once
#ifndef DECODER_H
#define DECODER_H
#include "common/benchmarkProfiler.h"
#include "common/common.h"
#include "sampler/include/sampler.h"
#include <NvInferRuntime.h>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    int32_t max_position_embeddings = 32768;
    int32_t qwen2vl_hiddendims = 3584;
};

struct GenerationConfig
{
    int64_t maxLength; // Equivalent to maxNewTokens + Length of input
    int64_t minLength;
    float topP;
    int64_t topK;
};

template <typename T>
class Decoder
{
public:
    Decoder()
        : mStream{nullptr}
        , mEngine{nullptr}
        , mContextExecutionContext{nullptr}
        , mGenerationExecutionContext{nullptr}
        , isSetup{false}
        , mConfig{0, 0, 0, 0, 0, 0, 0}
        , mDeviceBuffer{}
        , mSampler{nullptr}
    {
    }
    bool setup(std::filesystem::path const& fp, cudaStream_t& stream);
    void generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
        std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds = -1,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr,
        std::optional<TensorInfo> const& image_embeds = std::nullopt,
        std::optional<TensorInfo> const& mropeRotaryCosSin = std::nullopt,
        std::optional<TensorInfo> const& mropePositionDeltas = std::nullopt);

    std::vector<T> const& getLastHostLogits();
    size_t getDeviceMemorySize() const noexcept;
    int64_t getModelBatchSize() const noexcept;
    int64_t getMaxContextLength() const noexcept;
    ~Decoder()
    {
        for (auto deviceMem : mDeviceBuffer)
        {
            cudaFree(deviceMem.second);
        }
        for (auto hostMem : mHostBuffer)
        {
            free(hostMem.second);
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
    std::map<std::string, void*> mHostBuffer;
    bool validateAndFillConfig();
    bool checkStaticShape(std::string& name);
    void allocateBuffer();
    Sampler<half>* mSampler;
    // These are used as debugging functions
    std::string printKVCache(int64_t contextLength);
    std::string printLogits();
};

#endif