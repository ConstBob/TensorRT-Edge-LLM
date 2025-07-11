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

#pragma once

#include "common/benchmarkProfiler.h"
#include "common/common.h"
#include "common/safetensors_loader/safetensorsLoader.h"
#include "common/trtUtils.h"
#include "sampler/include/sampler.h"
#include <NvInferRuntime.h>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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
        , mUseCudaGraph{false}
        , mCudaGraphCaptured{false}
        , mGenerationGraph{nullptr}
        , mGenerationGraphExec{nullptr}
        , mIsEagle{false}
    {
    }
    bool setup(std::filesystem::path const& fp, cudaStream_t& stream, bool useCudaGraph = false, int64_t batchSize = 1,
        bool isEagle = false);
    void setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs);

    void generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> contextLengths,
        std::vector<std::vector<int64_t>>& outputIds, GenerationConfig generationConfig, int64_t endIds = -1,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr);

    void generateForContext(void* inputIds, std::vector<int32_t>& contextLengths,
        std::vector<int64_t> const& lastTokenIds, const nvinfer1::Dims inputDims = {});

    void generateForDecode(std::vector<int32_t>& contextLengths, std::vector<int64_t>& lastTokenIds);
    void addNewBuffer(std::string const& name, const nvinfer1::Dims dimsContext, int sizeOfByte);

    void* getDeviceBuffer(std::string const& name);
    const ModelConfig getModelConfig() const noexcept;

    void getLastHostLogits(std::vector<T>& hostLogits);
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
    bool addLora(std::string const& name, std::string const& filePath);
    bool switchLora(std::string const& name);
    std::vector<std::string> getLoraNames() const;

private:
    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;

    bool isSetup;
    ModelConfig mConfig;
    std::map<std::string, void*> mDeviceBuffer;
    std::map<std::string, void*> mHostBuffer;
    bool validateAndFillConfig(int64_t batchSize = 1);
    bool checkStaticShape(std::string& name);
    void allocateBuffer();
    void allocateBufferForKVCache();
    void allocateExtraBufferForEagle();
    void allocateExtraBufferForVanilla();
    void allocateCommonBuffers();
    void initCudaGraph();
    std::unique_ptr<Sampler<T>> mSampler;
    // These are used as debugging functions
    std::string printKVCache();
    std::string printLogits();
    std::string mEnginePath;

    bool mUseCudaGraph{true};
    bool mCudaGraphCaptured{false};
    cudaGraph_t mGenerationGraph;
    cudaGraphExec_t mGenerationGraphExec;
    // Flag indicating Eagle pattern mode(target + draft models)
    bool mIsEagle;
    std::unordered_map<std::string, std::unique_ptr<drivellm::SafeTensorsLoader>> mLoraWeights;
};