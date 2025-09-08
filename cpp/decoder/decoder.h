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

#pragma once

#include "common/benchmarkProfiler.h"
#include "common/safetensorsUtils.h"
#include "common/trtUtils.h"
#include <NvInferRuntime.h>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace drivellm
{
namespace rt
{

struct ModelConfig
// This is the model config inferred from optimization profiles
{
    int64_t batchSize{0};
    int64_t numHead{0};
    int64_t hiddenSizePerHead{0};
    int64_t rotaryDim{0};
    int64_t minSupportedInputLength{0};
    int64_t maxSupportedInputLength{0};
    int64_t maxLength{0}; // Equivalent to maxOutputLength;
    int64_t numLayers{0};
    int64_t vocabSize{0};
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
    using LogitsType = float;
    using KVCacheType = half;
    using LoraWeightType = half;

    Decoder()
        : mStream{nullptr}
        , mEngine{nullptr}
        , mContextExecutionContext{nullptr}
        , mGenerationExecutionContext{nullptr}
        , isSetup{false}
        , mConfig{}
        , mDeviceBuffer{}
        , mUseCudaGraph{false}
        , mCudaGraphCaptured{false}
        , mGenerationGraph{nullptr}
        , mGenerationGraphExec{nullptr}
        , mIsEagle{false}
    {
    }
    bool setup(std::string modelDir, int64_t batchSize = 1, bool isEagle = false, std::string modelType = "",
        bool useCudaGraph = false, cudaStream_t stream = nullptr);
    void setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs);
    void setupRopeCosSin();

    void generate(std::vector<int32_t> const& inputIds, std::vector<int32_t> contextLengths,
        std::vector<std::vector<int32_t>>& outputIds, GenerationConfig generationConfig, int32_t endIds = -1,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr);

    void generateForContext(void* inputIds, std::vector<int32_t>& contextLengths,
        std::vector<int64_t> const& lastTokenIds, nvinfer1::Dims const inputDims = {});

    void generateForDecode(std::vector<int32_t>& contextLengths, std::vector<int64_t>& lastTokenIds);
    void addNewBuffer(std::string const& name, nvinfer1::Dims const dimsContext, int sizeOfByte);
    void initDecodingPhaseCudaGraph();

    void* getDeviceBuffer(std::string const& name);
    ModelConfig const getModelConfig() const noexcept;

    void getLastHostLogits(std::vector<LogitsType>& hostLogits);
    size_t getDeviceMemorySize() const noexcept;
    int64_t getModelBatchSize() const noexcept;

    int64_t getMaxSupportedInputLength() const noexcept;
    int64_t getMinSupportedInputLength() const noexcept;

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

    // These are used as debugging functions
    std::string printKVCache();
    std::string mEnginePath;
    std::string mModelDir;
    std::string mModelType;

    bool mUseCudaGraph{true};
    bool mCudaGraphCaptured{false};
    cudaGraph_t mGenerationGraph;
    cudaGraphExec_t mGenerationGraphExec;
    // Flag indicating Eagle pattern mode(target + draft models)
    bool mIsEagle;
    std::unordered_map<std::string, std::vector<drivellm::rt::Tensor>> mLoraWeights;
};

} // namespace rt
} // namespace drivellm
