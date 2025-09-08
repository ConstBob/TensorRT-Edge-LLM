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
#include "decoder/decoder.h"
#include "kernels/speculative/eagleUtilKernels.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
using Json = nlohmann::json;

namespace drivellm
{
namespace rt
{

class Eagle
{
public:
    using LogitsType = float;
    using KVCacheType = half;
    using HiddenStatesType = half;

    Eagle(std::unique_ptr<Decoder> baseModel, std::unique_ptr<Decoder> draftModel, cudaStream_t stream,
        std::string baseModelDir, std::string draftModelDir, int32_t maxPathLen = 6, int32_t topK = 10,
        bool isEagle3 = false, int32_t maxDecodingTokens = 60)
        : mBaseModel(std::move(baseModel))
        , mDraftModel(std::move(draftModel))
        , mStream(stream)
    {

        auto modelConfig = mBaseModel->getModelConfig();
        mBatchSize = modelConfig.batchSize;
        // for eagle plus 1
        mMaxSeqLen = modelConfig.maxLength + 1;
        mVocabSize = modelConfig.vocabSize;
        mMaxInputLength = modelConfig.maxSupportedInputLength;
        // don't contain root node
        mMaxDraftTokens = maxDecodingTokens - 1;
        mMaxDecodingTokens = maxDecodingTokens;
        mMaxPathLen = maxPathLen;
        mTopK = topK;
        mIsEagle3 = isEagle3;
        mBaseModelDir = baseModelDir;
        mDraftModelDir = draftModelDir;
        mMaxDraftTokensPerStep = mMaxPathLen * mTopK;

        // Load config from base model directory (base_config.json)
        Json jsonConfig;
        std::string baseConfigPath = baseModelDir + "/base_config.json";
        std::ifstream baseConfigFileStream(baseConfigPath);
        if (!baseConfigFileStream.is_open())
        {
            LOG_ERROR("Eagle Decoder: Failed to open base config file: %s", baseConfigPath.c_str());
            throw std::runtime_error("Eagle Decoder: Failed to open base config file: " + baseConfigPath);
        }

        try
        {
            jsonConfig = Json::parse(baseConfigFileStream);
        }
        catch (Json::parse_error const& e)
        {
            LOG_ERROR("Failed to parse base config file: %s", e.what());
            throw std::runtime_error("Eagle: Fail to parse base config file to obtain model parameters");
        }

        mHiddenDim = jsonConfig["hidden_size"].get<int32_t>();
        mTargetOutputHiddenDim = isEagle3 ? mHiddenDim * 3 : mHiddenDim;

        // For draft vocab size, try to get from draft config first, then fall back to base config
        if (isEagle3)
        {
            std::string draftConfigPath = draftModelDir + "/draft_config.json";
            std::ifstream draftConfigFileStream(draftConfigPath);
            if (draftConfigFileStream.is_open())
            {
                try
                {
                    Json draftJsonConfig = Json::parse(draftConfigFileStream);
                    if (draftJsonConfig.contains("draft_vocab_size"))
                    {
                        mDraftVocabSize = draftJsonConfig["draft_vocab_size"].get<int32_t>();
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Eagle3 Decoder: draft_vocab_size not found in draft config file: " + draftConfigPath);
                    }
                }
                catch (Json::parse_error const& e)
                {
                    throw std::runtime_error("Eagle3 Decoder: Failed to parse draft config file: " + draftConfigPath);
                }
            }
            else
            {
                throw std::runtime_error("Eagle3 Decoder: Failed to open draft config file: " + draftConfigPath);
            }
        }
        else
        {
            mDraftVocabSize = jsonConfig["vocab_size"].get<int32_t>();
        }
        eagleCommonParamsInit();
        allocateEagleBuffer();
        addNewBufferForModelIO();
        setupExtraInputsForBaseModel();
        setupExtraInputsForDraftModelDecode();
    };

    void generate(std::vector<int32_t> const& inputIds, std::vector<int32_t> contextLengths,
        std::vector<std::vector<int32_t>>& outputIds, GenerationConfig generationConfig, int32_t endIds = -1,
        std::shared_ptr<BenchmarkProfiler> const profiler = nullptr, std::vector<int32_t>* newTokens = nullptr,
        std::vector<int32_t>* iterNumbers = nullptr);
    size_t getDeviceMemorySize() const noexcept;
    void getLastHostLogits(std::vector<LogitsType>& hostLogits);
    int64_t getModelBatchSize() const noexcept;
    int64_t getMinSupportedInputLength() const noexcept;
    int64_t getMaxSupportedInputLength() const noexcept;
    ModelConfig getBaseModelConfig() const noexcept;
    void setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs);
    void setupRopeCosSin();

    ~Eagle()
    {
        for (auto deviceMem : mEagleDeviceBuffer)
        {
            cudaFree(deviceMem.second);
        }
        for (auto hostMem : mEagleHostBuffer)
        {
            free(hostMem.second);
        }
        mEagleDeviceBuffer.clear();
        mEagleHostBuffer.clear();
    }

private:
    void addNewBufferForModelIO();
    void invokeSamplingAndAccept(int32_t* draftIds, int32_t const curTokensPerStep, int32_t endIds);
    void invokeUpdateDraInputIdsAndHSAndTrMaAndPosIdsAndInterScores(int32_t layerIdx, HiddenStatesType* hs_draft);
    void invokeUpdateCumScoresAndParentsIds(int32_t layerIdx);
    void invokeAssembleDraftIdsAndPathAndMaskAndPositionIds();
    void invokeInitializeAttentionMaskCausal();
    void draftDecodePostProcess(int32_t layerIdx);
    void draftModelDecodeInfer(std::vector<int32_t> ContextLengthForDraft);
    void updateGenerationStatus(
        int32_t& generationIter, int64_t& unfinishedBatchNum, std::vector<int32_t>& contextLengths);
    void allocateEagleBuffer();
    void invokeUpdateKVCacheAndHiddenStatesAndTreePositionIds();
    void initDecodingPhaseCudaGraph();

    void eagleCommonParamsInit();
    void initDraftVoc();
    void setupExtraInputsForBaseModel();
    void setupExtraInputsForDraftModelContext(std::vector<int32_t> const& contextLengths);
    void setupExtraInputsForDraftModelDecode();
    std::unique_ptr<Decoder> mBaseModel;
    std::unique_ptr<Decoder> mDraftModel;

    std::map<std::string, void*> mEagleDeviceBuffer;
    std::map<std::string, void*> mEagleHostBuffer;
    drivellm::kernel::EagleCommonParams mEagleCommonParams;
    std::string mBaseModelDir;
    std::string mDraftModelDir;
    std::vector<int64_t> acceptedLengthsHost{mBatchSize};
    std::vector<int64_t> lastLogitsOffsetHost{mBatchSize};

    int64_t mBatchSize;
    int32_t mMaxSeqLen;
    int32_t mHiddenDim;
    int32_t mVocabSize;
    int32_t mMaxInputLength;
    int32_t mTargetOutputHiddenDim;
    int32_t mMaxDraftTokensPerStep;
    int32_t mMaxDraftTokens;
    int32_t mMaxDecodingTokens;
    // depth+1
    int32_t mMaxPathLen;
    int32_t mTopK;
    int32_t mDraftVocabSize;
    bool mIsEagle3;

    cudaStream_t mStream;
};

} // namespace rt
} // namespace drivellm