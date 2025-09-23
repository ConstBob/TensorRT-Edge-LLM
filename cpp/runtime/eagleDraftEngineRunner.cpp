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

#include "runtime/eagleDraftEngineRunner.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "kernels/speculative/newEagleUtilKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
using namespace nvinfer1;

namespace
{
std::string formatEngineConfig(drivellm::rt::EagleDraftEngineRunnerConfig const& config)
{
    std::stringstream ss;
    ss << std::boolalpha;
    ss << "EagleDraftEngineRunnerConfig:"
       << "  numDecoderLayers: " << config.numDecoderLayers << "  numKVHeads: " << config.numKVHeads
       << "  headDim: " << config.headDim << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  kvCacheCapacityLength: " << config.kvCacheCapacityLength
       << "  draftModelVocabSize: " << config.draftModelVocabSize << "  maxDraftTreeSize: " << config.maxDraftTreeSize
       << "  baseModelHiddenDim: " << config.baseModelHiddenDim
       << "  draftModelHiddenDim: " << config.draftModelHiddenDim;
    return ss.str();
}

} // namespace

namespace drivellm
{
namespace rt
{
static constexpr int32_t kDRAFT_MODEL_CONTEXT_PROFILE_INDEX{0};
static constexpr int32_t kDRAFT_MODEL_GENERATION_PROFILE_INDEX{1};

// In the implementation, we only support batch size of 1 which will be further extended.
static constexpr int32_t kRUNTIME_BATCH_SIZE{1};

std::string const inputIdsName{"input_ids"};
std::string const contextLengthsName{"context_lengths"};
std::string const selectTokenIndicesName{"select_token_indices"};
std::string const logitsName{"logits"};
std::string const ropeCosSinName{"rope_rotary_cos_sin"};
std::string const baseModelHiddenStatesName{"hidden_states_input"};
std::string const draftModelHiddenStatesName{"hidden_states_from_draft"};
std::string const outputHiddenStatesName{"hidden_states"};
std::string const attentionMaskName{"attention_mask"};
std::string const attentionPosIdName{"attention_pos_id"};

EagleDraftEngineRunner::EagleDraftEngineRunner(
    std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream)
{
    LOG_INFO("Initializing EagleDraftEngineRunner from engine file: %s", enginePath.string().c_str());
    LOG_INFO("Using config file %s", configPath.string().c_str());

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("Failed to use MMap to read engine from file path: %s", enginePath.string());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    this->initializeConfigFromEngine();

    // Instantiate the KVCache instance of the EngineRunner.
    this->mLinearKVCache = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers,
        kRUNTIME_BATCH_SIZE, mConfig.maxSupportedInputLength, mConfig.numKVHeads, mConfig.headDim});

    // By design for tree attention kernel we use, the tree mask will be packed into in32_t values where each bit
    // represents the relationship between two
    int32_t const packedTreeMaskLen = static_cast<int64_t>(divUp(mConfig.maxDraftTreeSize, 32));
    // Instantiate other GPU memory input that needed by the Engine execution.
    this->mPosEncCosSinCache = rt::Tensor(
        {kRUNTIME_BATCH_SIZE, mConfig.kvCacheCapacityLength, mConfig.headDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    this->mSelectTokenIndices
        = rt::Tensor({kRUNTIME_BATCH_SIZE * mConfig.maxDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT64);
    this->mSequenceContextLengths = rt::Tensor({kRUNTIME_BATCH_SIZE}, rt::DeviceType::kGPU, DataType::kINT32);
    this->mDraftTreePositionIds
        = rt::Tensor({kRUNTIME_BATCH_SIZE, mConfig.maxDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    this->mPackedTreeMask = rt::Tensor(
        {kRUNTIME_BATCH_SIZE, mConfig.maxDraftTreeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
    this->mDummyInput = rt::Tensor({1}, rt::DeviceType::kGPU, DataType::kINT32);

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
        LOG_DEBUG("Initialize persistent Rope CosSinCache.");
        this->mPosEncCosSinCache
            = rt::Tensor({1, mConfig.kvCacheCapacityLength, mConfig.headDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
        bool const initRopeStatus = initializeRopeCosSinCache(mPosEncCosSinCache, ropeConfig, configJson, stream);
        if (!initRopeStatus)
        {
            LOG_ERROR("Failed to initialize persistent Rope CosSinCache.");
            throw std::runtime_error("Failed to initialize persistent Rope CosSinCache.");
        }
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(mPosEncCosSinCache.rawPointer(), 0, mPosEncCosSinCache.getMemoryCapacity(), stream));
    }

    // Currently we always run batch size of 1, we can pre-bind inputs tensor.
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(ropeCosSinName.c_str(), mPosEncCosSinCache.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        ropeCosSinName.c_str(), mPosEncCosSinCache.getShape().getTRTDims());

    setEngineIOStatus &= this->bindKVCacheToEngine(kRUNTIME_BATCH_SIZE);
    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input tensors.");
        throw std::runtime_error("Failed to bind engine input tensors.");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EagleDraftEngineRunner::initializeConfigFromEngine()
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
                mConfig.kvCacheCapacityLength = tensorDim.d[3];
                mConfig.headDim = tensorDim.d[4];
            }
            ++nbKVCacheInputs;
        }
    }
    mConfig.numDecoderLayers = nbKVCacheInputs;

    Dims const maxInputCtxIdsShape
        = mEngine->getProfileShape(inputIdsName.c_str(), kDRAFT_MODEL_CONTEXT_PROFILE_INDEX, OptProfileSelector::kMAX);
    Dims const maxInputGenIdsShape = mEngine->getProfileShape(
        inputIdsName.c_str(), kDRAFT_MODEL_GENERATION_PROFILE_INDEX, OptProfileSelector::kMAX);
    mConfig.maxSupportedInputLength = maxInputGenIdsShape.d[1];
    mConfig.maxDraftTreeSize = maxInputCtxIdsShape.d[1];

    Dims const logitsDim = mEngine->getTensorShape(logitsName.c_str());
    Dims const baseHiddenStatesDim = mEngine->getTensorShape(baseModelHiddenStatesName.c_str());
    Dims const draftHiddenStatesDim = mEngine->getTensorShape(draftModelHiddenStatesName.c_str());
    mConfig.draftModelVocabSize = logitsDim.d[1];
    mConfig.baseModelHiddenDim = baseHiddenStatesDim.d[2];
    mConfig.draftModelHiddenDim = draftHiddenStatesDim.d[2];

    LOG_INFO("Loaded EagleDraftEngineRunner with config: %s", formatEngineConfig(mConfig).c_str());
}

rt::EagleDraftEngineRunnerConfig EagleDraftEngineRunner::getDraftEngineConfig() const
{
    return mConfig;
}

rt::Tensor& EagleDraftEngineRunner::getRopeCosSinCacheTensor()
{
    return mPosEncCosSinCache;
}

rt::LinearKVCache& EagleDraftEngineRunner::getLinearKVCache()
{
    return mLinearKVCache;
}

bool EagleDraftEngineRunner::prefillStepInputValidation(rt::Tensor const& inputIds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor const& outputLogits,
    rt::Tensor const& outputHiddenStates)
{
    bool const checkInputsGPUTensor = inputIds.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. All inputs and outputs shall reside on GPU.");
        return false;
    }

    bool const isInputTypeValid = inputIds.getDataType() == DataType::kINT32
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Invalid data type of I/O tensors. Inputs shall be INT32, base model hidden states shall be FLOAT16, "
            "draft model hidden states shall be FLOAT16, output logits shall be FLOAT32, output hidden states shall be "
            "FLOAT16.");
        return false;
    }

    bool const isBatchValid = inputIds.getShape()[0] == kRUNTIME_BATCH_SIZE
        && baseModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && draftModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be 1, "
            "current inputIds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            inputIds.getShape().formatString(), baseModelHiddenStates.getShape().formatString(),
            draftModelHiddenStates.getShape().formatString());
        return false;
    }

    int64_t const inputSequenceLength = inputIds.getShape()[1];
    bool const sequenceLengthValid = inputSequenceLength <= mConfig.maxSupportedInputLength
        && baseModelHiddenStates.getShape()[1] == inputSequenceLength
        && draftModelHiddenStates.getShape()[1] == inputSequenceLength;
    if (!sequenceLengthValid)
    {
        LOG_ERROR(
            "Invalid sequence length of the input tensors. Sequence length shall be consistent and smaller than max "
            "supported length %d, current inputIds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates "
            "shape: %s",
            mConfig.maxSupportedInputLength, inputIds.getShape().formatString(),
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    bool const isInputHiddenSizeValid = baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isInputHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    bool const isOutputShapeValid = outputLogits.getShape()[0] == kRUNTIME_BATCH_SIZE
        && outputLogits.getShape()[1] == mConfig.draftModelVocabSize
        && outputHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && outputHiddenStates.getShape()[1] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [1, %d], hidden states shape shall be [1, %d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            mConfig.draftModelVocabSize, mConfig.draftModelHiddenDim, outputLogits.getShape().formatString(),
            outputHiddenStates.getShape().formatString());
        return false;
    }
    return true;
}

bool EagleDraftEngineRunner::executeEaglePrefillStep(rt::Tensor const& inputIds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->prefillStepInputValidation(
        inputIds, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Prefill request not performed due to invalid input tensors.");
        return false;
    }

    // Prepare the input for the engine execution.
    int32_t const inputSequenceLength = static_cast<int32_t>(inputIds.getShape()[1]);
    constexpr int32_t kCONTEXT_SELECT_TOKEN_LENGTH{1};
    mSequenceContextLengths.reshape({kRUNTIME_BATCH_SIZE});
    mSelectTokenIndices.reshape({kRUNTIME_BATCH_SIZE * kCONTEXT_SELECT_TOKEN_LENGTH});
    kernel::prepareEaglePrefillInputs(mSequenceContextLengths, mSelectTokenIndices, inputSequenceLength, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(inputIdsName.c_str(), const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(inputIdsName.c_str(), inputIds.getShape().getTRTDims());
    setEngineIOStatus &= mContextExecutionContext->setTensorAddress(
        baseModelHiddenStatesName.c_str(), const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        baseModelHiddenStatesName.c_str(), baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mContextExecutionContext->setTensorAddress(
        draftModelHiddenStatesName.c_str(), const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        draftModelHiddenStatesName.c_str(), draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(selectTokenIndicesName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mContextExecutionContext->setInputShape(
        selectTokenIndicesName.c_str(), mSelectTokenIndices.getShape().getTRTDims());

    // attention-pos-id and attention-mask are unused during the execution. We set the dummy input tensor with zero
    // shape.
    rt::Coords const emptyPosIdShape{kRUNTIME_BATCH_SIZE, 0};
    rt::Coords const emptyMaskShape{kRUNTIME_BATCH_SIZE, 0, 1};
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(attentionPosIdName.c_str(), mDummyInput.rawPointer());
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(attentionPosIdName.c_str(), emptyPosIdShape.getTRTDims());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(attentionMaskName.c_str(), mDummyInput.rawPointer());
    setEngineIOStatus
        &= mContextExecutionContext->setInputShape(attentionMaskName.c_str(), emptyMaskShape.getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mContextExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());
    setEngineIOStatus
        &= mContextExecutionContext->setTensorAddress(outputHiddenStatesName.c_str(), outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mContextExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT prefill stage enqueueV3() call.");
        return false;
    }
    // In prefill step. commit all KVCache generated during the execution.
    mLinearKVCache.commitPrefillRequest(mSequenceContextLengths, stream);

    LOG_DEBUG("Prefill stage execution completed for request with batch size %d.", kRUNTIME_BATCH_SIZE);
    return true;
}

bool EagleDraftEngineRunner::draftProposalStepInputValidation(rt::Tensor const& draftTreeInputIds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask, rt::Tensor const& outputLogits,
    rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = draftTreeInputIds.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
        && draftTreeMask.getDeviceType() == rt::DeviceType::kGPU && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. All inputs and outputs shall reside on GPU.");
        return false;
    }
    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = draftTreeInputIds.getDataType() == DataType::kINT32
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && draftTreeLength.getDataType() == DataType::kINT32
        && draftTreeMask.getDataType() == DataType::kINT8 && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Input token ids shall be INT32, hidden states I/O shall be FLOAT16, "
            "draft tree length shall be INT32, draft tree mask shall be INT8, output logits shall be FLOAT32.");
        return false;
    }
    // Validate shapes of the input tensors.
    bool const isBatchValid = draftTreeInputIds.getShape()[0] == kRUNTIME_BATCH_SIZE
        && baseModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && draftModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && draftTreeLength.getShape()[0] == kRUNTIME_BATCH_SIZE && draftTreeMask.getShape()[0] == kRUNTIME_BATCH_SIZE;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be 1, current draft tree input ids shape: %s, "
            "base model hidden states shape: %s, draft model hidden states shape: %s, draft tree length shape: %s, "
            "draft tree mask shape: %s",
            draftTreeInputIds.getShape().formatString(), baseModelHiddenStates.getShape().formatString(),
            draftModelHiddenStates.getShape().formatString(), draftTreeLength.getShape().formatString(),
            draftTreeMask.getShape().formatString());
        return false;
    }

    int64_t const paddedDraftTreeSize = draftTreeInputIds.getShape()[1];
    bool const isPaddedDraftTreeSizeValid = paddedDraftTreeSize <= mConfig.maxDraftTreeSize
        && draftTreeMask.getShape()[1] == paddedDraftTreeSize && draftTreeMask.getShape()[2] == paddedDraftTreeSize
        && baseModelHiddenStates.getShape()[1] == paddedDraftTreeSize
        && draftModelHiddenStates.getShape()[1] == paddedDraftTreeSize;
    if (!isPaddedDraftTreeSizeValid)
    {
        LOG_ERROR(
            "Invalid padded draft tree size of the input tensors. Padded draft tree size shall be smaller than max "
            "limit %d and be consistent among input tensors, current draft tree mask shape: %s, base model hidden "
            "states shape: %s, draft model hidden states shape: %s",
            mConfig.maxDraftTreeSize, draftTreeMask.getShape().formatString(),
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    bool const isHiddenSizeValid = baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    bool const isOutputShapeValid = outputLogits.getShape()[0] == outputHiddenStates.getShape()[0]
        && outputLogits.getShape()[1] == mConfig.draftModelVocabSize
        && outputHiddenStates.getShape()[1] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [select-token-size, %d], hidden states shape "
            "shall be [select-token-size, %d], current outputLogits shape: %s, outputHiddenStates shape: %s",
            mConfig.draftModelVocabSize, mConfig.draftModelHiddenDim, outputLogits.getShape().formatString(),
            outputHiddenStates.getShape().formatString());
        return false;
    }

    return true;
}

bool EagleDraftEngineRunner::executeEagleDraftProposalStep(rt::Tensor const& draftTreeInputIds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->draftProposalStepInputValidation(draftTreeInputIds, baseModelHiddenStates,
        draftModelHiddenStates, draftTreeLength, draftTreeMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Draft proposal request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const paddedDraftTreeSize = static_cast<int32_t>(draftTreeInputIds.getShape()[1]);
    int32_t const selectTokenSize = static_cast<int32_t>(outputLogits.getShape()[0]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(paddedDraftTreeSize, 32));

    // Prepare extra input for engine execution. Assemble packed tree mask, position indices, select token indices,
    // sequence context lengths.
    mSelectTokenIndices.reshape({kRUNTIME_BATCH_SIZE * selectTokenSize});
    mSequenceContextLengths.reshape({kRUNTIME_BATCH_SIZE});
    mDraftTreePositionIds.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize});
    mPackedTreeMask.reshape({kRUNTIME_BATCH_SIZE, paddedDraftTreeSize, packedTreeMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();
    kernel::prepareEagleDraftProposalInputs(draftTreeMask, draftTreeLength, sequenceStartIndex, mPackedTreeMask,
        mDraftTreePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        inputIdsName.c_str(), const_cast<void*>(draftTreeInputIds.rawPointer()));
    setEngineIOStatus
        &= mGenerationExecutionContext->setInputShape(inputIdsName.c_str(), draftTreeInputIds.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        baseModelHiddenStatesName.c_str(), const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        baseModelHiddenStatesName.c_str(), baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        draftModelHiddenStatesName.c_str(), const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        draftModelHiddenStatesName.c_str(), draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        selectTokenIndicesName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        selectTokenIndicesName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
    // Differs from prefill step, draft proposal step needs to take real packed mask and position indices.
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(attentionMaskName.c_str(), mPackedTreeMask.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionMaskName.c_str(), mPackedTreeMask.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        attentionPosIdName.c_str(), mDraftTreePositionIds.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionPosIdName.c_str(), mDraftTreePositionIds.getShape().getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        outputHiddenStatesName.c_str(), outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT draft proposal stage enqueueV3() call.");
        return false;
    }

    // Note in the draft token proposal step we explicitly don't commit the KVCache since we process the "whole tree" in
    // these steps.
    LOG_DEBUG("Draft proposal stage execution completed for request with batch size");
    return true;
}

bool EagleDraftEngineRunner::acceptDecodeTokenStepInputValidation(rt::Tensor const& acceptedTokens,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor const& outputLogits,
    rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = acceptedTokens.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. All inputs and outputs shall reside on GPU.");
        return false;
    }

    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = acceptedTokens.getDataType() == DataType::kINT32
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Invalid data type of I/O tensors. Accepted tokens shall be INT32, hidden states I/O shall be FLOAT16, "
            "output logits shall be FLOAT32, output hidden states shall be FLOAT16.");
        return false;
    }

    bool const isBatchValid = acceptedTokens.getShape()[0] == kRUNTIME_BATCH_SIZE
        && baseModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && draftModelHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be 1, current accepted tokens shape: %s, base "
            "model hidden states shape: %s, draft model hidden states shape: %s",
            acceptedTokens.getShape().formatString(), baseModelHiddenStates.getShape().formatString(),
            draftModelHiddenStates.getShape().formatString());
        return false;
    }

    int64_t const acceptedTokenNum = acceptedTokens.getShape()[1];
    bool const isAcceptedTokenNumValid = acceptedTokenNum <= mConfig.maxDraftTreeSize
        && baseModelHiddenStates.getShape()[1] == acceptedTokenNum
        && draftModelHiddenStates.getShape()[1] == acceptedTokenNum;
    if (!isAcceptedTokenNumValid)
    {
        LOG_ERROR(
            "Invalid accepted token number of the input tensors. Accepted token number shall be smaller than max limit "
            "%d, And be consistent among input tensors, current accepted tokens shape: %s, base model hidden states "
            "shape: %s, draft model hidden states shape: %s",
            mConfig.maxDraftTreeSize, acceptedTokens.getShape().formatString(),
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    bool const isHiddenSizeValid = baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            baseModelHiddenStates.getShape().formatString(), draftModelHiddenStates.getShape().formatString());
        return false;
    }

    // When accept committed tokens, we only need to collect the logits and hidden states from the "last" token.
    bool const isOutputShapeValid = outputLogits.getShape()[0] == kRUNTIME_BATCH_SIZE
        && outputHiddenStates.getShape()[0] == kRUNTIME_BATCH_SIZE
        && outputLogits.getShape()[1] == mConfig.draftModelVocabSize
        && outputHiddenStates.getShape()[1] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [1, %d], hidden states shape shall be [1, %d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            mConfig.draftModelVocabSize, mConfig.draftModelHiddenDim, outputLogits.getShape().formatString(),
            outputHiddenStates.getShape().formatString());
        return false;
    }

    return true;
}

bool EagleDraftEngineRunner::executeEagleAcceptDecodeTokenStep(rt::Tensor const& acceptedTokens,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->acceptDecodeTokenStepInputValidation(
        acceptedTokens, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Accept decode token request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const acceptedTokenNum = static_cast<int32_t>(acceptedTokens.getShape()[1]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(acceptedTokenNum, 32));
    constexpr int32_t kACCEPT_DECODE_SELECT_TOKEN_LENGTH{1};

    // Prepare extra input for engine execution. Assemble packed tree mask, position indices, select token indices,
    // sequence context lengths.
    mSelectTokenIndices.reshape({kRUNTIME_BATCH_SIZE * kACCEPT_DECODE_SELECT_TOKEN_LENGTH});
    mSequenceContextLengths.reshape({kRUNTIME_BATCH_SIZE});
    mDraftTreePositionIds.reshape({kRUNTIME_BATCH_SIZE, acceptedTokenNum});
    mPackedTreeMask.reshape({kRUNTIME_BATCH_SIZE, acceptedTokenNum, packedTreeMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();
    kernel::prepareEagleAcceptDecodeTokenInputs(sequenceStartIndex, mPackedTreeMask, mDraftTreePositionIds,
        mSelectTokenIndices, mSequenceContextLengths, acceptedTokenNum, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        inputIdsName.c_str(), const_cast<void*>(acceptedTokens.rawPointer()));
    setEngineIOStatus
        &= mGenerationExecutionContext->setInputShape(inputIdsName.c_str(), acceptedTokens.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        baseModelHiddenStatesName.c_str(), const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        baseModelHiddenStatesName.c_str(), baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        draftModelHiddenStatesName.c_str(), const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        draftModelHiddenStatesName.c_str(), draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        contextLengthsName.c_str(), mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        contextLengthsName.c_str(), mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        selectTokenIndicesName.c_str(), mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        selectTokenIndicesName.c_str(), mSelectTokenIndices.getShape().getTRTDims());
    // Differs from prefill step, draft proposal step needs to take real packed mask and position indices.
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(attentionMaskName.c_str(), mPackedTreeMask.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionMaskName.c_str(), mPackedTreeMask.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        attentionPosIdName.c_str(), mDraftTreePositionIds.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        attentionPosIdName.c_str(), mDraftTreePositionIds.getShape().getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(logitsName.c_str(), outputLogits.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        outputHiddenStatesName.c_str(), outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT accept decode token stage enqueueV3() call.");
        return false;
    }

    // Commit the KVCache for accepted tokens.
    mLinearKVCache.commitDecodeRequest(acceptedTokenNum, stream);

    LOG_DEBUG("Accept decode token stage execution completed for request with batch size");
    return true;
}

bool EagleDraftEngineRunner::bindKVCacheToEngine(int32_t activeBatchSize)
{
    // Prepare special input binding shape for prefill stage KVCache input.
    // TODO: Unify the semantics to always pass full KVCache shape.
    Dims const kvCacheDimPrefillIn = {5, {activeBatchSize, 2, mConfig.numKVHeads, 0, mConfig.headDim}};
    Dims const kvCacheDimDecodeIn
        = {5, {activeBatchSize, 2, mConfig.numKVHeads, mConfig.maxDraftTreeSize, mConfig.headDim}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string const pastKeyValuesName = format::fmtstr("past_key_values.%d", i);
        std::string const presentKeyValuesName = format::fmtstr("present_key_values.%d", i);

        rt::Tensor kvCacheBlock = mLinearKVCache.getKVCacheForDecoderLayer(i);
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

} // namespace rt
} // namespace drivellm