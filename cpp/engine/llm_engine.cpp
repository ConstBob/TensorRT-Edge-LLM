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

#include "llm_engine.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

EngineConfig::EngineConfig(std::string const& engine_dir, bool use_cuda_graph, int32_t batch_size)
    : engineDir(engine_dir)
    , useCudaGraph(use_cuda_graph)
    , batchSize(batch_size)
{
}

EngineConfig::EngineConfig(std::string const& engine_dir, std::string const& base_model_dir,
    std::string const& draft_model_dir, int32_t max_path_len, int32_t top_k, bool is_eagle3,
    int32_t max_decoding_tokens, bool use_cuda_graph)
    : engineDir(engine_dir)
    , baseModelDir(base_model_dir)
    , draftModelDir(draft_model_dir)
    , maxPathLen(max_path_len)
    , topK(top_k)
    , isEagle3(is_eagle3)
    , maxDecodingTokens(max_decoding_tokens)
    , useCudaGraph(use_cuda_graph)
{
}

LLMEngine::LLMEngine(EngineConfig const& config, cudaStream_t stream)
{
    try
    {
        mIsEagle3 = config.isEagle3;
        mModel = createModel(config, stream);
        // get context length and batch size
        if (isEagleModel())
        {
            auto& eagle = std::get<std::unique_ptr<Eagle>>(mModel);
            mMaxSupportedInputLength = eagle->getMaxSupportedInputLength();
            mMinSupportedInputLength = eagle->getMinSupportedInputLength();
            mBatchSize = eagle->getModelBatchSize();
        }
        else
        {
            auto& decoder = std::get<std::unique_ptr<Decoder>>(mModel);
            mMaxSupportedInputLength = decoder->getMaxSupportedInputLength();
            mMinSupportedInputLength = decoder->getMinSupportedInputLength();
            mBatchSize = decoder->getModelBatchSize();
        }
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("Failed to initialize LLMEngine: " + std::string(e.what()));
    }
}

bool LLMEngine::isEagleModel() const
{
    return std::holds_alternative<std::unique_ptr<Eagle>>(mModel);
}

int32_t LLMEngine::getMinSupportedInputLength() const
{
    return mMinSupportedInputLength;
}

int32_t LLMEngine::getMaxSupportedInputLength() const
{
    return mMaxSupportedInputLength;
}

int64_t LLMEngine::getBatchSize() const
{
    return mBatchSize;
}

std::unique_ptr<Decoder>& LLMEngine::getDecoder()
{
    if (!isEagleModel())
    {
        return std::get<std::unique_ptr<Decoder>>(mModel);
    }
    throw std::runtime_error("getDecoder() called in Eagle mode - use getEagle() instead");
}

std::unique_ptr<Eagle>& LLMEngine::getEagle()
{
    if (isEagleModel())
    {
        return std::get<std::unique_ptr<Eagle>>(mModel);
    }
    throw std::runtime_error("getEagle() called in standard mode - use getDecoder() instead");
}

int64_t LLMEngine::getDeviceMemorySize()
{
    if (isEagleModel())
    {
        auto& eagle = getEagle();
        return eagle->getDeviceMemorySize();
    }
    else
    {
        auto& decoder = getDecoder();
        return decoder->getDeviceMemorySize();
    }
}

void LLMEngine::setupExtraInputs(std::vector<EngineInputDesc> const& extraInputs)
{
    if (isEagleModel())
    {
        auto& eagle = getEagle();
        eagle->setupExtraInputs(extraInputs);
    }
    else
    {
        auto& decoder = getDecoder();
        decoder->setupExtraInputs(extraInputs);
    }
}

void LLMEngine::setupRopeCosSin()
{
    if (isEagleModel())
    {
        auto& eagle = getEagle();
        eagle->setupRopeCosSin();
    }
    else
    {
        auto& decoder = getDecoder();
        decoder->setupRopeCosSin();
    }
}

void LLMEngine::getLastHostLogits(std::vector<LogitsType>& hostLogits)
{
    if (isEagleModel())
    {
        auto& eagle = getEagle();
        eagle->getLastHostLogits(hostLogits);
    }
    else
    {
        auto& decoder = getDecoder();
        decoder->getLastHostLogits(hostLogits);
    }
}

ModelConfig LLMEngine::getBaseModelConfig()
{
    if (isEagleModel())
    {
        auto& eagle = getEagle();
        return eagle->getBaseModelConfig();
    }
    else
    {
        auto& decoder = getDecoder();
        return decoder->getModelConfig();
    }
}

void LLMEngine::generate(std::vector<int32_t> const& inputIds, std::vector<int32_t> const& contextLengths,
    std::vector<std::vector<int32_t>>& outputIds, GenerationConfig const& generationConfig,
    std::vector<int32_t>* newTokensNumbers, std::vector<int32_t>* iterNumbers,
    std::shared_ptr<BenchmarkProfiler> const profiler, Tokenizer* tokenizer, bool autoDecode)
{

    if (isEagleModel())
    {
        auto& eagle = getEagle();
        if (!tokenizer)
        {
            LOG_ERROR("tokenizer must be provided under eagle mode for generate()");
            throw std::runtime_error("tokenizer must be provided under eagle mode for generate()");
        }
        eagle->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId(), mIsEagle3,
            profiler, newTokensNumbers, iterNumbers);
        if (autoDecode)
        {
            for (int i = 0; i < mBatchSize; ++i)
            {
                std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
            }
        }
    }
    else
    {
        auto& decoder = getDecoder();
        int32_t eosId = -1;
        if (tokenizer)
        {
            eosId = tokenizer->getEosId();
        }
        if (autoDecode && tokenizer)
        {
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig, eosId);
            for (int i = 0; i < mBatchSize; ++i)
            {
                std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
            }
        }
        else
        {
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig, eosId, profiler);
        }
    }
}

std::vector<int32_t> LLMEngine::processInputSequence(std::vector<std::string> const& inputStrings, Tokenizer* tokenizer,
    std::vector<int32_t>& contextLengths, int32_t padId)
{
    // Process and tokenize input string, then store tokens in the appropriate batch position
    int32_t batchSize = static_cast<int32_t>(inputStrings.size());
    if (batchSize != mBatchSize)
    {
        throw std::runtime_error("Batch size mismatch for engine setup config.");
    }
    contextLengths.resize(batchSize, 0);
    std::vector<std::vector<int32_t>> batchInputIds;
    for (int32_t i = 0; i < batchSize; ++i)
    {
        auto tokenizedInput = tokenizer->encode(inputStrings[i], true);
        batchInputIds.emplace_back(tokenizedInput);
        contextLengths[i] = static_cast<int32_t>(tokenizedInput.size());
    }
    int32_t maxInputLengthInBatch = *std::max_element(contextLengths.begin(), contextLengths.end());

    if (maxInputLengthInBatch > mMaxSupportedInputLength)
    {
        throw std::runtime_error("Input length exceeds max supported input context length");
    }

    // Depends on the engine config, we may either pad input ids to maxSupportedInputLength
    // or pad inputs to largest input length in the batch.
    bool const useMaxSupportedISLPadding = mMinSupportedInputLength == mMaxSupportedInputLength;
    int32_t sequenceStride = useMaxSupportedISLPadding ? mMaxSupportedInputLength
                                                       : std::max(mMinSupportedInputLength, maxInputLengthInBatch);

    std::vector<int32_t> result(sequenceStride * batchSize, padId);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        std::copy(batchInputIds[i].begin(), batchInputIds[i].end(), result.begin() + i * sequenceStride);
    }
    return result;
}

typename LLMEngine::ModelPtr LLMEngine::createModel(EngineConfig const& config, cudaStream_t stream)
{
    auto baseDecoder = std::make_unique<Decoder>();

    if (config.baseModelDir.empty() && config.draftModelDir.empty())
    {
        // Single decoder mode (naive decoding)
        baseDecoder->setup(config.engineDir, config.batchSize, false, "", config.useCudaGraph, stream);
        return std::move(baseDecoder);
    }
    else
    {
        // Eagle mode - requires both base and draft decoders
        baseDecoder->setup(config.baseModelDir, 1, true, "base", config.useCudaGraph, stream);
        auto draftDecoder = std::make_unique<Decoder>();
        draftDecoder->setup(config.draftModelDir, 1, true, "draft", config.useCudaGraph, stream);

        return std::make_unique<Eagle>(std::move(baseDecoder), std::move(draftDecoder), stream, config.baseModelDir,
            config.draftModelDir, config.maxPathLen, config.topK, config.isEagle3, config.maxDecodingTokens);
    }
}