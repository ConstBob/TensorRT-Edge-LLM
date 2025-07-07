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

EngineConfig::EngineConfig(std::string const& base_engine_path)
    : baseEnginePath(base_engine_path)
{
}

EngineConfig::EngineConfig(std::string const& base_engine_path, std::string const& eagle_engine_path,
    int32_t max_path_len, int32_t top_k, bool is_eagle3, int32_t max_decoding_tokens, bool use_cuda_graph)
    : baseEnginePath(base_engine_path)
    , eagleEnginePath(eagle_engine_path)
    , maxPathLen(max_path_len)
    , topK(top_k)
    , isEagle3(is_eagle3)
    , maxDecodingTokens(max_decoding_tokens)
    , useCudaGraph(use_cuda_graph)
{
}

template <typename T>
LLMEngine<T>::LLMEngine(EngineConfig const& config, cudaStream_t stream)
{
    try
    {
        mIsEagle3 = config.isEagle3;
        mModel = createModel(config, stream);
        // get context length and batch size
        updateModelDimensions();
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("Failed to initialize LLMEngine: " + std::string(e.what()));
    }
}

template <typename T>
bool LLMEngine<T>::isEagleModel() const
{
    return std::holds_alternative<std::unique_ptr<Eagle<T>>>(mModel);
}

template <typename T>
int32_t LLMEngine<T>::getMaxContextLength() const
{
    return mMaxContextLength;
}

template <typename T>
int64_t LLMEngine<T>::getBatchSize() const
{
    return mBatchSize;
}

template <typename T>
std::unique_ptr<Decoder<T>>& LLMEngine<T>::getDecoder()
{
    if (!isEagleModel())
    {
        return std::get<std::unique_ptr<Decoder<T>>>(mModel);
    }
    throw std::runtime_error("getDecoder() called in Eagle mode - use getEagle() instead");
}

template <typename T>
std::unique_ptr<Eagle<T>>& LLMEngine<T>::getEagle()
{
    if (isEagleModel())
    {
        return std::get<std::unique_ptr<Eagle<T>>>(mModel);
    }
    throw std::runtime_error("getEagle() called in standard mode - use getDecoder() instead");
}

template <typename T>
int64_t LLMEngine<T>::getDeviceMemorySize()
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

template <typename T>
void LLMEngine<T>::getLastHostLogits(std::vector<T>& hostLogits)
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

template <typename T>
void LLMEngine<T>::generate(std::vector<int64_t> const& inputIds, std::vector<int32_t> const& contextLengths,
    std::vector<std::vector<int64_t>>& outputIds, GenerationConfig const& generationConfig,
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
        if (autoDecode && tokenizer)
        {
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());
            for (int i = 0; i < mBatchSize; ++i)
            {
                std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
            }
        }
        else
        {
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        }
    }
}

template <typename T>
void LLMEngine<T>::processInputSequence(std::string const& inputString, Tokenizer* tokenizer,
    std::vector<int32_t>& contextLengths, std::vector<int64_t>& inputIds, int64_t batchIdx, int64_t padId,
    bool truncate, bool padding)
{
    // Process and tokenize input string, then store tokens in the appropriate batch position
    std::vector<int64_t> batchInputIds = tokenizer->encode(inputString, true);
    int32_t inputSize = static_cast<int32_t>(batchInputIds.size());

    if (inputSize > mMaxContextLength)
    {
        if (truncate)
        {
            std::cout << "Warning: input length > max context length. The last tokens will be truncated." << std::endl;
        }
        else
        {
            throw std::runtime_error("Input length exceeds max context length");
        }
    }
    contextLengths[batchIdx] = std::min(inputSize, mMaxContextLength);
    if (padding)
    {
        batchInputIds.resize(mMaxContextLength, padId);
        std::copy(batchInputIds.begin(), batchInputIds.end(), inputIds.begin() + batchIdx * mMaxContextLength);
    }
    else
    {
        int64_t offset = calculatePrefixSum(contextLengths, batchIdx);
        std::copy(batchInputIds.begin(), batchInputIds.end(), inputIds.begin() + offset);
    }
}

template <typename T>
int64_t LLMEngine<T>::calculatePrefixSum(std::vector<int32_t> const& contextLengths, int32_t i) const
{
    // Calculate the sum of context lengths from index 0 to i-1 (prefix sum)
    if (i <= 0)
        return 0;
    return std::accumulate(contextLengths.begin(), contextLengths.begin() + i, 0LL);
}

template <typename T>
typename LLMEngine<T>::ModelPtr LLMEngine<T>::createModel(EngineConfig const& config, cudaStream_t stream)
{
    auto baseDecoder = std::make_unique<Decoder<T>>();

    if (config.eagleEnginePath.empty())
    {
        // Single decoder mode
        baseDecoder->setup(config.baseEnginePath, stream, config.useCudaGraph);
        return std::move(baseDecoder);
    }
    else
    {
        // Eagle mode - requires both base and draft decoders
        baseDecoder->setup(config.baseEnginePath, stream, config.useCudaGraph, 1, true);
        auto draftDecoder = std::make_unique<Decoder<T>>();
        draftDecoder->setup(config.eagleEnginePath, stream, config.useCudaGraph, 1, true);
        return std::make_unique<Eagle<T>>(std::move(baseDecoder), std::move(draftDecoder), stream,
            config.eagleEnginePath, config.maxPathLen, config.topK, config.isEagle3, config.maxDecodingTokens);
    }
}

template <typename T>
void LLMEngine<T>::updateModelDimensions()
{
    if (isEagleModel())
    {
        // Extract dimensions (context length and batch size) from Eagle model (Eagle mode)
        auto& eagle = std::get<std::unique_ptr<Eagle<T>>>(mModel);
        mMaxContextLength = eagle->getMaxContextLength();
        mBatchSize = eagle->getModelBatchSize();
    }
    else
    {
        // Extract dimensions  (context length and batch size) from standard Decoder model (non-Eagle mode)
        auto& decoder = std::get<std::unique_ptr<Decoder<T>>>(mModel);
        mMaxContextLength = decoder->getMaxContextLength();
        mBatchSize = decoder->getModelBatchSize();
    }
}

template class LLMEngine<half>;