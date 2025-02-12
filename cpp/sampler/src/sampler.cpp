/*
 * Copyright 2024 The TensorRT-LLM Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
 
#include "sampler/include/sampler.h"
#include "sampler/include/topKSamplingLayer.h"
#include <cstdint>
#include <memory>

template <typename T>
Sampler<T>::Sampler(int64_t batchSize, int64_t vocabSize)
    : mDecoderDomain(batchSize, 1, vocabSize)
{
    auto setupParams = std::make_shared<SamplingSetupParams>();

    // Greedy search setup
    setupParams->randomSeed = std::make_optional<std::vector<uint64_t>>({uint64_t(42)});
    setupParams->runtimeTopK = std::make_optional<std::vector<std::int32_t>>({1});
    setupParams->runtimeTopP = std::nullopt;
    setupParams->topPDecay = std::nullopt;
    setupParams->topPMin = std::nullopt;
    setupParams->topPResetIds = std::nullopt;

    mLayer = std::make_unique<TopKSamplingLayer<T>>(mDecoderDomain);

    // If we need to support top k, top p in the future, we need a separate setup
    // function to adjust top k, top p value between different request.
    mLayer->setup(batchSize, 1, nullptr, setupParams);
    mOutputIds.resize(batchSize);
    auto workspaceSize = mLayer->getWorkspaceSize();
    CUDA_CHECK(cudaMalloc(&mWorkspace, sizeof(int8_t) * workspaceSize));
    CUDA_CHECK(cudaMalloc(&mDevStates, sizeof(curandState) * batchSize));
    CUDA_CHECK(cudaMalloc(&mOutputIdsDevice, sizeof(int64_t) * batchSize));
}

template <typename T>
Sampler<T>::~Sampler()
{
    cudaFree(mWorkspace);
    cudaFree(mDevStates);
    cudaFree(mOutputIdsDevice);
}

template <typename T>
std::vector<int64_t> const& Sampler<T>::greedySample(T const* logits)
{
    int64_t batchSize = mOutputIds.size();

    auto inputs = std::make_shared<SamplingInputs>(batchSize);
    inputs->logits = std::make_shared<TensorWrapper const>(
        const_cast<T*>(logits), std::vector<int64_t>{batchSize, mDecoderDomain.getVocabSize()}, TRTDataType<T>::value);
    inputs->probsComputed = false;
    inputs->curandStates = mDevStates;
    inputs->samplingWorkspace = mWorkspace;

    auto outputs = std::make_shared<BaseDecodingOutputs>(std::make_shared<TensorWrapper>(
        mOutputIdsDevice, std::vector<int64_t>{mDecoderDomain.getBatchSize(), 1}, TRTDataType<int64_t>::value));
    outputs->maxSeqLen = 1;

    mLayer->forwardAsync(outputs, inputs);

    cudaDeviceSynchronize();

    cudaMemcpy(mOutputIds.data(), mOutputIdsDevice, sizeof(int64_t) * batchSize, cudaMemcpyDeviceToHost);

    return mOutputIds;
}

template class Sampler<half>;