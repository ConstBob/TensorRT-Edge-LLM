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

#pragma once

#include "baseLayer.h"
#include <cstdint>
#include <memory>
#include <vector>

template <typename T>
class Sampler
{
public:
    Sampler(int64_t batchSize, int64_t vocabSize);
    ~Sampler();

    std::vector<int64_t> const& greedySample(T const* logits);

    Sampler(Sampler const&) = delete;
    Sampler& operator=(Sampler const&) = delete;

private:
    std::unique_ptr<BaseLayer> mLayer;
    void* mWorkspace;
    curandState* mDevStates;
    DecoderDomain mDecoderDomain;
    std::vector<int64_t> mOutputIds;
    int64_t* mOutputIdsDevice;
};