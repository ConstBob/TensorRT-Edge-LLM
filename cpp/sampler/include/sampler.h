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