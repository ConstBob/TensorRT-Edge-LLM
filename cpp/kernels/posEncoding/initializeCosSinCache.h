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

namespace drivellm
{
namespace kernel
{

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

void initializeLongRopeCosSin(float* shortCosSinCache, float* longCosSinCache, float* shortFactor, float* longFactor,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t maxPositionEmbeddings,
    int32_t originalMaxPositionEmbeddings, cudaStream_t stream);

void initializeMRopeCosSin(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, int32_t batchSize, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm