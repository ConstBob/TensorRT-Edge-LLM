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

#include <cstdint>
#include <cuda_fp16.h>
#include <optional>
#include <set>
#include <vector>

std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k, std::vector<half> const& v,
    int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    std::optional<std::vector<int32_t>> const& treeAttnMask = std::nullopt);

std::vector<half> ropeRef(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, int32_t const seqIdx, float const ropeScale, float const ropeTheta, bool const permute);

std::vector<half> ropeRefCosSin(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, std::vector<float> const& cos, std::vector<float> const& sin, bool const permute);

// Sampling reference functions
std::vector<float> softmaxRef(std::vector<float> const& logits, float temperature = 1.0f);

std::set<int32_t> getTopKAllowedTokensRef(std::vector<float> const& logits, int32_t topK);

std::set<int32_t> getTopPAllowedTokensRef(std::vector<float> const& logits, float topP, float temperature);

std::set<int32_t> getCombinedAllowedTokensRef(
    std::vector<float> const& logits, int32_t topK, float topP, float temperature);

std::vector<std::pair<float, int32_t>> getTopKElementsRef(std::vector<float> const& logits, int32_t topK);

// Unified reference function that handles all cases
std::vector<std::pair<float, int32_t>> returnAllTopKReference(
    std::vector<float> const& input, int32_t topK, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs);

void computeLongRopeReference(std::vector<float>& shortCosSinCache, std::vector<float>& longCosSinCache,
    std::vector<float> const& shortFactor, std::vector<float> const& longFactor, float rotaryBaseFrequency,
    int32_t rotaryDim, int32_t kvCacheCapacity, int32_t rotaryEmbeddingMaxPositions,
    int32_t originalMaxPositionEmbeddings);

void computeMRopeReference(std::vector<float>& mropeRotaryCosSin, std::vector<int64_t> const& mropePositionIds,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize);

// Embedding lookup reference functions
std::vector<half> embeddingLookupRef(std::vector<int32_t> const& inputIds, std::vector<half> const& embeddingTable,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    std::optional<std::vector<half>> const& imageEmbeds = std::nullopt, int64_t imageTokenLen = 0);
