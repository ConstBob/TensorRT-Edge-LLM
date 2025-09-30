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

#include "references.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k, std::vector<half> const& v,
    int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    std::optional<std::vector<int32_t>> const& treeAttnMask)
{
    assert(qlen <= kvlen);
    int32_t const numQheadPerKV = numQHeads / numKVHeads;
    float qkScale = 1.0f / std::sqrt(static_cast<float>(headSize));

    auto qoIndexer = [numQHeads, headSize](int32_t tokenIdx, int32_t qHeadIdx, int32_t valIdx) {
        // Q and Out Tensor has layout of [QToken, Qhead, featureVal]
        return tokenIdx * numQHeads * headSize + qHeadIdx * headSize + valIdx;
    };
    auto kvIndexer = [kvlen, headSize](int32_t kvSequenceIdx, int32_t kvHeadIdx, int32_t valIdx) {
        // KV Tensor has layout of [KVhead, kv_sequence, featureVal]
        return kvHeadIdx * kvlen * headSize + kvSequenceIdx * headSize + valIdx;
    };

    std::vector<half> result(numQHeads * headSize * qlen);
    for (int32_t tokenIdx = 0; tokenIdx < qlen; ++tokenIdx)
    {
        for (int32_t qHeadIdx = 0; qHeadIdx < numQHeads; ++qHeadIdx)
        {
            std::vector<float> attnScores(kvlen, 0.0F);
            float maxVal = -std::numeric_limits<half>::infinity();
            int32_t kvHeadIdx = qHeadIdx / numQheadPerKV;
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                for (int32_t valIdx = 0; valIdx < headSize; ++valIdx)
                {
                    float qVal = __half2float(q[qoIndexer(tokenIdx, qHeadIdx, valIdx)]);
                    float kvVal = __half2float(k[kvIndexer(kvIdx, kvHeadIdx, valIdx)]);
                    attnScores[kvIdx] += qVal * kvVal * qkScale;
                }

                maxVal = std::max(maxVal, attnScores[kvIdx]);
            }
            // Apply Mask for casual and tree mask
            if (qlen > 1 && treeAttnMask.has_value())
            {
                int32_t const kvStartIdxForQ = kvlen - qlen;
                auto const treeMasks = treeAttnMask.value();
                for (int32_t maskQIdx = 0; maskQIdx < qlen; ++maskQIdx)
                {
                    int32_t const mask = treeMasks[tokenIdx * qlen + maskQIdx];
                    if (mask == 0)
                    {
                        // Set to -1e5 to make softmax result close to 0
                        attnScores[kvStartIdxForQ + maskQIdx] = -5e5;
                    }
                }
            }
            // Compute softmax using attnScores - maxVal
            float sumExp = 0.0F;
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                attnScores[kvIdx] = std::exp(attnScores[kvIdx] - maxVal);
                sumExp += attnScores[kvIdx];
            }
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                attnScores[kvIdx] /= sumExp;
            }

            // Compute BMM2 Attn_score @ V
            for (int32_t valIdx = 0; valIdx < headSize; ++valIdx)
            {
                float outVal = 0.0F;
                for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
                {
                    outVal += attnScores[kvIdx] * __half2float(v[kvIndexer(kvIdx, kvHeadIdx, valIdx)]);
                }
                result[qoIndexer(tokenIdx, qHeadIdx, valIdx)] = __float2half(outVal);
            }
        }
    }

    return result;
}

std::vector<half> ropeRef(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, int32_t const seqIdx, float const ropeScale, float const ropeTheta, bool const permute)
{
    std::vector<half> result;
    for (int32_t i = 0; i < numHeads; i++)
    {
        std::vector<half> x(input.begin() + headSize * i, input.begin() + headSize * (i + 1));
        std::vector<half> y(headSize);
        for (int32_t j = 0; j < rotaryDim / 2; j++)
        {
            int32_t leftIndex, rightIndex;
            // Determine whether to apply gpt-neox style rope to permute.
            if (permute)
            {
                leftIndex = j;
                rightIndex = rotaryDim / 2 + j;
            }
            else
            {
                leftIndex = j * 2;
                rightIndex = j * 2 + 1;
            }
            float invFreq = (seqIdx * ropeScale) / std::pow(ropeTheta, 2 * j / float(rotaryDim));
            float cos = std::cos(invFreq);
            float sin = std::sin(invFreq);
            y[leftIndex] = __half2float(x[leftIndex]) * cos - __half2float(x[rightIndex]) * sin;
            y[rightIndex] = __half2float(x[leftIndex]) * sin + __half2float(x[rightIndex]) * cos;
        }
        // Insert RoPE part
        result.insert(result.end(), y.begin(), y.begin() + rotaryDim);
        // Copy the remaining part of the input vector
        result.insert(result.end(), x.begin() + rotaryDim, x.end());
    }
    return result;
}

std::vector<half> ropeRefCosSin(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, std::vector<float> const& cosCache, std::vector<float> const& sinCache, bool const permute)
{
    std::vector<half> result;
    for (int32_t i = 0; i < numHeads; i++)
    {
        std::vector<half> x(input.begin() + headSize * i, input.begin() + headSize * (i + 1));
        std::vector<half> y(headSize);
        for (int32_t j = 0; j < rotaryDim / 2; j++)
        {
            int32_t leftIndex, rightIndex;
            // Determine whether to apply gpt-neox style rope to permute.
            if (permute)
            {
                leftIndex = j;
                rightIndex = rotaryDim / 2 + j;
            }
            else
            {
                leftIndex = j * 2;
                rightIndex = j * 2 + 1;
            }
            float cos = cosCache[j];
            float sin = sinCache[j];
            y[leftIndex] = __half2float(x[leftIndex]) * cos - __half2float(x[rightIndex]) * sin;
            y[rightIndex] = __half2float(x[leftIndex]) * sin + __half2float(x[rightIndex]) * cos;
        }
        // Insert RoPE part
        result.insert(result.end(), y.begin(), y.begin() + rotaryDim);
        // Copy the remaining part of the input vector
        result.insert(result.end(), x.begin() + rotaryDim, x.end());
    }
    return result;
}

std::vector<float> softmaxRef(std::vector<float> const& logits, float temperature)
{
    std::vector<float> scaledLogits(logits.size());
    float invTemp = (temperature < 1e-3f) ? 1000.0f : 1.0f / temperature;

    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaledLogits[i] = logits[i] * invTemp;
    }

    // Find max for numerical stability
    float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());

    // Compute softmax
    std::vector<float> probs(logits.size());
    float sumExp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] = std::exp(scaledLogits[i] - maxLogit);
        sumExp += probs[i];
    }

    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] /= sumExp;
    }

    return probs;
}

std::set<int32_t> getTopKAllowedTokensRef(std::vector<float> const& logits, int32_t topK)
{
    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + kLimit, logitPairs.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i < kLimit; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::set<int32_t> getTopPAllowedTokensRef(std::vector<float> const& logits, float topP, float temperature)
{
    // When temperature = 0.0f, we should always pick the highest probability token
    // This matches the behavior in SamplingParams constructor
    if (temperature < 1e-3f)
    {
        int32_t idxMax = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        // Return only the highest probability token
        return std::set<int32_t>{idxMax};
    }

    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    std::sort(logitPairs.begin(), logitPairs.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

    // Extract all logits and compute probabilities
    std::vector<float> allLogits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        allLogits[i] = logitPairs[i].first;
    }
    auto allProbs = softmaxRef(allLogits, temperature);

    // Handle edge case: topP = 0.0 means only the highest probability token
    if (topP <= 0.0f)
    {
        std::set<int32_t> allowedTokens;
        allowedTokens.insert(logitPairs[0].second);
        return allowedTokens;
    }

    // Find the cutoff point for top-p
    float cumsum = 0.0f;
    int32_t cutoff = 0;
    for (size_t i = 0; i < allProbs.size(); ++i)
    {
        cumsum += allProbs[i];
        cutoff = i + 1;
        if (cumsum >= topP)
        {
            break;
        }
    }

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i < cutoff; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::set<int32_t> getCombinedAllowedTokensRef(
    std::vector<float> const& logits, int32_t topK, float topP, float temperature)
{
    // When temperature = 0.0f, we should always pick the highest probability token
    // This matches the behavior in SamplingParams constructor
    if (temperature < 1e-3f)
    {
        int32_t idxMax = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        // Return only the highest probability token
        return std::set<int32_t>{idxMax};
    }

    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    std::sort(logitPairs.begin(), logitPairs.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

    // Apply top-k constraint first
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));

    // Extract top-k logits and compute probabilities
    std::vector<float> topKLogits(kLimit);
    for (int32_t i = 0; i < kLimit; ++i)
    {
        topKLogits[i] = logitPairs[i].first;
    }
    auto topKProbs = softmaxRef(topKLogits, temperature);

    // Apply top-p constraint to the top-k elements
    float cumsum = 0.0f;
    int32_t cutoff = kLimit - 1;

    for (int32_t i = 0; i < kLimit; ++i)
    {
        cumsum += topKProbs[i];
        if (cumsum >= topP)
        {
            cutoff = i;
            break;
        }
    }

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i <= cutoff; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::vector<std::pair<float, int32_t>> getTopKElementsRef(std::vector<float> const& logits, int32_t topK)
{
    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + kLimit, logitPairs.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    logitPairs.resize(kLimit);
    return logitPairs;
}

// Unified reference function that handles all cases
std::vector<std::pair<float, int32_t>> returnAllTopKReference(
    std::vector<float> const& input, int32_t topK, bool returnLogProbs, bool normalizeLogProbs, bool inputHasProbs)
{
    // First get the top-k elements from the entire vocabulary
    auto topKElements = getTopKElementsRef(input, topK);

    std::vector<std::pair<float, int32_t>> result;

    if (!returnLogProbs)
    {
        // Return raw values (either logits or probabilities)
        for (auto const& element : topKElements)
        {
            int32_t idx = element.second;
            float value = input[idx];
            result.emplace_back(value, idx);
        }
        return result;
    }

    // Return log probabilities
    if (inputHasProbs)
    {
        // Input is already probabilities
        if (normalizeLogProbs)
        {
            // Normalize over top-k only
            std::vector<float> topKProbs;
            for (auto const& element : topKElements)
            {
                topKProbs.push_back(input[element.second]);
            }

            // Normalize the top-k probabilities
            float sum = 0.0f;
            for (float prob : topKProbs)
            {
                sum += prob;
            }

            for (size_t i = 0; i < topKElements.size(); ++i)
            {
                int32_t idx = topKElements[i].second;
                float normalizedProb = topKProbs[i] / sum;
                float logProb = std::log(normalizedProb);
                result.emplace_back(logProb, idx);
            }
        }
        else
        {
            // Just take log of original probabilities
            for (auto const& element : topKElements)
            {
                int32_t idx = element.second;
                float prob = input[idx];
                float logProb = std::log(prob);
                result.emplace_back(logProb, idx);
            }
        }
    }
    else
    {
        // Input is logits
        // Find max logit among the top-k elements for numerical stability
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (auto const& element : topKElements)
        {
            maxLogit = std::max(maxLogit, element.first);
        }

        if (normalizeLogProbs)
        {
            // Compute sum of exp(logit - maxLogit) for normalization
            float sum = 0.0f;
            for (auto const& element : topKElements)
            {
                sum += std::exp(element.first - maxLogit);
            }

            for (auto const& element : topKElements)
            {
                int32_t idx = element.second;
                float logit = element.first;
                float expLogit = std::exp(logit - maxLogit);
                float logProb = std::log(expLogit) - std::log(sum);
                result.emplace_back(logProb, idx);
            }
        }
        else
        {
            // Just output log(exp(value - maxLogit)) = value - maxLogit
            for (auto const& element : topKElements)
            {
                int32_t idx = element.second;
                float logit = element.first;
                float logProb = logit - maxLogit;
                result.emplace_back(logProb, idx);
            }
        }
    }

    return result;
}

void computeLongRopeReference(std::vector<float>& shortCosSinCache, std::vector<float>& longCosSinCache,
    std::vector<float> const& shortFactor, std::vector<float> const& longFactor, float rotaryBaseFrequency,
    int32_t rotaryDim, int32_t kvCacheCapacity, int32_t rotaryEmbeddingMaxPositions,
    int32_t originalMaxPositionEmbeddings)
{
    float scalingFactor = 1.0f;
    float scale = static_cast<float>(rotaryEmbeddingMaxPositions) / static_cast<float>(originalMaxPositionEmbeddings);
    if (scale > 1.0f)
    {
        scalingFactor = std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(originalMaxPositionEmbeddings)));
    }

    auto initCosSin = [&](std::vector<float> const& extFactors, std::vector<float>& cosSin, int32_t maxPositions) {
        for (int32_t pos = 0; pos < maxPositions; ++pos)
        {
            for (int32_t i = 0; i < rotaryDim / 2; ++i)
            {
                float invFreq = pos / (extFactors[i] * std::pow(rotaryBaseFrequency, 2 * i / float(rotaryDim)));
                float cos = std::cos(invFreq) * scalingFactor;
                float sin = std::sin(invFreq) * scalingFactor;
                cosSin[pos * rotaryDim + i] = cos;
                cosSin[pos * rotaryDim + i + rotaryDim / 2] = sin;
            }
        }
    };

    // LongCosSinCache for context length > originalMaxPositionEmbeddings
    // For all positions, use longFactor to compute cosSinCache
    initCosSin(longFactor, longCosSinCache, kvCacheCapacity);

    // ShortCosSinCache for context length <= originalMaxPositionEmbeddings
    // For positions <= originalMaxPositionEmbeddings, use shortFactor to compute cosSinCache
    // For positions > originalMaxPositionEmbeddings, use longFactor to compute cosSinCache. Copy from longCosSinCache.
    int32_t shortMaxPositions = std::min(originalMaxPositionEmbeddings, kvCacheCapacity);
    initCosSin(shortFactor, shortCosSinCache, shortMaxPositions);
    if (shortMaxPositions < kvCacheCapacity)
    {
        std::copy(longCosSinCache.begin() + shortMaxPositions * rotaryDim, longCosSinCache.end(),
            shortCosSinCache.begin() + shortMaxPositions * rotaryDim);
    }

    return;
}

void computeMRopeReference(std::vector<float>& mropeRotaryCosSin, std::vector<int32_t> const& mropePositionIds,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize)
{
    // mropePositionIds: (bs, 3, maxPositionEmbeddings)
    // mropeRotaryCosSin: (bs, maxPositionEmbeddings, rotaryDim)

    std::vector<float> invFreq;
    for (int i = 0; i < rotaryDim / 2; ++i)
    {
        float value = pow(rotaryBaseFrequency, 2 * i / (float) rotaryDim);
        invFreq.emplace_back(value);
    }

    std::vector<std::vector<float>> cosOri(rotaryEmbeddingMaxPositions, std::vector<float>(rotaryDim / 2));
    std::vector<std::vector<float>> sinOri(rotaryEmbeddingMaxPositions, std::vector<float>(rotaryDim / 2));
    for (int i = 0; i < rotaryEmbeddingMaxPositions; ++i)
    {
        for (int j = 0; j < (rotaryDim / 2); ++j)
        {
            cosOri[i][j] = std::cos(i / invFreq[j]);
            sinOri[i][j] = std::sin(i / invFreq[j]);
        }
    }

    std::vector<int> mRopeSections{0, 16, 40, 64}; // cumsum of {16, 24, 24}
    for (int b = 0; b < batchSize; ++b)
    {
        for (int sec = 0; sec < 3; ++sec)
        {
            for (int i = 0; i < rotaryEmbeddingMaxPositions; ++i)
            {
                int pos = mropePositionIds[b * 3 * rotaryEmbeddingMaxPositions + sec * rotaryEmbeddingMaxPositions + i];
                for (int j = mRopeSections[sec]; j < mRopeSections[sec + 1]; ++j)
                {
                    int cosDstIdx = b * rotaryEmbeddingMaxPositions * rotaryDim + i * rotaryDim + j;
                    int32_t sinOffset = rotaryDim / 2;
                    mropeRotaryCosSin[cosDstIdx] = cosOri[pos][j];
                    mropeRotaryCosSin[cosDstIdx + sinOffset] = sinOri[pos][j];
                }
            }
        }
    }
}

std::vector<half> embeddingLookupRef(std::vector<int32_t> const& inputIds, std::vector<half> const& embeddingTable,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    std::optional<std::vector<half>> const& imageEmbeds, int64_t imageTokenLen)
{
    std::vector<half> result(batchSize * seqLen * hiddenSize, __float2half(0.0f));

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
        {
            int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];
            bool const isImageToken = imageEmbeds.has_value() && tokenId > (vocabSize - 1);

            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = batchIdx * seqLen * hiddenSize + tokenIdx * hiddenSize + elementIdx;

                half embeddingValue;
                if (isImageToken)
                {
                    int32_t const visualTokenId = tokenId - vocabSize;
                    if (visualTokenId >= 0 && visualTokenId < imageTokenLen)
                    {
                        int64_t const imageEmbedIdx = visualTokenId * hiddenSize + elementIdx;
                        embeddingValue = imageEmbeds.value()[imageEmbedIdx];
                    }
                    else
                    {
                        embeddingValue = __float2half(0.0f);
                    }
                }
                else
                {
                    // For normal tokens, check bounds and use zero embedding for out-of-bounds
                    if (tokenId >= 0 && tokenId < vocabSize)
                    {
                        int64_t const embeddingIdx = tokenId * hiddenSize + elementIdx;
                        embeddingValue = embeddingTable[embeddingIdx];
                    }
                    else
                    {
                        embeddingValue = __float2half(0.0f);
                    }
                }

                result[resultIdx] = embeddingValue;
            }
        }
    }

    return result;
}

void assembleDraftTreeDescReference(std::vector<int8_t> const& draftTreeMask,
    std::vector<int32_t> const& draftTreeLength, std::vector<int32_t> const& sequenceStartIndex,
    std::vector<int32_t>& packedDraftTreeMask, std::vector<int32_t>& tensorPositionIndices, int32_t paddedDraftTreeSize)
{
    int32_t const kNUM_MASK_PER_ENTRY{32};
    size_t batchSize = draftTreeLength.size();
    int32_t const packedTreeMaskLen = (paddedDraftTreeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const actualDraftTreeSize = draftTreeLength[batchIdx];
        int32_t const sequenceStartIdx = sequenceStartIndex[batchIdx];

        for (size_t tokenIdx = 0; tokenIdx < actualDraftTreeSize; ++tokenIdx)
        {
            int32_t attendNodeNum = 0;
            int32_t const packedTreeMaskOffset
                = batchIdx * paddedDraftTreeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
            for (size_t i = 0; i <= tokenIdx; ++i)
            {
                int8_t const maskFlag = draftTreeMask[batchIdx * paddedDraftTreeSize * paddedDraftTreeSize
                    + tokenIdx * paddedDraftTreeSize + i];
                if (maskFlag)
                {
                    attendNodeNum += 1;
                    packedDraftTreeMask[packedTreeMaskOffset + i / kNUM_MASK_PER_ENTRY]
                        |= (1 << (i % kNUM_MASK_PER_ENTRY));
                }
            }
            // A token always attend to itself, subtract 1 to reflect its position in the sequence
            int32_t tensorPositionIdx = sequenceStartIdx + attendNodeNum - 1;
            tensorPositionIndices[batchIdx * paddedDraftTreeSize + tokenIdx] = tensorPositionIdx;
        }
    }
}

void prepareEagleDraftProposalMiscInputReference(std::vector<int32_t> const& draftTreeLength,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t selectTokenLength, int32_t paddedDraftTreeSize)
{
    size_t batchSize = draftTreeLength.size();

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const draftTreeSize = draftTreeLength[batchIdx];
        sequenceContextLengths[batchIdx] = sequenceStartIndex[batchIdx] + paddedDraftTreeSize;

        for (size_t i = 0; i < selectTokenLength; ++i)
        {
            selectTokenIndices[batchIdx * selectTokenLength + i] = draftTreeSize - selectTokenLength + i;
        }
    }
}

void prepareEaglePrefillInputReference(
    std::vector<int32_t>& sequenceContextLengths, std::vector<int64_t>& selectTokenIndices, int32_t sequenceLength)
{
    size_t const batchSize = sequenceContextLengths.size();
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        sequenceContextLengths[batchIdx] = sequenceLength;
        selectTokenIndices[batchIdx] = sequenceLength - 1;
    }
}

void prepareEagleAcceptDecodeTokenInputReference(std::vector<int32_t> const& sequenceStartIndices,
    std::vector<int32_t>& packedTreeMask, std::vector<int32_t>& tensorPositionIndices,
    std::vector<int64_t>& selectTokenIndices, std::vector<int32_t>& sequenceContextLengths, int32_t acceptedTokenNum)
{
    size_t const batchSize = sequenceStartIndices.size();
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        // Generate packed tree mask and tensor position indices for each accepted token
        for (int32_t tokenIdx = 0; tokenIdx < acceptedTokenNum; ++tokenIdx)
        {
            int32_t packedTreeMaskValue = 0;
            // Create casual attention mask: each token attends to all previous tokens including itself
            for (int32_t i = 0; i <= tokenIdx; ++i)
            {
                packedTreeMaskValue |= (1 << i);
            }

            int32_t const packedTreeMaskOffset = batchIdx * acceptedTokenNum + tokenIdx;
            packedTreeMask[packedTreeMaskOffset] = packedTreeMaskValue;
            tensorPositionIndices[packedTreeMaskOffset] = sequenceStartIndices[batchIdx] + tokenIdx;
        }

        // Set select token index to the last accepted token
        selectTokenIndices[batchIdx] = acceptedTokenNum - 1;
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + acceptedTokenNum;
    }
}

void prepareEagleBaseTreeDecodingInputReference(std::vector<int8_t> const& baseTreeDecodingMask,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& packedBaseTreeDecodingMask,
    std::vector<int32_t>& tensorPositionIndices, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t treeSize)
{
    // baseTreeDecodingMask: (bs, tree-size, tree-size)
    // sequenceStartIndex: (bs)
    // packedBaseTreeDecodingMask: (bs, tree-size, divup(tree-size, 32))
    // tensorPositionIndices: (bs, tree-size)
    // sequenceContextLengths: (bs)
    // selectTokenIndices: (bs, tree-size)

    int32_t const kNUM_MASK_PER_ENTRY{32};
    size_t batchSize = sequenceStartIndex.size();
    int32_t const packedTreeMaskLen = (treeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const sequenceStartIdx = sequenceStartIndex[batchIdx];
        sequenceContextLengths[batchIdx] = sequenceStartIdx + treeSize;

        for (size_t tokenIdx = 0; tokenIdx < treeSize; ++tokenIdx)
        {
            int32_t attendNodeNum = 0;
            int32_t const packedTreeMaskOffset = batchIdx * treeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
            for (size_t i = 0; i <= tokenIdx; ++i)
            {
                int8_t const maskFlag = baseTreeDecodingMask[batchIdx * treeSize * treeSize + tokenIdx * treeSize + i];
                if (maskFlag)
                {
                    attendNodeNum += 1;
                    packedBaseTreeDecodingMask[packedTreeMaskOffset + i / kNUM_MASK_PER_ENTRY]
                        |= (1 << (i % kNUM_MASK_PER_ENTRY));
                }
            }
            // A token always attend to itself, subtract 1 to reflect its position in the sequence
            int32_t tensorPositionIdx = sequenceStartIdx + attendNodeNum - 1;
            tensorPositionIndices[batchIdx * treeSize + tokenIdx] = tensorPositionIdx;
            selectTokenIndices[batchIdx * treeSize + tokenIdx] = tokenIdx;
        }
    }
}

void eagleBaseCommitKVCacheAndAssembleHiddenStateReference(std::vector<int32_t> const& acceptedIndices,
    std::vector<int32_t> const& acceptLengths, std::vector<half> const& kvCacheBuffer,
    std::vector<int32_t> const& kvCacheLengths, std::vector<half> const& hiddenState,
    std::vector<half>& kvCacheBufferOut, std::vector<half>& hiddenStateOut, int32_t const numLayers,
    int32_t const maxBatchSize, int32_t const numHeads, int32_t const maxSeqLen, int32_t const headDim,
    int32_t const maxDepth, int32_t const draftTreeSize, int32_t const baseHiddenDim)
{
    size_t const activeBatchSize = acceptLengths.size();

    for (int b = 0; b < activeBatchSize; b++)
    {
        int32_t const kvCacheLength = kvCacheLengths[b];
        int32_t const acceptLength = acceptLengths[b];

        // Start from 1 since the root position will always be accepted.
        for (int i = 1; i < acceptLength; i++)
        {
            int32_t const acceptedIdx = acceptedIndices[b * maxDepth + i];
            assert(acceptedIdx >= 0 && acceptedIdx + kvCacheLength < maxSeqLen && acceptedIdx < draftTreeSize
                && "Accepted index out of bounds");

            // kvCacheBuffer: [num-layers, max-batch-size, 2, num-heads, max-seq-len, hidden-size-per-head].
            for (int l = 0; l < numLayers; l++)
            {
                for (int k = 0; k < 2; k++)
                {
                    for (int h = 0; h < numHeads; h++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            int32_t const srcOffset = l * maxBatchSize * 2 * numHeads * maxSeqLen * headDim
                                + b * 2 * numHeads * maxSeqLen * headDim + k * numHeads * maxSeqLen * headDim
                                + h * maxSeqLen * headDim + (kvCacheLength + acceptedIdx) * headDim + d;

                            int32_t const dstOffset = l * maxBatchSize * 2 * numHeads * maxSeqLen * headDim
                                + b * 2 * numHeads * maxSeqLen * headDim + k * numHeads * maxSeqLen * headDim
                                + h * maxSeqLen * headDim + (kvCacheLength + i) * headDim + d;
                            kvCacheBufferOut[dstOffset] = kvCacheBuffer[srcOffset];
                        }
                    }
                }
            }

            // hiddenState: [batch, num-tokens, hidden-dim].
            for (int d = 0; d < baseHiddenDim; d++)
            {
                int32_t const srcOffset = b * draftTreeSize * baseHiddenDim + acceptedIdx * baseHiddenDim + d;
                int32_t const dstOffset = b * draftTreeSize * baseHiddenDim + i * baseHiddenDim + d;
                hiddenStateOut[dstOffset] = hiddenState[srcOffset];
            }
        }
    }
}

// Helper function to compute token depth - count total connections (sum of 1s)
int32_t computeTokenDepthRef(int32_t tokenIdx, std::vector<int8_t> const& attentionMask, int32_t numTokens)
{
    int32_t depth = 0;
    for (int32_t i = 0; i < numTokens; ++i)
    {
        if (attentionMask[tokenIdx * numTokens + i] == 1)
        {
            depth++;
        }
    }
    return depth;
}

EagleAcceptResult eagleAcceptRef(std::vector<float> const& logits, std::vector<int32_t> const& tokenIds,
    std::vector<int8_t> const& attentionMask, int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth)
{
    EagleAcceptResult result;
    result.acceptedTokenIds.resize(batchSize * maxDepth, -1);
    result.acceptedLogitsIndices.resize(batchSize * maxDepth, -1);
    result.acceptLengths.resize(batchSize, 0);

    int32_t maxAcceptLength = 0;

    // Process each batch
    for (int32_t b = 0; b < batchSize; ++b)
    {
        // Precompute token depths for this batch
        std::vector<int32_t> tokenDepths(numTokens);
        int32_t const batchMaskOffset = b * numTokens * numTokens;
        int32_t const batchTokenOffset = b * numTokens;
        for (int32_t i = 0; i < numTokens; ++i)
        {
            int32_t depth = 0;
            for (int32_t j = 0; j < numTokens; ++j)
            {
                if (attentionMask[batchMaskOffset + i * numTokens + j] == 1)
                {
                    depth++;
                }
            }
            tokenDepths[i] = depth;
        }

        int32_t currentDepth = 0;
        int32_t currentTokenIdx = 0;                    // Start with token[0], use logits[0] to predict next
        int32_t expectedNextDepth = tokenDepths[0] + 1; // Next depth should be current token's depth + 1

        // Start with accept length 0, will be set to at least 1 in the loop
        result.acceptLengths[b] = 0;

        // Process tokens - always accept at least one
        for (int32_t step = 0; step < maxDepth && currentTokenIdx < numTokens; ++step)
        {
            // Step 1: Find top-1 token from logits[b][currentTokenIdx]
            int32_t const logitsOffset = b * numTokens * vocabSize + currentTokenIdx * vocabSize;

            // Find argmax with consistent tie-breaking (prefer lower indices)
            float maxLogit = -std::numeric_limits<float>::infinity();
            int32_t selectedTokenId = 0; // Default to token 0 for consistent behavior

            for (int32_t v = 0; v < vocabSize; ++v)
            {
                if (logits[logitsOffset + v] > maxLogit)
                {
                    maxLogit = logits[logitsOffset + v];
                    selectedTokenId = v;
                }
            }

            // Step 2: Always accept the selected token
            result.acceptedTokenIds[b * maxDepth + currentDepth] = selectedTokenId;
            result.acceptedLogitsIndices[b * maxDepth + currentDepth] = currentTokenIdx;
            result.acceptLengths[b] = currentDepth + 1;
            currentDepth++;

            // Step 3: Check if the selected token exists in the tree to continue
            int32_t nextTokenIdx = -1;

            for (int32_t checkIdx = 1; checkIdx < numTokens; ++checkIdx)
            {
                if (tokenIds[batchTokenOffset + checkIdx] == selectedTokenId
                    && tokenDepths[checkIdx] == expectedNextDepth)
                {
                    // Check attention mask: does checkIdx attend to currentTokenIdx?
                    int32_t maskOffset = batchMaskOffset + checkIdx * numTokens + currentTokenIdx;
                    if (attentionMask[maskOffset] == 1)
                    {
                        // Found a valid next token in tree
                        nextTokenIdx = checkIdx;
                        break; // Take the first match
                    }
                }
            }

            // Step 4: Update for next iteration
            if (nextTokenIdx != -1)
            {
                // Found valid next token in tree, continue from there
                currentTokenIdx = nextTokenIdx;
                expectedNextDepth++;
            }
            else
            {
                // No valid next token found in tree, stop
                break;
            }
        }

        // Ensure at least 1 token is always accepted
        if (result.acceptLengths[b] == 0)
        {
            // This should never happen, but as a safety net, accept the first predicted token
            int32_t const logitsOffset = b * numTokens * vocabSize + 0 * vocabSize;
            float maxLogit = -std::numeric_limits<float>::infinity();
            int32_t selectedTokenId = 0; // Default to token 0 for consistent behavior
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                if (logits[logitsOffset + v] > maxLogit)
                {
                    maxLogit = logits[logitsOffset + v];
                    selectedTokenId = v;
                }
            }
            result.acceptedTokenIds[b * maxDepth + 0] = selectedTokenId;
            result.acceptedLogitsIndices[b * maxDepth + 0] = 0;
            result.acceptLengths[b] = 1;
        }

        // Update max accept length
        maxAcceptLength = std::max(maxAcceptLength, result.acceptLengths[b]);
    }

    // Ensure maxAcceptLength is at least 1
    maxAcceptLength = std::max(maxAcceptLength, 1);
    result.maxAcceptLength = maxAcceptLength;

    // Reshape the result vectors to [batchSize, maxAcceptLength]
    std::vector<int32_t> reshapedTokenIds(batchSize * maxAcceptLength, -1);
    std::vector<int32_t> reshapedLogitsIndices(batchSize * maxAcceptLength, -1);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < result.acceptLengths[b] && i < maxAcceptLength; ++i)
        {
            reshapedTokenIds[b * maxAcceptLength + i] = result.acceptedTokenIds[b * maxDepth + i];
            reshapedLogitsIndices[b * maxAcceptLength + i] = result.acceptedLogitsIndices[b * maxDepth + i];
        }
    }

    result.acceptedTokenIds = std::move(reshapedTokenIds);
    result.acceptedLogitsIndices = std::move(reshapedLogitsIndices);

    return result;
}

void transposeToPatchQwenReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const T, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const temporalPatchSize, int32_t const patchSize, int32_t const mergeSize)
{
    assert(originalImage.size() == T * height * width * channels);
    assert(patch.size() == T * height * width * channels);

    int const gridT = T / temporalPatchSize;
    int const gridH = height / (mergeSize * patchSize);
    int const gridW = width / (mergeSize * patchSize);

    for (int gt = 0; gt < gridT; ++gt)
    {
        for (int gh = 0; gh < gridH; ++gh)
        {
            for (int gw = 0; gw < gridW; ++gw)
            {
                for (int mergeH = 0; mergeH < mergeSize; ++mergeH)
                {
                    for (int mergeW = 0; mergeW < mergeSize; ++mergeW)
                    {
                        for (int c = 0; c < channels; ++c)
                        {
                            for (int t = 0; t < temporalPatchSize; ++t)
                            {
                                for (int patchH = 0; patchH < patchSize; ++patchH)
                                {
                                    for (int patchW = 0; patchW < patchSize; ++patchW)
                                    {
                                        // src dimensions: (T, H, W, C) => (gridT, temporalPatchSize, gridH, mergeSize,
                                        // patchSize, gridW, mergeSize, patchSize, C)
                                        int originalT = gt * temporalPatchSize + t;
                                        int originalH = gh * mergeSize * patchSize + mergeH * patchSize + patchH;
                                        int originalW = gw * mergeSize * patchSize + mergeW * patchSize + patchW;
                                        half value = originalImage[originalT * height * width * channels
                                            + originalH * width * channels + originalW * channels + c];

                                        // dst dimensions: (gridT, gridH, gridW, mergeSize, mergeSize) x (channels,
                                        // temporalPatchSize, patchSize, patchSize)
                                        int dstHW = gt * gridH * gridW * mergeSize * mergeSize
                                            + gh * gridW * mergeSize * mergeSize + gw * mergeSize * mergeSize
                                            + mergeH * mergeSize + mergeW;
                                        int dstDim = c * temporalPatchSize * patchSize * patchSize
                                            + t * patchSize * patchSize + patchH * patchSize + patchW;
                                        patch[dstHW * channels * temporalPatchSize * patchSize * patchSize + dstDim]
                                            = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void transposeToPatchInternVLReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const blockSizeH, int32_t const blockSizeW)
{
    assert(originalImage.size() == height * width * channels);
    assert(patch.size() == height * width * channels);

    for (int gridH = 0; gridH < height / blockSizeH; ++gridH)
    {
        for (int gridW = 0; gridW < width / blockSizeW; ++gridW)
        {
            for (int blockH = 0; blockH < blockSizeH; ++blockH)
            {
                for (int blockW = 0; blockW < blockSizeW; ++blockW)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        // src dimensions: (H, W, C) => (gridH, blockSizeH, gridW, blockSizeW, C)
                        int originalH = gridH * blockSizeH + blockH;
                        int originalW = gridW * blockSizeW + blockW;
                        half value = originalImage[originalH * width * channels + originalW * channels + c];

                        // dst dimensions: (gridH*gridW, C, blockSizeH, blockSizeW)
                        int dstNumBlocks = gridH * (width / blockSizeW) + gridW;
                        patch[dstNumBlocks * channels * blockSizeH * blockSizeW + c * blockSizeH * blockSizeW
                            + blockH * blockSizeW + blockW]
                            = value;
                    }
                }
            }
        }
    }
}
