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
    float invTemp = (temperature == 0.0f) ? 0.0f : 1.0f / temperature;

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

void computeMRopeReference(std::vector<float>& mropeRotaryCosSin, std::vector<int64_t> const& mropePositionIds,
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
