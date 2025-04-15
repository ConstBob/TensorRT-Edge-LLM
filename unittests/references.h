#pragma once

#include <vector>
#include <cuda_fp16.h>


inline std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k, std::vector<half> const& v,
                                            int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
                                            std::optional<std::vector<int32_t>> const& treeAttnMask = std::nullopt)
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