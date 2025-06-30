#pragma once

#include <cuda_fp16.h>
#include <vector>
#include <optional>
#include <cstdint>

std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k,
    std::vector<half> const& v, int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, std::optional<std::vector<int32_t>> const& treeAttnMask = std::nullopt);

std::vector<half> ropeRef(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const seqIdx, float const ropeScale, float const ropeTheta, bool const permute);

std::vector<half> ropeRefCosSin(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    std::vector<float> const& cos, std::vector<float> const& sin, bool const permute);
