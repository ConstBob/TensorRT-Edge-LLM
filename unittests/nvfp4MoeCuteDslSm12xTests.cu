/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeRunner.h"
#include "testUtils.h"

using namespace trt_edgellm;

namespace
{

// ---------------------------------------------------------------------------
// SM12x NVFP4 fused MoE runner constants. The geometry (FP4 levels, FP8 SF
// vector size, 6D MMA swizzle, alpha-with-scale dequant) is shared with the
// SM110 plugin; the GeForce kernel differs in the FC1 SwiGLU output layout
// (plain [up_all, gate_all] concat -- see _concat_qwen3_swiglu_fc1 in
// tensorrt_edgellm/checkpoint/repacking.py).
// ---------------------------------------------------------------------------
constexpr int32_t kSfVecSize = 16;
constexpr int32_t kRowTile = 128;
constexpr int32_t kNumExperts = 128;
constexpr int32_t kTopK = 8;

constexpr std::array<float, 16> kFp4Levels{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ---------------------------------------------------------------------------
// FP8 E4M3 helpers (match torch.float8_e4m3fn round-to-nearest-even encoding).
// ---------------------------------------------------------------------------
inline float fp8BytesToFloat(uint8_t raw)
{
    __nv_fp8_e4m3 fp8;
    fp8.__x = raw;
    return static_cast<float>(static_cast<__half>(fp8));
}

inline uint8_t floatToFp8Byte(float value)
{
    __nv_fp8_e4m3 fp8(value);
    return fp8.__x;
}

// ---------------------------------------------------------------------------
// 6D NVFP4 scale-factor layout helpers (same swizzle as SM110).
// ---------------------------------------------------------------------------
struct ScaleShape
{
    int32_t experts;
    int32_t mOuter;
    int32_t kOuter;
    int32_t inner0; // 32
    int32_t inner1; // 4
    int32_t inner2; // 4
    int64_t volume() const
    {
        return static_cast<int64_t>(experts) * mOuter * kOuter * inner0 * inner1 * inner2;
    }
};

inline ScaleShape scaleShape(int32_t rows, int32_t cols, int32_t experts = kNumExperts)
{
    int32_t const sfCols = cols / kSfVecSize;
    int32_t const mOuter = (rows + kRowTile - 1) / kRowTile;
    int32_t const kOuter = (sfCols + 3) / 4;
    return {experts, mOuter, kOuter, 32, 4, 4};
}

inline int64_t atomOffset(int32_t row, int32_t sfCol, int32_t sfCols)
{
    int64_t const innerK = sfCol % 4;
    int64_t const innerM = (row % kRowTile) / 32;
    int64_t const outerM = row % 32;
    int64_t const kTile = sfCol / 4;
    int64_t const numKTiles = (sfCols + 3) / 4;
    int64_t const mTile = row / kRowTile;
    return mTile * numKTiles * 512 + kTile * 512 + outerM * 16 + innerM * 4 + innerK;
}

// ---------------------------------------------------------------------------
// Deterministic test-input generators (seeded PRNG so the C++ reference always
// reproduces the same bytes the kernel sees).
// ---------------------------------------------------------------------------
std::vector<uint8_t> makeQWeights(std::mt19937& rng, size_t numBytes)
{
    std::vector<uint8_t> out(numBytes);
    std::uniform_int_distribution<int32_t> dist(0, 255);
    for (size_t i = 0; i < numBytes; ++i)
    {
        out[i] = static_cast<uint8_t>(dist(rng));
    }
    return out;
}

std::vector<uint8_t> makeScaleTensorUniform(int32_t rows, int32_t cols, int32_t experts = kNumExperts)
{
    ScaleShape const shape = scaleShape(rows, cols, experts);
    std::vector<uint8_t> out(static_cast<size_t>(shape.volume()), 0);
    uint8_t const byteValue = floatToFp8Byte(0.0078125f);
    std::fill(out.begin(), out.end(), byteValue);
    return out;
}

// ---------------------------------------------------------------------------
// FP4 quantize / dequantize helpers.
// ---------------------------------------------------------------------------
inline uint8_t roundFp4(float value)
{
    uint8_t best = 0;
    float bestDist = std::fabs(value - kFp4Levels[0]);
    for (uint8_t i = 1; i < kFp4Levels.size(); ++i)
    {
        float const dist = std::fabs(value - kFp4Levels[i]);
        if (dist < bestDist)
        {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

std::vector<float> fp4RoundtripLinearSfRaw(float const* values, int32_t rows, int32_t cols, float globalScale)
{
    std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
    float const scale = std::max(globalScale, 1e-12f);
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t begin = 0; begin < cols; begin += kSfVecSize)
        {
            float const* block = values + static_cast<size_t>(row) * cols + begin;
            float vecMax = 0.0f;
            for (int32_t i = 0; i < kSfVecSize; ++i)
            {
                vecMax = std::max(vecMax, std::fabs(block[i]));
            }
            if (vecMax == 0.0f)
            {
                continue;
            }
            float const sfValue = (vecMax / 6.0f) / scale;
            float const sfBack = fp8BytesToFloat(floatToFp8Byte(sfValue));
            float const effectiveScale = sfBack * scale;
            if (effectiveScale <= 0.0f || !std::isfinite(effectiveScale))
            {
                continue;
            }
            for (int32_t i = 0; i < kSfVecSize; ++i)
            {
                uint8_t const code = roundFp4(block[i] / effectiveScale);
                out[static_cast<size_t>(row) * cols + begin + i] = kFp4Levels[code] * sfBack;
            }
        }
    }
    return out;
}

std::vector<float> dequantWeight(
    uint8_t const* qWeights, uint8_t const* scale6d, int32_t expert, int32_t rows, int32_t cols, float alpha)
{
    int32_t const sfCols = cols / kSfVecSize;
    size_t const packedPerExpert = static_cast<size_t>(rows) * (cols / 2);
    size_t const sfPerExpert = static_cast<size_t>(scaleShape(rows, cols).volume()) / kNumExperts;
    uint8_t const* packed = qWeights + static_cast<size_t>(expert) * packedPerExpert;
    uint8_t const* sfBase = scale6d + static_cast<size_t>(expert) * sfPerExpert;

    std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t col = 0; col < cols; col += 2)
        {
            uint8_t const byte = packed[static_cast<size_t>(row) * (cols / 2) + col / 2];
            uint8_t const lo = byte & 0x0Fu;
            uint8_t const hi = (byte >> 4) & 0x0Fu;
            float const sfLo = fp8BytesToFloat(sfBase[atomOffset(row, col / kSfVecSize, sfCols)]);
            float const sfHi = fp8BytesToFloat(sfBase[atomOffset(row, (col + 1) / kSfVecSize, sfCols)]);
            out[static_cast<size_t>(row) * cols + col] = kFp4Levels[lo] * sfLo * alpha;
            out[static_cast<size_t>(row) * cols + col + 1] = kFp4Levels[hi] * sfHi * alpha;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Activations.
// ---------------------------------------------------------------------------
inline float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

// GeForce SwiGLU FC1 layout: projection[:I] are the "up" rows, projection[I:2I]
// are the "gate" rows (plain concat -- NOT the SM110 64-row interleave).
// Matches _concat_qwen3_swiglu_fc1 in tensorrt_edgellm/checkpoint/repacking.py.
std::vector<float> applySwigluConcat(float const* projection, int32_t intermediateSize)
{
    std::vector<float> out(static_cast<size_t>(intermediateSize), 0.0f);
    for (int32_t i = 0; i < intermediateSize; ++i)
    {
        out[i] = projection[i] * silu(projection[i + intermediateSize]);
    }
    return out;
}

// ReLU2: activation[i] = max(projection[i], 0)^2. FC1 output is [I] for ReLU2
// (no up/gate doubling) -- mirrors the Python reference.
std::vector<float> applyReLU2(float const* projection, int32_t intermediateSize)
{
    std::vector<float> out(static_cast<size_t>(intermediateSize), 0.0f);
    for (int32_t i = 0; i < intermediateSize; ++i)
    {
        float const v = std::max(projection[i], 0.0f);
        out[i] = v * v;
    }
    return out;
}

constexpr int32_t kActivationTypeSwiGLU = 2;
constexpr int32_t kActivationTypeReLU2 = 4;

inline int32_t fc1InputN(int32_t intermediateSize, int32_t activationType)
{
    return activationType == kActivationTypeSwiGLU ? 2 * intermediateSize : intermediateSize;
}

inline CuteDslMoeActivation toRunnerActivation(int32_t activationType)
{
    switch (activationType)
    {
    case kActivationTypeSwiGLU: return CuteDslMoeActivation::kSwiGLU;
    case kActivationTypeReLU2: return CuteDslMoeActivation::kReLU2;
    default: return CuteDslMoeActivation::kSwiGLU;
    }
}

std::vector<float> applyFc1Activation(float const* projection, int32_t intermediateSize, int32_t activationType)
{
    switch (activationType)
    {
    case kActivationTypeSwiGLU: return applySwigluConcat(projection, intermediateSize);
    case kActivationTypeReLU2: return applyReLU2(projection, intermediateSize);
    default: throw std::runtime_error("unsupported activation_type in SM12x test reference");
    }
}

// ---------------------------------------------------------------------------
// Test case definition + deterministic generator (mirrors the SM110 helpers
// but the SwiGLU layout is the plain GeForce concat).
// ---------------------------------------------------------------------------
struct MoeCase
{
    std::string name;
    int32_t numTokens;
    int32_t hiddenSize;
    int32_t intermediateSize;
    int32_t activationType; // 2 = SwiGLU; only SwiGLU is exercised here.
    CuteDslMoeBackend backend;
    uint64_t seed;
};

struct CaseData
{
    MoeCase config;
    int32_t n1;
    std::vector<__half> hiddenFp16;   // [T, H]
    std::vector<int32_t> topkIds;     // [T, topK]
    std::vector<float> topkWeights;   // [T, topK]
    std::vector<uint8_t> fc1QWeights; // [E, n1, H/2]
    std::vector<uint8_t> fc1Scale;
    std::vector<uint8_t> fc2QWeights; // [E, H, I/2]
    std::vector<uint8_t> fc2Scale;
    std::vector<float> fc1Alpha;
    std::vector<float> fc2Alpha;
    std::vector<float> inputGlobalScale;
    std::vector<float> downInputScale;
};

CaseData buildCase(MoeCase const& cfg)
{
    CaseData c{};
    c.config = cfg;
    c.n1 = fc1InputN(cfg.intermediateSize, cfg.activationType);

    std::mt19937 rng(cfg.seed);

    c.hiddenFp16.resize(static_cast<size_t>(cfg.numTokens) * cfg.hiddenSize);
    {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        float maxAbs = 0.0f;
        for (size_t i = 0; i < c.hiddenFp16.size(); ++i)
        {
            float const v = dist(rng);
            c.hiddenFp16[i] = __float2half_rn(v);
            maxAbs = std::max(maxAbs, std::fabs(v));
        }
        float const inputFloor = std::max(maxAbs / (448.0f * 6.0f), 1e-12f);
        c.inputGlobalScale.assign(kNumExperts, inputFloor);
    }

    c.topkIds.resize(static_cast<size_t>(cfg.numTokens) * kTopK);
    c.topkWeights.resize(static_cast<size_t>(cfg.numTokens) * kTopK);
    {
        std::vector<int32_t> pool(kNumExperts);
        for (int32_t i = 0; i < kNumExperts; ++i)
        {
            pool[i] = i;
        }
        std::uniform_real_distribution<float> wDist(0.05f, 1.0f);
        for (int32_t t = 0; t < cfg.numTokens; ++t)
        {
            std::shuffle(pool.begin(), pool.end(), rng);
            float wSum = 0.0f;
            for (int32_t k = 0; k < kTopK; ++k)
            {
                c.topkIds[static_cast<size_t>(t) * kTopK + k] = pool[k];
                float const w = wDist(rng);
                c.topkWeights[static_cast<size_t>(t) * kTopK + k] = w;
                wSum += w;
            }
            for (int32_t k = 0; k < kTopK; ++k)
            {
                c.topkWeights[static_cast<size_t>(t) * kTopK + k] /= wSum;
            }
        }
    }

    c.fc1QWeights = makeQWeights(rng, static_cast<size_t>(kNumExperts) * c.n1 * (cfg.hiddenSize / 2));
    c.fc2QWeights = makeQWeights(rng, static_cast<size_t>(kNumExperts) * cfg.hiddenSize * (cfg.intermediateSize / 2));
    c.fc1Scale = makeScaleTensorUniform(c.n1, cfg.hiddenSize);
    c.fc2Scale = makeScaleTensorUniform(cfg.hiddenSize, cfg.intermediateSize);
    c.fc1Alpha.assign(kNumExperts, 0.85f);
    c.fc2Alpha.assign(kNumExperts, 0.75f);
    c.downInputScale.assign(kNumExperts, 1.0e-4f);
    return c;
}

// ---------------------------------------------------------------------------
// CPU reference: float[T, H]. For each (token, top-K slot) the path is
// FP4-quantize hidden -> FC1 (dequantized weight @ hidden) * input_global_scale
// -> SwiGLU (plain [up_all, gate_all] concat) or ReLU2 -> FP4-quantize
// activation -> FC2 -> router-weighted scatter-add. FP4 / FP8 SF / alpha math
// matches the SM12x fused wrappers so the kernel and reference agree up to
// FP4 rounding noise.
// ---------------------------------------------------------------------------
std::vector<float> computeReference(CaseData const& c)
{
    int32_t const T = c.config.numTokens;
    int32_t const H = c.config.hiddenSize;
    int32_t const I = c.config.intermediateSize;
    int32_t const n1 = c.n1;

    std::vector<float> hiddenFp32(static_cast<size_t>(T) * H);
    for (size_t i = 0; i < hiddenFp32.size(); ++i)
    {
        hiddenFp32[i] = __half2float(c.hiddenFp16[i]);
    }

    std::vector<float> output(static_cast<size_t>(T) * H, 0.0f);

    for (int32_t t = 0; t < T; ++t)
    {
        for (int32_t slot = 0; slot < kTopK; ++slot)
        {
            int32_t const expert = c.topkIds[static_cast<size_t>(t) * kTopK + slot];
            float const routerWeight = c.topkWeights[static_cast<size_t>(t) * kTopK + slot];
            float const inputScale = c.inputGlobalScale[expert];
            float const downScale = c.downInputScale[expert];

            std::vector<float> hiddenRaw
                = fp4RoundtripLinearSfRaw(hiddenFp32.data() + static_cast<size_t>(t) * H, 1, H, inputScale);

            std::vector<float> fc1W
                = dequantWeight(c.fc1QWeights.data(), c.fc1Scale.data(), expert, n1, H, c.fc1Alpha[expert]);
            std::vector<float> projection(n1, 0.0f);
            for (int32_t row = 0; row < n1; ++row)
            {
                float acc = 0.0f;
                float const* wRow = fc1W.data() + static_cast<size_t>(row) * H;
                for (int32_t k = 0; k < H; ++k)
                {
                    acc += wRow[k] * hiddenRaw[k];
                }
                projection[row] = acc * inputScale;
            }

            std::vector<float> activated = applyFc1Activation(projection.data(), I, c.config.activationType);

            std::vector<float> activatedRaw = fp4RoundtripLinearSfRaw(activated.data(), 1, I, downScale);
            std::vector<float> fc2W
                = dequantWeight(c.fc2QWeights.data(), c.fc2Scale.data(), expert, H, I, c.fc2Alpha[expert]);
            float* outRow = output.data() + static_cast<size_t>(t) * H;
            for (int32_t row = 0; row < H; ++row)
            {
                float acc = 0.0f;
                float const* wRow = fc2W.data() + static_cast<size_t>(row) * I;
                for (int32_t k = 0; k < I; ++k)
                {
                    acc += wRow[k] * activatedRaw[k];
                }
                outRow[row] += routerWeight * downScale * acc;
            }
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Runner driver: upload inputs, run the kernel, return FP32 output [T*H].
// ---------------------------------------------------------------------------
struct RunResult
{
    int32_t status;
    std::vector<float> outputFp32;
};

RunResult runCase(CaseData const& c)
{
    using rt::Coords;
    using rt::DeviceType;
    int32_t const T = c.config.numTokens;
    int32_t const H = c.config.hiddenSize;
    int32_t const I = c.config.intermediateSize;
    int32_t const n1 = c.n1;

    rt::Tensor hidden({T, H}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor topkIds({T, kTopK}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor topkWeights({T, kTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fc1Q({kNumExperts, n1, H / 2}, DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor fc2Q({kNumExperts, H, I / 2}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    auto const fc1Shape = scaleShape(n1, H);
    auto const fc2Shape = scaleShape(H, I);
    rt::Tensor fc1Scale(
        {fc1Shape.experts, fc1Shape.mOuter, fc1Shape.kOuter, fc1Shape.inner0, fc1Shape.inner1, fc1Shape.inner2},
        DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor fc2Scale(
        {fc2Shape.experts, fc2Shape.mOuter, fc2Shape.kOuter, fc2Shape.inner0, fc2Shape.inner1, fc2Shape.inner2},
        DeviceType::kGPU, nvinfer1::DataType::kINT8);

    rt::Tensor fc1Alpha({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fc2Alpha({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor inputScale({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor downScale({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor output({T, H}, DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Identity tables required by the SM12x runner ABI: weightExpertIds and
    // globalToLocalExpertIds must be [0, 1, ..., E-1] and MUST NOT alias each
    // other (the kernel collapses cos_sim if they alias -- see runner header).
    rt::Tensor weightExpertIds({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor globalToLocalExpertIds({kNumExperts}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    std::vector<int32_t> identity(kNumExperts);
    for (int32_t i = 0; i < kNumExperts; ++i)
    {
        identity[i] = i;
    }

    CUDA_CHECK(cudaMemcpy(
        hidden.rawPointer(), c.hiddenFp16.data(), c.hiddenFp16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(topkIds.rawPointer(), c.topkIds.data(), c.topkIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        topkWeights.rawPointer(), c.topkWeights.data(), c.topkWeights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc1Q.rawPointer(), c.fc1QWeights.data(), c.fc1QWeights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc2Q.rawPointer(), c.fc2QWeights.data(), c.fc2QWeights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc1Scale.rawPointer(), c.fc1Scale.data(), c.fc1Scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc2Scale.rawPointer(), c.fc2Scale.data(), c.fc2Scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        fc1Alpha.rawPointer(), c.fc1Alpha.data(), c.fc1Alpha.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        fc2Alpha.rawPointer(), c.fc2Alpha.data(), c.fc2Alpha.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputScale.rawPointer(), c.inputGlobalScale.data(), c.inputGlobalScale.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(downScale.rawPointer(), c.downInputScale.data(), c.downInputScale.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weightExpertIds.rawPointer(), identity.data(), identity.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(globalToLocalExpertIds.rawPointer(), identity.data(), identity.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(output.rawPointer(), 0, static_cast<size_t>(T) * H * sizeof(__half)));

    int32_t const maxRoutedRows = T * kTopK;
    size_t const workspaceBytes = CuteDslNvfp4MoeRunner::getWorkspaceSize(T, maxRoutedRows, kNumExperts, kTopK, H, I);
    EXPECT_GT(workspaceBytes, 0u);
    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));
    auto workspaceGuard = Defer([&] { cudaFree(workspace); });

    CuteDslNvfp4MoeParams params{};
    params.numTokens = T;
    params.numExperts = kNumExperts;
    params.topK = kTopK;
    params.hiddenSize = H;
    params.moeInterSize = I;
    params.maxRoutedRows = maxRoutedRows;
    params.hiddenStates = hidden.rawPointer();
    params.topkIds = topkIds.dataPointer<int32_t>();
    params.topkWeights = topkWeights.dataPointer<float>();
    params.fc1QWeights = fc1Q.rawPointer();
    params.fc1BlocksScale = fc1Scale.rawPointer();
    params.fc1Alpha = fc1Alpha.dataPointer<float>();
    params.fc2QWeights = fc2Q.rawPointer();
    params.fc2BlocksScale = fc2Scale.rawPointer();
    params.fc2Alpha = fc2Alpha.dataPointer<float>();
    params.inputGlobalScale = inputScale.dataPointer<float>();
    params.downInputScale = downScale.dataPointer<float>();
    params.weightExpertIds = weightExpertIds.dataPointer<int32_t>();
    params.globalToLocalExpertIds = globalToLocalExpertIds.dataPointer<int32_t>();
    params.output = output.rawPointer();
    params.activation = toRunnerActivation(c.config.activationType);
    params.ioDtype = CuteDslMoeIoDtype::kFP16;
    params.backend = c.config.backend;

    cudaStream_t stream = nullptr;
    int32_t const ret = CuteDslNvfp4MoeRunner{}.run(params, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> hostFp16(static_cast<size_t>(T) * H);
    CUDA_CHECK(
        cudaMemcpy(hostFp16.data(), output.rawPointer(), hostFp16.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> hostFp32(hostFp16.size());
    for (size_t i = 0; i < hostFp16.size(); ++i)
    {
        hostFp32[i] = __half2float(hostFp16[i]);
    }
    return {ret, std::move(hostFp32)};
}

// ---------------------------------------------------------------------------
// Per-token median-cosine + magnitude-ratio summary.
// ---------------------------------------------------------------------------
struct Summary
{
    double medianCosine;
    double magRatio;
    double maxAbs;
};

Summary summarize(std::vector<float> const& got, std::vector<float> const& ref, int32_t T, int32_t H)
{
    std::vector<double> cosines;
    cosines.reserve(static_cast<size_t>(T));
    double gotNorm2 = 0.0;
    double refNorm2 = 0.0;
    double maxAbs = 0.0;
    for (int32_t t = 0; t < T; ++t)
    {
        double dot = 0.0;
        double gn = 0.0;
        double rn = 0.0;
        for (int32_t h = 0; h < H; ++h)
        {
            float const g = got[static_cast<size_t>(t) * H + h];
            float const r = ref[static_cast<size_t>(t) * H + h];
            dot += static_cast<double>(g) * static_cast<double>(r);
            gn += static_cast<double>(g) * static_cast<double>(g);
            rn += static_cast<double>(r) * static_cast<double>(r);
            maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(g - r)));
        }
        gotNorm2 += gn;
        refNorm2 += rn;
        double const denom = std::sqrt(gn) * std::sqrt(rn);
        cosines.push_back(denom > 0.0 ? dot / std::max(denom, 1e-30) : 1.0);
    }
    std::sort(cosines.begin(), cosines.end());
    double const median = cosines.empty()
        ? 1.0
        : (cosines.size() % 2 == 0 ? 0.5 * (cosines[cosines.size() / 2 - 1] + cosines[cosines.size() / 2])
                                   : cosines[cosines.size() / 2]);
    double const mag = std::sqrt(refNorm2) > 0.0 ? std::sqrt(gotNorm2) / std::max(std::sqrt(refNorm2), 1e-30) : 0.0;
    return {median, mag, maxAbs};
}

bool isSupportedSm()
{
    int32_t const sm = getSMVersion();
    return sm == 120 || sm == 121;
}

bool checkRequirementsAndLoad()
{
    if (!isSupportedSm())
    {
        return false;
    }
    int32_t const sm = getSMVersion();
    if (!CuteDslNvfp4MoeRunner::canImplement(/*hiddenSize=*/1024, /*moeInterSize=*/768, kNumExperts, kTopK, sm,
            CuteDslMoeActivation::kSwiGLU, CuteDslMoeIoDtype::kFP16, CuteDslMoeBackend::kAuto))
    {
        return false;
    }
    return CuteDslNvfp4MoeRunner::loadKernelModules();
}

std::vector<MoeCase> defaultCases()
{
    return {
        // num_tokens=1, kAuto -> decode backend (T*K=8 well under the 640
        // cutover in CuteDslNvfp4MoeRunner::resolveBackend).
        {/*name=*/"decode_h1024_i768_t1_swiglu", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActivationTypeSwiGLU,
            /*backend=*/CuteDslMoeBackend::kAuto, /*seed=*/0xC0FFEEu},
        // num_tokens=8 with explicit kPrefill so we exercise the prefill
        // backend at a small (cheap-reference) token count rather than
        // waiting for kAuto's 640-row cutover.
        {/*name=*/"prefill_h1024_i768_t8_swiglu", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActivationTypeSwiGLU,
            /*backend=*/CuteDslMoeBackend::kPrefill, /*seed=*/0xDEADBEEFu},
        // ReLU2 path: fc1InputN = I (no up/gate doubling), reference uses
        // square(max(x, 0)). canImplement enforces moeInterSize % kLevelTileN(128) == 0
        // -> I=768 ok.
        {/*name=*/"decode_h1024_i768_t1_relu2", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActivationTypeReLU2,
            /*backend=*/CuteDslMoeBackend::kAuto, /*seed=*/0xFEEDFACEu},
        {/*name=*/"prefill_h1024_i768_t8_relu2", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActivationTypeReLU2,
            /*backend=*/CuteDslMoeBackend::kPrefill, /*seed=*/0xBADCAFEu},
    };
}

} // namespace

TEST(CuteDslNvfp4MoeSm12xTest, smoke)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "SM12x NVFP4 fused MoE CuTeDSL runner test requires Spark/GB10 (SM120/SM121), got SM=" << sm;
    }
    if (!checkRequirementsAndLoad())
    {
        GTEST_SKIP() << "Failed to load SM12x NVFP4 fused MoE CuTeDSL kernel modules or canImplement returned false";
    }

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData c = buildCase(cfg);
        RunResult const result = runCase(c);
        ASSERT_EQ(result.status, 0) << "runner returned non-zero status for " << cfg.name;
        ASSERT_EQ(result.outputFp32.size(), static_cast<size_t>(cfg.numTokens) * cfg.hiddenSize);
        bool allZero = true;
        for (float v : result.outputFp32)
        {
            ASSERT_TRUE(std::isfinite(v)) << "non-finite output in " << cfg.name;
            if (v != 0.0f)
            {
                allZero = false;
            }
        }
        ASSERT_FALSE(allZero) << "output is all-zero for " << cfg.name;
    }
}

TEST(CuteDslNvfp4MoeSm12xTest, accuracy)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "SM12x NVFP4 fused MoE CuTeDSL runner test requires Spark/GB10 (SM120/SM121), got SM=" << sm;
    }
    if (!checkRequirementsAndLoad())
    {
        GTEST_SKIP() << "Failed to load SM12x NVFP4 fused MoE CuTeDSL kernel modules or canImplement returned false";
    }

    // Loose-but-meaningful bands for NVFP4 MoE: cosine >= 0.94 catches sign /
    // routing / activation bugs; magnitude ratio in [0.25, 3.0] catches gross
    // scale drift (per-block FP8 SF, per-expert alpha, FC1/FC2 global scales)
    // without flagging the expected FP4-quantization noise floor.
    constexpr double kMinCosine = 0.94;
    constexpr double kMinMagRatio = 0.25;
    constexpr double kMaxMagRatio = 3.00;

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData c = buildCase(cfg);
        std::vector<float> const ref = computeReference(c);
        RunResult const result = runCase(c);
        ASSERT_EQ(result.status, 0) << "runner returned non-zero status for " << cfg.name;
        ASSERT_EQ(result.outputFp32.size(), ref.size());

        Summary const s = summarize(result.outputFp32, ref, cfg.numTokens, cfg.hiddenSize);
        std::cout << "[" << cfg.name << "] median_cos=" << s.medianCosine << " mag_ratio=" << s.magRatio
                  << " max_abs=" << s.maxAbs << std::endl;
        EXPECT_GE(s.medianCosine, kMinCosine)
            << "median cosine " << s.medianCosine << " below threshold " << kMinCosine << " for " << cfg.name;
        EXPECT_GE(s.magRatio, kMinMagRatio)
            << "magnitude ratio below band [" << kMinMagRatio << ", " << kMaxMagRatio << "] for " << cfg.name;
        EXPECT_LE(s.magRatio, kMaxMagRatio)
            << "magnitude ratio above band [" << kMinMagRatio << ", " << kMaxMagRatio << "] for " << cfg.name;
    }
}

#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
