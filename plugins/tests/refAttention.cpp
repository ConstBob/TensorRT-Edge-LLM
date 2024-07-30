/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "refAttention.h"


template <typename T>
Vec<float, validElemsPerHead> toF32Head(Vec<T, validElemsPerHead> const& src) {
    Vec<float, validElemsPerHead> dst;
    for (uint32_t i = 0; i < validElemsPerHead; i++) {
        dst[i] = float(src[i]);
    }
    return dst;
}

inline float dot(Vec<float, validElemsPerHead> const& q, Vec<float, validElemsPerHead> const& k) {
    float acc = 0;
    for (uint32_t i = 0; i < validElemsPerHead; i++) {
        acc += q[i] * k[i];
    }
    return acc;
};

#if EIGEN_WORLD_VERSION < 3 || (EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION < 4)
namespace Eigen {
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;
}
#endif

template <typename MathElem>
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention(IOHead const* q, CacheSeq const& k, CacheSeq const& v, uint32_t seqLen, float kvScale, float xScale) {
    
    float const rcpXScale = 1.f/xScale;
    float const qkScale = 1/sqrtf(validElemsPerHead) * kvScale;

    Eigen::Matrix<float, headGrpSize, Eigen::Dynamic, Eigen::RowMajor> gemm0Acc(headGrpSize, seqLen);
    Vec<Vec<float, validElemsPerHead>, headGrpSize> qF32;
    for (uint32_t i = 0; i < headGrpSize; i++) {
        qF32[i] = toF32Head(q[i]);
    }
    for (uint32_t j = 0; j < seqLen; j++) {
        auto const kF32 = toF32Head(k[j]);
        for (uint32_t i = 0; i < headGrpSize; i++) {
            gemm0Acc(i, j) = dot(qF32[i], kF32) * qkScale;
        }
    }

    Eigen::Vector<float, headGrpSize> const rowMax = gemm0Acc.rowwise().maxCoeff().eval();

    Eigen::Matrix<float, headGrpSize, Eigen::Dynamic, Eigen::RowMajor> x = (gemm0Acc.colwise() - rowMax).array().exp().eval();
    Eigen::Vector<float, headGrpSize> const rowSum = x.rowwise().sum().eval();

    std::for_each(x.data(), x.data() + x.size(), [&](float& e){ e = float(MathElem(e * rcpXScale));});
    
    auto gemm1Acc = Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>::Zero().eval();
    for (uint32_t j = 0; j < seqLen; j++) {
        auto const vF32 = toF32Head(v[j]);
        for (uint32_t i = 0; i < headGrpSize; i++) {
            for (uint32_t k = 0; k < validElemsPerHead; k++) {
                gemm1Acc(i, k) += vF32[k] * x(i, j);
            }
        }
    }
    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> out = gemm1Acc.array().colwise() * (xScale * kvScale / rowSum.array());
    std::for_each(out.data(), out.data() + out.size(), [](float& e){ e = float(InputElem(e));});
    return out;
}
#define INSTANTIATE_refAttention(prec) template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention<prec>(IOHead const* q, CacheSeq const& k, CacheSeq const& v, uint32_t seqLen, float kvScale, float xScale)
INSTANTIATE_refAttention(InputElem);

template <typename MathElem, uint32_t tileSize>
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refFlashAttention(IOHead const* q, CacheSeq const& k, CacheSeq const& v, uint32_t seqLen, float kvScale, float xScale) {
    uint32_t const nbTiles = divUp(seqLen, tileSize);
    auto gemm1Acc = Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>::Zero().eval();
    Eigen::Vector<float, headGrpSize> rowMax, rowSum;
    rowMax.fill(-INFINITY);
    rowSum.fill(0);
    float const rcpXScale = 1.f/xScale;
    float const qkScale = 1/sqrtf(validElemsPerHead) * kvScale;
    for (uint32_t idxTile = 0; idxTile < nbTiles; idxTile++) {
        Eigen::Matrix<float, headGrpSize, tileSize, Eigen::RowMajor> gemm0Acc;
        Vec<Vec<float, validElemsPerHead>, headGrpSize> qF32;
        for (uint32_t i = 0; i < headGrpSize; i++) {
            qF32[i] = toF32Head(q[i]);
        }
        for (uint32_t j = 0; j < tileSize; j++) {
            if (tileSize * idxTile + j < seqLen) {
                auto const kF32 = toF32Head(k[tileSize * idxTile + j]);
                for (uint32_t i = 0; i < headGrpSize; i++) {
                    gemm0Acc(i, j) = dot(qF32[i], kF32) * qkScale;
                }
            }
            else {
                gemm0Acc.col(j).fill(-INFINITY);
            }
        }

        Eigen::Vector<float, headGrpSize> const tileRowMax = gemm0Acc.rowwise().maxCoeff().cwiseMax(rowMax).eval();

        Eigen::Matrix<float, headGrpSize, tileSize, Eigen::RowMajor> tileX = (gemm0Acc.colwise() - tileRowMax).array().exp().eval();
        Eigen::Vector<float, headGrpSize> const tileRowSum = tileX.rowwise().sum().eval();

        std::for_each(tileX.data(), tileX.data() + tileX.size(), [&](float& e){ e = float(MathElem(e * rcpXScale));});

        assert((rowMax.array() <= tileRowMax.array()).eval().all());
        if ((rowMax.array() < tileRowMax.array()).any()) {
            Eigen::Vector<float, headGrpSize> const scale = (rowMax - tileRowMax).array().exp();
            gemm1Acc.array().colwise() *= scale.array();
            rowSum.array().colwise() *= scale.array();
            rowMax = tileRowMax;
        }
        
        for (uint32_t j = 0; j < std::min(tileSize, seqLen - tileSize * idxTile); j++) {
            auto const vF32 = toF32Head(v[tileSize * idxTile + j]);
            for (uint32_t i = 0; i < headGrpSize; i++) {
                for (uint32_t k = 0; k < validElemsPerHead; k++) {
                    gemm1Acc(i, k) += vF32[k] * tileX(i, j);
                }
            }
        }
        rowSum += tileRowSum;
    }
    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> out = gemm1Acc.array().colwise() * (xScale * kvScale / rowSum.array());
    std::for_each(out.data(), out.data() + out.size(), [](float& e){ e = float(InputElem(e));});
    return out;
}

#define INSTANTIATE_refFlashAttention(prec, tileSize) template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refFlashAttention<prec, tileSize>(IOHead const* q, CacheSeq const& k, CacheSeq const& v, uint32_t seqLen, float kvScale, float xScale)
INSTANTIATE_refFlashAttention(CacheElem, 64);
INSTANTIATE_refFlashAttention(CacheElem, 128);