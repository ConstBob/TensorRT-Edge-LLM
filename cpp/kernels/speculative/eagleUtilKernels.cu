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

#include "eagleUtilKernels.h"

#include "common/common.h"
#include "memoryUtils.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <set>
#include <stdexcept>
#include <string>

namespace drivellm
{
namespace kernel
{

__inline__ __device__ int32_t packMaskBits(bool* mask, int32_t startIdx, int32_t maxLength)
{
    constexpr int32_t numBitsPerPackedMask = 32;
    int32_t packedMask = 0;

    for (int32_t k = 0; k < numBitsPerPackedMask; k++)
    {
        if (k < maxLength)
        {
            int32_t const maskFlag = mask[startIdx + k];
            packedMask |= maskFlag << k;
        }
    }
    return packedMask;
}

template <int32_t BLOCK_SIZE>
__global__ void acceptDraftTokensByIdsWithPaths(int32_t* outputIds, int32_t* inputIdsDraftDecode,
    int32_t const* draftIds, int32_t const* targetIds, int32_t* sequenceLengths, int64_t* acceptedLen,
    int32_t* bestPathIds, int64_t* finishedFinal, int32_t const* paths, int32_t const endId,
    int32_t const curTokensPerStep, int64_t const batchSize, int32_t const maxSeqLen, int32_t const maxPathLen,
    int32_t const maxDecodingTokens)
{
    auto const batchIdx = static_cast<int32_t>(blockIdx.x);
    if (batchIdx >= batchSize)
        return;

    auto const inputLength = sequenceLengths == nullptr ? -1 : sequenceLengths[batchIdx];
    auto const numTokensPerStep = curTokensPerStep;
    int4 partialMax{-1, -1, 0, 0};
    if (numTokensPerStep == 1 && draftIds == nullptr)
    {
        auto const acceptedLength = numTokensPerStep;
        auto const targetSrcTokenIdx = batchIdx * maxDecodingTokens;
        auto const outputTokenIdx = batchIdx * maxSeqLen + inputLength;
        auto const targetToken = targetIds[targetSrcTokenIdx];
        outputIds[outputTokenIdx] = targetToken;
        inputIdsDraftDecode[batchIdx * maxDecodingTokens] = targetToken;

        if (threadIdx.x == 0)
        {

            bestPathIds[batchIdx] = 0;
            acceptedLen[batchIdx] = acceptedLength;
            if (sequenceLengths)
            {
                sequenceLengths[batchIdx] += acceptedLength;
            }
        }
    }
    else
    {

        for (auto pathIdx = threadIdx.x; pathIdx < maxDecodingTokens; pathIdx += blockDim.x)
        {
            auto acceptedLength = maxPathLen + 1;

            auto const pathOffset = batchIdx * maxDecodingTokens * (maxPathLen + 1) + pathIdx * (maxPathLen + 1);
            auto const tokenId = paths[pathOffset];
            bool hasEnd = false;

            if (tokenId == -1)
            {

                continue;
            }

            auto const targetTokenIdx = batchIdx * maxDecodingTokens + tokenId;

            auto targetToken = targetIds[targetTokenIdx];
            auto nextIdx = tokenId;

            for (int32_t ti = 1; ti < maxPathLen + 1; ti++)
            {
                auto const tokenId = paths[pathOffset + ti];
                if (tokenId == -1)
                {
                    // check if last token is EOS when path terminates.
                    hasEnd = targetToken == endId;
                    acceptedLength = hasEnd ? ti - 1 : ti;
                    break;
                }
                // draftIds contains root token
                auto const draftTokenIdx = batchIdx * maxDecodingTokens + tokenId;
                auto const draftToken = draftIds[draftTokenIdx];
                bool const accepted = draftToken == targetToken;
                hasEnd = targetToken == endId;
                if (!accepted || hasEnd)
                {
                    acceptedLength = hasEnd ? ti - 1 : ti;
                    break;
                }

                auto const targetTokenIdx = batchIdx * maxDecodingTokens + tokenId;
                targetToken = targetIds[targetTokenIdx];
                nextIdx = tokenId;
            }

            // Get longest path of the thread
            if (partialMax.x < acceptedLength)
            {
                partialMax.x = acceptedLength;
                partialMax.y = pathIdx;
                partialMax.z = hasEnd;
                partialMax.w = nextIdx;
            }
        }
        // Get the longest path of the block (request)
        typedef cub::BlockReduce<int4, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        int4 total = BlockReduce(tempStorage).Reduce(partialMax, reduceMaxInt4);

        __shared__ int4 totalShared;
        if (threadIdx.x == 0)
        {
            totalShared = total;
        }
        __syncthreads();

        auto const acceptedLength = totalShared.x;
        auto const bestPathIdx = totalShared.y;
        auto const pathOffset = batchIdx * maxDecodingTokens * (maxPathLen + 1) + bestPathIdx * (maxPathLen + 1);

        for (auto ti = static_cast<int32_t>(threadIdx.x); ti < acceptedLength + 1;
            ti += static_cast<int32_t>(blockDim.x))
        {
            auto tokenId = paths[pathOffset + ti];
            auto const targetSrcTokenIdx = batchIdx * maxDecodingTokens + tokenId;
            auto const outputTokenIdx = batchIdx * maxSeqLen + inputLength + ti;
            auto const targetToken = targetIds[targetSrcTokenIdx];
            outputIds[outputTokenIdx] = targetToken;
            inputIdsDraftDecode[batchIdx * maxDecodingTokens + ti] = targetToken;
        }

        if (threadIdx.x == 0)
        {
            auto const hasEnd = totalShared.z;
            if (hasEnd && finishedFinal)
            {
                finishedFinal[batchIdx] = 1;
            }
            if (sequenceLengths)
            {
                sequenceLengths[batchIdx] += acceptedLength;
            }

            bestPathIds[batchIdx] = bestPathIdx;
            acceptedLen[batchIdx] = acceptedLength;
        }
    }
}

void dispatchAcceptDraftTokensByIdsWithPaths(
    AcceptDraftTokensByIdsWithPathsParams const& params, EagleCommonParams const& commonParams)
{
    int32_t constexpr BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(commonParams.batchSize);
    acceptDraftTokensByIdsWithPaths<BLOCK_SIZE><<<grid, block, 0, commonParams.stream>>>(params.outputIds,
        params.inputIdsDraftDecode, params.draftIds, params.targetIds, params.contextLengths, params.acceptedLengths,
        params.bestPathIds, params.finishedFinal, params.paths, params.endIds, params.curTokensPerStep,
        commonParams.batchSize, commonParams.maxSeqLen, commonParams.maxPathLen, commonParams.maxDecodingTokens);
}

template <typename T>
__global__ void updateDraftInputIdsAndTreeMaskAndPositionIds(int32_t* outputIdsAllDraft,
    int32_t* selectedOutputIdsDraft, int32_t* treeIndices, bool* treeMaskInput, bool* treeMaskInit,
    bool* treeMaskUpdate, bool* treeMaskUpdateforAttention, int32_t* packedTreeMaskUpdateforAttention,
    int32_t* treePositionIds, int32_t* curContextLengths, int32_t* packedTreeMaskUpdateforAttentionNoPadding,
    float* intermediateScores, float* cumScoresForThirdTopk, int32_t* outputIdsForThirdTopk, int32_t* allTokens,
    int32_t const* draftVoc, int32_t topK, int32_t layerIdx, int32_t maxLength, int32_t batchSize, int32_t maxPathLen)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const bs = idx / (maxLength * topK);
    auto const outId = (idx / (maxLength)) % topK;
    auto const lengthIdx = idx % (maxLength);

    if (idx >= batchSize * maxLength * topK)
    {
        return;
    }
    if (layerIdx == 0)
    {
        if (lengthIdx == 0 && outId < topK)
        {
            treePositionIds[bs * maxLength + outId] = curContextLengths[bs] - 1;
            auto const idx = flat_index3(bs, outId, outId, topK, maxLength);
            treeMaskInput[idx] = true;
            treeMaskUpdate[idx] = true;
            treeMaskUpdateforAttention[idx] = true;
            auto const idxforTreeInit = flat_index3(bs, outId, outId, topK, topK);
            treeMaskInit[idxforTreeInit] = true;
            auto const idsIdxSrc = flat_index2(bs, outId, topK);
            auto const selectedOutId = outputIdsAllDraft[idsIdxSrc];
            selectedOutputIdsDraft[bs * maxLength + outId]
                = draftVoc == nullptr ? selectedOutId : selectedOutId + draftVoc[selectedOutId];
            allTokens[bs * maxPathLen * topK * topK + outId]
                = draftVoc == nullptr ? selectedOutId : selectedOutId + draftVoc[selectedOutId];
        }
        __syncthreads();
        if (lengthIdx % 32 == 0 && lengthIdx < topK && outId < topK)
        {

            auto const indexPacked = flat_index3(bs, outId, lengthIdx / 32, topK, divUp(topK, 32));
            int32_t const numPackedMasksPerToken = (topK + 31) / 32;

            int const packedIdx = flat_index3(bs, outId, lengthIdx / 32, maxLength, numPackedMasksPerToken);
            int32_t remainingBits = topK - lengthIdx;
            if (remainingBits > 32)
            {
                remainingBits = 32;
            }
            auto const packValue
                = packMaskBits(treeMaskInit + bs * topK * topK, outId * topK + lengthIdx, remainingBits);
            packedTreeMaskUpdateforAttention[packedIdx] = packValue;

            packedTreeMaskUpdateforAttentionNoPadding[packedIdx] = packValue;
        }
    }
    else
    {
        if (lengthIdx == 0 && outId < topK)
        {
            auto const outIdx = bs * topK * topK + treeIndices[bs * topK + outId];
            auto const selectedOutId = outputIdsAllDraft[outIdx];
            selectedOutputIdsDraft[bs * topK + layerIdx * topK + outId]
                = draftVoc == nullptr ? selectedOutId : selectedOutId + draftVoc[selectedOutId];
            treePositionIds[bs * maxLength + layerIdx * topK + outId] = curContextLengths[bs] + layerIdx - 1;
            intermediateScores[bs * topK + outId] = cumScoresForThirdTopk[outputIdsForThirdTopk[bs * topK + outId]];

            for (unsigned int i = 0; i < topK; i++)
            {
                auto outputIdsAllDraftIdx = flat_index3(bs, i, outId, topK, topK);
                auto const curOutId = outputIdsAllDraft[outputIdsAllDraftIdx];
                auto curAllTokenIdx
                    = bs * maxPathLen * topK * topK + (layerIdx - 1) * topK * topK + topK + i * topK + outId;
                allTokens[curAllTokenIdx] = draftVoc == nullptr ? curOutId : curOutId + draftVoc[curOutId];
            }
        }

        // Update tree mask
        if (lengthIdx < layerIdx * topK)
        {

            auto const treeIdx = static_cast<unsigned int>(treeIndices[bs * topK + outId] / topK);
            auto const inputIdx = flat_index3(bs, treeIdx, lengthIdx, topK, maxLength);
            auto const dstIdx = flat_index3(bs, outId, lengthIdx, topK, maxLength);
            treeMaskUpdate[dstIdx] = treeMaskInput[inputIdx];
            // layerIdx=1, update[0:10,60]
            // layerIdx=2, update[10:20,60]
            // layerIdx=3, update[20:30,60]
            auto const dstIdxforAttention = flat_index3(bs, layerIdx * topK + outId, lengthIdx, maxLength, maxLength);
            treeMaskUpdateforAttention[dstIdxforAttention] = treeMaskInput[inputIdx];
        }
        // last 10 thread 50-60
        if (lengthIdx >= maxLength - topK)
        {
            auto const initIdx = lengthIdx - (maxLength - topK);
            auto const dstIdx = flat_index3(bs, outId, layerIdx * topK + initIdx, maxLength, maxLength);

            treeMaskUpdate[dstIdx] = treeMaskInit[bs * topK * topK + outId * topK + initIdx];

            auto const dstIdxforAttention
                = flat_index3(bs, layerIdx * topK + outId, layerIdx * topK + initIdx, maxLength, maxLength);
            treeMaskUpdateforAttention[dstIdxforAttention] = treeMaskInit[bs * topK * topK + outId * topK + initIdx];
        }
        if (lengthIdx < layerIdx * topK && outId == 0)
        {
            auto const idx = flat_index3(bs, lengthIdx, lengthIdx, maxLength, maxLength);
            treeMaskUpdateforAttention[idx] = true;
        }
        __syncthreads();

        // packedTreeMaskUpdateforAttention:[bs,maxLength,ceil(maxLength/32]--not continuous memory, has padding
        if (lengthIdx % 32 == 0 && outId == 0)
        {
            // Calculate the number of packed masks per token
            int32_t const numPackedMasksPerToken = (maxLength + 31) / 32;
            int32_t remainingBits = maxLength - lengthIdx;
            if (remainingBits > 32)
            {
                remainingBits = 32;
            }
            for (int i = 0; i < (layerIdx + 1) * topK; i++)
            {
                int const packedIdx = flat_index3(static_cast<unsigned int>(bs), static_cast<unsigned int>(i),
                    static_cast<unsigned int>(lengthIdx / 32), static_cast<int32_t>(maxLength),
                    static_cast<int32_t>(numPackedMasksPerToken));
                packedTreeMaskUpdateforAttention[packedIdx] = packMaskBits(
                    treeMaskUpdateforAttention + bs * maxLength * maxLength, i * maxLength + lengthIdx, remainingBits);
            }
        }
        __syncthreads();
        if (lengthIdx < (layerIdx + 1) * topK && outId == 0)
        {

            auto const actualMaskLength = divUp((layerIdx + 1) * topK, 32);
            auto const actualTokenLength = (layerIdx + 1) * topK;
            auto const numPackedMasksPerToken = (maxLength + 31) / 32;

            auto const dstIdx = flat_index3(
                int32_t(bs), int32_t(lengthIdx), 0, int32_t(actualTokenLength), int32_t(actualMaskLength));
            for (int i = 0; i < actualMaskLength; i++)
            {
                auto const srcIdx = flat_index3(
                    int32_t(bs), int32_t(lengthIdx), 0, int32_t(maxLength), int32_t(numPackedMasksPerToken));

                packedTreeMaskUpdateforAttentionNoPadding[dstIdx + i] = packedTreeMaskUpdateforAttention[srcIdx + i];
            }
        }
    }
}

template <typename T>
__global__ void updateHiddenStates(T* inputHiddenStatesDraft, T* outputHiddenStatesDraft, int32_t* treeIndices,
    int32_t batchSize, int32_t topK, int32_t hiddenDim, int32_t layerIdx, int32_t maxPathLen)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const bs = idx / (topK * hiddenDim);
    auto const outId = (idx / hiddenDim) % topK;
    auto const hiddenIdx = idx % hiddenDim;

    if (idx >= batchSize * topK * hiddenDim)
    {
        return;
    }
    if (layerIdx == 0)
    {
        auto const inputIdx = flat_index2(bs, hiddenIdx, hiddenDim);
        auto const outputIdx
            = flat_index4(bs, static_cast<unsigned int>(layerIdx), outId, hiddenIdx, maxPathLen, topK, hiddenDim);
        outputHiddenStatesDraft[outputIdx] = inputHiddenStatesDraft[inputIdx];
    }
    else
    {
        auto const treeIdx = static_cast<unsigned int>(treeIndices[bs * topK + outId] / topK);
        auto const inputIdx = flat_index3(bs, treeIdx, hiddenIdx, topK, hiddenDim);
        //[bs,maxPathLen,topk,hidden_dim]
        auto const outputIdx
            = flat_index4(bs, static_cast<unsigned int>(layerIdx), outId, hiddenIdx, maxPathLen, topK, hiddenDim);
        outputHiddenStatesDraft[outputIdx] = inputHiddenStatesDraft[inputIdx];
    }
}

template <typename T>
void dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores(
    UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<T> const& params,
    EagleCommonParams const& commonParams)
{
    constexpr int32_t BLOCK_SIZE = 128;
    int64_t gridSize = (commonParams.batchSize * commonParams.topK * params.maxLength + BLOCK_SIZE - 1) / BLOCK_SIZE;

    updateDraftInputIdsAndTreeMaskAndPositionIds<T>
        <<<gridSize, BLOCK_SIZE, 0, commonParams.stream>>>(params.outputIdsAllDraft, params.selectedOutputIdsDraft,
            params.treeIndices, params.treeMaskInput, params.treeMaskInit, params.treeMaskUpdate,
            params.treeMaskUpdateforAttention, params.packedTreeMaskUpdateforAttention, params.treePositionIds,
            params.curContextLengths, params.packedTreeMaskUpdateforAttentionNoPadding, params.intermediateScores,
            params.cumScoresForThirdTopk, params.outputIdsForThirdTopk, params.allTokens, params.draftVoc,
            commonParams.topK, params.layerIdx, params.maxLength, commonParams.batchSize, commonParams.maxPathLen);

    CUDA_CHECK(cudaMemcpyAsync(params.treeMaskInput, params.treeMaskUpdate,
        commonParams.batchSize * commonParams.topK * params.maxLength * sizeof(bool), cudaMemcpyDeviceToDevice,
        commonParams.stream));
    CUDA_CHECK(cudaMemsetAsync(params.treeMaskUpdateforAttention, 0,
        commonParams.batchSize * params.maxLength * params.maxLength * sizeof(bool), commonParams.stream));

    gridSize = (commonParams.batchSize * commonParams.topK * commonParams.hiddenDim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateHiddenStates<T><<<gridSize, BLOCK_SIZE, 0, commonParams.stream>>>(params.inputHiddenStatesDraft,
        params.outputHiddenStatesDraft, params.treeIndices, commonParams.batchSize, commonParams.topK,
        commonParams.hiddenDim, params.layerIdx, commonParams.maxPathLen);
}
template void dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores<float>(
    UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<float> const& params,
    EagleCommonParams const& commonParams);
template void dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores<half>(
    UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<half> const& params,
    EagleCommonParams const& commonParams);

inline __device__ void insertionSortOutputIds(int32_t* outputIds, int32_t n)
{
    for (int32_t ii = 1; ii < n; ++ii)
    {
        int32_t key = outputIds[ii];
        int32_t jj = ii - 1;

        while (jj >= 0 && outputIds[jj] > key)
        {
            outputIds[jj + 1] = outputIds[jj];
            jj--;
        }
        outputIds[jj + 1] = key;
    }
}
inline __device__ int64_t findAncestorIndex(int32_t* draftIds, int32_t tokenIdx, int32_t topK, int32_t maxDraftTokens)
{
    int target = tokenIdx - 1;
    int left = 0;
    int right = maxDraftTokens;
    while (left < right)
    {
        int mid = (left + right) / 2;
        if (draftIds[mid] < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

__global__ void assembleDraftIdsAndTreeMaskAndPositionIdsAndPredecessors(int32_t const* thirdTopKIds,
    int32_t const* allDraftIds, int64_t const* allDraftIdsAncestors, int32_t const* modelInputIds,
    int32_t const* contextLengths, bool* treeMask, int32_t* positionIds, int32_t* draftIds, int64_t* draftIdsAncestors,
    int32_t* packedTreeMaskVerification, int32_t const batchSize, int32_t const maxDraftTokens,
    int32_t const maxDecodingTokens, int32_t const topK, int64_t const maxSeqLen)
{

    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const totalElements = batchSize * maxDecodingTokens;

    if (idx < totalElements)
    {
        int const batchIdx = idx / maxDecodingTokens;
        int const tokenIdx = idx % maxDecodingTokens;

        int32_t const* curThirdTopKIds = thirdTopKIds + batchIdx * maxDraftTokens;
        int32_t const* curAllDraftIds = allDraftIds + batchIdx * maxDraftTokens;
        int64_t const* curAllDraftIdsAncestors = allDraftIdsAncestors + batchIdx * maxDraftTokens;
        bool* curTreeMask = treeMask + batchIdx * maxDecodingTokens * maxDecodingTokens;
        int32_t* curPositionIds = positionIds + batchIdx * maxDecodingTokens;
        int32_t* curDraftIds = draftIds + batchIdx * maxDecodingTokens;
        int64_t* curDraftIdsAncestors = draftIdsAncestors + batchIdx * maxDraftTokens;
        int32_t const* curModelInputIds = modelInputIds + batchIdx * maxSeqLen;

        extern __shared__ char smem[];
        bool* curMask = (bool*) (smem) + threadIdx.x * maxDecodingTokens;

        int32_t* sortedIds = (int32_t*) (smem + blockDim.x * maxDecodingTokens * sizeof(bool));

        // The first thread in each block is responsible for sorting
        if (threadIdx.x == 0)
        {

            for (int i = 0; i < maxDraftTokens; i++)
            {
                sortedIds[i] = curThirdTopKIds[i];
            }
            // Sort the thirdTopKIds for the current batch
            insertionSortOutputIds(sortedIds, maxDraftTokens);
        }

        __syncthreads();

        // Fist part: calculate draftIds and draftIdsPredecessors
        if (tokenIdx < maxDraftTokens)
        {
            int layerIdx = sortedIds[tokenIdx];

            int draftAncestor = curAllDraftIdsAncestors[layerIdx / topK];

            if (draftAncestor == 0)
            {
                curDraftIdsAncestors[tokenIdx] = 0;
            }
            else
            {

                int draftAncestorIdx = findAncestorIndex(sortedIds, draftAncestor, topK, maxDraftTokens);
                curDraftIdsAncestors[tokenIdx] = draftAncestorIdx + 1;
            }
        }
        __syncthreads();

        // Second part: use calculated mask_index to build tree_mask and tree_position_ids
        if (tokenIdx < maxDecodingTokens)
        {
            if (tokenIdx == 0)
            {

                curDraftIds[0] = curModelInputIds[contextLengths[batchIdx] - 1]; // sample_token
            }
            else
            {
                int idx = sortedIds[tokenIdx - 1];
                curDraftIds[tokenIdx] = curAllDraftIds[idx];
            }

            for (int j = 0; j < maxDecodingTokens; j++)
            {
                curMask[j] = false;
            }

            if (tokenIdx > 0)
            {

                curMask[0] = true;
                curMask[tokenIdx] = true;

                int parent = curDraftIdsAncestors[tokenIdx - 1];
                while (parent >= 0)
                {
                    curMask[parent] = true;
                    if (parent == 0)
                        break;
                    parent = curDraftIdsAncestors[parent - 1];
                }
            }
            else
            { // root node
                curMask[0] = true;
            }
            __syncthreads();
            for (int j = 0; j < maxDecodingTokens; j++)
            {
                curTreeMask[tokenIdx * maxDecodingTokens + j] = curMask[j];
            }

            // calculate position id
            int sum = 0;
            for (int j = 0; j < maxDecodingTokens; j++)
            {
                sum += (int) curMask[j];
            }
            curPositionIds[tokenIdx] = sum - 1;
        }
        __syncthreads();
        // one thread compute one token's packedTreeMaskVerification
        int32_t const numPackedMasksPerToken = (maxDecodingTokens + 31) / 32;
        for (int i = 0; i < numPackedMasksPerToken; i++)
        {
            int32_t const packedIdx = flat_index3(batchIdx, tokenIdx, i, maxDecodingTokens, numPackedMasksPerToken);
            int32_t remainingBits = maxDecodingTokens - i * 32;
            if (remainingBits > 32)
            {
                remainingBits = 32;
            }
            auto curMaskIdx = flat_index3(batchIdx, tokenIdx, 0, maxDecodingTokens, maxDecodingTokens);
            packedTreeMaskVerification[packedIdx] = packMaskBits(treeMask + curMaskIdx, i * 32, remainingBits);
        }
    }
}

__global__ void reconstructPath(int64_t const* draftIdsAncestors, int32_t* depthIds, int32_t* path,
    int32_t* validPathNum, int32_t const* contextLengths, int32_t const batchSize, int32_t const maxDraftTokens,
    int32_t const maxPathLen, int32_t const maxDecodingTokens)
{

    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;
    if (tokenIdx == 0)
    {
        validPathNum[batchIdx] = 0;
    }
    if (tokenIdx < maxDecodingTokens)
    {
        for (int i = 0; i < maxPathLen + 1; i++)
        {
            auto const pathIdx = flat_index3(batchIdx, tokenIdx, i, maxDecodingTokens, maxPathLen + 1);
            path[pathIdx] = -1;
        }
    }
    __syncthreads();

    if (tokenIdx < maxDecodingTokens)
    {

        int64_t const* curDraftIdsAncestors = draftIdsAncestors + batchIdx * maxDraftTokens;
        int32_t const* curDepthIds = depthIds + batchIdx * maxDecodingTokens;
        int32_t* curPath = path + batchIdx * maxDecodingTokens * (maxPathLen + 1);

        // check if it is a leaf node
        bool isLeaf = true;
        for (int j = 0; j < maxDraftTokens; j++)
        {
            if (curDraftIdsAncestors[j] == tokenIdx)
            {
                isLeaf = false;
                break;
            }
        }

        if (isLeaf)
        {

            int rid = atomicAdd(&validPathNum[batchIdx], 1);
            int depth = curDepthIds[tokenIdx];
            int cid = tokenIdx;

            for (int32_t j = depth; j >= 0; j--)
            {

                curPath[rid * (maxPathLen + 1) + j] = cid;
                if (cid > 0)
                {                                        // if not root node
                    cid = curDraftIdsAncestors[cid - 1]; // ancestor index
                }
                else
                {
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (tokenIdx < maxDecodingTokens)
    {
        int32_t* curPositionIds = depthIds + batchIdx * maxDecodingTokens;
        curPositionIds[tokenIdx] += contextLengths[batchIdx] - 1;
    }
}

void dispatchAssembleDraftIdsAndPathAndMaskAndPositionIds(
    AssembleDraftIdsAndPathAndMaskAndPositionIdsParams const& params, EagleCommonParams const& commonParams)
{
    constexpr int32_t BLOCK_SIZE = 128;

    auto gridSize = (commonParams.maxDecodingTokens * commonParams.batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t smemSize = (BLOCK_SIZE * commonParams.maxDecodingTokens * sizeof(bool)
                          + commonParams.maxDecodingTokens * sizeof(int32_t) + 15)
        & ~15;
    assembleDraftIdsAndTreeMaskAndPositionIdsAndPredecessors<<<gridSize, BLOCK_SIZE, smemSize, commonParams.stream>>>(
        params.fourthTopKIds, params.allDraftIds, params.allDraftIdsAncestors, params.modelInputIds,
        params.contextLengths, params.treeMask, params.positionIds, params.draftIds, params.draftIdsAncestors,
        params.packedTreeMaskVerification, commonParams.batchSize, commonParams.maxDraftTokens,
        commonParams.maxDecodingTokens, commonParams.topK, commonParams.maxSeqLen);

    gridSize = commonParams.batchSize;
    auto const blockSize = commonParams.maxDecodingTokens;
    reconstructPath<<<gridSize, blockSize, 0, commonParams.stream>>>(params.draftIdsAncestors, params.positionIds,
        params.paths, params.validPathNum, params.contextLengths, commonParams.batchSize, commonParams.maxDraftTokens,
        commonParams.maxPathLen, commonParams.maxDecodingTokens);
}

__global__ void updateCumScoresAndParentsIds(float* outputLogProbsAllDraft, float* intermediateScores, float* cumScores,
    int32_t* outputIdsCurrentDraft, int64_t* parentsIds, int32_t bias, int32_t layerIdx, int32_t topK,
    int32_t batchSize, int32_t maxPathLen)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const bs = idx / (topK * topK);
    auto const tokenIdx = (idx / topK) % topK;
    auto const candIdx = idx % topK;
    if (idx >= batchSize * topK * topK)
    {

        return;
    }

    if (layerIdx == 0 && tokenIdx == 0)
    {
        auto const idxSrc = flat_index3(bs, static_cast<unsigned int32_t>(0), candIdx, topK, topK);
        auto const idxDst = flat_index2(bs, candIdx, topK);
        intermediateScores[idxDst] = outputLogProbsAllDraft[idxSrc];
    }
    else if (layerIdx > 0)
    {
        // Update parents indices
        if (candIdx == 0)
        {
            int parentIdx = bs * topK * topK + layerIdx * topK - topK + 1 + tokenIdx;
            if (layerIdx == 1)
            {
                parentsIds[parentIdx] = tokenIdx + bias;
            }
            else
            {
                auto const src = outputIdsCurrentDraft[bs * topK + tokenIdx];

                parentsIds[parentIdx] = src + bias;
            }
        }

        // Update cumulative scores: cu_scores = topk_p + scores[:, None]
        auto const score_idx = bs * topK + tokenIdx;
        auto const prob_idx = (bs * topK + tokenIdx) * topK + candIdx;
        cumScores[prob_idx] = outputLogProbsAllDraft[prob_idx] + intermediateScores[score_idx];
    }
}

void dispatchUpdateCumScoresAndParentsIds(
    UpdateCumScoresAndParentsIdsParams const& params, EagleCommonParams const& commonParams)
{
    constexpr int32_t BLOCK_SIZE = 128;
    int64_t gridSize = (commonParams.batchSize * commonParams.topK * commonParams.topK + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateCumScoresAndParentsIds<<<gridSize, BLOCK_SIZE, 0, commonParams.stream>>>(params.outputLogProbsAllDraft,
        params.intermediateScores, params.cumScores, params.outputIdsCurrentDraft, params.parentsIds, params.bias,
        params.layerIdx, commonParams.topK, commonParams.batchSize, commonParams.maxPathLen);
}

template <typename T>
__global__ void updateKVCache(T* kvCache, int32_t const* paths, int64_t const* acceptedLengths,
    int32_t const* bestPathIds, int32_t* contextLengths, int32_t const maxPathLen, int32_t const batchSize,
    int32_t const numHead, int32_t const hiddenSizePerHead, int32_t const maxDecodingTokens, int32_t const maxSeqLen,
    int32_t const numLayers)
{

    constexpr int32_t vec_size = std::is_same<T, half>::value ? 2 : 1;
    assert(hiddenSizePerHead % vec_size == 0);

    int32_t const layerIdx = blockIdx.z;
    int32_t const headIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t const hVecIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t const hIdx = hVecIdx * vec_size;

    if (layerIdx >= numLayers || headIdx >= numHead || hVecIdx >= hiddenSizePerHead / vec_size)
        return;

    int32_t const headOffset = headIdx * maxSeqLen * hiddenSizePerHead;
    int32_t const hiddenStride = hiddenSizePerHead;
    // [B, 2, H, S, D]--->kv cache
    int32_t const layerOffset = layerIdx * batchSize * 2 * numHead * maxSeqLen * hiddenSizePerHead;

    using vec_t = typename std::conditional<std::is_same<T, half>::value, half2, float>::type;

    for (int32_t bs = 0; bs < batchSize; ++bs)
    {
        if (acceptedLengths[bs] == 0)
            continue;

        int32_t dstIdxBase = contextLengths[bs] - acceptedLengths[bs] - 1;
        int32_t const bestPath = bestPathIds[bs];
        int32_t const acceptedLen = acceptedLengths[bs];

        int32_t const pathOffset = bs * maxDecodingTokens * (maxPathLen + 1) + bestPath * (maxPathLen + 1);
        int32_t const kvCacheOffset = layerOffset + bs * 2 * numHead * maxSeqLen * hiddenSizePerHead;

        for (int pathIdx = 0; pathIdx < acceptedLen; ++pathIdx)
        {
            int32_t const validTokenIdx = paths[pathOffset + pathIdx];

            if (validTokenIdx == -1)
                break;
            int32_t const srcPos = contextLengths[bs] + validTokenIdx - acceptedLengths[bs] - 1;
            int32_t const keyDstOffset = kvCacheOffset + headOffset + dstIdxBase * hiddenStride + hIdx;

            int32_t const keySrcOffset = kvCacheOffset + headOffset + srcPos * hiddenStride + hIdx;

            if constexpr (std::is_same<T, half>::value)
            {
                *reinterpret_cast<vec_t*>(&kvCache[keyDstOffset]) = *reinterpret_cast<vec_t*>(&kvCache[keySrcOffset]);
            }
            else
            {
                kvCache[keyDstOffset] = kvCache[keySrcOffset];
            }

            int32_t const valueDstOffset = kvCacheOffset + 1 * numHead * maxSeqLen * hiddenSizePerHead + headOffset
                + dstIdxBase * hiddenStride + hIdx;

            int32_t const valueSrcOffset = kvCacheOffset + 1 * numHead * maxSeqLen * hiddenSizePerHead + headOffset
                + srcPos * hiddenStride + hIdx;

            if constexpr (std::is_same<T, half>::value)
            {
                *reinterpret_cast<vec_t*>(&kvCache[valueDstOffset])
                    = *reinterpret_cast<vec_t*>(&kvCache[valueSrcOffset]);
            }
            else
            {
                kvCache[valueDstOffset] = kvCache[valueSrcOffset];
            }

            dstIdxBase++;
        }
    }
}

__global__ void updateContextLengthsAndTreePositionIds(int32_t* contextLengths, int64_t const* acceptedLengths,
    int32_t* treePositionIds, int32_t batchSize, int32_t maxDecodingTokens)
{
    int32_t const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batchSize)
        return;

    for (int i = 0; i < acceptedLengths[tid]; i++)
    {
        treePositionIds[tid * maxDecodingTokens + i] = contextLengths[tid] - acceptedLengths[tid] - 1 + i;
    }
}

template <typename T>
__global__ void updateHiddenStatesInputs(T* hiddenStatesInputs, T* hiddenStates, int32_t const* paths,
    int32_t const* bestPathIds, int64_t const* acceptedLengths, int32_t maxPathLen, int32_t batchSize,
    int32_t maxDecodingTokens, int32_t hiddenDim)
{
    constexpr int32_t vec_size = std::is_same<T, half>::value ? 2 : 1;
    assert(hiddenDim % vec_size == 0);

    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const bs = tid / (hiddenDim / vec_size);
    int32_t const hiddenIdx = (tid % (hiddenDim / vec_size)) * vec_size;

    if (bs >= batchSize || hiddenIdx >= hiddenDim)
        return;

    using vec_t = typename std::conditional<std::is_same<T, half>::value, half2, float>::type;

    if (acceptedLengths[bs] == 0)
        return;

    int32_t const bestPath = bestPathIds[bs];
    int32_t const acceptedLen = acceptedLengths[bs];
    int32_t const pathOffset = bs * maxDecodingTokens * (maxPathLen + 1) + bestPath * (maxPathLen + 1);

    int32_t dstIdx = 0;
    for (int32_t pathIdx = 0; pathIdx < acceptedLen; ++pathIdx)
    {
        int32_t const validTokenIdx = paths[pathOffset + pathIdx];
        if (validTokenIdx == -1)
            break;

        int32_t const srcOffset = flat_index3(bs, validTokenIdx, hiddenIdx, maxDecodingTokens, hiddenDim);
        int32_t const dstOffset = flat_index3(bs, dstIdx, hiddenIdx, maxDecodingTokens, hiddenDim);
        if constexpr (std::is_same<T, half>::value)
        {
            *reinterpret_cast<vec_t*>(&hiddenStatesInputs[dstOffset])
                = *reinterpret_cast<vec_t const*>(&hiddenStates[srcOffset]);
        }
        else
        {
            hiddenStatesInputs[dstOffset] = hiddenStates[srcOffset];
        }

        dstIdx++;
    }
}

template <typename T>
void dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds(
    UpdateKVCacheParams<T> const& params, EagleCommonParams const& commonParams)
{
    dim3 blockSize(16, 8);
    constexpr int vec_size = std::is_same<T, half>::value ? 2 : 1;
    dim3 gridSize(((commonParams.hiddenSizePerHead / vec_size) + blockSize.x - 1) / blockSize.x,
        (commonParams.numHead + blockSize.y - 1) / blockSize.y, commonParams.numLayers);

    updateKVCache<T><<<gridSize, blockSize, 0, commonParams.stream>>>(params.KVCache, params.paths,
        params.acceptedLengths, params.bestPathIds, params.contextLengths, commonParams.maxPathLen,
        commonParams.batchSize, commonParams.numHead, commonParams.hiddenSizePerHead, commonParams.maxDecodingTokens,
        commonParams.kvCacheSeqLen, commonParams.numLayers);

    int threadsPerBlock = 256;

    int32_t const totalThreads = commonParams.batchSize * (commonParams.targetHiddenDim / vec_size);
    int32_t blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    updateHiddenStatesInputs<T><<<blocksPerGrid, threadsPerBlock, 0, commonParams.stream>>>(params.hiddenStatesInputs,
        params.hiddenStates, params.paths, params.bestPathIds, params.acceptedLengths, commonParams.maxPathLen,
        commonParams.batchSize, commonParams.maxDecodingTokens, commonParams.targetHiddenDim);

    blocksPerGrid = (commonParams.batchSize + threadsPerBlock - 1) / threadsPerBlock;
    updateContextLengthsAndTreePositionIds<<<blocksPerGrid, threadsPerBlock, 0, commonParams.stream>>>(
        params.contextLengths, params.acceptedLengths, params.treePositionIds, commonParams.batchSize,
        commonParams.maxDecodingTokens);
}

template void dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds<float>(
    UpdateKVCacheParams<float> const& params, EagleCommonParams const& commonParams);
template void dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds<half>(
    UpdateKVCacheParams<half> const& params, EagleCommonParams const& commonParams);

__global__ void initCausalAttentionMask(bool* mask, int batchSize, int maxPathLen, int32_t* packedMask)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = batchSize * maxPathLen * maxPathLen;
    if (idx >= total)
    {
        return;
    }
    int const b = idx / (maxPathLen * maxPathLen);
    int const i = (idx % (maxPathLen * maxPathLen)) / maxPathLen;
    int const j = idx % maxPathLen;

    mask[idx] = (j <= i);
    __syncthreads();

    int32_t const numPackedMasksPerToken = (maxPathLen + 31) / 32;
    auto const packedIdx = flat_index3(b, i, j / 32, maxPathLen, numPackedMasksPerToken);
    if (j % 32 == 0 && j < maxPathLen)
    {
        int32_t remainingBits = maxPathLen - j;
        if (remainingBits > 32)
        {
            remainingBits = 32;
        }
        packedMask[packedIdx] = packMaskBits(mask + b * maxPathLen * maxPathLen + i * maxPathLen, j, remainingBits);
    }
}

void dispatchInitializeAttentionMaskCausal(
    InitCausalAttentionMaskParams const& params, EagleCommonParams const& commonParams)
{
    int const blockSize = 256;
    int const total = commonParams.batchSize * commonParams.maxDecodingTokens * commonParams.maxDecodingTokens;
    int const gridSize = (total + blockSize - 1) / blockSize;

    initCausalAttentionMask<<<gridSize, blockSize, 0, commonParams.stream>>>(
        params.mask, commonParams.batchSize, commonParams.maxDecodingTokens, params.packedMask);
}

__global__ void getLastLogitsOffsetKernel(int64_t* lastLogitsOffset, int32_t const* paths, int32_t const* bestPathIds,
    int64_t const* acceptedLengths, int32_t batchSize, int32_t maxPathLen, int32_t maxDecodingTokens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;

    int bestPath = bestPathIds[idx];
    int64_t acceptedLen = acceptedLengths[idx];
    int pathOffset = idx * maxDecodingTokens * (maxPathLen + 1) + bestPath * (maxPathLen + 1);

    lastLogitsOffset[idx] = paths[pathOffset + acceptedLen - 1];
}

void dispatchGetLastLogitsOffset(GetLastLogitsOffsetParams const& params, EagleCommonParams const& commonParams)
{
    int const blockSize = 128;
    int const gridSize = (commonParams.batchSize + blockSize - 1) / blockSize;

    getLastLogitsOffsetKernel<<<gridSize, blockSize, 0, commonParams.stream>>>(params.lastLogitsOffset, params.paths,
        params.bestPathIds, params.acceptedLengths, commonParams.batchSize, commonParams.maxPathLen,
        commonParams.maxDecodingTokens);
}

} // namespace kernel
} // namespace drivellm