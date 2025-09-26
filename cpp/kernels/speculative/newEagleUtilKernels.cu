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

#include "kernels/common/vectorizedTypes.cuh"
#include "newEagleUtilKernels.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"

using namespace nvinfer1;

namespace drivellm
{
namespace kernel
{

__global__ void prepareEaglePrefillInputKernel(
    int32_t* sequenceContextLengths, int64_t* selectTokenIndices, int32_t sequenceLength)
{
    int32_t const batchIdx = blockIdx.x;
    if (threadIdx.x == 0)
    {
        sequenceContextLengths[batchIdx] = sequenceLength;
        selectTokenIndices[batchIdx] = sequenceLength - 1;
    }
}

__global__ void prepareEagleDraftProposalMiscInputKernel(int32_t const* draftTreeSizes,
    int32_t const* sequenceStartIndices, int32_t* sequenceContextLengths, int64_t* selectTokenIndices,
    int32_t selectTokenLength, int32_t paddedDraftTreeSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const blockSize = blockDim.x;
    int32_t const tIdx = threadIdx.x;
    // Use draftTreeSizes if provided, otherwise use paddedDraftTreeSize for all batches
    int32_t const draftTreeSize = (draftTreeSizes != nullptr) ? draftTreeSizes[batchIdx] : paddedDraftTreeSize;

    if (tIdx == 0)
    {
        // Special handling eagle, add padded size to meet the tree-attentionkernel implementation.
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + paddedDraftTreeSize;
    }

    // Select token indices is in format of [batch, select-token-length]. With current implementation,
    // we always put the whole tree into the computation and select the "current round" logits/hidden states
    // to proceed with the next round.
    for (int32_t i = tIdx; i < selectTokenLength; i += blockSize)
    {
        selectTokenIndices[batchIdx * selectTokenLength + i] = draftTreeSize - selectTokenLength + i;
    }
}

__global__ void assembleDraftTreeDescKernel(int8_t const* draftTreeMask, int32_t const* draftTreeSizes,
    int32_t const* sequenceStartIndices, int32_t* packedDraftTreeMask, int32_t* tensorPositionIndices,
    int32_t const paddedDraftTreeSize)
{
    // Supports up to draft tree size of 128 {4 x 32}, should be sufficient for now.
    constexpr int32_t kNUM_MASK_PER_ENTRY{32};
    constexpr int32_t kMAX_DRAFT_PACKED_TREE_SIZE{4};

    // Each thread will handle one token in the draft tree to setup the mask and tensor position indices.
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    int32_t packedTreeMask[kMAX_DRAFT_PACKED_TREE_SIZE] = {0};
    int32_t const packedTreeMaskLen = (paddedDraftTreeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;
    int32_t const actualDraftTreeSize = (draftTreeSizes != nullptr) ? draftTreeSizes[batchIdx] : paddedDraftTreeSize;
    int32_t const sequenceStartIndex = sequenceStartIndices[batchIdx];

    // Unpacked tree mask formulate in the format of [batch, padded-draft-tree-size, padded-draft-tree-size].
    // Packed tree mask len is in format of [batch, padded-draft-tree-size, divup(padded-draft-tree-size, 32)].
    // Tensor position indices is in format of [batch, padded-draft-tree-size].
    int32_t const unpackedTreeMaskOffset
        = batchIdx * paddedDraftTreeSize * paddedDraftTreeSize + tokenIdx * paddedDraftTreeSize;
    int32_t const packedTreeMaskOffset
        = batchIdx * paddedDraftTreeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
    int32_t const tensorPositionOffset = batchIdx * paddedDraftTreeSize + tokenIdx;

    int32_t tensorPositionIdx{0};
    if (tokenIdx < actualDraftTreeSize)
    {
        // With causal attention, the node will only attend to nodes "prior" to itself.
        int32_t attendNodeNum{0};
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            int8_t const maskFlag = draftTreeMask[unpackedTreeMaskOffset + i];
            if (maskFlag)
            {
                attendNodeNum += 1;
                packedTreeMask[i / kNUM_MASK_PER_ENTRY] |= (1 << (i % kNUM_MASK_PER_ENTRY));
            }
        }
        // A token always attend to itself, subtract 1 to reflect its position in the sequence.
        tensorPositionIdx = sequenceStartIndex + attendNodeNum - 1;
    }

    // Write result to the output. For node outside the "real" draft tree size, we will write 0 value to keep mask
    // well-formed.
    if (tokenIdx < paddedDraftTreeSize)
    {
        tensorPositionIndices[tensorPositionOffset] = tensorPositionIdx;
        for (int32_t i = 0; i < packedTreeMaskLen; ++i)
        {
            packedDraftTreeMask[packedTreeMaskOffset + i] = packedTreeMask[i];
        }
    }
}

__global__ void assembleCasualTreeAndSelectIndicesKernel(int32_t const* sequenceStartIndices, int32_t* packedTreeMasks,
    int32_t* tensorPositionIndices, int64_t* selectTokenIndices, int32_t* sequenceContextLengths,
    int32_t const acceptedTokenNum)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    // 32 should be sufficient for accepted tokens from base model.
    int32_t packedTreeMask{0};
    if (tokenIdx < acceptedTokenNum)
    {
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            packedTreeMask |= (1 << i);
        }

        // Packed tree mask shall have layout of [batch, accepted-token-num, divup(accepted-token-num, 32)].
        // Here the accepted token num should be strictly smaller than 32.
        // tensor position indices have layout of [batch, accepted-token-num], the offset will be identical to packed
        // tree mask.
        int32_t const packedTreeMaskOffset = batchIdx * acceptedTokenNum + tokenIdx;
        packedTreeMasks[packedTreeMaskOffset] = packedTreeMask;
        tensorPositionIndices[packedTreeMaskOffset] = sequenceStartIndices[batchIdx] + tokenIdx;
    }
    if (threadIdx.x == 0)
    {
        selectTokenIndices[batchIdx] = acceptedTokenNum - 1;
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + acceptedTokenNum;
    }
}

void prepareEaglePrefillInputs(rt::Tensor& sequenceContextLengths, rt::Tensor& selectTokenIndices,
    int32_t const sequenceLength, cudaStream_t stream)
{
    check::check(sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for the input tensors.");
    check::check(sequenceContextLengths.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64,
        "Context-length input shall be INT32 and select-token-indices shall be INT64.");
    uint32_t const batchSize = sequenceContextLengths.getShape()[0];

    // Assign one warp for each batch.
    dim3 const blockDim{32};
    dim3 const gridDim{batchSize};
    prepareEaglePrefillInputKernel<<<gridDim, blockDim, 0, stream>>>(
        sequenceContextLengths.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(), sequenceLength);
}

void prepareEagleDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
    rt::Tensor const& sequenceStartIndices, rt::Tensor& packedDraftTreeMask, rt::Tensor& tensorPositionIndices,
    rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths, cudaStream_t stream)
{
    check::check(draftTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
            && sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedDraftTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftTreeMask.getDataType() == DataType::kINT8 && draftTreeLength.getDataType() == DataType::kINT32
            && sequenceStartIndices.getDataType() == DataType::kINT32
            && packedDraftTreeMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64
            && sequenceContextLengths.getDataType() == DataType::kINT32,
        "Data type check failed for the input tensors.");

    uint32_t const batchSize = draftTreeMask.getShape()[0];
    int32_t const paddedDraftTreeSize = draftTreeMask.getShape()[1];
    int32_t const selectTokenLength = selectTokenIndices.getShape()[1];

    check::check(tensorPositionIndices.getShape()[1] == paddedDraftTreeSize,
        "Select token indices shall have shape [batch, padded-draft-tree-size].");

    // Round up block size to multiple of warp
    uint32_t const blocksize = divUp(paddedDraftTreeSize, 32) * 32;
    // Perform tree mask packing and tensor position indices.
    dim3 const blockDim1{blocksize};
    dim3 const gridDim1{batchSize};
    assembleDraftTreeDescKernel<<<gridDim1, blockDim1, 0, stream>>>(draftTreeMask.dataPointer<int8_t>(),
        draftTreeLength.dataPointer<int32_t>(), sequenceStartIndices.dataPointer<int32_t>(),
        packedDraftTreeMask.dataPointer<int32_t>(), tensorPositionIndices.dataPointer<int32_t>(), paddedDraftTreeSize);

    // Perform misc input setup, assign one warp for each batch since selectTokenLength is around 8 ~ 12.
    dim3 const blockDim2{32};
    dim3 const gridDim2{batchSize};
    prepareEagleDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(draftTreeLength.dataPointer<int32_t>(),
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), selectTokenLength, paddedDraftTreeSize);
}

void prepareEagleAcceptDecodeTokenInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor& packedTreeMask,
    rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths,
    int32_t const acceptedTokenNum, cudaStream_t stream)
{
    check::check(sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(sequenceStartIndices.getDataType() == DataType::kINT32
            && packedTreeMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64,
        "Data type shall all be INT32 for these tensors.");
    check::check(packedTreeMask.getShape()[1] == acceptedTokenNum && acceptedTokenNum < 32,
        "Current kernel implementation support accepted token <= 32 per batch. "
        "Packed tree mask shall have shape [batch, accepted-token-num, 1].");
    uint32_t const batchSize = sequenceStartIndices.getShape()[0];
    // Round up block size to multiple of warp size.
    uint32_t const blocksize = divUp(acceptedTokenNum, 32) * 32;
    // Perform casual tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};

    assembleCasualTreeAndSelectIndicesKernel<<<gridDim, blockDim, 0, stream>>>(
        sequenceStartIndices.dataPointer<int32_t>(), packedTreeMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(),
        sequenceContextLengths.dataPointer<int32_t>(), acceptedTokenNum);
}

__global__ void prepareEagleBaseTreeDecodingInputKernel(int8_t const* baseTreeDecodingMask,
    int32_t const* sequenceStartIndices, int32_t* packedTreeMask, int32_t* tensorPositionIndices,
    int32_t* sequenceContextLengths, int64_t* selectTokenIndices, int32_t const treeSize)
{
    constexpr int32_t kNUM_MASK_PER_ENTRY{32};

    // Each thread will handle one token in the tree to setup the mask and tensor position indices.
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    if (tokenIdx == 0)
    {
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + treeSize;
    }

    int32_t const packedTreeMaskLen = (treeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;
    int32_t const sequenceStartIndex = sequenceStartIndices[batchIdx];

    // Unpacked tree mask formulate in the format of [batch, tree-size, tree-size].
    // Packed tree mask len is in format of [batch, tree-size, divup(tree-size, 32)].
    // Tensor position indices is in format of [batch, tree-size].
    int32_t const unpackedTreeMaskOffset = batchIdx * treeSize * treeSize + tokenIdx * treeSize;
    int32_t const packedTreeMaskOffset = batchIdx * treeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
    int32_t const tensorPositionOffset = batchIdx * treeSize + tokenIdx;
    int32_t const selectTokenOffset = batchIdx * treeSize + tokenIdx;

    if (tokenIdx < treeSize)
    {
        // With causal attention, the node will only attend to nodes "prior" to itself.
        int32_t attendNodeNum{0};
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            int8_t const maskFlag = baseTreeDecodingMask[unpackedTreeMaskOffset + i];
            if (maskFlag)
            {
                attendNodeNum += 1;
                packedTreeMask[packedTreeMaskOffset + i / kNUM_MASK_PER_ENTRY] |= (1 << (i % kNUM_MASK_PER_ENTRY));
            }
        }
        // A token always attend to itself, subtract 1 to reflect its position in the sequence.
        tensorPositionIndices[tensorPositionOffset] = sequenceStartIndex + attendNodeNum - 1;
        selectTokenIndices[selectTokenOffset] = tokenIdx;
    }
}

void prepareEagleBaseTreeDecodingInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
    rt::Tensor& packedBaseTreeDecodingMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream)
{
    check::check(baseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
            && sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedBaseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(baseTreeDecodingMask.getDataType() == DataType::kINT8
            && sequenceStartIndices.getDataType() == DataType::kINT32
            && packedBaseTreeDecodingMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64
            && sequenceContextLengths.getDataType() == DataType::kINT32,
        "Data type check failed for the input tensors.");

    uint32_t const batchSize = baseTreeDecodingMask.getShape()[0];
    int32_t const treeSize = baseTreeDecodingMask.getShape()[1];

    check::check(tensorPositionIndices.getShape()[1] == treeSize,
        "Tensor position indices shall have shape [batch, tree-size].");

    // Round up block size to multiple of warp
    uint32_t const blocksize = divUp(treeSize, 32) * 32;
    // Perform tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};
    assembleDraftTreeDescKernel<<<gridDim, blockDim, 0, stream>>>(baseTreeDecodingMask.dataPointer<int8_t>(), nullptr,
        sequenceStartIndices.dataPointer<int32_t>(), packedBaseTreeDecodingMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), treeSize);

    // Perform misc input setup, assign one warp for each batch since selectTokenLength is around 8 ~ 12.
    dim3 const blockDim2{32};
    dim3 const gridDim2{batchSize};
    prepareEagleDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(nullptr,
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), treeSize, treeSize);
}

template <int32_t HEAD_DIM, int32_t MAX_PATH>
__global__ void eagleBaseCommitKVCacheKernel(int32_t const* acceptedIndices, int32_t const* acceptLengths,
    int32_t const* kvCacheLengths, half* kvCacheBuffer, int32_t const activeBatchSize, int32_t const maxDepth,
    int32_t const numLayers, int32_t const maxBatchSize, int32_t const numHeads, int32_t const maxSeqLen)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Only HEAD_DIM = 64 or 128 are supported");
    DVec<half> tempBuffer[MAX_PATH];

    // The kernel have assumptions that:
    //     1. Each CTA will handle multiple heads.
    //     2. Each thread will copy 16 bytes of data (half[8]), each warp will copy 512 bytes (half[256]) data per
    //     iteration.
    //     3. Each CTA contains 128 threads (4 warps).
    //         blockDim.x = number of threads used to process 1 head = HEAD_DIM / DVec<half>::vec_size
    //         blockDim.y = number of heads handled by each CTA = 128 / blockDim.x
    //     4. The KVCache buffer has layout of [numLayers, maxBatchSize, 2, numHeads, maxSeqLen, HEAD_DIM]

    int32_t const tIdx = threadIdx.x;
    int32_t const tIdy = threadIdx.y;
    int32_t const bIdx = blockIdx.x;
    int32_t const headIdx = bIdx * blockDim.y + tIdy;

    int32_t const kvLayerIdx = headIdx / (activeBatchSize * 2 * numHeads);
    int32_t const kvBatchIdx = (headIdx % (activeBatchSize * 2 * numHeads)) / (2 * numHeads);
    int32_t const kvHeadIdx = headIdx % (2 * numHeads);

    int32_t const actualAcceptLength = acceptLengths[kvBatchIdx];
    int32_t const pastKvCacheLength = kvCacheLengths[kvBatchIdx];
    int32_t const kvCacheOffset = kvLayerIdx * maxBatchSize * 2 * numHeads * maxSeqLen * HEAD_DIM
        + kvBatchIdx * 2 * numHeads * maxSeqLen * HEAD_DIM + kvHeadIdx * maxSeqLen * HEAD_DIM
        + pastKvCacheLength * HEAD_DIM;

    // PHASE 1: Collect all accepted data into local temp buffer
    // Start from 1 since the root position will always be accepted.
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const acceptedIdx = acceptedIndices[kvBatchIdx * maxDepth + i];
        if (acceptedIdx >= 0 && acceptedIdx + pastKvCacheLength < maxSeqLen)
        {
            int32_t const srcOffset = kvCacheOffset + acceptedIdx * HEAD_DIM + tIdx * DVec<half>::vec_size;
            tempBuffer[i].load(kvCacheBuffer + srcOffset);
        }
    }

    // PHASE 2: Write from local temp buffer to final positions
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const dstOffset = kvCacheOffset + i * HEAD_DIM + tIdx * DVec<half>::vec_size;
        tempBuffer[i].store(kvCacheBuffer + dstOffset);
    }
}

template <int32_t MAX_PATH>
__global__ void eagleBaseAssembleHiddenStateKernel(int32_t const* acceptedIndices, int32_t const* acceptLengths,
    half* hiddenState, int32_t const batchSize, int32_t const maxDepth, int32_t const numTokens,
    int32_t const hiddenDim)
{
    DVec<half> tempBuffer[MAX_PATH];

    // The kernel have assumptions that:
    //     1. Each thread will copy 16 bytes of data (half[8]), each warp will copy 512 bytes (half[256]) data per
    //     iteration.
    //     2. Each CTA contains 128 threads (4 warps), a total of 128*8=1024 elements.
    //         Since hiddenDim can be very large, each CTA will handle part of a batch.
    //     3. The acceptedIndices has layout of [batch, max-depth]
    //     4. The hiddenState buffer has layout of [batch, num-tokens, hidden-dim]

    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int32_t const startIdx = dimIdx * DVec<half>::vec_size;

    if (startIdx >= hiddenDim)
    {
        return;
    }

    int32_t const actualAcceptLength = acceptLengths[batchIdx];
    int32_t const hiddenStateOffset = batchIdx * numTokens * hiddenDim;

    // PHASE 1: Collect all accepted data into local temp buffer
    // Start from 1 since the root position will always be accepted.
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const acceptedIdx = acceptedIndices[batchIdx * maxDepth + i];
        if (acceptedIdx >= 0 && acceptedIdx < numTokens)
        {
            int32_t const srcOffset = hiddenStateOffset + acceptedIdx * hiddenDim + dimIdx * DVec<half>::vec_size;
            tempBuffer[i].load(hiddenState + srcOffset);
        }
    }

    // PHASE 2: Write from local temp buffer to final positions
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const dstOffset = hiddenStateOffset + i * hiddenDim + dimIdx * DVec<half>::vec_size;
        tempBuffer[i].store(hiddenState + dstOffset);
    }
}

void eagleBaseCommitKVCacheAndAssembleHiddenState(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, rt::Tensor& kvCacheBuffer, rt::Tensor& hiddenState, cudaStream_t stream)
{
    check::check(acceptedIndices.getDeviceType() == rt::DeviceType::kGPU
            && acceptLengths.getDeviceType() == rt::DeviceType::kGPU
            && kvCacheBuffer.getDeviceType() == rt::DeviceType::kGPU
            && kvCacheLengths.getDeviceType() == rt::DeviceType::kGPU
            && hiddenState.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(acceptedIndices.getDataType() == DataType::kINT32 && acceptLengths.getDataType() == DataType::kINT32
            && kvCacheBuffer.getDataType() == DataType::kHALF && kvCacheLengths.getDataType() == DataType::kINT32
            && hiddenState.getDataType() == DataType::kHALF,
        "Data type validation failed: acceptedIndices, acceptLengths, and kvCacheLengths should be INT32; "
        "kvCacheBuffer and hiddenState should be HALF.");

    auto const acceptIndicesShape = acceptedIndices.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    auto const kvCacheBufferShape = kvCacheBuffer.getShape();
    auto const kvCacheLengthsShape = kvCacheLengths.getShape();
    auto const hiddenStateShape = hiddenState.getShape();
    check::check(acceptIndicesShape.getNumDims() == 2, "acceptedIndices should be 2D tensor [batch, max-depth].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths should be 1D tensor [batch].");
    check::check(kvCacheBufferShape.getNumDims() == 6,
        "kvCacheBuffer should be 6D tensor [num-layers, batch, 2, num-heads, max-seq-len, hidden-size-per-head].");
    check::check(kvCacheLengthsShape.getNumDims() == 1, "kvCacheLengths should be 1D tensor [batch].");
    check::check(hiddenStateShape.getNumDims() == 3,
        "hiddenState should be 3D tensor [batch, draft-tree-size, base-hidden-dim].");

    uint32_t const batchSize = acceptIndicesShape[0];
    int32_t const maxDepth = acceptIndicesShape[1];
    check::check(acceptLengthsShape[0] == batchSize, "acceptLengths should have same batch size as acceptedIndices.");

    constexpr int32_t MAX_PATH{8};
    check::check(maxDepth <= (MAX_PATH + 1), "maxDepth > 9 is not supported by the kernel.");

    // Each CTA has 128 threads, each thread will handle vecSize elements.
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t threadsPerBlock = 128;

    // Commit KVCache
    int32_t const numLayers = kvCacheBufferShape[0];
    int32_t const maxBatchSize = kvCacheBufferShape[1];
    int32_t const numHeads = kvCacheBufferShape[3];
    int32_t const maxSeqLen = kvCacheBufferShape[4];
    int32_t const headDim = kvCacheBufferShape[5];

    uint32_t const bDimX = headDim / vecSize;
    uint32_t const headPerBlock = threadsPerBlock * vecSize / headDim;
    uint32_t const totalNumHeads = numLayers * batchSize * 2 * numHeads;
    uint32_t const totalNumBlocks = (totalNumHeads + headPerBlock - 1) / headPerBlock;

    dim3 const blockDim1(bDimX, headPerBlock);
    dim3 const gridDim1{totalNumBlocks};

    switch (headDim)
    {
    case 64:
        eagleBaseCommitKVCacheKernel<64, MAX_PATH>
            <<<gridDim1, blockDim1, 0, stream>>>(acceptedIndices.dataPointer<int32_t>(),
                acceptLengths.dataPointer<int32_t>(), kvCacheLengths.dataPointer<int32_t>(),
                kvCacheBuffer.dataPointer<half>(), batchSize, maxDepth, numLayers, maxBatchSize, numHeads, maxSeqLen);
        break;
    case 128:
        eagleBaseCommitKVCacheKernel<128, MAX_PATH>
            <<<gridDim1, blockDim1, 0, stream>>>(acceptedIndices.dataPointer<int32_t>(),
                acceptLengths.dataPointer<int32_t>(), kvCacheLengths.dataPointer<int32_t>(),
                kvCacheBuffer.dataPointer<half>(), batchSize, maxDepth, numLayers, maxBatchSize, numHeads, maxSeqLen);
        break;
    default:
        throw std::runtime_error(
            "Only HEAD_DIM = 64 or 128 are supported by eagleBaseCommitKVCacheAndAssembleHiddenState, current HEAD_DIM "
            "= "
            + std::to_string(headDim));
    }

    // Assemble Hidden State
    int32_t const numTokens = hiddenStateShape[1];
    int32_t const hiddenDim = hiddenStateShape[2];
    check::check(hiddenDim % vecSize == 0, "hiddenDim must be divisible by vecSize.");

    uint32_t const dimPerBlock = threadsPerBlock * vecSize;
    uint32_t const gridY = (hiddenDim + dimPerBlock - 1) / dimPerBlock;
    dim3 const blockDim2(threadsPerBlock);
    dim3 const gridDim2{batchSize, gridY};

    eagleBaseAssembleHiddenStateKernel<MAX_PATH><<<gridDim2, blockDim2, 0, stream>>>(
        acceptedIndices.dataPointer<int32_t>(), acceptLengths.dataPointer<int32_t>(), hiddenState.dataPointer<half>(),
        batchSize, maxDepth, numTokens, hiddenDim);
}

} // namespace kernel
} // namespace drivellm