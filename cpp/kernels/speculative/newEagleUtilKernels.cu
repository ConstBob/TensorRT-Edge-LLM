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
    int32_t const draftTreeSize = draftTreeSizes[batchIdx];

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

__global__ void assembleDraftTreeDescKernel(int8_t const* draftTreeMask, int32_t const* draftTreeSize,
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
    int32_t const actualDraftTreeSize = draftTreeSize[batchIdx];
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
    uint32_t const batchSize = static_cast<uint32_t>(sequenceContextLengths.getShape()[0]);

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

    uint32_t const batchSize = static_cast<uint32_t>(draftTreeMask.getShape()[0]);
    int32_t const paddedDraftTreeSize = static_cast<int32_t>(draftTreeMask.getShape()[1]);
    int32_t const selectTokenLength = static_cast<int32_t>(selectTokenIndices.getShape()[1]);

    check::check(tensorPositionIndices.getShape()[1] == paddedDraftTreeSize,
        "Select token indices shall have shape [batch, padded-draft-tree-size].");

    // Round up block size to multiple of warp
    uint32_t const blocksize = static_cast<uint32_t>(divUp(paddedDraftTreeSize, 32) * 32);
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
    uint32_t const batchSize = static_cast<uint32_t>(sequenceStartIndices.getShape()[0]);
    // Round up block size to multiple of warp size.
    uint32_t const blocksize = static_cast<uint32_t>(divUp(acceptedTokenNum, 32) * 32);
    // Perform casual tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};

    assembleCasualTreeAndSelectIndicesKernel<<<gridDim, blockDim, 0, stream>>>(
        sequenceStartIndices.dataPointer<int32_t>(), packedTreeMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(),
        sequenceContextLengths.dataPointer<int32_t>(), acceptedTokenNum);
}

} // namespace kernel
} // namespace drivellm