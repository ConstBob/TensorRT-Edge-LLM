/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/state/kvPageTable.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

namespace trt_edgellm
{
namespace rt
{

namespace
{
//! V id of a K id: the sentinel maps to itself, a live id shifts by numPages.
int32_t deriveV(int32_t k, int32_t numPages)
{
    return k == kUNUSED_PAGE_ENTRY ? kUNUSED_PAGE_ENTRY : k + numPages;
}
} // namespace

KVPageTable::KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages)
    : mMaxBatch(maxBatch)
    , mMaxPagesPerSeq(maxPagesPerSeq)
    , mNumPages(numPages)
{
    check::check(maxBatch > 0, "KVPageTable: maxBatch must be positive.");
    check::check(maxPagesPerSeq > 0, "KVPageTable: maxPagesPerSeq must be positive.");
    check::check(numPages > 0, "KVPageTable: numPages must be positive.");

    mHost.assign(static_cast<size_t>(maxBatch) * 2 * maxPagesPerSeq, kUNUSED_PAGE_ENTRY);
    mDevice = rt::Tensor(
        Coords{maxBatch, 2, maxPagesPerSeq}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "KVPageTable::kernelView");
}

void KVPageTable::setIdentity()
{
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        int32_t* kRow = mutableHostRow(b);
        int32_t* vRow = kRow + mMaxPagesPerSeq;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = b * mMaxPagesPerSeq + j;
            kRow[j] = k;
            vRow[j] = deriveV(k, mNumPages);
        }
    }
    mIsIdentity = true;
}

void KVPageTable::setRow(int32_t slot, int32_t const* kPageIds, int32_t count)
{
    check::check(slot >= 0 && slot < mMaxBatch, "KVPageTable::setRow: slot out of range.");
    check::check(count >= 0 && count <= mMaxPagesPerSeq, "KVPageTable::setRow: count out of range.");

    mIsIdentity = false;
    int32_t* kRow = mutableHostRow(slot);
    int32_t* vRow = kRow + mMaxPagesPerSeq;
    for (int32_t j = 0; j < count; ++j)
    {
        int32_t const k = kPageIds[j];
        kRow[j] = k;
        vRow[j] = deriveV(k, mNumPages);
    }
    for (int32_t j = count; j < mMaxPagesPerSeq; ++j)
    {
        kRow[j] = kUNUSED_PAGE_ENTRY;
        vRow[j] = kUNUSED_PAGE_ENTRY;
    }
}

bool KVPageTable::checkInvariants(std::string& error) const
{
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        int32_t const* kRow = hostRow(b);
        bool sawSentinel = false;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = kRow[j];
            if (k == kUNUSED_PAGE_ENTRY)
            {
                sawSentinel = true;
                continue;
            }
            if (sawSentinel)
            {
                error = "KVPageTable: live page id " + std::to_string(k) + " follows sentinel at slot "
                    + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
            if (k < 0 || k >= mNumPages)
            {
                error = "KVPageTable: page id " + std::to_string(k) + " out of range [0, " + std::to_string(mNumPages)
                    + ") at slot " + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
        }
    }
    return true;
}

void KVPageTable::upload(cudaStream_t stream)
{
    std::string error;
    ELLM_CHECK(checkInvariants(error), "KVPageTable::upload: " + error);

    CUDA_CHECK(cudaMemcpyAsync(
        mDevice.rawPointer(), mHost.data(), mHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

rt::Tensor const& KVPageTable::kernelView() const
{
    return mDevice;
}

rt::Tensor& KVPageTable::kernelView()
{
    return mDevice;
}

int32_t const* KVPageTable::hostRow(int32_t slot) const
{
    return mHost.data() + static_cast<size_t>(slot) * 2 * mMaxPagesPerSeq;
}

int32_t* KVPageTable::mutableHostRow(int32_t slot)
{
    return mHost.data() + static_cast<size_t>(slot) * 2 * mMaxPagesPerSeq;
}

} // namespace rt
} // namespace trt_edgellm
