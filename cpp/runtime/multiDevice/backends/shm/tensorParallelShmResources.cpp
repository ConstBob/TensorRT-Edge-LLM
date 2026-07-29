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

#include "runtime/multiDevice/backends/shm/tensorParallelShmResources.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "kernels/multiDeviceKernels/shmAllReduce.h"
#include "runtime/multiDevice/multiDevicePluginCommRegistry.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

class TensorParallelShmResources final : public PluginAllReducePathResources
{
public:
    TensorParallelShmResources(int32_t tpSize, std::vector<int32_t> localRanks, std::vector<int32_t> localDevices,
        ShmAllReduceConfig const& shmAllReduceConfig, std::string shmSessionName)
        : mTpSize(tpSize)
        , mLocalRanks(std::move(localRanks))
        , mLocalDevices(std::move(localDevices))
        , mShmAllReduceConfig(shmAllReduceConfig)
        , mShmSessionName(std::move(shmSessionName))
    {
        try
        {
            initializeState();
            ELLM_CHECK(registerWithPlugin(), "Failed to register SHM resources with the AllReduce plugin.");
        }
        catch (...)
        {
            unregisterFromPlugin();
            destroyState();
            throw;
        }
    }

    ~TensorParallelShmResources() noexcept override
    {
        unregisterFromPlugin();
        destroyState();
    }

    AllReducePathType type() const noexcept override
    {
        return AllReducePathType::kShm;
    }

    bool registered() const noexcept override
    {
        return mShmState != nullptr && !mLocalRanks.empty() && mRegisteredCount == mLocalRanks.size();
    }

private:
    bool registerWithPlugin() noexcept
    {
        if (registered())
        {
            return true;
        }

        while (mRegisteredCount < mLocalRanks.size())
        {
            size_t const index = mRegisteredCount;
            int32_t const rank = mLocalRanks[index];
            int32_t const device = mLocalDevices[index];
            if (!registerShmAllReduceForPlugin(mShmState, device, rank))
            {
                unregisterFromPlugin();
                return false;
            }
            ++mRegisteredCount;
            LOG_INFO(
                "[TP rank %d/%d] Registered SHM AllReduce state for device %d "
                "(maxElements=%lld, allReduceElementThreshold=%lld, fp8SmallPathElementThreshold=%lld).",
                rank, mTpSize, device, static_cast<long long>(mShmState->maxElements),
                static_cast<long long>(mShmState->allReduceElementThreshold),
                static_cast<long long>(mShmState->fp8SmallPathElementThreshold));
        }
        return registered();
    }

    bool unregisterFromPlugin() noexcept
    {
        bool success = true;
        while (mRegisteredCount > 0)
        {
            size_t const index = mRegisteredCount - 1;
            int32_t const rank = mLocalRanks[index];
            int32_t const device = mLocalDevices[index];
            if (!unregisterShmAllReduceForPlugin(mShmState, device, rank))
            {
                success = false;
            }
            --mRegisteredCount;
        }
        return success;
    }

    void initializeState()
    {
        ELLM_CHECK(mTpSize == 2, format::fmtstr("SHM AllReduce supports TP=2 only: tpSize=%d", mTpSize));
        ELLM_CHECK(isFullLocalParallelGroup(mTpSize, mLocalRanks),
            format::fmtstr("SHM AllReduce requires all TP ranks to be local and ordered: localRanks=%zu, tpSize=%d",
                mLocalRanks.size(), mTpSize));
        ELLM_CHECK(mShmAllReduceConfig.shmMaxElements > 0,
            format::fmtstr("SHM AllReduce max elements must be positive: maxAllReduceElements=%lld",
                static_cast<long long>(mShmAllReduceConfig.shmMaxElements)));

        CudaDeviceGuard deviceGuard(mLocalDevices.front());
        mShmState = kernels::shmAllReduceInit(mTpSize, mShmAllReduceConfig.shmMaxElements,
            mShmAllReduceConfig.shmAllReduceElementThreshold, mShmAllReduceConfig.shmFp8SmallPathElementThreshold,
            mShmSessionName.c_str());
        ELLM_CHECK(mShmState != nullptr, "shmAllReduceInit failed.");
    }

    void destroyState() noexcept
    {
        if (mShmState != nullptr)
        {
            kernels::shmAllReduceDestroy(mShmState);
            mShmState = nullptr;
        }
    }

    int32_t mTpSize{1};
    std::vector<int32_t> mLocalRanks{};
    std::vector<int32_t> mLocalDevices{};
    ShmAllReduceConfig mShmAllReduceConfig{};
    std::string mShmSessionName{};
    kernels::ShmAllReduceState* mShmState{nullptr};
    size_t mRegisteredCount{0};
};

} // namespace

std::unique_ptr<PluginAllReducePathResources> createTensorParallelShmResources(int32_t tpSize,
    std::vector<int32_t> localRanks, std::vector<int32_t> localDevices, ShmAllReduceConfig const& config,
    std::string sessionName)
{
    return std::make_unique<TensorParallelShmResources>(
        tpSize, std::move(localRanks), std::move(localDevices), config, std::move(sessionName));
}

} // namespace rt
} // namespace trt_edgellm
