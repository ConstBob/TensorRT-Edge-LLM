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

#pragma once

#include "checkMacros.h"
#include "cuda_runtime_api.h"
#include <memory>
#include <type_traits>
namespace drivellm
{

class CudaEvent
{

public:
    using pointer = cudaEvent_t;

    explicit CudaEvent(unsigned int flags = cudaEventDefault)
    {
        pointer event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
        mEvent = EventPtr(event, Deleter());
    }

    pointer get() const
    {
        return mEvent.get();
    }

    void synchronize() const
    {
        CUDA_CHECK(cudaEventSynchronize(get()));
    }

    void record() const
    {
        CUDA_CHECK(cudaEventRecord(get()));
    }

private:
    class Deleter
    {
    public:
        Deleter() = default;

        constexpr void operator()(pointer event) const
        {
            if (event != nullptr)
            {
                CUDA_CHECK(cudaEventDestroy(event));
            }
        }
    };

    using element_type = std::remove_pointer_t<pointer>;
    using EventPtr = std::unique_ptr<element_type, Deleter>;

    EventPtr mEvent;
};

} // namespace drivellm
