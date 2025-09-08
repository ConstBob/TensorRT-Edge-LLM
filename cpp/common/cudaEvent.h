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
