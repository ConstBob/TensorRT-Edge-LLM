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

#include "common/checkMacros.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <numeric>
#include <tuple>

namespace drivellm
{

inline int getDevice()
{
    int current_dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

template <typename T1, typename T2>
inline size_t divUp(const T1& a, const T2& n)
{
    size_t tmp_a = static_cast<size_t>(a);
    size_t tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

inline bool isCudaLaunchBlocking()
{
    static bool firstCall = true;
    static bool result = false;

    if (firstCall)
    {
        char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
        result = env != nullptr && std::string(env) == "1";
        firstCall = false;
    }

    return result;
}

/// Get the memory info
/// \return The free and total amount of memory in bytes
inline std::tuple<size_t, size_t> getDeviceMemoryInfo()
{
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return {free, total};
}

/// Get the SM version of GPU
/// \return The SM version of current GPU
inline int getSMVersion()
{
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

#ifdef NDEBUG
#define sync_check_cuda_error()                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t result = cudaDeviceSynchronize();                                                                  \
        CUDA_CHECK(result);                                                                                            \
    } while (0)
#else
#define sync_check_cuda_error()
#endif

} // namespace drivellm
