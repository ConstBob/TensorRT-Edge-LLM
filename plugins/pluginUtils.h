/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

inline void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorName(err));
        throw std::runtime_error(cudaGetErrorName(err));
    }
}

inline void checkCu(CUresult err)
{
    if (err != CUDA_SUCCESS)
    {
        char const* str = nullptr;
        if (cuGetErrorName(err, &str) != CUDA_SUCCESS)
        {
            str = "A cuda driver API error happened, but we failed to query the error name\n";
        }
        printf("%s\n", str);
        throw std::runtime_error(str);
    }
}

inline void check(bool condition, std::string errorMsg)
{
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

inline int getSMVersion()
{
    int device{-1};
    checkCuda(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    checkCuda(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    checkCuda(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}