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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <set>
#include <stdexcept>
#include <string>

#include <NvInferRuntime.h>

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