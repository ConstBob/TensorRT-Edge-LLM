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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(SanityCheck, InitializeCUDA)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    // Ensure the function executes successfully
    ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
    // We assume at least one GPU is present in the system
    ASSERT_GT(device_count, 0) << "CUDA device is not available";

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        // Check if properties were fetched successfully
        ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceProperties failed for device " << i << ": "
                                    << cudaGetErrorString(err);
    }
}