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

#include <cuda_fp16.h>
#include <stdint.h>

namespace drivellm
{
namespace kernel
{

void gemv_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);

void gemm_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);
} // namespace kernel
} // namespace drivellm