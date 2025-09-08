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