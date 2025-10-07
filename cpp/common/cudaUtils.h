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
#include <cuda_runtime.h>

namespace drivellm
{

template <typename T1, typename T2>
inline size_t divUp(const T1& a, const T2& n)
{
    size_t tmp_a = static_cast<size_t>(a);
    size_t tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

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
} // namespace drivellm
