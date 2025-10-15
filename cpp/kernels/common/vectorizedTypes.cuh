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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace drivellm
{
namespace kernel
{

//! Data vectorization helper to load/store data from global memory.
//! This provides efficient vectorized memory access for CUDA kernels.
template <typename T>
struct DVec
{
    static constexpr uint32_t vec_size = 0;
    inline T& operator[](uint32_t idx);
    inline T const& operator[](uint32_t idx) const;
    inline void load(T const* ptr);
    inline void store(T* ptr) const;
};

//! Specialization for float[8] - aligns with load/store of activation data.
//! Use this to load cos/sin cache and other float data.
template <>
struct DVec<float>
{
    float4 data[2];
    static constexpr uint32_t vec_size = 8;

    __device__ __forceinline__ float& operator[](uint32_t idx)
    {
        return ((float*) (data))[idx];
    }

    __device__ __forceinline__ float const& operator[](uint32_t idx) const
    {
        return ((float const*) (data))[idx];
    }

    __device__ __forceinline__ void load(float const* ptr)
    {
        data[0] = *(reinterpret_cast<float4 const*>(ptr));
        data[1] = *(reinterpret_cast<float4 const*>(ptr + 4));
    }

    __device__ __forceinline__ void store(float* ptr) const
    {
        *(reinterpret_cast<float4*>(ptr)) = data[0];
        *(reinterpret_cast<float4*>(ptr + 4)) = data[1];
    }
};

//! Specialization for half[8] - enforces granularity of 16 bytes load/store from global memory.
//! This is the most commonly used vectorization for half precision data.
template <>
struct DVec<half>
{
    uint4 data;
    static constexpr uint32_t vec_size = 8;

    __device__ __forceinline__ half& operator[](uint32_t idx)
    {
        return reinterpret_cast<half*>(&data)[idx];
    }

    __device__ __forceinline__ half const& operator[](uint32_t idx) const
    {
        return reinterpret_cast<half const*>(&data)[idx];
    }

    __device__ __forceinline__ void load(half const* ptr)
    {
        data = *(reinterpret_cast<uint4 const*>(ptr));
    }

    __device__ __forceinline__ void store(half* ptr) const
    {
        *(reinterpret_cast<uint4*>(ptr)) = data;
    }
};

} // namespace kernel
} // namespace drivellm
