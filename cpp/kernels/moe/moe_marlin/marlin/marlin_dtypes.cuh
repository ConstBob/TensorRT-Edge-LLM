/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Adapted from https://github.com/vllm-project/vllm/blob/v0.14.0/csrc/quantization/gptq_marlin/marlin_dtypes.cuh
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */
#ifndef _data_types_cuh
#define _data_types_cuh
#include "common/cudaMacros.h"
#include "marlin.cuh"
#include "scalar_type.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME
{

template <long scalar_type_id>
class MarlinScalarType
{
};

template <>
class MarlinScalarType<trt_edgellm::marlin_dtypes::kFloat16.id()>
{
public:
    using scalar_t = half;
    using scalar_t2 = half2;
    using scalar_t4 = half2;
    using scalar_32bit_t = half2;

    // Matrix fragments for tensor core instructions; their precise layout is
    // documented here:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    using FragA = Vec<half2, 4>;
    using FragB = Vec<half2, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<half2, 1>;
#if SUPPORTS_FP8
    using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
#endif
    using FragZP = Vec<half2, 4>;

    static __device__ float inline num2float(half const x)
    {
        return __half2float(x);
    }

    static __device__ half2 inline num2num2(half const x)
    {
        return __half2half2(x);
    }

    static __device__ half2 inline nums2num2(half const x1, half const x2)
    {
        return __halves2half2(x1, x2);
    }

    static __host__ __device__ half inline float2num(float const x)
    {
        return __float2half(x);
    }

    static __host__ __device__ float2 inline num22float2(half2 const x)
    {
        return __half22float2(x);
    }
};

template <>
class MarlinScalarType<trt_edgellm::marlin_dtypes::kBFloat16.id()>
{
public:
    using scalar_t = nv_bfloat16;
    using scalar_t2 = nv_bfloat162;
    using scalar_t4 = nv_bfloat162;
    using scalar_32bit_t = nv_bfloat162;

    using FragA = Vec<nv_bfloat162, 4>;
    using FragB = Vec<nv_bfloat162, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<nv_bfloat162, 1>;
#if SUPPORTS_FP8
    using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
#endif
    using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static __device__ float inline num2float(nv_bfloat16 const x)
    {
        return __bfloat162float(x);
    }

    static __device__ nv_bfloat162 inline num2num2(nv_bfloat16 const x)
    {
        return __bfloat162bfloat162(x);
    }

    static __device__ nv_bfloat162 inline nums2num2(nv_bfloat16 const x1, nv_bfloat16 const x2)
    {
        return __halves2bfloat162(x1, x2);
    }

    static __host__ __device__ nv_bfloat16 inline float2num(float const x)
    {
        return __float2bfloat16(x);
    }

    static __host__ __device__ float2 inline num22float2(nv_bfloat162 const x)
    {
        return __bfloat1622float2(x);
    }
#endif
};

#if SUPPORTS_FP8
template <>
class MarlinScalarType<trt_edgellm::marlin_dtypes::kFE4M3fn.id()>
{
public:
    using scalar_t = __nv_fp8_e4m3;
    using scalar_t2 = __nv_fp8x2_e4m3;
    using scalar_t4 = __nv_fp8x4_e4m3;
    using scalar_32bit_t = __nv_fp8x4_e4m3;

    using FragA = Vec<__nv_fp8x4_e4m3, 4>;
    using FragB = Vec<__nv_fp8x4_e4m3, 2>;
    using FragC = Vec<float, 4>;
    using FragZP = Vec<__nv_fp8x2_e4m3, 4>;

    static __host__ __device__ float2 inline num22float2(__nv_fp8x2_e4m3 const x)
    {
        return (float2) x;
    }
};
#endif

template <>
class MarlinScalarType<trt_edgellm::marlin_dtypes::kS8.id()>
{
public:
    using scalar_t = int8_t;
    using scalar_t2 = int16_t;
    using scalar_t4 = int32_t;
    using scalar_32bit_t = int32_t;

    using FragA = Vec<int32_t, 4>;
    using FragB = Vec<int32_t, 2>;
    using FragC = Vec<float, 4>;
    using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2
{
};

template <>
class MarlinScalarType2<half> : public MarlinScalarType<trt_edgellm::marlin_dtypes::kFloat16.id()>
{
};

template <>
class MarlinScalarType2<nv_bfloat16> : public MarlinScalarType<trt_edgellm::marlin_dtypes::kBFloat16.id()>
{
};

#if SUPPORTS_FP8
template <>
class MarlinScalarType2<__nv_fp8_e4m3> : public MarlinScalarType<trt_edgellm::marlin_dtypes::kFE4M3fn.id()>
{
};
#endif
template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<trt_edgellm::marlin_dtypes::kS8.id()>
{
};

} // namespace MARLIN_NAMESPACE_NAME

#endif
