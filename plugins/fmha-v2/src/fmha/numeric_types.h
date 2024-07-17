/***************************************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdint>
#pragma once

#if CUDART_VERSION >= 11080
// TODO Better way?
#define FMHA_CUDA_SUPPORTS_FP8 true
#endif
#include <cuda_bf16.h>
#if FMHA_CUDA_SUPPORTS_FP8
#include <cuda_fp8.h>
#endif
namespace fmha {

using fp16_t = uint16_t;
using fp32_t = float;
using tf32_t = uint32_t;
using bf16_t = nv_bfloat16;
#if FMHA_CUDA_SUPPORTS_FP8
using e4m3_t = __nv_fp8_e4m3;
using e5m2_t = __nv_fp8_e5m2;
#else
using e4m3_t = char;
using e5m2_t = char;
#endif

static constexpr float MAX_E4M3 =   448.f; //0x7E 2^8  * 1.75
static constexpr float MAX_E5M2 = 57344.f; //0x7B 2^15 * 1.75

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha

