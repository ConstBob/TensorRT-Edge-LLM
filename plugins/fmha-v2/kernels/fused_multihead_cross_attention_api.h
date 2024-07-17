/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cuda.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>
#include <tuple>

using Params_v1   = bert::Fused_multihead_attention_params_v1;
using Params_v2   = bert::Fused_multihead_attention_params_v2;
using Params_mhca = bert::Fused_multihead_attention_params_mhca;
using Launch_params = bert::Fused_multihead_attention_launch_params;

void run_fmha_v2_fp16_64_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_128_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_256_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_64_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_128_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_256_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_384_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_512_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_384_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_512_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_get_max_heads_per_wave(int*);

inline void run_fmha_v1(Params_v1 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        int sm,
                        cudaStream_t stream=0){
const size_t s                 = params.s;
const size_t b                 = params.b;
const size_t d                 = params.d;
const bool force_unroll        = launch_params.force_unroll;
const bool ignore_b1opt        = launch_params.ignore_b1opt;

const bool use_flash_attention = false;

if( false ) {}
else {
    assert(false && "Unsupported config.");
}

}

// Note: transitioning to moving kernel launch parameters into launch_params to reduce the
// occurrences the interface needs to be modified
inline void run_fmha_v2(Params_v2 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        int sm,
                        cudaStream_t stream=0) {

const size_t s = params.s;
const size_t b = params.b;
const size_t h = params.h;
const size_t d = params.d;

const bool interleaved         = launch_params.interleaved;
const bool force_unroll        = launch_params.force_unroll;
const bool ignore_b1opt        = launch_params.ignore_b1opt;
const bool force_fp32_acc      = launch_params.force_fp32_acc;
const bool warp_specialization = launch_params.warp_specialization;
const bool use_tma             = launch_params.use_tma;
const bool use_flash_attention = launch_params.flash_attention;
// tiled variant uses ldgsts
const bool use_tiled = launch_params.use_granular_tiling;

if( data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_64_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_64_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_128_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_128_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_256_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_256_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_64_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_64_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_128_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_128_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_256_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_256_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_384_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_384_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_512_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_512_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_384_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_384_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90
    && !use_flash_attention && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_512_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_512_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90
    && use_flash_attention && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90
    && use_flash_attention && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90
    && use_flash_attention && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90
    && use_flash_attention && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80
    && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80
    && !use_tiled && use_flash_attention && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_nl(params, launch_params, stream);

} 
else {
    assert(false && "Unsupported config.");
}

}

#if true // fmhca api header

inline void run_fmhca(Params_mhca &params,
                      const Launch_params &launch_params,
                      Data_type data_type,
                      int sm,
                      cudaStream_t stream=0) {

const size_t s_kv   = params.s;
const size_t b      = params.b;
const size_t d      = params.d_padded;

const bool interleaved  = launch_params.interleaved;
const bool force_unroll = launch_params.force_unroll;
const bool ignore_b1opt = launch_params.ignore_b1opt;

if( false ) {}
else {
    assert(false && "Unsupported config");
}

}

#endif // fmhca api header

inline std::tuple<size_t, size_t, size_t> get_warps(Launch_params& launch_params,
                                                    int sm,
                                                    Data_type data_type,
                                                    size_t s,
                                                    size_t b,
                                                    size_t d,
                                                    int version) {
    size_t warps_m, warps_n, warps_k = 1;
    const bool interleaved           = launch_params.interleaved;
    const bool use_tma               = launch_params.use_tma;
    const bool force_unroll          = launch_params.force_unroll;
    const bool ignore_b1opt          = launch_params.ignore_b1opt;
    const bool use_flash_attention   = launch_params.flash_attention;
    // tiled variant uses ldgsts
    const bool use_tiled             = launch_params.use_granular_tiling;
    const bool warp_specialization   = launch_params.warp_specialization;

if( data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else {
	assert(false && "Unsupported config");
}

    return std::make_tuple(warps_m, warps_n, warps_k);
}

// The constant is defined in "setup.py".
constexpr int MAX_STGS_PER_LOOP = 4;

// The number of CTAs and threads per CTA to launch the kernel.
inline void get_grid_size(int &heads_per_wave,
                          int &ctas_per_head,
                          int sm,
                          Data_type data_type,
                          size_t b,
                          size_t s,
                          size_t h,
                          size_t d,
                          bool use_multi_ctas,
                          int version) {

    // Determine the number of CTAs per head (kernel constant).
    int max_heads_per_wave = 0;
    ctas_per_head = 1;
    heads_per_wave = b*h;


    // Adjust the number of heads per wave.
    if( heads_per_wave > max_heads_per_wave ) {
        heads_per_wave = max_heads_per_wave;
    }
}

