/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include "mha_stdheaders.h"

#define STATIC_NB_K_HEADS 0
#if STATIC_NB_K_HEADS
#define NB_K_HEADS 2
#endif

// allowed values are multiples of 16 in range [16, 256]
#ifndef HEAD_ELEMS
#define HEAD_ELEMS 128
#endif

// 1 means fp16 and 0 means bf16 input/output
#ifndef INPUT_FP16
#define INPUT_FP16 1
#endif

// Don't modify
#if INPUT_FP16
#define INPUT_ELEM half
#define INPUT_ELEM2 half2
#else
#define INPUT_ELEM __nv_bfloat16
#define INPUT_ELEM2 __nv_bfloat162
#endif

// For beam search. Allowed values: 1, 4
#ifndef BEAM_WIDTH
#define BEAM_WIDTH 1
#endif

// nbQHeads / nbKHeads for MQA/GQA
#ifndef HEAD_GRP_SIZE
#define HEAD_GRP_SIZE 4
#endif

// 0: half/bf16 based on INPUT_FP16; 1: int8_t; 2: __nv_fp8_e4m3
#ifndef CACHE_ELEM_ENUM
#define CACHE_ELEM_ENUM 0
#endif

// don't modify
#define USE_KV_CACHE true

// don't modify
#ifndef ALLOW_MULTI_BLOCK_MODE
#define ALLOW_MULTI_BLOCK_MODE true
#endif

// For paged KV cache. Allowed values: 0, 16, 32, 64, 128
// 0 means contiguous KV cache (non-paged).
#ifndef TOKENS_PER_PAGE
#define TOKENS_PER_PAGE 64
#endif

// don't modify
#ifndef USE_PAGED_KV_CACHE
#define USE_PAGED_KV_CACHE (TOKENS_PER_PAGE > 0)
#endif

// don't modify
#define USE_BEAM_SEARCH (BEAM_WIDTH > 1)

#if CACHE_ELEM_ENUM == 0
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll")
#else
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll(1)")
#endif

// good for short sequence length but bad for long sequence length.
#ifndef SHORT_SEQ_OPT
#define SHORT_SEQ_OPT 1
#endif

// true should be better if warpTile.x * cacheElemSize < 128. otherwise use false.
#define GRP_LOAD_V (CACHE_ELEM_ENUM != 0) || (HEAD_ELEMS == 256 && USE_PAGED_KV_CACHE && BEAM_WIDTH > 1)

// use custom barrier for NVRTC to avoid pulling in many headers
#ifndef USE_CUSTOM_BARRIER
#define USE_CUSTOM_BARRIER 1
#endif

#define DBG_BATCH_SIZE 2
#define DBG_SEQ_LEN 256 * 4 + 3
#define DBG_NB_CTAS_PER_SEQ 8

#include <cuda_fp16.h>
#include <cuda_fp8.h>
template <int32_t elemTypeEnum>
using ElemType = mha::conditional_t<elemTypeEnum == 0, INPUT_ELEM,
    mha::conditional_t<elemTypeEnum == 1, int8_t, mha::conditional_t<elemTypeEnum == 2, __nv_fp8_e4m3, void>>>;
