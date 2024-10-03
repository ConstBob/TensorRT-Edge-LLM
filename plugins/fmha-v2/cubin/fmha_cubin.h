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
#include <stdint.h>

extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm101_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm101_cu_cubin[];
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm101_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm101_cu_cubin_len;

namespace fmha_v2
{

enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_87 = 87;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_101 = 101;

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mStepQ;
    unsigned int mStepKV;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    int mAttentionMaskType;
    int mAttentionInputLayout;
    bool mInterleaved;
    bool mFlashAttention;
    bool mWarpSpecialization;
    bool mFP32Accumulation;
    bool mAlibiSupported;
    bool mTiled;
    bool mEnableQKTanhScale;
} sMhaKernelMetaInfosV2[] = {
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_causal_sm89_kernel_nl_tiled", 81920, 128, 64, 1, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_custom_mask_sm89_kernel_nl_tiled", 81920, 128, 64, 3, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm89_kernel_nl", 32768, 128, 64, 1, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_custom_mask_sm89_kernel_nl", 32768, 128, 64, 3, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_causal_sm80_kernel_nl_tiled", 81920, 128, 64, 1, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_custom_mask_sm80_kernel_nl_tiled", 81920, 128, 64, 3, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm80_kernel_nl", 32768, 128, 64, 1, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_custom_mask_sm80_kernel_nl", 32768, 128, 64, 3, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_causal_sm86_kernel_nl_tiled", 81920, 128, 64, 1, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_custom_mask_sm86_kernel_nl_tiled", 81920, 128, 64, 3, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm86_kernel_nl", 32768, 128, 64, 1, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_custom_mask_sm86_kernel_nl", 32768, 128, 64, 3, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_causal_sm87_kernel_nl_tiled", 81920, 128, 64, 1, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_custom_mask_sm87_kernel_nl_tiled", 81920, 128, 64, 3, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm87_kernel_nl", 32768, 128, 64, 1, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_custom_mask_sm87_kernel_nl", 32768, 128, 64, 3, 0, false, true, false, false, true, false, false},
{ DATA_TYPE_FP16, 0, 64, 128, 128, kSM_101,  cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm101_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm101_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_qkv_128_causal_sm101_kernel_nl_tiled", 81920, 128, 64, 1, 0, false, true, false, false, true, true, false},
{ DATA_TYPE_FP16, 0, 64, 32, 128, kSM_101,  cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm101_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm101_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm101_kernel_nl", 32768, 128, 64, 1, 0, false, true, false, false, true, false, false}
};

} // fmha_v2

