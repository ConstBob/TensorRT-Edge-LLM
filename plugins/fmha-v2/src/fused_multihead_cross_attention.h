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
#include <vector>
#include <fused_multihead_attention_utils.h>
#include <fused_multihead_attention.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace bert {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if USE_DEMO_BERT_PARAMS

//TODO TRT plugins use a different parameter struct taken from the old XMMA fork.
//     Until all cubins in the plugin are replaced with new kernels, we need to conform to that.
#include <fused_multihead_attention_demo_bert_params.h>

#endif // USE_DEMO_BERT_PARAMS

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gmem_params {
    // The matrix.
    void *ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t stride_in_bytes;

    // The number of heads
    int h;

    // Hidden dim per head
    int d;

    // array of length b+1 holding prefix sum of actual sequence lenghts.
    int *cu_seqlens;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_mhca : Fused_multihead_attention_params_v2 {
    // Sequence length of Q
    int s_q;
    int d_padded;
    bool force_unroll;
    Gmem_params gmem_q_params;
    Gmem_params gmem_kv_params;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace bert

