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

#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>
#include <fmha/utils.h>
#include <fmha/gmem_tile_qkv_packed.h>
#include <fmha/smem_tile_qkv.h>
#include <fmha/gmem_tile_qkv.h>
#include <fmha/gmem_tile_ps.h>
#include <fmha/mask.h>
#include <fmha/softmax.h>
#include <fmha/smem_tile_v.h>
#include <fmha/smem_tile_o.h>
#include <fmha/gmem_tile_o_packed.h>
#include <fmha/gmem_tile_o.h>

namespace fused_multihead_attention {

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// The kernel implemented here reads the matrices K, Q and V of size (column-major):
//
// - K : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
// - Q : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
// - V : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
//
// It does the following operations:
//
// - P = norm * K^T * Q , where norm is the normalization term (a scalar),
// - S = Softmax(P) over the columns of P,
// - O = V * S.
//
// The intermediate matrices have the following sizes (column-major):
//
// - P : SEQUENCE_LENGTH * SEQUENCE_LENGTH (384 * 384 for BERT-Large),
// - O : EMBEDDING_SIZE  * SEQUENCE_LENGTH ( 64 * 384 for BERT-Large).
//
// To be able to hold the matrices on the SM we iterate over the SEQUENCE_LENGHT dimension of the
// O matrix (i.e. its columns). The matrices K and V are kept in registers on the SM whereas Q is
// read over the different iterations of the loop.
//
// To be able to operate entirely from registers (on Turing and Ampere) for the V * S product, we
// actually compute P^T = Q^T * K (remember that (AB)^T = B^T A^T).
//

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// U T I L S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int FMHA_VERSION> 
struct Single_cta {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
struct Single_cta<1> {

    // Ctor.
    template< typename Params >
    inline __device__ Single_cta(const Params &params, int bidb, int bidh, int bidn, int tidx)
        : bidb(bidb), bidh(bidh), bidn(bidn) {
        sum_s = params.b * params.s;
        actual_seqlen = params.s;
        bidx = bidb * params.h + bidh;
    }

    // Should we do an early exit? No.
    inline __device__ bool stop_early(int = 0) const {
        return false;
    }

    // The length of the sequence.
    int actual_seqlen;
    // The indices of the block (batch, head, linear index).
    int bidb, bidh, bidn, bidx;
    // The total number of tokens.
    int sum_s;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
struct Single_cta<2> {

    // Ctor for fmhca params. TODO: consolidate
    template< typename Params >
    inline __device__ Single_cta(const Params &params, int bidb, int bidh, int bidn, int tidx)
        : bidb(bidb), bidh(bidh), bidn(bidn), num_heads(params.h) {
        sum_s = params.cu_seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
        bidx = sum_s * params.h + bidh;
    }

    // Ctor.
    inline __device__ Single_cta(
        const bert::Fused_multihead_attention_params_v2 &params,
        int bidb,
        int bidh,
        int bidn,
        int tidx)
        : bidb(bidb), bidh(bidh), bidn(bidn), num_heads(params.h) {
        if( params.is_s_padded ) {
            sum_s = params.s * bidb;
        } else {
            sum_s = params.cu_seqlens[bidb];
        }
        actual_seqlen = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
        bidx = sum_s * params.h + bidh;
    }

    // Skip empty sequences.
    inline __device__ bool stop_early(int loop = 0) const {
        return loop >= actual_seqlen;
    }

    // The length of the sequence.
    int actual_seqlen;
    // The indices of the block (batch, head, linear index).
    int bidb, bidh, bidn, bidx;
    // The total number of tokens.
    int sum_s;
    int num_heads;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int VERSION >
struct Multi_cta : public Single_cta<VERSION> {

    // The base class.
    using Base = Single_cta<VERSION>;

    // Ctor.
    template< typename Params >
    inline __device__ Multi_cta(const Params &params, int bidb, int bidh, int bidn, int tidx)
        : Base(params, bidb, bidh, bidn, tidx) { 
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout [Batch, Sequence Length]
template< int THREADS_PER_CTA, bool SEQUENCES_INTERLEAVED = false >
struct Block_info_padded {

    template<typename Params>
    __device__ inline Block_info_padded(const Params &params,
                                        const int bidb,
                                        const int bidh,
                                        const int tidx)
        : bidb(bidb), bidh(bidh), bidn(0), num_heads(params.h) {

        hidx =  bidb * params.h + bidh;

        // The block index.
        sum_s = params.cu_seqlens[bidb];
        //actual_seqlen = params.seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
        bidx = sum_s * params.h + bidh;

        tidx_global = hidx * THREADS_PER_CTA + tidx;
    }

    __device__ inline bool stop_early() const {
        return actual_seqlen == 0;
    }

    template<int M_PER_ITER>
    __device__ inline int get_steps(const int begin) const {
        return ((actual_seqlen - begin) + M_PER_ITER - 1) / M_PER_ITER;
    }

    int actual_seqlen;
    int bidx;
    int sum_s;
    int bidh;
    int bidb;
    int bidn;
    int hidx;
    int num_heads;
    int tidx_global;
    int next_seq_offset_factor = 1;
};

// Layout [Sequence Length, Batch]
template< int THREADS_PER_CTA >
struct Block_info_padded <THREADS_PER_CTA, true> {

    template<typename Params>
    __device__ inline Block_info_padded(const Params &params,
                                        const int bidb,
                                        const int bidh,
                                        const int tidx)
        : bidb(bidb), bidh(bidh), bidn(0), num_heads(params.h) {
        
        hidx =  bidb * params.h + bidh;

        // The block index.
        sum_s = bidb;
        //actual_seqlen = params.seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
        bidx = sum_s * params.h + bidh;

        next_seq_offset_factor = params.b;

        tidx_global = hidx * THREADS_PER_CTA + tidx;
    }

    __device__ inline bool stop_early() const {
        return actual_seqlen == 0;
    }

    template<int M_PER_ITER>
    __device__ inline int get_steps(const int begin) const {
        return ((actual_seqlen - begin) + M_PER_ITER - 1) / M_PER_ITER;
    }

    int actual_seqlen;
    int bidx;
    int sum_s;
    int bidh;
    int bidb;
    int bidn;
    int hidx;
    int num_heads;
    int tidx_global;
    int next_seq_offset_factor;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fused_multihead_attention

