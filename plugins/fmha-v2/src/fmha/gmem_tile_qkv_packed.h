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
#include <fmha/traits.h>
#include <fused_multihead_attention.h>

namespace fmha {
namespace v2 {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int USE_LDGSTS >
struct Ldgsts_helper {
    template< typename This, typename Smem_tile, int LDGS >
    static inline __device__ void 
    load(This *this_, Smem_tile &smem_tile, const void* (&ptrs)[LDGS], uint32_t (&preds)[LDGS]) {
        fmha::pack_predicates(this_->preds_, preds);
        smem_tile.store(ptrs, this_->preds_);
    } 
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Ldgsts_helper<0> {
    template< typename This, typename Smem_tile, int LDGS >
    static inline __device__ void 
    load(This *this_, Smem_tile &smem_tile, const void* (&ptrs)[LDGS], uint32_t (&preds)[LDGS]) {
#if 0
        fmha::pack_predicates(this_->preds_, preds);
        fmha::ldg(this_->fetch_, ptrs, this_->preds_);
#else
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            this_->fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
        }
        // not packing predicates removes restrictions (e.g. FP16 384, 4 warps)
        Ldg_functor<uint4, LDGS> fct(this_->fetch_, ptrs);
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            fct.ldgsts(ii, preds[ii]);
        }
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interelaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 3
>
struct Gmem_tile_qkv {

    // The size of each LDG.
    enum { BYTES_PER_LDG = 16 };
    // The number of bits/bytes of element
    enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ / 8 };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8 };
    // The number of threads to load a "row" of the matrix.
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };
    // The valid size of a row in bytes (without paddings).
    enum { VALID_COLS = VALID_COLS_ };
    // The amount of bytes that are valid per row.
    enum { VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8 };
    // The number of "rows" loaded per LDG.
    enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of rows.
    enum { ROWS = ROWS_ };
    // The number of LDGs needed to load a chunk of the Q matrix.
    enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };
    // The number of predicate registers.
    enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

    // Is it Hopper?
    enum { IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true };
    // Make sure we use a single register to store predicates. Do not throw for Hopper for now. 
    static_assert( !USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "" );
    // We do not use LDGSTS (for the moment).
    enum { USE_LDGSTS = USE_LDGSTS_ };

    // Ctor for bert::Fused_multihead_attention_params_v2 class
    template<typename Block_info>
    inline __device__ Gmem_tile_qkv(const bert::Fused_multihead_attention_params_v2 &params,
                                    int qkv_offset,
                                    const Block_info &binfo,
                                    int tidx,
                                    int cta_row_offset = 0,
                                    int cta_col_offset_in_bytes = 0)
        : Gmem_tile_qkv(params.qkv_ptr,
                        params.qkv_stride_in_bytes,
                        params.h,
                        qkv_offset,
                        binfo,
                        tidx,
                        params.h_kv,
                        cta_row_offset,
                        cta_col_offset_in_bytes) {
    }

    // Ctor for other param classes (such as Qkv_params in train_ops)
    template<typename Params,
             typename Block_info>
    inline __device__ Gmem_tile_qkv(const Params &params,
                                    int qkv_offset,
                                    const Block_info &binfo,
                                    int tidx,
                                    int cta_row_offset = 0,
                                    int cta_col_offset_in_bytes = 0)
        : Gmem_tile_qkv(params.qkv_ptr,
                        params.qkv_stride_in_bytes,
                        params.h,
                        qkv_offset,
                        binfo,
                        tidx,
                        cta_row_offset,
                        cta_col_offset_in_bytes) {
    }

    // Ctor.
    template< typename Block_info >
    inline __device__ Gmem_tile_qkv(void *qkv_ptr, 
                                    size_t qkv_stride_in_bytes,
                                    int num_heads,
                                    int qkv_offset, 
                                    const Block_info &binfo, 
                                    int tidx,
                                    int num_kv_heads = 0,
                                    int cta_row_offset = 0,
                                    int cta_col_offset_in_bytes = 0)
        : params_qkv_stride_in_bytes_(qkv_stride_in_bytes)
        , actual_seqlen_(binfo.actual_seqlen)
        , qkv_ptr_(reinterpret_cast<char *>(qkv_ptr)) {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Do not load/store if the thread is in the padded area
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params_qkv_stride_in_bytes_;
        // Add the block index.
        int64_t idx;

        // Both MQA and GQA will use non HEADS_INTERLEAVED layout
        if( num_kv_heads < num_heads ) {
            const int num_qkv_heads = num_heads + 2 * num_kv_heads;
            const int head_id = binfo.bidh;
            const int kv_head_id = binfo.bidh / (num_heads / num_kv_heads);
            // QKV layout [b, s, [q_hd, k_h'd, v_h'd]]
            if( qkv_offset == 0 ) {         // Q tensor
                idx = binfo.sum_s * num_qkv_heads + head_id;
            } else if( qkv_offset == 1 ) {  // K tensor
                idx = binfo.sum_s * num_qkv_heads + num_heads + kv_head_id;
            } else if( qkv_offset == 2 ) {  // V tensor
                idx = binfo.sum_s * num_qkv_heads + num_heads + num_kv_heads + kv_head_id;
            }
        } else if( HEADS_INTERLEAVED ) {
            // [b, s, h, [q_d, k_d, v_d]] aka bsh3d
            // bidx = sum_s * params.h + bidh;
            idx = binfo.bidx * NUM_MATS + qkv_offset;
        } else {
            // [b, s, [q_hd, k_hd, v_hd]] aka bs3hd
            idx = (binfo.sum_s * NUM_MATS + qkv_offset) * num_heads + binfo.bidh;
        }

        // Assemble the final pointer.
        qkv_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col_in_bytes_;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;

        // Set the initial seq_len and qkv_offset in case of reinterating
        actual_seqlen_init_ = actual_seqlen_;
        qkv_ptr_init_ = qkv_ptr_;

    }

    // Store data to shared memory.
    template<typename Smem_tile>
    inline __device__ void commit(Smem_tile &smem_tile) {
        if( !USE_LDGSTS ) {
            smem_tile.store(fetch_);
        }
    }

    // Load data from memory.
    template< typename Smem_tile > 
    inline __device__ void load(Smem_tile &smem_tile) {
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
        }

        // Prepare the load pointers.
        const void *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    // Load data from memory.
    inline __device__ void load() {
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
        }

        // Prepare the load pointers.
        const void *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
        }

        // Trigger the LDGs.
        if ( col_in_bytes_ < VALID_BYTES_PER_ROW ) {
            fmha::pack_predicates(preds_, preds);
            fmha::ldg(fetch_, ptrs, preds_);
        } else {
            #pragma unroll
            for ( int ii = 0; ii < LDGS; ++ii ) {
                fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }

    // Move the pointer to the next row location.
    inline __device__ void move(const int steps = 1) {
        qkv_ptr_ += (int64_t) ROWS * params_qkv_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Move the pointer to the next row location by the offset (not step).
    inline __device__ void move_by_offset(const int offset) {
        qkv_ptr_ = qkv_ptr_init_ + (int64_t) offset * params_qkv_stride_in_bytes_;
        actual_seqlen_ = actual_seqlen_init_ - (int) offset;
    }

    // Move the pointer to the next column location
    inline __device__ void move_col(const int steps = 1) {
        qkv_ptr_ += (int64_t) COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void reset() {
        qkv_ptr_ = qkv_ptr_init_;
        actual_seqlen_ = actual_seqlen_init_;
    }

    // Rewind the pointer back to previous column location
    inline __device__ void rewind_col(const int steps) {
        qkv_ptr_ -= COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void move_to(const int step) {
        qkv_ptr_ = qkv_ptr_init_ + (int64_t) ROWS * params_qkv_stride_in_bytes_ * step;
        actual_seqlen_ = actual_seqlen_init_ - (int) ROWS * step;
    }

    // Store data to memory.
    inline __device__ void store(const uint4 (&data)[LDGS]) {
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            char *ptr = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            if( ((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_)) &&
                col_in_bytes_ < VALID_BYTES_PER_ROW /*TODO: double check*/) {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // The stride between rows for the QKV matrice.
    int64_t params_qkv_stride_in_bytes_;
    // The pointer.
    char *qkv_ptr_;
    char *qkv_ptr_init_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    int col_in_bytes_;
    // The sequence length.
    int actual_seqlen_;
    int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interelaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 1
>
struct Gmem_tile_q_kv {

    // The size of each LDG.
    enum { BYTES_PER_LDG = 16 };
    // The padded to the next power of 2 number of columns
    enum { COLS_PADDED = Next_power_of_two<COLS>::VALUE };
    // The padded size of a row in bytes.
    enum { BYTES_PER_ROW_PADDED = COLS_PADDED * BITS_PER_ELEMENT / 8 };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8 };
    // The number of threads to load a padded "row" of the matrix.
    enum { THREADS_PER_ROW_PADDED = BYTES_PER_ROW_PADDED / BYTES_PER_LDG };
    // The number of threads to load a "row" of the matrix.
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };
    // The number of "rows" loaded per LDG.
    enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW_PADDED };
    // The number of rows.
    enum { ROWS = ROWS_ };
    // The number of LDGs needed to load a chunk of the Q matrix.
    enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };
    // The number of predicate registers.
    enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

    // Is it Hopper?
    enum { IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true };
    // Make sure we use a single register to store predicates. Do not throw for Hopper for now. 
    static_assert( !USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "" );
    // We do not use LDGSTS (for the moment).
    enum { USE_LDGSTS = USE_LDGSTS_ };

    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_q_kv(const Params &params, 
                                     int offset, 
                                     const Block_info &binfo, 
                                     int tidx,
                                     int cta_row_offset = 0)
        : params_stride_in_bytes_(params.stride_in_bytes)
        , actual_seqlen_(binfo.actual_seqlen)
        , ptr_(reinterpret_cast<char *>(params.ptr)) {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW_PADDED;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW_PADDED;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Mask for predicate if the channels are in the padded area
        const int bytes_per_row_non_padded = params.d * BITS_PER_ELEMENT / 8;
        mask_ = col < bytes_per_row_non_padded / BYTES_PER_LDG;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params.stride_in_bytes;
        // Add the block index. 
        int64_t idx;
        if( HEADS_INTERLEAVED ) {
            idx = binfo.bidx * NUM_MATS + offset;
        } else {
            idx = (binfo.sum_s * NUM_MATS + offset) * params.h + binfo.bidh;
        }
        // Assemble the final pointer.
        ptr_ += row_offset + idx * bytes_per_row_non_padded + col * BYTES_PER_LDG;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;
    }

    // Store data to shared memory.
    template<typename Smem_tile>
    inline __device__ void commit(Smem_tile &smem_tile) {
        if( !USE_LDGSTS ) {
            smem_tile.store(fetch_);
        }
    }

    // Load data from memory.
    template< typename Smem_tile > 
    inline __device__ void load(Smem_tile &smem_tile) {
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = (row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_)) && mask_;
        }

        // Prepare the load pointers.
        const void *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = ptr_ + (int64_t) ii * ROWS_PER_LDG * params_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    inline __device__ void move(const int steps = 1) {
        ptr_ += (int64_t) ROWS * params_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Store data to memory.
    inline __device__ void store(const uint4 (&data)[LDGS]) {
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            char *ptr = ptr_ + (int64_t)ii * ROWS_PER_LDG * params_stride_in_bytes_;
            if( (row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_) ) {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // The stride between rows for the matrix.
    int64_t params_stride_in_bytes_;
    // The pointer.
    char *ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    // Keep track of predicate state that depends only on the initialization state.
    int mask_;
    // The sequence length.
    int actual_seqlen_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_
>
struct Gmem_tile_qkv_interleaved {

    // The vectorization width for NC/32HW32.
    enum { VEC = 32 };

    // The size of each LDG.
    enum { BYTES_PER_LDG = 16 };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = VEC * BITS_PER_ELEMENT / 8 };

    // DEBUG.
    static_assert(BYTES_PER_ROW == 32, "");
    // END OF DEBUG.

    // The number of threads to load a "row" of the matrix.
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

    // DEBUG.
    static_assert(THREADS_PER_ROW == 2, "");
    // END OF DEBUG.

    // The number of "rows" loaded per LDG.
    enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of slices. It is either 1 for DIM_PER_HEAD == 32 and 2 for DIM_PER_HEAD == 64.
    enum { NUM_SLICES = COLS / VEC };

    // DEBUG.
    static_assert(NUM_SLICES == 1 ||  NUM_SLICES == 2, "");
    // END OF DEBUG.

    // The number of rows in a slice.
    enum { ROWS = ROWS_ };
    // The number of LDGs needed to load a chunk of the Q matrix.
    enum { LDGS = fmha::Div_up<ROWS * NUM_SLICES, ROWS_PER_LDG>::VALUE };

    // The number of predicate registers.
    enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };
    // Make sure we use a single register to store predicates.
    static_assert(PRED_REGS == 1, "");

    // Do we use LDGSTS on Ampere?
    enum { USE_LDGSTS = USE_LDGSTS_ };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_qkv_interleaved(const Params &params,
                                                int qkv_select,
                                                const Block_info &block_info,
                                                int tidx,
                                                int cta_row_offset = 0)
        : actual_seqlen_(block_info.actual_seqlen - cta_row_offset)
        , total_(params.qkv_stride_in_bytes)
        , qkv_ptr_(reinterpret_cast<const char *>(params.qkv_ptr)) {

        int bidh = block_info.bidh;
        int sum_s = block_info.sum_s;

        // We must keep track of the row to repack predicates in load.
        row_ = tidx / THREADS_PER_ROW;
        // The column.
        int col = tidx % THREADS_PER_ROW;

        // h is N
        // d is H
        // we get the data in as: 3 x h x (d/32) x total x 32 (think 3 x h x (d/32)
        // x b x s x 32)

        // Loading qkv: ignore slice for now.
        int qkv_offset = qkv_select * params.h * NUM_SLICES * total_;
        // bidh * GROUPS * B * S + b * S.
        int block_offset = bidh * NUM_SLICES * total_ + sum_s;  
        // The row offset.
        int row_offset = (qkv_offset + block_offset + cta_row_offset) * BYTES_PER_ROW;

        // That's the pointer to load from (see "load").
        qkv_ptr_ += row_offset + col * BYTES_PER_LDG;

        init_actual_seqlen_ = actual_seqlen_;
        init_qkv_ptr_ = qkv_ptr_;
    }

    // Store data to shared memory.
    template< typename Smem_tile > 
    inline __device__ void commit(Smem_tile &smem_tile) {
        if( !USE_LDGSTS ) {
            smem_tile.store(fetch_);
        }
    }

    // Load data from memory.
    template< typename Smem_tile > 
    inline __device__ void load(Smem_tile &smem_tile) {
        const void *ptrs[LDGS];
        uint32_t preds[LDGS];

        // We precompute slice offsets and predicates
        #pragma unroll
        for( int ii = 0; ii < LDGS; ii++ ) {
            // the next row
            int row_i = row_ + ii * ROWS_PER_LDG;

            // Decompose the current row in slice and original row
            int slice = row_i / ROWS;
            // The position in the slice.
            int row_in_slice = row_i % ROWS;

            // Update the predicate.
            preds[ii] = row_in_slice < min(actual_seqlen_, ROWS);
            // Compute the pointer.
            ptrs[ii] = &qkv_ptr_[(slice * total_ + row_in_slice) * BYTES_PER_ROW];
        }

        // Update the predicate register.
        fmha::pack_predicates(preds_, preds);

        // Trigger the loads.
        if( USE_LDGSTS ) {
            smem_tile.store(ptrs, preds_);
        } else {
            fmha::ldg(fetch_, ptrs, preds_);
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move(const int steps = 1) {
        qkv_ptr_ += (int64_t) ROWS * BYTES_PER_ROW * steps;
        actual_seqlen_ -= ROWS * steps;
    }

    // Reset to the initial location.
    inline __device__ void reset() {
        qkv_ptr_  = init_qkv_ptr_;
        actual_seqlen_ = init_actual_seqlen_;
    }

    // The pointer.
    const char *qkv_ptr_;
    const char *init_qkv_ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // keep track of the row the thread is processing as we move the tile
    int row_;
    // The sequence length.
    int actual_seqlen_;
    int init_actual_seqlen_;
    // The number of rows per slice??
    int total_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace v2
}  // namespace fmha
