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
#include <fmha/hopper/fragment.h>
#include <fmha/gmem_tile_o_packed.h>

namespace fmha {

namespace v2 {

template <typename Traits, typename Cta_tile, int WARPS_K>
struct Gmem_tile_o_hopper {};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Not super proud of this. Need to refactor.
// A not optimized way of storing tile_O, without SMEM swizzle.
// STG.32 is going to be used.
template< typename Traits,
          typename Cta_tile>
struct Gmem_tile_o_hopper_16bits
 {
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of elements per STG.
    enum { ELEMENTS_PER_STG = 2 };
    // The size in bytes of each element.
    enum { BYTES_PER_ELEMENT = 2 };
    // The size of each STG.
    enum { BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT };

    // The number of rows accessed by each thread.
    enum { ROWS_PER_THREAD = Mma_tile::M_PER_MMA / 8 / Cta_tile::WARPS_PER_CTA };

    enum { ROWS = Cta_tile::M };
    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum { COLS_PER_THREAD = Mma_tile::N_PER_MMA / 4 / 2};

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum { ELTS_PER_THREAD = ROWS_PER_THREAD * COLS_PER_THREAD };

    // Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape.
    static_assert(Mma_tile::M_PER_MMA == Cta_tile::M &&
                  Mma_tile::N_PER_MMA * Mma_tile::MMAS_N == Cta_tile::N,
                  "Currently, we assume for o matrix, GMMA M shape matches CTA M shape. ");

    // Step N for one quad
    enum { STEP_N = 8 * BYTES_PER_ELEMENT };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper_16bits(const Params &params,
                                         const Block_info &block_info,
                                         int tidx,
                                         int cta_row_offset = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen)
        , o_ptr_(reinterpret_cast<char *>(params.o_ptr)) {
        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        //int warpgroup_idx = warp / 4;
        int warp_idx_within_warpgroup = warp % 4;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
        // Store the row to update the predicates in load.
        row_ = cta_row_offset + row;
        // Compute the position of the thread in the row.
        int col = lane % 4 * ELEMENTS_PER_STG;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_ + block_info.bidx * BYTES_PER_ROW;
        // Finalize the pointer.
        o_ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(const Accumulators (&acc)[M][N]) {
        const int64_t step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
        #pragma unroll
        for(int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx) {
            if( row_ + row_idx * 8 >= actual_seqlen_ ) {
                break;
            }
            #pragma unroll
            for(int mma_ni = 0; mma_ni < Mma_tile::MMAS_N; ++mma_ni) {
                #pragma unroll
                for(int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx) {
                    uint32_t acc_0 = acc[0][mma_ni].reg(col_idx * ROWS_PER_THREAD + row_idx);

                    int64_t offset = (int64_t)row_idx * step_m +
                        (int64_t)(col_idx + mma_ni * COLS_PER_THREAD) * STEP_N;
                    fmha::stg(o_ptr_ + offset, acc_0);
                } // row_idx
            } // col_idx
        } // mma_ni
    }

    // Move to the next location.
    inline __device__ void move() {
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }


    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The pointer.
    char *o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;

    // The row loaded by this thread.
    int row_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          1>  // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                        Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>, Cta_tile>;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                         const Block_info &block_info,
                                         int tidx,
                                         int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          1>  // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                        Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                         const Block_info &block_info,
                                         int tidx,
                                         int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(const Accumulators (&acc)[M][N]) {
        const int64_t step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
        #pragma unroll
        for(int mma_ni = 0; mma_ni < Mma_tile::MMAS_N; ++mma_ni) {
            #pragma unroll
            for(int col_idx = 0; col_idx < Base::COLS_PER_THREAD; ++col_idx) {
                #pragma unroll
                for(int row_idx = 0; row_idx < Base::ROWS_PER_THREAD; ++row_idx) {
                    if( this->row_ + row_idx * 8 >= this->actual_seqlen_ ) {
                        break;
                    }
                    // 2 denotes as fp32 --> fp16
                    float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                    float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                    uint32_t out = fmha::float2_to_half2(reg0, reg1);

                    int64_t offset = (int64_t)row_idx * step_m +
                        (int64_t)(col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                    fmha::stg(this->o_ptr_ + offset, out);
                } // row_idx
            } // col_idx
        } // mma_ni
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          1>  // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                        Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                         const Block_info &block_info,
                                         int tidx,
                                         int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(const Accumulators (&acc)[M][N]) {
        const int64_t step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
        #pragma unroll
        for(int mma_ni = 0; mma_ni < Mma_tile::MMAS_N; ++mma_ni) {
            #pragma unroll
            for(int col_idx = 0; col_idx < Base::COLS_PER_THREAD; ++col_idx) {
                #pragma unroll
                for(int row_idx = 0; row_idx < Base::ROWS_PER_THREAD; ++row_idx) {
                    if( this->row_ + row_idx * 8 >= this->actual_seqlen_ ) {
                        break;
                    }
                    // 2 denotes as fp32 --> bf16
                    float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                    float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                    uint32_t out = fmha::float2_to_bf16_x2(reg0, reg1);

                    int64_t offset = (int64_t)row_idx * step_m +
                        (int64_t)(col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                    fmha::stg(this->o_ptr_ + offset, out);
                } // row_idx
            } // col_idx
        } // mma_ni
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          2>  // WARPS_K
                          : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                              Cta_tile,
                                              /*CTAS_PER_HEAD=*/ 1,
                                              /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                            GMMA_N,
                                            GMMA_K,
                                            GMMA_A_RF,
                                            GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                            const Block_info &block_info,
                                            int tidx,
                                            int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          2>  // WARPS_K
                          : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                              Cta_tile,
                                              /*CTAS_PER_HEAD=*/ 1,
                                              /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                            GMMA_N,
                                            GMMA_K,
                                            GMMA_A_RF,
                                            GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                            const Block_info &block_info,
                                            int tidx,
                                            int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                          Cta_tile,
                          2>  // WARPS_K
                          : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                              GMMA_N,
                                              GMMA_K,
                                              GMMA_A_RF,
                                              GMMA_B_RF>,
                                              Cta_tile,
                                              /*CTAS_PER_HEAD=*/ 1,
                                              /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                            GMMA_N,
                                            GMMA_K,
                                            GMMA_A_RF,
                                            GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper(const Params &params,
                                         const Block_info &block_info,
                                         int tidx,
                                         int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>,
                   Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                                               GMMA_N,
                                                               GMMA_K,
                                                               GMMA_A_RF,
                                                               GMMA_B_RF>,
                                           Cta_tile,
                                           Cta_tile::WARPS_K> {

    // The traits class.
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M,
                                                                   GMMA_N,
                                                                   GMMA_K,
                                                                   GMMA_A_RF,
                                                                   GMMA_B_RF>,
                                               Cta_tile,
                                               Cta_tile::WARPS_K>;
    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_o(const Params &params,
                                  const Block_info &block_info,
                                  int tidx,
                                  int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>,
                   Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                                               GMMA_N,
                                                               GMMA_K,
                                                               GMMA_A_RF,
                                                               GMMA_B_RF>,
                                           Cta_tile,
                                           Cta_tile::WARPS_K> {

    // The traits class.
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M,
                                                                   GMMA_N,
                                                                   GMMA_K,
                                                                   GMMA_A_RF,
                                                                   GMMA_B_RF>,
                                               Cta_tile,
                                               Cta_tile::WARPS_K>;
    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_o(const Params &params,
                                  const Block_info &block_info,
                                  int tidx,
                                  int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>,
                   Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                                               GMMA_N,
                                                               GMMA_K,
                                                               GMMA_A_RF,
                                                               GMMA_B_RF>,
                                           Cta_tile,
                                           Cta_tile::WARPS_K> {

    // The traits class.
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                                  GMMA_N,
                                                  GMMA_K,
                                                  GMMA_A_RF,
                                                  GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M,
                                                                   GMMA_N,
                                                                   GMMA_K,
                                                                   GMMA_A_RF,
                                                                   GMMA_B_RF>,
                                               Cta_tile,
                                               Cta_tile::WARPS_K>;
    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_o(const Params &params,
                                  const Block_info &block_info,
                                  int tidx,
                                  int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Traits, typename Cta_tile, int NUM_MATS=1, bool HEADS_INTERLEAVED=false>
struct Gmem_tile_o_gmma_32bit_8bit
{
    static_assert(sizeof(typename Traits::Accumulator_type) == 4);
    static_assert(sizeof(typename Traits::C_type) == 1);
    // This is for non-splitk GMMA BMM2.
    static_assert(Cta_tile::WARPS_K == 1);
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of elements per STG.
    enum { ELEMENTS_PER_STG = 4 };
    // The size in bytes of each element.
    enum { BYTES_PER_ELEMENT = 1 };
    // The size of each STG.
    enum { BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT };

    enum { ROWS = Cta_tile::M };
    // The number of rows accessed by each thread.
    enum { ROWS_PER_THREAD = Mma_tile::M_PER_MMA / 8 / Cta_tile::WARPS_M };
    static_assert(ROWS_PER_THREAD == 2);
    static_assert(ROWS_PER_THREAD == Mma_tile::ROWS_PER_THREAD);

    // The number of columns access by each thread.
    // The number of core matrices in N.
    enum { COLS_PER_THREAD = Mma_tile::N_PER_MMA / 4 / 2}; // N_PER_MMA = GMMA_N
    static_assert(COLS_PER_THREAD == Mma_tile::COLS_PER_THREAD / 2);
    // Assume there is an even number of core matrices, such that we can pack two
    static_assert(COLS_PER_THREAD % 2 == 0);

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum { ELTS_PER_THREAD = ROWS_PER_THREAD * COLS_PER_THREAD * 2 };

    // Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape.
    static_assert(Mma_tile::M_PER_MMA == Cta_tile::M &&
                  Mma_tile::N_PER_MMA == Cta_tile::N,
                  "Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape. ");

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_gmma_32bit_8bit(const Params &params,
                                                  const Block_info &block_info,
                                                  int tidx,
                                                  int cta_row_offset = 0)
        : Gmem_tile_o_gmma_32bit_8bit(params.o_ptr,
                                      params.o_stride_in_bytes,
                                      block_info,
                                      tidx,
                                      params.scale_bmm2,
                                      cta_row_offset) {
    }

    template <typename Block_info>
    inline __device__ Gmem_tile_o_gmma_32bit_8bit(void * o_ptr,
                                                  int o_stride_in_bytes,
                                                  const Block_info &block_info,
                                                  int tidx,
                                                  uint32_t scale_bmm2,
                                                  int cta_row_offset = 0,
                                                  int mat_offset = 0)
        : params_o_stride_in_bytes_(o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen)
        , o_ptr_(reinterpret_cast<char *>(o_ptr))
        , params_scale_bmm2_(scale_bmm2) {

        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        //int warpgroup_idx = warp / 4;
        int warp_idx_within_warpgroup = warp % 4;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
        // Store the row to update the predicates in load.
        row_ = cta_row_offset + row;
        // Compute the position of the thread in the row.
        int col = lane % 4 * ELEMENTS_PER_STG;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        //int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_ + block_info.bidx * BYTES_PER_ROW;
        // Finalize the pointer.
        //o_ptr_ += row_offset + col * BYTES_PER_ELEMENT;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_;
        // Add the block index.
        int64_t idx = block_info.bidx;
        if(NUM_MATS > 1) {
            if( HEADS_INTERLEAVED ) {
                idx = block_info.bidx * NUM_MATS + mat_offset;
            } else {
                idx = (block_info.sum_s * NUM_MATS + mat_offset) * block_info.num_heads + block_info.bidh;
            }
        }
        // Assemble the final pointer.
        o_ptr_ += row_offset + idx * BYTES_PER_ROW + col * BYTES_PER_ELEMENT;


    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(const Accumulators (&acc)[M][N]) {

        static_assert(COLS_PER_THREAD == 8); // FOR D=64.
        static_assert(Accumulators::NUM_ELTS == ELTS_PER_THREAD);
        static_assert(COLS_PER_THREAD / 2  * ROWS_PER_THREAD * 4 == ELTS_PER_THREAD);

        const int64_t step_m = 8 * (this->params_o_stride_in_bytes_);
        // 4 threads per row writing 4 elements.
        const int64_t step_n = 16 * BYTES_PER_ELEMENT;
        // we assume M = N = 1. some shortcuts.
        static_assert(M == 1);
        static_assert(N == 1);

        #pragma unroll
        for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
            if( row_ + ri * 8 >= actual_seqlen_ ) {
                break;
            }
            // Iterate over 16 columns to pack 4 values per thread.
            #pragma unroll
            for( int ci = 0; ci < COLS_PER_THREAD / 2; ++ci ) {
                // Assuming EVEN,EVEN,ODD,ODD column pattern due to packing of V.
                uint4 src;
                src.x = acc[0][0].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 0); // 0
                src.y = acc[0][0].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 0); // 4
                src.z = acc[0][0].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 1); // 1
                src.w = acc[0][0].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 1); // 5

                using Src_type = typename Traits::Accumulator_type;
                using Dst_type = typename Traits::C_type;
                // Packs the 32bit values to 8bit.
                // Depending on the type, applies extra scaling with parameter scale_bmm2.
                uint32_t dst = Acc_packer<Src_type, Dst_type>::run(this, src);

                int64_t offset = (int64_t)ri * step_m + (int64_t)ci * step_n;
                fmha::stg(o_ptr_ + offset, dst);
            }  // ri
        }      // ci
    }

    // Move to the next location.
    inline __device__ void move() {
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }


    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The pointer.
    char *o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;

    // The row loaded by this thread.
    int row_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;
    uint32_t params_scale_bmm2_;

    const bool params_enable_i2f_trick_ = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits,
         typename Cta_tile,
         int WARPS_K>
struct Gmem_tile_o_hopper_32bit_8bit {};

template<typename Traits,
         typename Cta_tile>
struct Gmem_tile_o_hopper_32bit_8bit<Traits, Cta_tile, 1>
    : public Gmem_tile_o_gmma_32bit_8bit<Traits, Cta_tile> {


    // The Base class.
    using Base =  Gmem_tile_o_gmma_32bit_8bit<Traits, Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper_32bit_8bit(const Params &params,
                                                    const Block_info &block_info,
                                                    int tidx,
                                                    int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}

    template <typename Block_info>
    inline __device__ Gmem_tile_o_hopper_32bit_8bit(void * o_ptr,
                                                    int o_stride_in_bytes,
                                                    const Block_info &block_info,
                                                    int tidx,
                                                    uint32_t scale_bmm2,
                                                    int cta_row_offset = 0)
        : Base(o_ptr, o_stride_in_bytes, block_info, tidx, scale_bmm2, cta_row_offset) {}

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits,
         typename Cta_tile>
struct Gmem_tile_o_hopper_32bit_8bit<Traits, Cta_tile, 2>
    : public Gmem_tile_o_8bit<Traits, Cta_tile, /*CTAS_PER_HEAD=*/ 1> {

    // The Base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, /*CTAS_PER_HEAD=*/ 1>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper_32bit_8bit(const Params &params,
                                                    const Block_info &block_info,
                                                    int tidx,
                                                    int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {}

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M,
                                                       GMMA_N,
                                                       GMMA_K,
                                                       GMMA_A_RF,
                                                       GMMA_B_RF>,
                   Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper_32bit_8bit<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M,
                                                                               GMMA_N,
                                                                               GMMA_K,
                                                                               GMMA_A_RF,
                                                                               GMMA_B_RF>,
                                           Cta_tile,
                                           Cta_tile::WARPS_K> {

    // The traits class.
    using Traits = fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M,
                                                       GMMA_N,
                                                       GMMA_K,
                                                       GMMA_A_RF,
                                                       GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_32bit_8bit<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M,
                                                                                   GMMA_N,
                                                                                   GMMA_K,
                                                                                   GMMA_A_RF,
                                                                                   GMMA_B_RF>,
                                               Cta_tile,
                                               Cta_tile::WARPS_K>;
    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_o(const Params &params,
                                  const Block_info &block_info,
                                  int tidx,
                                  int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {
    }

    template <typename Block_info>
    inline __device__ Gmem_tile_o(void * o_ptr,
                                  int o_stride_in_bytes,
                                  const Block_info &block_info,
                                  int tidx,
                                  uint32_t scale_bmm2,
                                  int cta_row_offset = 0)
        : Base(o_ptr, o_stride_in_bytes, block_info, tidx, scale_bmm2, cta_row_offset) {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_igmma_int8_int32_traits<GMMA_M,
                                                        GMMA_N,
                                                        GMMA_K,
                                                        GMMA_A_RF,
                                                        GMMA_B_RF>,
                   Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper_32bit_8bit<fmha::Hopper_igmma_int8_int32_traits<GMMA_M,
                                                                                GMMA_N,
                                                                                GMMA_K,
                                                                                GMMA_A_RF,
                                                                                GMMA_B_RF>,
                                           Cta_tile,
                                           Cta_tile::WARPS_K> {

    // The traits class.
    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M,
                                                        GMMA_N,
                                                        GMMA_K,
                                                        GMMA_A_RF,
                                                        GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_32bit_8bit<fmha::Hopper_igmma_int8_int32_traits<GMMA_M,
                                                                                    GMMA_N,
                                                                                    GMMA_K,
                                                                                    GMMA_A_RF,
                                                                                    GMMA_B_RF>,
                                               Cta_tile,
                                               Cta_tile::WARPS_K>;
    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_o(const Params &params,
                                  const Block_info &block_info,
                                  int tidx,
                                  int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset) {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////


}  // namespace v2

} // namespace fmha
