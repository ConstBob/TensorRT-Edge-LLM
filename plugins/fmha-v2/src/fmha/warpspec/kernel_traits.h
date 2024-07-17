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
#include <cuda/std/array>
#include <fmha/hopper/gmem_tile_o_packed.h>
#include <fmha/warpspec/epilogue.h>
#include <fmha/warpspec/circular_buffer.h>

namespace fmha {
namespace ws {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
         // The instruction trait template for initializing BMM1 and BMM2 traits.
         template<int, int, int, bool, bool> class Instruction_traits,
         // The step size in query sequence dimension (M of BMM1 and BMM2).
         int STEP_Q_,
         // The step size in key/value sequence dimension (N of BMM1 and K of BMM2).
         int STEP_KV_,
         // The head dimension.
         int D_,
         // The number of smem buffers for Q tiles.
         int Q_BUFFERS_,
         // The number of smem buffers for K, and V tiles.
         int KV_BUFFERS_,
         // The number of compute warpgroups (128 threads per warpgroup).
         int NUM_COMPUTE_GROUPS_,
         // The number of data warpgroups (TMA).
         int DMA2COMPUTE_DEPTH_,
         // The attention mask type: padding (0), causal (1), limited_past_causal (2).
         // See fused_multihead_attention_kernel.h for description.
         int ATTENTION_MASK_TYPE_ = 0,
         // Is head interleaved ? 
         // (head_interleaved means input [bxs, h, 3, d], otherwise [bx3, 3, h, d]).
         bool HEADS_INTERLEAVED_ = true>
struct Kernel_traits {

    // The step size in query sequence dimension (M of BMM1 and BMM2).
    enum { STEP_Q = STEP_Q_ };

    // The step size in key/value sequence dimension (N of BMM1 and K of BMM2).
    enum { STEP_KV = STEP_KV_ };

    // The acutal head dimension.
    enum { D = D_ };

    // The number of smem buffers for Q tiles.
    enum { Q_BUFFERS = Q_BUFFERS_ };

    // The number of smem buffers for K, and V tiles.
    enum { KV_BUFFERS = KV_BUFFERS_ };

    // The number of compute warpgroups (128 threads per warpgroup).
    enum { NUM_COMPUTE_GROUPS = NUM_COMPUTE_GROUPS_ };

    // The number of data warpgroups (TMA).
    enum { DMA2COMPUTE_DEPTH = DMA2COMPUTE_DEPTH_ };

    // The number of ctas per cluster.
    enum { CTAS_PER_CGA = 1 };

    // The total number of threads per block,
    enum { THREADS = 128 + NUM_COMPUTE_GROUPS * 128 };

    // The number of warps in the M dimension.
    enum { WARPS_M = 4 };

    // The number of warpgroups in the M dimensions.
    enum { WARP_GROUP_M = WARPS_M / 4 };

    // The number of warps in the N dimension.
    enum { WARPS_N = 1 };

    // The number of warpgroups in the N dimension.
    enum { WARP_GROUP_N = WARPS_N };

    // The number of warpgroups in the K dimension.
    enum { WARP_GROUP_K = 1 };

    // The attention mask type: padding (0), causal (1), limited_past_causal (2).
    enum { CAUSAL_MASK           = (ATTENTION_MASK_TYPE_ == 1 || ATTENTION_MASK_TYPE_ == 2) };
    enum { LIMITED_PAST_SEQUENCE = ATTENTION_MASK_TYPE_ == 2 };

    // Is head interleaved ? 
    // (head_interleaved means input [bxs, h, 3, d], otherwise [bx3, 3, h, d]).
    enum { HEADS_INTERLEAVED = HEADS_INTERLEAVED_ };

    // The instruction traits for the BMM1.
    using Traits_p = Instruction_traits<STEP_Q, STEP_KV, 16, false, false>;

    // The bytes per element.
    enum { ELEMENT_BYTES = sizeof(typename Traits_p::A_type) };
    // The bytes of head dimension.
    enum { D_BYTES = D * ELEMENT_BYTES };

    // Split D into multiple groups in order to match the TMA swizzle mode (128B).
    // 1. BMM1: we split D into multiple K groups.
    // 2. BMM2: we split D into multiple N groups (MMAS_N).

    // The number of head_dimension groups.
    enum { D_GROUPS = fmha::Div_up<D_BYTES, 128>::VALUE };
    // The head_dimension per group.
    enum { D_PER_GROUP = D / D_GROUPS };
    static_assert( D_GROUPS * D_PER_GROUP == D );
    // The head_dimension bytes per group
    enum { D_BYTES_PER_GROUP = D_BYTES / D_GROUPS };

    // Set GMMA descriptor mode based on the head_size.
    static constexpr auto GMMA_DESC_MODE = 
        ( D_BYTES_PER_GROUP > 64 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
        : D_BYTES_PER_GROUP > 32 ? fmha::Gmma_descriptor_mode::SWIZZLE_64B
        :                          fmha::Gmma_descriptor_mode::SWIZZLE_32B);

    // The instruction traits for the BMM2.
    using Traits_o = Instruction_traits<STEP_Q, D_PER_GROUP, 16, true, false>;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Traits_p::template Cta_tile<STEP_Q,
                                                            STEP_KV,
                                                            D,
                                                            WARP_GROUP_M,
                                                            WARP_GROUP_N,
                                                            WARP_GROUP_K>;

    // The CTA description for BMM1 (after head_dimension is split).
    using Cta_tile_p_split_d = typename Traits_p::template Cta_tile<STEP_Q,
                                                                    STEP_KV,
                                                                    D_PER_GROUP,
                                                                    WARP_GROUP_M,
                                                                    WARP_GROUP_N,
                                                                    WARP_GROUP_K>;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Traits_o::template Cta_tile<STEP_Q,
                                                            D,
                                                            STEP_KV,
                                                            WARP_GROUP_M,
                                                            WARP_GROUP_K,
                                                            WARP_GROUP_N>;

    // The CTA description for BMM2 (after head dimension is split).
    using Cta_tile_o_split_d = typename Traits_o::template Cta_tile<STEP_Q,
                                                                    D_PER_GROUP,
                                                                    STEP_KV,
                                                                    WARP_GROUP_M,
                                                                    WARP_GROUP_K,
                                                                    WARP_GROUP_N>;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // Smem_tiles are currently only used as meta data for the compute tile.
    // The Q shared memory tile.
    using Smem_tile_q = fmha::Smem_tile_hopper_a<Traits_p,
                                                 Cta_tile_p_split_d,
                                                 fmha::Row,
                                                 16,
                                                 Q_BUFFERS * D_GROUPS,
                                                 GMMA_DESC_MODE,
                                                 true, // USE_TMA_Q
                                                 Traits_p::GMMA_A_RF>;

    // The K shared memory tile.
    using Smem_tile_k = fmha::Smem_tile_hopper_b<Traits_p,
                                                 Cta_tile_p_split_d,
                                                 fmha::Col,
                                                 16,
                                                 KV_BUFFERS * D_GROUPS,
                                                 GMMA_DESC_MODE,
                                                 true // USE_TMA_K
                                                 >;

    // The V shared memory tile.
    using Smem_tile_v = fmha::Smem_tile_hopper_b<Traits_o,
                                                 Cta_tile_o_split_d,
                                                 fmha::Row,
                                                 16,
                                                 KV_BUFFERS,
                                                 GMMA_DESC_MODE,
                                                 true // USE_TMA_V
                                                 >;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = fmha::Compute_tile_with_gmma<Traits_p,
                                                        Cta_tile_p,
                                                        Smem_tile_q,
                                                        Smem_tile_k,
                                                        Traits_p::GMMA_A_RF,
                                                        Traits_p::GMMA_B_RF>;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = fmha::Compute_tile_with_gmma<Traits_o,
                                                        Cta_tile_o,
                                                        Smem_tile_q,
                                                        Smem_tile_v,
                                                        Traits_o::GMMA_A_RF,
                                                        Traits_o::GMMA_B_RF>;
    // The global memory tile for O.
    using Gmem_tile_o = fmha::v2::Gmem_tile_o_hopper<Traits_o,
                                                     Cta_tile_o,
                                                     Cta_tile_o::WARPS_K>;

    // The q, k, v tile buffer.
    using Data_type = typename Traits_p::A_type;
    using Buffer_q_t = cuda::std::array<Data_type, D * STEP_Q * Q_BUFFERS>;
    using Buffer_k_t = cuda::std::array<Data_type, D * STEP_KV * KV_BUFFERS>;
    using Buffer_v_t = cuda::std::array<Data_type, D * STEP_KV * KV_BUFFERS>;

    // The smem bytes of q, k, v tiles.
    enum {
        SMEM_BYTES_Q = sizeof(Buffer_q_t),
        SMEM_BYTES_K = sizeof(Buffer_k_t),
        SMEM_BYTES_V = sizeof(Buffer_v_t),
    };

    // The reader/writer (consumer/producer) barriers.
    using Circular_buffer_kv_reader = typename CircularBuffer<KV_BUFFERS, CTAS_PER_CGA>::Reader;
    using Circular_buffer_kv_writer = typename CircularBuffer<KV_BUFFERS, CTAS_PER_CGA>::Writer;
    using Circular_buffer_q_reader = typename CircularBuffer<Q_BUFFERS, CTAS_PER_CGA>::Reader;
    using Circular_buffer_q_writer = typename CircularBuffer<Q_BUFFERS, CTAS_PER_CGA>::Writer;

    // The struct of shared memory buffers.
    struct Shared {

        // The smem buffer of q, k, v tiles
        Buffer_q_t smem_q[NUM_COMPUTE_GROUPS];
        Buffer_k_t smem_k;
        Buffer_v_t smem_v;

        // The head info to be shared among compute groups
        struct Head_info {
            // How many steps to execute.
            int q_steps;
            // The start step for query.
            int q_step_offset;
            // How many steps to execute.
            int kv_steps;
            // The actual sequence length (variable sequence length).
            int actual_seqlen;
            // The batch/head index.
            int bidx;
        };

        // DMA to Compute:
        // In this use case it probably makes sense to have the same number of BUFFERS in both queues.
        // - barriers to wait for K+V loads to complete.
        CircularBuffer<KV_BUFFERS, CTAS_PER_CGA> tma_k_tracker;
        CircularBuffer<KV_BUFFERS, CTAS_PER_CGA> tma_v_tracker;
        CircularBuffer<Q_BUFFERS, CTAS_PER_CGA> tma_q_tracker[NUM_COMPUTE_GROUPS];
        CircularBufferWithData<DMA2COMPUTE_DEPTH, Head_info> head_info_tracker[NUM_COMPUTE_GROUPS];

        // Mutex
        OrderedMutex compute_mutex;

        inline __device__ void init(int tid0) {

            #pragma unroll
            for( int i = 0; i < NUM_COMPUTE_GROUPS; i++ ) {
                tma_q_tracker[i].init(tid0, 1, CTAS_PER_CGA);
                head_info_tracker[i].init(tid0, /*producer_threads=*/1, /*consumer_threads=*/128);
            }

            tma_k_tracker.init(tid0, 1, NUM_COMPUTE_GROUPS);
            tma_v_tracker.init(tid0, 1, NUM_COMPUTE_GROUPS);

            compute_mutex.init(tid0, 128, 128);
        }
    };

    enum { BYTES_PER_SMEM = sizeof(Shared) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ws
}  // namespace fmha
