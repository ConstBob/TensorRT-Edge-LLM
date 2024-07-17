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

namespace fmha {
namespace ws {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Template instruction traits to specialize structs
    template<int, int, int, bool, bool> class Instruction_traits,
    // Kernel Traits
    typename Kernel_traits>
struct Compute {

    // The shared struct.
    using Shared = typename Kernel_traits::Shared;

    // The q, or kv tile reader.
    using Circular_buffer_q_reader = typename Kernel_traits::Circular_buffer_q_reader;
    using Circular_buffer_kv_reader = typename Kernel_traits::Circular_buffer_kv_reader;

    // The instruction traits for BMM1.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The Q shared memory tile.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    // The K shared memory tile.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    // The V shared memory tile.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Kernel_traits::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // The global memory tile for storing BMM2 output.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;

    // Softmax
    using Softmax = Softmax<Instruction_traits, Kernel_traits>;

    // BMM2 epilogue
    using Tile_o_epilogue = Tile_o_epilogue<Instruction_traits, Kernel_traits>;

    // The step size of Q loop.
    enum { STEP_Q = Kernel_traits::STEP_Q };
    // The step size of KV loop.
    enum { STEP_KV = Kernel_traits::STEP_KV };
    // The number of compute groups (currently fixed at 2).
    enum { NUM_COMPUTE_GROUPS = Kernel_traits::NUM_COMPUTE_GROUPS };
    // Whether use the causal mask.
    enum { CAUSAL_MASK           = Kernel_traits::CAUSAL_MASK };
    // Whether we will ignore the long distance tokens in the beginning.
    enum { LIMITED_PAST_SEQUENCE = Kernel_traits::LIMITED_PAST_SEQUENCE };

    // The head_dimension groups.
    enum { D_GROUPS = Kernel_traits::D_GROUPS };
    // The MMA_K groups (corresponding to head_dimension groups).
    enum { MMA_K_GROUPS = Kernel_traits::D_GROUPS };
    // The number of MMAS_K for each head_dimension group.
    enum { MMAS_K_PER_GROUP = Mma_tile_p::MMAS_K / Kernel_traits::D_GROUPS };

#define K_TILE_WAIT()                                                                          \
    int ready_k = cbr_k.peek();                                                                \
    if( !ready_k ) {                                                                           \
        cbr_k.wait();                                                                          \
    }                                                                                          \

#define KV_TILE_COMPLETE()                                                                     \
    if( tidx == 0 ) {                                                                          \
        cbr_k.complete(cbr_k.ptr());                                                           \
        cbr_v.complete(cbr_v.ptr());                                                           \
    }                                                                                          \
    cbr_k.advance();                                                                           \
    cbr_v.advance();                                                                           \

#define COMPUTE_SINGLE_TILE(IS_FIRST_COL, APPLY_MASK)                                           \
    compute_single_tile<IS_FIRST_COL, APPLY_MASK>(ctile_p,                                      \
                                                  softmax,                                      \
                                                  ctile_o,                                      \
                                                  p_max,                                        \
                                                  p_sum,                                        \
                                                  tidx,                                         \
                                                  actual_seqlen,                                \
                                                  params.max_past_s,                            \
                                                  q_step_offset + q_step_idx,                   \
                                                  kv_step_idx,                                  \
                                                  cbr,                                          \
                                                  cbr_v,                                        \
                                                  mutex_accessor,                               \
                                                  kv_step_idx == (kv_idx_end - 1));

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template< typename Params >
    inline __device__ void run(int warpgroup_id, int tidx, Shared *shared, const Params &params) {

        auto head_tracker = shared->head_info_tracker[warpgroup_id].createReader();
        auto cbr = shared->tma_q_tracker[warpgroup_id].createReader();

        auto cbr_k = shared->tma_k_tracker.createReader();
        auto cbr_v = shared->tma_v_tracker.createReader();

        // Ctile_p initalize (relies on q_stage, kv_stage).
        char *smem_q = reinterpret_cast<char *>(&shared->smem_q[warpgroup_id][0]);
        char *smem_k = reinterpret_cast<char *>(&shared->smem_k[0]);
        Compute_tile_p ctile_p(smem_q, smem_k);

        // Softmax
        Softmax softmax(params, tidx);

        // Ctile_o initalize (relies on kv_stage).
        uint32_t smem_v = __cvta_generic_to_shared(&shared->smem_v[0]);
        Compute_tile_o ctile_o(0, smem_v);

        // BMM2 epilogue
        Tile_o_epilogue tile_o_epilogue;

        // Mutex between two compute groups.
        OrderedMutexAccessor mutex_accessor(shared->compute_mutex, warpgroup_id);
        // Notify warpgroup 0 to execute HGMMA first (overlap HGMMA and Softmax Math Instructions).
        if( warpgroup_id == 1 ) {
            mutex_accessor.arrive();
        }

        // While loop for different heads.
        while( true ) {

            typename Shared::Head_info head_info = head_tracker.pop();

            if( head_info.kv_steps == -1 ) {
                break;
            }

            const int kv_steps = head_info.kv_steps;
            const int q_steps = head_info.q_steps;
            const int q_step_offset = head_info.q_step_offset;
            const int actual_seqlen = head_info.actual_seqlen;

            // Need to check if it is masked when max_past_s is samller than what actual_seqlen.
            static_assert(STEP_KV >= STEP_Q, "");
            // There will only be one kv tile needed to be masked since STEP_KV >= STEP_Q;
            const bool apply_start_mask = LIMITED_PAST_SEQUENCE ?
                actual_seqlen > params.max_past_s : false;

            int q_step_idx = warpgroup_id;

            // Compute work.
            for( ; q_step_idx < q_steps; q_step_idx += NUM_COMPUTE_GROUPS ) {

                // Check whether it is a valid run of q steps.
                const bool valid_run =
                    (q_step_idx + q_step_offset) * STEP_Q < actual_seqlen;

                // KV tile is shared by two q tiles, 
                // so we need to consider the last compute group's q tile.
                const int q_step_bound =
                    (q_step_idx + (warpgroup_id ^ 1) + q_step_offset + 1) * STEP_Q;

                // Skip initial kv tiles due to limited past squence length.
                // Consider the last 2 tiles.
                const int kv_idx_start = LIMITED_PAST_SEQUENCE ?
                    (max(0, q_step_bound - 2 * STEP_Q - params.max_past_s) / STEP_KV) : 0;

                const int kv_idx_start_mask_end = LIMITED_PAST_SEQUENCE ?
                    (max(0, q_step_bound - (warpgroup_id ^ 1) * STEP_Q - params.max_past_s) / STEP_KV) : 0;

                // We will skip unnecessary kv steps for causal mask.
                // Exclusive end.
                const int kv_idx_end = CAUSAL_MASK ?
                    ((q_step_bound + STEP_KV - 1) / STEP_KV) : kv_steps;
                const int valid_kv_idx_end = STEP_Q == STEP_KV
                    ? (CAUSAL_MASK
                        ? ((q_step_bound - (warpgroup_id ^ 1) * STEP_Q + STEP_KV - 1) / STEP_KV)
                        : kv_steps)
                    : kv_idx_end;
                const int valid_kv_steps = valid_kv_idx_end - kv_idx_start;

                Gmem_tile_o gmem_o(
                    params, head_info, tidx, (q_step_idx + q_step_offset) * STEP_Q);

                // Q ready to use in smem.
                int ready = cbr.peek();
                if( !ready ) {
                    cbr.wait();
                }

                static_assert(Mma_tile_p::CORES_M == 2);
                float p_max[Mma_tile_p::CORES_M];
                float p_sum[Mma_tile_p::CORES_M];

                int kv_step_idx = kv_idx_start;
                // First K tiles ready to use in smem.
                K_TILE_WAIT();
                // Need to apply mask if only kv tile exists.
                if( valid_kv_steps == 1 || apply_start_mask ) {
                    COMPUTE_SINGLE_TILE(true, true);
                }
                else {
                    COMPUTE_SINGLE_TILE(true, false);
                }
                KV_TILE_COMPLETE();

                for( kv_step_idx += 1; kv_step_idx < valid_kv_idx_end - 1; ++kv_step_idx ) {

                    // Current step's K tiles ready to use in smem.
                    K_TILE_WAIT();

                    // Move kv tile to next buffer.
                    if( D_GROUPS > 1 ) {
                        ctile_p.increment_gmma_desc_group();
                    } else {
                        ctile_p.increment_gmma_desc_b_group();
                    }

                    ctile_o.increment_gmma_desc_group();

                    if( apply_start_mask && kv_step_idx <= kv_idx_start_mask_end ) {
                        COMPUTE_SINGLE_TILE(false, true);
                    }
                    else {
                        COMPUTE_SINGLE_TILE(false, false);
                    }

                    KV_TILE_COMPLETE();
                }

                // We need to apply mask for the final KV tile (assume KV step size >= Q step size).
                static_assert(STEP_KV >= STEP_Q, "");

                for( ; kv_step_idx < kv_idx_end; ++kv_step_idx ) {
                    // Current step's K tiles ready to use in smem.
                    K_TILE_WAIT();

                    // Move kv tile to next buffer.
                    if( D_GROUPS > 1 ) {
                        ctile_p.increment_gmma_desc_group();
                    } else {
                        ctile_p.increment_gmma_desc_b_group();
                    }

                    ctile_o.increment_gmma_desc_group();

                    COMPUTE_SINGLE_TILE(false, true);

                    KV_TILE_COMPLETE();
                }

                if( valid_run ) {
                    // Final step's update.
                    tile_o_epilogue.scale(ctile_o, p_sum);
                    // Store o_tile to gmem.
                    gmem_o.store(ctile_o.acc_);
                }

                // Move q, kv to next buffer.
                ctile_p.increment_gmma_desc_a_group();
                ctile_p.increment_gmma_desc_b_group();
                ctile_o.increment_gmma_desc_group();
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template< bool IS_FIRST_COL, bool APPLY_MASK >
    inline __device__ void compute_single_tile(Compute_tile_p &ctile_p,
                                               Softmax &softmax,
                                               Compute_tile_o &ctile_o,
                                               float (&p_max)[Mma_tile_p::CORES_M],
                                               float (&p_sum)[Mma_tile_p::CORES_M],
                                               const int tidx,
                                               const int actual_seqlen,
                                               const int start_seqlen,
                                               const int ri,
                                               const int ci,
                                               Circular_buffer_q_reader &cbr,
                                               Circular_buffer_kv_reader &cbr_v,
                                               OrderedMutexAccessor &mutex,
                                               bool complete = false) {
        // Wait until another warpgroup has already executed HGMMA.
        mutex.wait();

        // Ctile_p is only used once by each n step.
        ctile_p.clear();

        // BMM1 (Q x K').
        warpgroup_arrive();

        // Only single K groups when sizeof(D) <= 128B.
        #pragma unroll
        for( int kbi = 0; kbi < MMA_K_GROUPS - 1; kbi++ ) {
            #pragma unroll
            for( int ki = 0; ki < MMAS_K_PER_GROUP; ki++ ) {
                ctile_p.compute(ki, false, ki == MMAS_K_PER_GROUP - 1);
            }
            ctile_p.increment_gmma_desc_group();
        }

        #pragma unroll
        for( int ki = 0; ki < MMAS_K_PER_GROUP - 1; ki++ ) {
            ctile_p.compute(ki);
        }

        ctile_p.compute(MMA_K_GROUPS * MMAS_K_PER_GROUP - 1, true, true);

        warpgroup_wait<0>();

        // Arrive when the last tile consumes the q tile.
        if( complete ) {
            if( tidx == 0 ) {
                cbr.complete(cbr.ptr());
            }
            cbr.advance();
        }

        // Notify another warpgroup to execute HGMMA.
        mutex.arrive();

        // Frgament p for BMM2 input
        Fragment_p frag_p[Mma_tile_o::MMAS_K];

        // Unpack o elements and apply the mask
        softmax.unpack_and_apply_mask<IS_FIRST_COL, APPLY_MASK>(
            ctile_p, p_max, actual_seqlen, start_seqlen, ri, ci);

        // Softmax Exp, max/sum, and update scales.
        softmax.compute_and_update_scale<IS_FIRST_COL>(p_max, p_sum);

        // Update flash attention scales and pack it for BMM2
        softmax.pack<IS_FIRST_COL>(ctile_o, frag_p);

        // Wait until v buffer is ready.
        int ready = cbr_v.peek();
        if( !ready ) {
            cbr_v.wait();
        }

        // BMM2 (S * V).
        warpgroup_arrive();

        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K - 1; ++ki ) {
            ctile_o.fill_frag_a(frag_p[ki]);
            ctile_o.compute(ki);
        }
        ctile_o.fill_frag_a(frag_p[Mma_tile_o::MMAS_K - 1]);
        ctile_o.compute(Mma_tile_o::MMAS_K - 1, true, true);

        warpgroup_wait<0>();
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ws
}  // namespace fmha
