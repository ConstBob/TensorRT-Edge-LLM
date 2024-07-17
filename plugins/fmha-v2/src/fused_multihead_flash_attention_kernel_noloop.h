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

#include <fused_multihead_attention_kernel.h>
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>

namespace fused_multihead_attention {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Kernel_traits, typename Params >
inline __device__ void device_flash_attention_nl(const Params &params) {

    // The instruction traits.
    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_o = typename Kernel_traits::Traits_o;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    // Do we use LDGSTS for Q, K or V?
    enum { USE_LDGSTS_Q = Kernel_traits::USE_LDGSTS_Q };
    enum { USE_LDGSTS_K = Kernel_traits::USE_LDGSTS_K };
    enum { USE_LDGSTS_V = Kernel_traits::USE_LDGSTS_V };

    // Do we use LDGSTS for any of the 3 input matrices.
    enum { USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V };

    // If either K or V uses LDGSTS, they cannot share a buffer.
    static_assert(!(USE_LDGSTS_K || USE_LDGSTS_V) || !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

    // Fragment double buffer (reduce register pressure)
    enum { FRAGMENT_QK_SIZE_IN_K_DIM = (Kernel_traits::LIMIT_QK_FRAGMENTS) * 2
            + !(Kernel_traits::LIMIT_QK_FRAGMENTS) * Mma_tile_p::MMAS_K };
    static_assert(!(Kernel_traits::LIMIT_QK_FRAGMENTS && USE_LDGSTS_K), "");
    static_assert(!(Kernel_traits::SHARE_SMEM_FOR_K_AND_V &&
                    (Kernel_traits::LIMIT_QK_FRAGMENTS || Kernel_traits::LIMIT_V_FRAGMENTS)),
                  "");

    enum { FRAGMENT_V_SIZE_IN_K_DIM = (Kernel_traits::LIMIT_V_FRAGMENTS) * 2
            + !(Kernel_traits::LIMIT_V_FRAGMENTS) * Mma_tile_o::MMAS_K };

    // Shared memory.
    extern __shared__ char smem_[];

    // The loop -- each CTA works on a different loop iteration.
    const int q_loop = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    // The block info.
    const Single_cta<Kernel_traits::VERSION> binfo(params, bidb, bidh, 0, tidx);
    if( binfo.stop_early(q_loop * Gmem_tile_q::ROWS) ) {
        return;
    }

    // Create the object to control the masks.
    fmha::Mask<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    // Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_loop * Gmem_tile_q::ROWS);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);

    // The base pointer of smem_v;
    char *smem_v_ = nullptr;
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        smem_v_ = &smem_[Smem_tile_q::BYTES_PER_TILE];
    } else {
        smem_v_ = &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE];
    }
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the global memory tile loader for O.
    // Gmem_tile_o gmem_o(params, binfo, tidx);
    Gmem_tile_o gmem_o(params, binfo, tidx, q_loop * Gmem_tile_o::ROWS);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Kernel_traits::NO_LOOP ? 0 : Smem_tile_q::BYTES_PER_TILE], tidx);

    // How many loops we need to apply mask.
    constexpr int MASK_LOOPS = (Cta_tile_p::M + Cta_tile_p::N - 1) / Cta_tile_p::N;
    static_assert(MASK_LOOPS * Cta_tile_p::N == Cta_tile_p::M || Cta_tile_p::N >= Cta_tile_p::M, "");

    // The start/end step of kv loops.
    // Do we need to mask out the tokens that is far away from the beginning.
    const bool MASK_PAST = Kernel_traits::LIMITED_PAST_SEQUENCE &&
        binfo.actual_seqlen > params.max_past_s;
    const int valid_seqlen = Kernel_traits::CAUSAL_MASK
        ? min((q_loop + 1) * Cta_tile_p::M, binfo.actual_seqlen)
        : binfo.actual_seqlen;

    const int kv_loop_end =
        ((valid_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N) * Cta_tile_p::N;
    const int kv_loop_start = MASK_PAST ?
        (max(0, q_loop * Cta_tile_p::M - params.max_past_s) / Cta_tile_p::N) * Cta_tile_p::N : 0;
    const int kv_loop_limited_past_mask = MASK_PAST ?
        (max(0, (q_loop + 1) * Cta_tile_p::M - 1 - params.max_past_s) / Cta_tile_p::N) * Cta_tile_p::N : 0;

    static_assert(Cta_tile_p::M >= Cta_tile_p::N, "");

    // Move K and V tiles.
    // We need offset here since we split single k loops into finer granularity.
    gmem_k.move_by_offset(kv_loop_start);
    gmem_v.move_by_offset(kv_loop_start);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);
    // Trigger the loads for K.
    gmem_v.load(smem_v);

    // Push the LDGDEPBAR instruction after the loads for Q, K and V.
    fmha::ldgdepbar<USE_LDGSTS>();

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Commit the data for V to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_v.commit(smem_v);
    }

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS, 1>(); __syncthreads();

    // Load the fragments for Q.
    // NOTE: lds full frag_q as it will be used by all kv loops
    typename Smem_tile_q::Fragment frag_q[FRAGMENT_QK_SIZE_IN_K_DIM][Mma_tile_p::MMAS_M];
    if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
        smem_q.load(frag_q[0], 0);
    } else {
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            smem_q.load(frag_q[ki], ki);
        }
    }

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[FRAGMENT_QK_SIZE_IN_K_DIM][Mma_tile_p::MMAS_N];
    if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
        smem_k.load(frag_k[0], 0);
    } else {
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            smem_k.load(frag_k[ki], ki);
        }
    }

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_v.commit(smem_v);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[FRAGMENT_V_SIZE_IN_K_DIM][Mma_tile_o::VALID_MMAS_N];
    if( Kernel_traits::LIMIT_V_FRAGMENTS ) {
        // v fragment load is deferred to after BMM1
    } else {
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
            smem_v.load(frag_v[ki], ki);
        }
    }

    // Store/load P to/from memory (for debugging).
#if defined(STORE_P)
    enum { BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8 };
    using Gmem_tile_p = fmha::Gmem_tile_p<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
    Gmem_tile_p gmem_p(params.p_ptr,
                       params.p_stride_in_bytes,
                       params.scale_bmm1,
                       tidx,
                       Kernel_traits::NO_LOOP ? q_loop * Cta_tile_p::M : 0);
#endif

    // Store S to memory (for debugging). NOTE: We use A_type as C_type is int32 for IMMA???
#if defined(STORE_S)
    enum { BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8 };
    using Gmem_tile_s = fmha::Gmem_tile_s<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
    Gmem_tile_s gmem_s(params.s_ptr,
                       params.s_stride_in_bytes,
                       params.scale_softmax,
                       tidx,
                       Kernel_traits::NO_LOOP ? q_loop * Cta_tile_p::M : 0);
#endif

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[Smem_tile_q::BYTES_PER_TILE], bidb, tidx);

    // Prefetch next kv buffer to share memory
    enum { PREFETCH_K_BUFFER_TO_SMEM = !Kernel_traits::LIMIT_QK_FRAGMENTS &&
        !Softmax::USE_SHARED_MEMORY };
    enum { PREFETCH_V_BUFFER_TO_SMEM = !Kernel_traits::SHARE_SMEM_FOR_K_AND_V };
    enum { PREFETCH_KV_BUFFER_TO_SMEM = PREFETCH_K_BUFFER_TO_SMEM && PREFETCH_V_BUFFER_TO_SMEM };

    // The number of threads per row.
    // enum { THREADS_PER_ROW = Cta_tile_p::WARPS_N * 8 };
    // DEBUG.
    // static_assert(THREADS_PER_ROW == 32, "");
    // END OF DEBUG.
    enum { THREADS_PER_ROW = 32 };

    // Load the mask for that iteration.
    mask.load(q_loop);

    // Declare the accumulators for the 1st gemm.
    fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
    using Acc_type_o = typename Traits_o::Accumulator_type;
    fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

    // Flash attention updater
    fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer;

    float global_max[Softmax::ROWS_PER_THREAD];
    float global_sum[Softmax::ROWS_PER_THREAD];

    for( int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += Cta_tile_p::N ) {

        const bool first_step = (kv_loop == kv_loop_start);
        // Need to set nans to 0 when limited past sequence is enabled
        //  as it is possible that all tokens are masked out.
        const bool apply_exp_mask = (MASK_PAST && kv_loop <= kv_loop_limited_past_mask);
        const bool apply_mask = params.has_alibi ||
                                (kv_loop >= kv_loop_end - MASK_LOOPS * Cta_tile_p::N) ||
                                apply_exp_mask;

        // Trigger the load for the next K/V values (smem_k double buffer).
        if( kv_loop + Cta_tile_p::N < kv_loop_end && PREFETCH_KV_BUFFER_TO_SMEM ) {

            // Make sure we are done reading the data (smem).
            if( !(Smem_tile_k::BUFFERS_PER_TILE > 1 && Smem_tile_v::BUFFERS_PER_TILE > 1) ) {
                __syncthreads();
            }

            gmem_k.move();
            smem_k.move_to_next_write_buffer();
            gmem_k.load(smem_k);

            gmem_v.move();
            smem_v.move_to_next_write_buffer();
            gmem_v.load(smem_v);

            // Push the LDGDEPBAR instruction after the loads for K.
            fmha::ldgdepbar<USE_LDGSTS>();

            gmem_k.commit(smem_k);
            if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
                gmem_v.commit(smem_v);
            }
        }

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        using Acc_type_p = typename Traits_p::Accumulator_type;
        fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {

            int q_ki = (ki - 1); // frag index for mma
            int k_ki = (ki - 1);
            if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                q_ki = (ki - 1) & 1;
                smem_q.load(frag_q[(ki & 1)], ki);
            }
            if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                k_ki = (ki - 1) & 1;
                smem_k.load(frag_k[(ki & 1)], ki);
            }

            // Do the math for the values already in registers.
            if( ki <= Mma_tile_p::VALID_MMAS_K ) {
                fmha::gemm(acc_p, frag_q[q_ki], frag_k[k_ki]);
            }
        }

        if( Mma_tile_p::MMAS_K <= Mma_tile_p::VALID_MMAS_K ) {
            int ki = Mma_tile_p::MMAS_K;
            int k_ki = (ki - 1);
            if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                k_ki = (ki - 1) & 1;
            }
            fmha::gemm(acc_p, frag_q[k_ki], frag_k[k_ki]);
        }

        // Store the P matrix.
#if defined(STORE_P)
        gmem_p.store(acc_p);
#endif

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack(acc_p);

        // Apply the mask.
        if( apply_mask ) {
            // Move mask offset.
            mask.move_to_offset(kv_loop);

            if( params.has_alibi ) {
                softmax.apply_mask_alibi_log2e(mask, bidh, params.alibi_params);
            } else {
                softmax.apply_mask(mask);
            }
        }

        // Make sure we are done reading the data (smem_v).
        if( (Kernel_traits::SHARE_SMEM_FOR_K_AND_V ||
             Kernel_traits::LIMIT_QK_FRAGMENTS) &&
             Softmax::USE_SHARED_MEMORY ) {
            __syncthreads();
        }

        // First step of the flash attention
        if( first_step ) {
            // Compute the max.
            softmax.template reduce<fmha::Max_>(global_max);

            if (Softmax::USE_SHARED_MEMORY) {
                // Make sure we are done reading shared memory.
                __syncthreads();
            }

            if( apply_exp_mask ) {
                softmax.template apply_exp2<true>(global_max);
            } else {
                softmax.template apply_exp2<false>(global_max);
            }

            // Compute the sum.
            softmax.template reduce<fmha::Sum_>(global_sum);
        }
        else {
            float tmp[Softmax::ROWS_PER_THREAD];

            #pragma unroll
            for(int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
                tmp[i] = global_max[i];
            }

            // Compute the max.
            softmax.template reduce<fmha::Max_>(global_max);

            if (Softmax::USE_SHARED_MEMORY) {
                // Make sure we are done reading shared memory.
                __syncthreads();
            }

            if( apply_exp_mask ) {
                // Update last step's acc_o.
                acc_o_normalizer.template update<true>(acc_o, global_max, tmp, global_sum);
                // Apply expf of softmax.
                softmax.template apply_exp2<true>(global_max);
            } else {
                // Update last step's acc_o.
                acc_o_normalizer.template update<false>(acc_o, global_max, tmp, global_sum);
                // Apply expf of softmax.
                softmax.template apply_exp2<false>(global_max);
            }

            // Update the global sum.
            #pragma unroll
            for(int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
                tmp[i] = global_sum[i];
                global_sum[i] = 0.f;
            }

            // Compute the sum.
            softmax.template reduce<fmha::Sum_>(global_sum);

            #pragma unroll
            for(int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
                global_sum[i] += tmp[i];
            }

        }

        // Trigger the load for the next K/V values.
        if( kv_loop + Cta_tile_p::N < kv_loop_end && !PREFETCH_KV_BUFFER_TO_SMEM ) {

            gmem_k.move();
            gmem_v.move();
            smem_k.move_to_next_write_buffer();
            smem_v.move_to_next_write_buffer();
            gmem_k.load(smem_k);
            gmem_v.load(smem_v);

            // Push the LDGDEPBAR instruction after the loads for Q, K and V.
            fmha::ldgdepbar<USE_LDGSTS>();

            if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V && Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                // Make sure K is fully consumed before committing
                __syncthreads();
            }
            gmem_k.commit(smem_k);
            if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V && !Kernel_traits::LIMIT_V_FRAGMENTS ) {
                // Sync before write in case it is not synced before
                if ( !Kernel_traits::LIMIT_QK_FRAGMENTS ) __syncthreads();
                // Only commit after V is done loading, else defer committing until after O = P*V
                gmem_v.commit(smem_v);
            }
        }

        // Store the P matrix.
#if defined(STORE_S)
        softmax.store(gmem_s);
#endif

#if defined(STORE_P)
        gmem_p.move_n();
#endif

#if defined(STORE_S)
        gmem_s.move();
#endif

        // Prefetch LDS Q/K
        if( kv_loop + Cta_tile_p::N < kv_loop_end ) {

            fmha::depbar<USE_LDGSTS, 1>(); __syncthreads();

            if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                smem_q.load(frag_q[0], 0);
            }

            smem_k.move_to_next_read_buffer();
            if( Kernel_traits::LIMIT_QK_FRAGMENTS ) {
                smem_k.load(frag_k[0], 0);
            } else {
                #pragma unroll
                for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
                    smem_k.load(frag_k[ki], ki);
                }
            }
        }

        // Prefetch LDS V for this loop
        if ( Kernel_traits::LIMIT_V_FRAGMENTS ) {
            // Sync before read in case it was not sync before
            if (kv_loop + Cta_tile_p::N >= kv_loop_end) __syncthreads();
            smem_v.load(frag_v[0], 0);
        }

        // Repack for the next BMM.
        fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        softmax.pack(frag_p);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_o::MMAS_K; ++ki ) {

            int p_ki = (ki - 1); // frag index for mma
            int v_ki = (ki - 1);
            if ( Kernel_traits::LIMIT_V_FRAGMENTS ) {
                v_ki = (ki - 1) & 1;
                smem_v.load(frag_v[(ki & 1)], ki);
            }
            fmha::gemm(acc_o, frag_p[p_ki], frag_v[v_ki]);
        }
        {
            int ki = Mma_tile_o::MMAS_K;
            int p_ki = (ki - 1);
            int v_ki = (ki - 1);
            if( Kernel_traits::LIMIT_V_FRAGMENTS ) {
                v_ki = (ki - 1) & 1;
            }
            fmha::gemm(acc_o, frag_p[p_ki], frag_v[v_ki]);
        }

        // Commit V to shared memory
        if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V && Kernel_traits::LIMIT_V_FRAGMENTS ) {
            __syncthreads();
            gmem_v.commit(smem_v);
        }

        // Prefetch LDS V for next loop
        if( kv_loop + Cta_tile_p::N < kv_loop_end ) {

            // Commit the data for V to shared memory if it has not been done already.
            if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
                // Make sure we are done loading the fragments for K.
                __syncthreads();

                // Commit the data to shared memory for V.
                gmem_v.commit(smem_v);

                // Make sure the data is in shared memory.
                __syncthreads();
            }

            smem_v.move_next_read_buffer();

            if( Kernel_traits::LIMIT_V_FRAGMENTS ) {
                // To avoid excessive spill, prefetch LDS V is delayed until after 1st GEMM
            } else {
                #pragma unroll
                for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
                    smem_v.load(frag_v[ki], ki);
                }
            }
        }
    } // Inner loop over the key/value sequence length.

    // Update acc_o of flash attention
    acc_o_normalizer.final_update(acc_o, global_sum);

    // Wait for last round of LDS K/V to finish
    __syncthreads();

    // Loop over MMAS_M.
    #pragma unroll
    for( int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii ) {

        // Swizzle the elements and do the final reduction.
        smem_o.store(acc_o, ii);

        // Make sure the data is in shared memory.
        __syncthreads();

        // Load from shared memory.
        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        smem_o.load(out);

        // Make sure the data was read from shared memory.
        if( ii < Gmem_tile_o::LOOPS - 1 ) {
            __syncthreads();
        }

        // Output the values.
        gmem_o.store(out, ii);
    }
}  // device_flash_attention_1xN

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fused_multihead_attention

