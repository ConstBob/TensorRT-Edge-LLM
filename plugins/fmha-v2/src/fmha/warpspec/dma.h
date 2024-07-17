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
#include <fmha/hopper/utils_tma.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/hopper/tma_descriptor.h>

namespace fmha {
namespace ws {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
struct DMA {

    // The shared struct.
    using Shared = typename Kernel_traits::Shared;
    // The kv buffer writer.
    using Circular_buffer_kv_writer = typename Kernel_traits::Circular_buffer_kv_writer;

    // The step size of Q loop.
    enum { STEP_Q = Kernel_traits::STEP_Q };
    // The step size of KV loop.
    enum { STEP_KV = Kernel_traits::STEP_KV };

    // The tile size of Q.
    enum { TILE_SIZE_Q = STEP_Q * Kernel_traits::D };
    // The tile size of Q after head_dimension split.
    enum { TILE_SIZE_Q_PER_D_GROUP = STEP_Q * Kernel_traits::D_PER_GROUP };

    // The tile size of K.
    enum { TILE_SIZE_K = STEP_KV * Kernel_traits::D };
    // The tile size of K after head_dimension split.
    enum { TILE_SIZE_K_PER_D_GROUP = STEP_KV * Kernel_traits::D_PER_GROUP };

    // The tile size of V.
    enum { TILE_SIZE_V = TILE_SIZE_K };
    // The tile size of V after head_dimension split.
    enum { TILE_SIZE_V_PER_D_GROUP = TILE_SIZE_K_PER_D_GROUP };

    // Whether apply causal mask or not.
    enum { CAUSAL_MASK = Kernel_traits::CAUSAL_MASK };
    // Whether we will ignore the long distance tokens in the beginning.
    enum { LIMITED_PAST_SEQUENCE = Kernel_traits::LIMITED_PAST_SEQUENCE };

    // Is heads interleaved ?
    enum { HEADS_INTERLEAVED = Kernel_traits::HEADS_INTERLEAVED };

    struct Device {
        // The sum_s.
        int sum_s_;
        // multi_query_attention (multiple heads share the same key/value).
        bool multi_query_attention_;

        inline __device__ Device() {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Params>
        inline __device__ void run(const Params &params, Shared *shared) {
            // DMA.

            auto cbw0 = shared->tma_q_tracker[0].createWriter();
            auto cbw1 = shared->tma_q_tracker[1].createWriter();
            Circular_buffer_kv_writer cbw_k = shared->tma_k_tracker.createWriter();
            Circular_buffer_kv_writer cbw_v = shared->tma_v_tracker.createWriter();
            auto headinfo_tracker0 = shared->head_info_tracker[0].createWriter();
            auto headinfo_tracker1 = shared->head_info_tracker[1].createWriter();

            multi_query_attention_ = params.h_kv < params.h;

            int num_heads = params.h * params.b;
            for( int it = blockIdx.y; it < num_heads; it += gridDim.y ) {

                // If we do bidh = next_head % h, we'd guarantee b to be spread across CTAs.
                int bidh = it % params.h;
                int bidb = it / params.h;

                const cudaTmaDesc *desc_q = &params.tma_desc_q;
                const cudaTmaDesc *desc_k = &params.tma_desc_k;
                const cudaTmaDesc *desc_v = &params.tma_desc_v;
                int actual_seqlen;
                if( params.is_s_padded ) {
                    sum_s_ = bidb * params.s;
                    actual_seqlen = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
                } else {
                    sum_s_ = params.cu_seqlens[bidb];
                    actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s_;
                }

                // split work across M
                int q_steps = (actual_seqlen + STEP_Q - 1) / STEP_Q;

                // Q_steps may be distributed to multiple blocks to increase the occupacy
                // when b*h is small.
                // The number of q_steps needs to be multiple of 2.
                q_steps = (q_steps + gridDim.x - 1) / gridDim.x;
                q_steps += (q_steps & 1);
                // The last block may process fewer q_steps.
                int q_step_offset = q_steps * blockIdx.x;
                if( q_step_offset * STEP_Q >= actual_seqlen ) {
                    continue;
                }

                // Split work across N.
                const int kv_steps = (actual_seqlen + STEP_KV - 1) / STEP_KV;
                // Send head info.
                typename Shared::Head_info info{
                    q_steps, q_step_offset, kv_steps, actual_seqlen, sum_s_ * params.h + bidh
                };
                headinfo_tracker0.push(info);
                headinfo_tracker1.push(info);

                // Preload kv_tile.
                // Skip initial kv tiles due to max_past_s
                const int kv_idx_start = LIMITED_PAST_SEQUENCE ?
                    (max(0, (1 + q_step_offset - 1) * STEP_Q - params.max_past_s) / STEP_KV) : 0;
                load_kv(bidh, params.h, params.h_kv, kv_idx_start, desc_k, desc_v, shared, cbw_k, cbw_v);

                // Assign q_steps to two compute groups.
                for( int q_step_idx = 0; q_step_idx < q_steps; q_step_idx++ ) {
                    if( q_step_idx % 2 == 0 ) {
                        load_q(bidh, q_step_idx + q_step_offset, desc_q, shared->smem_q[0], cbw0);
                    } else {
                        load_q(bidh, q_step_idx + q_step_offset, desc_q, shared->smem_q[1], cbw1);
                    }

                    // Load kv tiles after two q tiles are issued.
                    if( ((q_step_idx % 2 == 1) || q_step_idx == q_steps - 1) ) {

                        // Skip initial kv tiles due to max_past_s
                        const int kv_idx_start = LIMITED_PAST_SEQUENCE ?
                            (max(0, (q_step_idx + q_step_offset - 1) * STEP_Q - params.max_past_s) / STEP_KV) :
                            0;

                        const int q_step_bound = (q_step_idx + q_step_offset + 1) * STEP_Q;
                        // Early stop when causal mask is enabled.
                        const int kv_idx_end = CAUSAL_MASK ?
                            ((q_step_bound + STEP_KV - 1) / STEP_KV) : kv_steps;
                        for( int kv_step_idx = kv_idx_start + 1; kv_step_idx < kv_idx_end; kv_step_idx++ ) {
                            load_kv(
                                bidh, params.h, params.h_kv, kv_step_idx, desc_k, desc_v, shared, cbw_k, cbw_v);
                        }
                        // Preload next q_step's kv tile.
                        if( q_step_idx != q_steps - 1 ) {
                            // Skip initial kv tiles due to max_past_s
                            const int next_kv_idx_start = LIMITED_PAST_SEQUENCE ?
                                (max(0, (q_step_idx + q_step_offset + 1) * STEP_Q - params.max_past_s) / STEP_KV) :
                                0;
                            load_kv(bidh, params.h, params.h_kv, next_kv_idx_start, desc_k, desc_v, shared, cbw_k, cbw_v);
                        }
                    }
                }
            }

            // Signal compute groups to break.
            headinfo_tracker0.push({ -1, -1, -1, -1, -1 });
            headinfo_tracker1.push({ -1, -1, -1, -1, -1 });
        }

        // Load q tiles from gmem to smem by TMA.
        template<typename BufferWriter, typename Smem_q>
        inline __device__ void load_q(int bidh,
                                      int q_step_idx,
                                      const cudaTmaDesc *desc_q,
                                      Smem_q &smem_q,
                                      BufferWriter &cbw) {

            int barrier_id = cbw.tmaReserve(TILE_SIZE_Q * Kernel_traits::ELEMENT_BYTES, 1);

            // coordinates: d, 3, h, s
            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
            const int32_t q_coord_dim1 = HEADS_INTERLEAVED ? 0 : bidh;
            const int32_t q_coord_dim2 = HEADS_INTERLEAVED ? bidh : 0;
            #pragma unroll
            for( int di = 0; di < Kernel_traits::D_GROUPS; ++di ) {
                const int32_t coords[4] = { di * Kernel_traits::D_PER_GROUP,
                                            multi_query_attention_ ? bidh : q_coord_dim1,
                                            multi_query_attention_ ? 0 : q_coord_dim2,
                                            sum_s_ + q_step_idx * STEP_Q };
                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(
                    desc_q,
                    __cvta_generic_to_shared(
                        &smem_q[barrier_id * TILE_SIZE_Q + di * TILE_SIZE_Q_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw.barrier_ptr(barrier_id)),
                    coords);
            }
        }

        // Load k,v tiles from gmem to smem by TMA.
        template<typename BufferWriter>
        inline __device__ void load_kv(int bidh,
                                       int h,
                                       int h_kv,
                                       int kv_step_idx,
                                       const cudaTmaDesc *desc_k,
                                       const cudaTmaDesc *desc_v,
                                       Shared *shared,
                                       BufferWriter &cbw_k,
                                       BufferWriter &cbw_v) {

            int k_barrier_id = cbw_k.tmaReserve((TILE_SIZE_K)*Kernel_traits::ELEMENT_BYTES, 1);

            int v_barrier_id = cbw_v.tmaReserve((TILE_SIZE_V)*Kernel_traits::ELEMENT_BYTES, 1);

            // Coordinates:
            // [d, 3, h, s] for head_interleaved, otherwise [d, h, 3, s]
            // for multi_query attention, it will be [d, h + 2, 1, s]
            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
            const int32_t k_coord_dim1 = HEADS_INTERLEAVED ? 1 : bidh;
            const int32_t k_coord_dim2 = HEADS_INTERLEAVED ? bidh : 1;
            const int32_t v_coord_dim1 = HEADS_INTERLEAVED ? 2 : bidh;
            const int32_t v_coord_dim2 = HEADS_INTERLEAVED ? bidh : 2;

            #pragma unroll
            for( int di = 0; di < Kernel_traits::D_GROUPS; ++di ) {
                const int32_t k_coords[4] = { di * Kernel_traits::D_PER_GROUP,
                                              multi_query_attention_ ? h + bidh / (h / h_kv) : k_coord_dim1,
                                              multi_query_attention_ ? 0 : k_coord_dim2,
                                              sum_s_ + kv_step_idx * STEP_KV };

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(
                    desc_k,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)),
                    k_coords);

                const int32_t v_coords[4] = { di * Kernel_traits::D_PER_GROUP,
                                              multi_query_attention_ ? h + h_kv + bidh / (h / h_kv) : v_coord_dim1,
                                              multi_query_attention_ ? 0 : v_coord_dim2,
                                              sum_s_ + kv_step_idx * STEP_KV };

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(
                    desc_v,
                    __cvta_generic_to_shared(
                        &shared->smem_v[di * Kernel_traits::KV_BUFFERS * TILE_SIZE_V_PER_D_GROUP +
                                        v_barrier_id * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_v.barrier_ptr(v_barrier_id)),
                    v_coords);
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////

    struct Host {
        Host() {
        }

        // Set TMA descriptors on host, and memcpy to device
        void init_params(bert::Fused_multihead_attention_params_v2 &params,
                         const bert::Fused_multihead_attention_launch_params &launch_params,
                         cudaStream_t stream) const {

            // Separate q, k, and v tma descriptors (continuous buffer).
            fmha::Multiple_tma_descriptor<4> qkv_tma_descriptor;

            // Per batch tensor size.
            uint32_t tensor_size_qkv[4];
            if( params.h_kv < params.h ) {
                // Take MQA as non-heads-interleaved.
                tensor_size_qkv[2] = 1;
                tensor_size_qkv[1] = (params.h + 2 * params.h_kv);
                tensor_size_qkv[0] = params.d;  //params.d;
            } else if( HEADS_INTERLEAVED ) {
                tensor_size_qkv[2] = params.h;
                tensor_size_qkv[1] = 3;
                tensor_size_qkv[0] = params.d;  //params.d;
            } else {
                tensor_size_qkv[2] = 3;
                tensor_size_qkv[1] = params.h;
                tensor_size_qkv[0] = params.d;  //params.d;
            }

            // Box size for k and v.
            uint32_t box_size[4];
            // Update this on device?
            box_size[2] = 1;
            box_size[1] = 1;
            box_size[0] = Kernel_traits::D_PER_GROUP;

            // Stride size in bytes. Assumes least significant dim is 1 (?)
            uint64_t tensor_stride_qkv[3];
            tensor_stride_qkv[0] = tensor_size_qkv[0] * Kernel_traits::ELEMENT_BYTES;  // d
            tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];          // d*h
            tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1];          // d*h*3

            // Traversal stride.
            uint32_t traversal_stride_qkv[4] = { 1, 1, 1, 1 };

            // OOB fill zeros.
            uint32_t oob_fill = 0;

            // FP32 to TF32 conversion disabled.
            uint32_t fp32_to_tf32 = 0;

            // GMMA descriptor mode.
            static constexpr int D_BYTES_PER_GROUP = Kernel_traits::D_BYTES_PER_GROUP;
            static constexpr fmha::cudaTmaDescSwizzle swizzle_mode =
                ( D_BYTES_PER_GROUP > 64 ? fmha::cudaTmaDescSwizzle::SWIZZLE_128B
                : D_BYTES_PER_GROUP > 32 ? fmha::cudaTmaDescSwizzle::SWIZZLE_64B
                :                          fmha::cudaTmaDescSwizzle::SWIZZLE_32B);

            static_assert(STEP_KV <= 256 && STEP_Q <= 256,
                          "max box size is 256");

            // QKV [TOTAL, 3, h, d].
            tensor_size_qkv[3] = 
                params.is_s_padded ? (params.b * params.s) : launch_params.seqlens[params.b];

            // QKV ptr.
            char *qkv_ptr = reinterpret_cast<char *>(params.qkv_ptr);

            // Q: STEP_Q.
            box_size[3] = STEP_Q;
            qkv_tma_descriptor.set_tma_desctriptor(
                qkv_ptr,
                fmha::cudaTmaDescFormat::F16_RN,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                tensor_size_qkv,
                tensor_stride_qkv,
                traversal_stride_qkv,
                box_size,
                oob_fill,
                fp32_to_tf32,
                &params.tma_desc_q);

            // K: STEP_KV.
            box_size[3] = STEP_KV;
            qkv_tma_descriptor.set_tma_desctriptor(
                qkv_ptr,
                fmha::cudaTmaDescFormat::F16_RN,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                tensor_size_qkv,
                tensor_stride_qkv,
                traversal_stride_qkv,
                box_size,
                oob_fill,
                fp32_to_tf32,
                &params.tma_desc_k);

            // V: STEP_KV.
            box_size[3] = STEP_KV;
            qkv_tma_descriptor.set_tma_desctriptor(
                qkv_ptr,
                fmha::cudaTmaDescFormat::F16_RN,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                tensor_size_qkv,
                tensor_stride_qkv,
                traversal_stride_qkv,
                box_size,
                oob_fill,
                fp32_to_tf32,
                &params.tma_desc_v);
        }
    };
};

}  // namespace ws
}  // namespace fmha