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

#include <fmha/utils.h>
#include <fmha/traits.h>

namespace fmha {
namespace ws {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Special Softmax struct to handle optimization tricks on Hopper Warp-Specialized Kernels.
template< template<int, int, int, bool, bool> class Traits,
          typename Kernel_traits >
struct Softmax_base {

    // The instruction traits for BMM1.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

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

    // Whether apply causal mask or not.
    enum { CAUSAL_MASK = Kernel_traits::CAUSAL_MASK };
    // Whether we will ignore the long distance tokens in the beginning.
    enum { LIMITED_PAST_SEQUENCE = Kernel_traits::LIMITED_PAST_SEQUENCE };

    // Ctor.
    template< typename Params >
    inline __device__ Softmax_base(Params params, int tidx) : scale_bmm1_(params.scale_bmm1) {

        int warp = tidx / 32;
        int lane = tidx % 32;
        // The corrsponding row/col for each thread after MMA.
        // fixed 4x1 warp layout.
        quad_col_ = lane % 4;
        if( CAUSAL_MASK ) {
            quad_row_ = warp * 16 + lane / 4;
        }
    }

    // Convert from FP16 fragments to floats.
    template< bool IS_FIRST_COL, bool APPLY_MASK >
    inline __device__ void unpack_and_apply_mask(Compute_tile_p &ctile_p,
                                                 float (&global_max)[Mma_tile_p::CORES_M],
                                                 int actual_seqlen,
                                                 int max_past_seqlen,
                                                 int ri,
                                                 int ci) {

        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::CORES_N; ni++ ) {

                bool v0 = true, v1 = true;
                if constexpr( APPLY_MASK ) {
                    if constexpr( CAUSAL_MASK ) {
                        // Causal Mask: we have to apply mask before getting max.
                        int row = ri * Cta_tile_p::M + quad_row_ + mi * 8;
                        int col = ci * Cta_tile_p::N + quad_col_ * 2 + ni * 8;
                        // Mask for the two N elements.
                        v0 = (col <= row);
                        v1 = (col + 1 <= row);

                        // Only pay attention to the last max-past-seqlen long tokens.
                        if constexpr( LIMITED_PAST_SEQUENCE ) {
                            int start_seqlen = max(0, row - max_past_seqlen);
                            v0 &= (col >= start_seqlen);
                            v1 &= (col + 1 >= start_seqlen);
                        }

                    } else {
                        // Dense Mask.
                        int col = ci * Cta_tile_p::N + quad_col_ * 2 + ni * 8;
                        v0 = (col < actual_seqlen);
                        v1 = (col + 1 < actual_seqlen);
                    }
                }

                // Satfinite in case of overflow (due to fp16 accumulation).
                float2 f2 = half2_to_float2(satfinite_h2(
                    ctile_p.acc_[0][0].reg(ni * Mma_tile_p::CORES_M + mi)));

                f2.x = v0 ? f2.x : -HUGE_VALF;
                f2.y = v1 ? f2.y : -HUGE_VALF;
                reinterpret_cast<float2 *>(&elt_[mi][2 * ni])[0] = f2;
            }
        }

    }

    // Calulate max/sum, and update flash-attention scales.
    template< bool IS_FIRST_COL >
    inline __device__ void compute_and_update_scale(float (&global_max)[Mma_tile_p::CORES_M],
                                                    float (&global_sum)[Mma_tile_p::CORES_M]) {

        float scale = reinterpret_cast<float&>(scale_bmm1_);

        // Row-wise max of current tile.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
            if( IS_FIRST_COL ) {
                local_max_[mi] = elt_[mi][0];
            } else {
                local_max_[mi] = fmaxf(global_max[mi], elt_[mi][0]);
            }
            #pragma unroll
            for( int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++ ) {
                local_max_[mi] = fmaxf(local_max_[mi], elt_[mi][ni]);
            }
            local_max_[mi] =
                fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 1), local_max_[mi]);
            local_max_[mi] =
                fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 2), local_max_[mi]);
        }

        // Softmax Exp.
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_p::CORES_N; ni++ ) {
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {

                float &p0 = elt_[mi][2 * ni + 0];
                float &p1 = elt_[mi][2 * ni + 1];

                float scaled_max = local_max_[mi] * scale;

                p0 = custom_expf(p0, scale, scaled_max);
                p1 = custom_expf(p1, scale, scaled_max);
                // Need to check if nans are generated as we may skip the whole kv tile for single row.
                if constexpr( LIMITED_PAST_SEQUENCE ) {
                    p0 = p0 != p0 ? 0.f : p0;
                    p1 = p1 != p1 ? 0.f : p1;
                }
            }
        }

        // Row-wise sum of current tile.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
            local_sum_[mi] = elt_[mi][0];
            #pragma unroll
            for( int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++ ) {
                local_sum_[mi] += elt_[mi][ni];
            }
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 1);
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 2);
        }

        // Initialize or update the global sum and max.
        if( IS_FIRST_COL ) {
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
                global_sum[mi] = local_sum_[mi];
                global_max[mi] = local_max_[mi];
            }

        } else {
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
                float max_old = global_max[mi];
                float max_new = local_max_[mi];
                float sum_old = global_sum[mi];
                float sum_new = local_sum_[mi];
                // Remove the old max and replace by the new one.
                correction_[mi] = exp2f((max_old - max_new) * scale);
                // Need to check if nans are generated as we may skip the whole kv tile for single row.
                if constexpr( LIMITED_PAST_SEQUENCE ) {
                    correction_[mi] = correction_[mi] != correction_[mi] ? 0.f : correction_[mi];
                }
                global_sum[mi] = sum_old * correction_[mi] + sum_new;

                // New max already takes into account old max.
                global_max[mi] = max_new;
            }
        }

    }

    // Update flash attention scales and pack elements for BMM2.
    template< bool IS_FIRST_COL >
    inline __device__ void pack(Compute_tile_o &ctile_o,
                                Fragment_p (&frag_p)[Mma_tile_o::MMAS_K]) {

        // Pack 4 cols for BMM2 A tile.
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_o::MMAS_K; ni++ ) {
            frag_p[ni].reg(0) = float2_to_half2(elt_[0][4 * ni + 0], elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) = float2_to_half2(elt_[1][4 * ni + 0], elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) = float2_to_half2(elt_[0][4 * ni + 2], elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) = float2_to_half2(elt_[1][4 * ni + 2], elt_[1][4 * ni + 3]);
        }

        if( !IS_FIRST_COL ) {
            // Correct accumulators to current max.
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
                const uint32_t scale = float_to_half2(correction_[mi]);

                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimention is split.
                #pragma unroll
                for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                    #pragma unroll
                    for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                        uint32_t &reg = ctile_o.acc_[0][mma_ni].reg(ni * Mma_tile_o::CORES_M + mi);
                        reg = hmul2(reg, scale);
                    }
                }
            }
        } else {
            ctile_o.clear();
        }
    }

    // BMM1 scale.
    uint32_t scale_bmm1_;
    // The col index for the mma thread layout.
    int quad_col_;
    // The row index for the mma thread layout.
    int quad_row_;

    // Unpacked BMM1 output buffer.
    float elt_[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N * 2];
    // Local max.
    float local_max_[Mma_tile_p::CORES_M];
    // Local sum.
    float local_sum_[Mma_tile_p::CORES_M];
    // Correction_ scales for ctil_o.
    float correction_[Mma_tile_p::CORES_M];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< template<int, int, int, bool, bool> class Traits,
          typename Kernel_traits >
struct Softmax {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp16_traits
template<typename Kernel_traits>
struct Softmax<Hopper_hgmma_fp16_traits, Kernel_traits> :
    public Softmax_base<Hopper_hgmma_fp16_traits, Kernel_traits> {

    // The Base class.
    using Base = Softmax_base<Hopper_hgmma_fp16_traits, Kernel_traits>;

    // Ctor.
    template< typename Params >
    inline __device__ Softmax(const Params &params, int tidx)
        : Base(params, tidx) { }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp32_traits
template<typename Kernel_traits>
struct Softmax<Hopper_hgmma_fp32_traits, Kernel_traits> :
    public Softmax_base<Hopper_hgmma_fp32_traits, Kernel_traits> {

    // The Base class.
    using Base = Softmax_base<Hopper_hgmma_fp32_traits, Kernel_traits>;

    // The instruction traits for BMM1.
    using Traits_p = typename Base::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Base::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Base::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Base::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Whether apply causal mask or not.
    enum { CAUSAL_MASK = Base::CAUSAL_MASK };
    // Whether we will ignore the long distance tokens in the beginning.
    enum { LIMITED_PAST_SEQUENCE = Kernel_traits::LIMITED_PAST_SEQUENCE };

    // Ctor.
    template< typename Params >
    inline __device__ Softmax(const Params &params, int tidx)
        : Base(params, tidx) { }

    // Unpack BMM1 float ouput.
    template< bool IS_FIRST_COL, bool APPLY_MASK >
    inline __device__ void unpack_and_apply_mask(Compute_tile_p &ctile_p,
                                                 float (&global_max)[Mma_tile_p::CORES_M],
                                                 int actual_seqlen,
                                                 int max_past_seqlen,
                                                 int ri,
                                                 int ci) {

        // Compute S and apply mask.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::CORES_N; ni++ ) {

                bool v0 = true, v1 = true;
                if constexpr( APPLY_MASK ) {
                    if constexpr( CAUSAL_MASK ) {
                        // Causal Mask: we have to apply mask before getting max.
                        int row = ri * Cta_tile_p::M + this->quad_row_ + mi * 8;
                        int col = ci * Cta_tile_p::N + this->quad_col_ * 2 + ni * 8;
                        // Mask for the two N elements.
                        v0 = (col <= row);
                        v1 = (col + 1 <= row);

                        // Only pay attention to the last max-past-seqlen long tokens.
                        if constexpr( LIMITED_PAST_SEQUENCE ) {
                            int start_seqlen = max(0, row - max_past_seqlen);
                            v0 &= (col >= start_seqlen);
                            v1 &= (col + 1 >= start_seqlen);
                        }

                    } else {
                        // Dense Mask.
                        int col = ci * Cta_tile_p::N + this->quad_col_ * 2 + ni * 8;
                        v0 = (col < actual_seqlen);
                        v1 = (col + 1 < actual_seqlen);
                    }
                }

                float reg0 =
                  ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi);
                float reg1 =
                  ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi + 1);

                this->elt_[mi][2 * ni] = v0 ? reg0: -HUGE_VALF;
                this->elt_[mi][2 * ni + 1] = v1 ? reg1 : -HUGE_VALF;
            }
        }
    }

    // Update flash attention scales and pack elements for BMM2.
    template< bool IS_FIRST_COL >
    inline __device__ void pack(Compute_tile_o &ctile_o,
                                Fragment_p (&frag_p)[Mma_tile_o::MMAS_K]) {

        // Pack 4 cols for BMM2 A tile.
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_o::MMAS_K; ni++ ) {
            frag_p[ni].reg(0) = float2_to_half2(
                this->elt_[0][4 * ni + 0], this->elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) = float2_to_half2(
                this->elt_[1][4 * ni + 0], this->elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) = float2_to_half2(
                this->elt_[0][4 * ni + 2], this->elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) = float2_to_half2(
                this->elt_[1][4 * ni + 2], this->elt_[1][4 * ni + 3]);
        }

        if( !IS_FIRST_COL ) {
            // Correct accumulators to current max.
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimention is split.
                #pragma unroll
                for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                    #pragma unroll
                    for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                        float &reg0 =
                          ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                        float &reg1 =
                          ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                        reg0 *= this->correction_[mi];
                        reg1 *= this->correction_[mi];
                    }
                }
            }
        } else {
            ctile_o.clear();
        }
    }

};
////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_bf16_traits
template<typename Kernel_traits>
struct Softmax<Hopper_hgmma_bf16_traits, Kernel_traits> :
    public Softmax_base<Hopper_hgmma_bf16_traits, Kernel_traits> {

    // The Base class.
    using Base = Softmax_base<Hopper_hgmma_bf16_traits, Kernel_traits>;

    // The instruction traits for BMM1.
    using Traits_p = typename Base::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Base::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Base::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Base::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Whether apply causal mask or not.
    enum { CAUSAL_MASK = Base::CAUSAL_MASK };
    // Whether we will ignore the long distance tokens in the beginning.
    enum { LIMITED_PAST_SEQUENCE = Kernel_traits::LIMITED_PAST_SEQUENCE };

    // Ctor.
    template< typename Params >
    inline __device__ Softmax(const Params &params, int tidx)
        : Base(params, tidx) { }

    // Unpack BMM1 float ouput.
    template< bool IS_FIRST_COL, bool APPLY_MASK >
    inline __device__ void unpack_and_apply_mask(Compute_tile_p &ctile_p,
                                                 float (&global_max)[Mma_tile_p::CORES_M],
                                                 int actual_seqlen,
                                                 int max_past_seqlen,
                                                 int ri,
                                                 int ci) {

        // Compute S and apply mask.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::CORES_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::CORES_N; ni++ ) {

                bool v0 = true, v1 = true;
                if constexpr( APPLY_MASK ) {
                    if constexpr( CAUSAL_MASK ) {
                        // Causal Mask: we have to apply mask before getting max.
                        int row = ri * Cta_tile_p::M + this->quad_row_ + mi * 8;
                        int col = ci * Cta_tile_p::N + this->quad_col_ * 2 + ni * 8;
                        // Mask for the two N elements.
                        v0 = (col <= row);
                        v1 = (col + 1 <= row);

                        // Only pay attention to the last max-past-seqlen long tokens.
                        if constexpr( LIMITED_PAST_SEQUENCE ) {
                            int start_seqlen = max(0, row - max_past_seqlen);
                            v0 &= (col >= start_seqlen);
                            v1 &= (col + 1 >= start_seqlen);
                        }

                    } else {
                        // Dense Mask.
                        int col = ci * Cta_tile_p::N + this->quad_col_ * 2 + ni * 8;
                        v0 = (col < actual_seqlen);
                        v1 = (col + 1 < actual_seqlen);
                    }
                }

                float reg0 =
                  ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi);
                float reg1 =
                  ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi + 1);

                this->elt_[mi][2 * ni] = v0 ? reg0 : -HUGE_VALF;
                this->elt_[mi][2 * ni + 1] = v1 ? reg1 : -HUGE_VALF;
            }
        }
    }

    // Update flash attention scales and pack elements for BMM2.
    template< bool IS_FIRST_COL >
    inline __device__ void pack(Compute_tile_o &ctile_o,
                                Fragment_p (&frag_p)[Mma_tile_o::MMAS_K]) {

        // Pack 4 cols for BMM2 A tile.
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_o::MMAS_K; ni++ ) {
            frag_p[ni].reg(0) =
                float2_to_bf16_x2(this->elt_[0][4 * ni + 0], this->elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) =
                float2_to_bf16_x2(this->elt_[1][4 * ni + 0], this->elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) =
                float2_to_bf16_x2(this->elt_[0][4 * ni + 2], this->elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) =
                float2_to_bf16_x2(this->elt_[1][4 * ni + 2], this->elt_[1][4 * ni + 3]);
        }

        if( !IS_FIRST_COL ) {
            // Correct accumulators to current max.
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimention is split.
                #pragma unroll
                for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                    #pragma unroll
                    for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                        float &reg0 =
                          ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                        float &reg1 =
                          ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                        reg0 *= this->correction_[mi];
                        reg1 *= this->correction_[mi];
                    }
                }
            }
        } else {
            ctile_o.clear();
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// BMM2 epilogue to apply scales (flash attention)
template< template<int, int, int, bool, bool> class Traits,
          typename Kernel_traits >
struct Tile_o_epilogue_base {

    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(Compute_tile_o &ctile_o,
                                 float (&global_sum)[Mma_tile_o::CORES_M]) {
        // Final step's update.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
            global_sum[mi] = global_sum[mi] == 0.f ? 1.f : 1.0f / global_sum[mi];
            const uint32_t scale = float_to_half2(global_sum[mi]);

            // Assume only N has multiple MMAs (MMAS_M = 1).
            #pragma unroll
            for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                #pragma unroll
                for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                    uint32_t &reg =
                        ctile_o.acc_[0][mma_ni].reg(ni * Mma_tile_o::CORES_M + mi);
                    reg = hmul2(reg, scale);
                }
            }
        }
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template< template<int, int, int, bool, bool> class Traits,
          typename Kernel_traits >
struct Tile_o_epilogue {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp16_traits
template<typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_fp16_traits, Kernel_traits> :
    public Tile_o_epilogue_base<Hopper_hgmma_fp16_traits, Kernel_traits> {

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_fp16_traits, Kernel_traits>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp32_traits
template<typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_fp32_traits, Kernel_traits> :
    public Tile_o_epilogue_base<Hopper_hgmma_fp32_traits, Kernel_traits> {

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_fp32_traits, Kernel_traits>;

    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(Compute_tile_o &ctile_o,
                                 float (&global_sum)[Mma_tile_o::CORES_M]) {
        // Final step's update.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
            global_sum[mi] = global_sum[mi] == 0.f ? 1.f : 1.0f / global_sum[mi];

            // Assume only N has multiple MMAs (MMAS_M = 1).
            #pragma unroll
            for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                #pragma unroll
                for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                    float &reg0 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                    float &reg1 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                    reg0 *= global_sum[mi];
                    reg1 *= global_sum[mi];
                }
            }
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_bf16_traits
template<typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_bf16_traits, Kernel_traits> :
    public Tile_o_epilogue_base<Hopper_hgmma_bf16_traits, Kernel_traits> {

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_bf16_traits, Kernel_traits>;

    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(Compute_tile_o &ctile_o,
                                 float (&global_sum)[Mma_tile_o::CORES_M]) {
        // Final step's update.
        #pragma unroll
        for( int mi = 0; mi < Mma_tile_o::CORES_M; mi++ ) {
            global_sum[mi] = global_sum[mi] == 0.f ? 1.f : 1.0f / global_sum[mi];

            // Assume only N has multiple MMAs (MMAS_M = 1).
            #pragma unroll
            for( int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++ ) {
                #pragma unroll
                for( int ni = 0; ni < Mma_tile_o::CORES_N; ni++ ) {
                    float &reg0 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                    float &reg1 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                    reg0 *= global_sum[mi];
                    reg1 *= global_sum[mi];
                }
            }
        }
    }

};

}  // namespace ws
}  // namespace fmha