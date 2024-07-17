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

////////////////////////////////////////////////////////////////////////////////////////////////////

// Ada hmma/imma reuses Ampere
template< typename Traits_ >
struct Traits_reuse {
    using Traits = Traits_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Traits_reuse<fmha::Ada_hmma_fp16_traits> {
    using Traits = fmha::Ampere_hmma_fp16_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Traits_reuse<fmha::Ada_hmma_fp32_traits> {
    using Traits = fmha::Ampere_hmma_fp32_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Traits_reuse<fmha::Ada_imma_int8_int32_traits> {
    using Traits = fmha::Ampere_imma_int8_int32_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_p,bool FORCE_EPILOGUE_FP16 >
struct Traits_o_adapter {
    using Traits = Traits_p;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool FORCE_EPILOGUE_FP16>
struct Traits_o_adapter<fmha::Volta_hmma_fp16_traits, FORCE_EPILOGUE_FP16> {
    using Traits = fmha::Volta_hmma_fp16_16x16x16_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// convert to fp16 before smem_o store
template<>
struct Traits_o_adapter<fmha::Ampere_hmma_fp32_traits, true> {
    using Traits = fmha::Ampere_hmma_fp16_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// convert to fp16 before smem_o store
template<>
struct Traits_o_adapter<fmha::Turing_hmma_fp32_traits, true> {
    using Traits = fmha::Turing_hmma_fp16_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// convert to bf16 before smem_o store
template<>
struct Traits_o_adapter<fmha::Ampere_hmma_bf16_traits, true> {
    using Traits = fmha::Ampere_hmma_bf16_bf16_traits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Instruction traits.
    typename Traits_,
    // The global memory tile for Q, K and V.
    template <typename, typename, int, int, int, int, bool, bool, int> class Gmem_tile_qkv_,
    // The global memory tile for the output.
    template <typename, typename, int> class Gmem_tile_o_,
    // Sequence length.
    int S,
    // The valid hidden dimension.
    int VALID_D_,
    // The iteration step of the outer loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags to control the behaviour of LDGs.
    uint32_t FLAGS,
    // The version of the kernel.
    int VERSION_,
    // The mask version of the kernel
    int MASK_VERSION_,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true
>
struct Kernel_traits_ {

    // The instruction traits for the Q*K product.
    using Traits_p = typename Traits_reuse<Traits_>::Traits;
    // The instruction traits for the P*V product. Hack to change the traits for Volta HMMA.
    using Traits_o = typename Traits_o_adapter<Traits_p, false>::Traits;
    // The instruction traits for the epilogue of the 2nd GEMM. Always use FP16.
    using Traits_e = typename Traits_o_adapter<Traits_p, BMM2_FP16_EPILOGUE>::Traits;

    // The padded D dimension
    enum { VALID_D = VALID_D_ };
    enum { D = Next_power_of_two<VALID_D>::VALUE };

    // TODO: expose these tiling params to the interface
    enum { USE_GRANULAR_TILING = (FLAGS & 0x1000) != 0u }; // TODO ANT: check FLAGS

    using Traits_tile_size = Traits_tile_size<(bool)USE_GRANULAR_TILING, STEP, S, D>;

    enum { CTA_P_TILE_M = Traits_tile_size::CTA_P_TILE_M };
    enum { CTA_P_TILE_N = Traits_tile_size::CTA_P_TILE_N };
    enum { CTA_P_TILE_K = Traits_tile_size::CTA_P_TILE_K };
    enum { CTA_O_TILE_M = Traits_tile_size::CTA_O_TILE_M };
    enum { CTA_O_TILE_N = Traits_tile_size::CTA_O_TILE_N };
    enum { CTA_O_TILE_K = Traits_tile_size::CTA_O_TILE_K };

    // Do we need to reload Q due to spliting the D.
    // The smem buffer is enough when D <= 128
    enum { RELOAD_Q = D > 128 };

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = typename Traits_p::template Cta_tile_extd<CTA_P_TILE_M,
                                                                 CTA_P_TILE_N,
                                                                 CTA_P_TILE_K,
                                                                 S,
                                                                 VALID_D,
                                                                 WARPS_M,
                                                                 WARPS_N,
                                                                 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = typename Traits_o::template Cta_tile_extd<CTA_O_TILE_M,
                                                                 CTA_O_TILE_N,
                                                                 CTA_O_TILE_K,
                                                                 VALID_D,
                                                                 S,
                                                                 WARPS_M,
                                                                 1,
                                                                 WARPS_N>;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // Constraints on the K dimension.
    static_assert(Mma_tile_p::K_PER_MMA <= static_cast<int>(D));
    static_assert(Mma_tile_o::K_PER_MMA <= S);

    // The version.
    enum { VERSION = VERSION_ };
    // The mask version: padding (2), causal (3), limited_past_causal (4).
    enum { MASK_VERSION = MASK_VERSION_ };
    // Whether use causal mask or not.
    enum { CAUSAL_MASK = MASK_VERSION_ >= 3 };
    // Whether limite the past sequence length for attention.
    enum { LIMITED_PAST_SEQUENCE = MASK_VERSION_ == 4 };

    // Do we use LDGSTS for Q, K or V.
    enum { USE_LDGSTS_Q = (FLAGS & 0x1u) != 0u };
    enum { USE_LDGSTS_K = (FLAGS & 0x2u) != 0u };
    enum { USE_LDGSTS_V = (FLAGS & 0x4u) != 0u };

    // Do we use one buffer for K and V.
    enum { SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x8u) != 0u };

    // Do we use the scale max trick.
    enum { USE_SCALE_MAX = (FLAGS & 0x10u) != 0u };

    // Are heads in QKV interleaved, i.e. total x h x 3 x d or total x 3 x h x d.
    enum { HEADS_INTERLEAVED = (FLAGS & 0x20u) == 0u };

    // Keep full K matrix in registers.
    enum { K_IN_REGS = (FLAGS & 0x40) == 0u };

    // Do we use only 2 fragments or full fragments for frag_q/k (only used by flash attention)
    enum { LIMIT_QK_FRAGMENTS = ((FLAGS & 0x80u) != 0u && !SHARE_SMEM_FOR_K_AND_V)};
    // Do we use only 2 fragments or full fragments for frag_v (only used by flash attention)
    enum { LIMIT_V_FRAGMENTS = ((FLAGS & 0x100u) != 0u && !SHARE_SMEM_FOR_K_AND_V)};
    // Limiting QK fragments implies SMEM_K has to reside in SMEM
    static_assert(!(LIMIT_QK_FRAGMENTS && SHARE_SMEM_FOR_K_AND_V), "");

    // Indicates that kernel does not loop over Q tensor, usually kernel name has _nl suffix
    enum { NO_LOOP = (FLAGS & 0x200u) != 0u };

    // Are sequences in one batch interleaved. i.e. s x b x ..., or b x s x ...
    enum { SEQUENCES_INTERLEAVED = (FLAGS & 0x400) != 0u };

    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    enum { CTAS_PER_HEAD = CTAS_PER_HEAD_ };

    // The number of shared memory buffers to build a software pipeline for Q, K and V.
    enum { BUFFERS_PER_TILE_SMEM_Q = (USE_GRANULAR_TILING && D > 64) || (USE_LDGSTS_Q && !NO_LOOP) ? 2 : 1 };
    enum { BUFFERS_PER_TILE_SMEM_K = USE_GRANULAR_TILING ? 2 : 1 };
    enum { BUFFERS_PER_TILE_SMEM_V = USE_GRANULAR_TILING ? 2 : 1 };

    // The global memory tile to load Q.
    using Gmem_tile_q = Gmem_tile_qkv_<Traits_p, 
                                       Cta_tile_p, 
                                       Traits_p::BITS_PER_ELEMENT_A, 
                                       CTA_P_TILE_M,
                                       CTA_P_TILE_K,
                                       VALID_D,
                                       USE_LDGSTS_Q,
                                       HEADS_INTERLEAVED,
                                       3 //NUM_MATS
                                       >;

    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_a<Traits_p,
                                          Cta_tile_p,
                                          fmha::Row,
                                          Gmem_tile_q::BYTES_PER_LDG,
                                          BUFFERS_PER_TILE_SMEM_Q>;

    // The global memory tile to load K.
    using Gmem_tile_k = Gmem_tile_qkv_<Traits_p, 
                                       Cta_tile_p, 
                                       Traits_p::BITS_PER_ELEMENT_B, 
                                       CTA_P_TILE_N,
                                       CTA_P_TILE_K,
                                       VALID_D,
                                       USE_LDGSTS_K,
                                       HEADS_INTERLEAVED,
                                       3 // NUM_MATS
                                       >;

    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_b<Traits_p,
                                          Cta_tile_p,
                                          fmha::Col,
                                          Gmem_tile_k::BYTES_PER_LDG,
                                          BUFFERS_PER_TILE_SMEM_K>;

    // The global memory tile to load V.
    using Gmem_tile_v = Gmem_tile_qkv_<Traits_o, 
                                       Cta_tile_o, 
                                       Traits_o::BITS_PER_ELEMENT_B, 
                                       CTA_O_TILE_K,
                                       CTA_O_TILE_N,
                                       VALID_D,
                                       USE_LDGSTS_V,
                                       HEADS_INTERLEAVED,
                                       3 // NUM_MATS
                                       >;

    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Traits_o, Cta_tile_o, BUFFERS_PER_TILE_SMEM_V>;

    // The global memory tile to store O.
    using Gmem_tile_o = Gmem_tile_o_<Traits_e, Cta_tile_o, CTAS_PER_HEAD>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o<Traits_e, Cta_tile_o>;

    // Make sure the number of threads match.
    static_assert((int) Gmem_tile_o::THREADS_PER_ROW == (int) Smem_tile_o::THREADS_PER_ROW, "");

    // The number of threads.
    enum { THREADS = Cta_tile_p::THREADS_PER_CTA };

    // Make sure the number of threads matches both CTAs.
    static_assert((int) THREADS == (int) Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    enum { BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE };
    // The extra amount of shared memory needed to load V.
    enum { BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE };
    // The amount of shared memory needed for Q, K and V..
    enum { BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V };
    // The amount of shared memory needed to load/store O.
    enum { BYTES_PER_SMEM_O = Smem_tile_o::BYTES_PER_TILE };
    // The amount of shared memory needed to load Q and store O.
    enum { BYTES_PER_SMEM_QO = NO_LOOP ? Smem_tile_o::BYTES_PER_TILE
                                       : Smem_tile_q::BYTES_PER_TILE + BYTES_PER_SMEM_O };

    // The amount of shared memory needed for Q, K, V and O.
    enum { BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE };

    // Make sure we have enough shared memory.
    static_assert( (NO_LOOP ? Smem_tile_o::BYTES_PER_TILE
                            : Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE)
                            <= BYTES_PER_SMEM, "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Instruction traits.
    typename Traits_,
    // The global memory tile for Q, K and V.
    template <typename, typename, int, int, int, bool, bool, int> class Gmem_tile_q_,
    // The global memory tile for the output.
    template <typename, typename, int> class Gmem_tile_o_,
    // Sequence length for K/V.
    int S_KV,
    // The hidden dimension.
    int D,
    // The iteration step of the outer loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags to control the behaviour of LDGs.
    uint32_t FLAGS,
    // The version of the kernel.
    int VERSION_,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true
>
struct Kernel_traits_fmhca_ {

    // The instruction traits for the Q*K product.
    using Traits_p = typename Traits_reuse<Traits_>::Traits;
    // The instruction traits for the P*V product. Hack to change the traits for Volta HMMA.
    using Traits_o = typename Traits_o_adapter<Traits_p, false>::Traits;
    // The instruction traits for the epilogue of the 2nd GEMM. Always use FP16.
    using Traits_e = typename Traits_o_adapter<Traits_p, BMM2_FP16_EPILOGUE>::Traits;

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = typename Traits_p::template Cta_tile_extd<STEP, 
                                                                 S_KV, 
                                                                 D, 
                                                                 S_KV,
                                                                 D,
                                                                 WARPS_M, 
                                                                 WARPS_N, 
                                                                 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = typename Traits_o::template Cta_tile_extd<STEP, 
                                                                 D, 
                                                                 S_KV, 
                                                                 D,
                                                                 S_KV,
                                                                 WARPS_M, 
                                                                 1, 
                                                                 WARPS_N>;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // Constraints on the K dimension.
    static_assert(Mma_tile_p::K_PER_MMA <= D, "");
    static_assert(Mma_tile_o::K_PER_MMA <= S_KV, "");

    // The version.
    enum { VERSION = VERSION_ };
    // The mask version
    enum { MASK_VERSION = VERSION_ };
    // Whether use causal mask or not.
    enum { CAUSAL_MASK = MASK_VERSION >= 3 };
    // Whether limite the past sequence length for attention.
    enum { LIMITED_PAST_SEQUENCE = MASK_VERSION == 4 };

    // Do we use LDGSTS for Q, K or V.
    enum { USE_LDGSTS_Q = (FLAGS & 0x1u) != 0u };
    enum { USE_LDGSTS_K = (FLAGS & 0x2u) != 0u };
    enum { USE_LDGSTS_V = (FLAGS & 0x4u) != 0u };

    // Do we use one buffer for K and V.
    enum { SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x8u) != 0u };

    // Do we use the scale max trick.
    enum { USE_SCALE_MAX = (FLAGS & 0x10u) != 0u };

    // Are heads in QKV interleaved, i.e. total x h x 3 x d or total x 3 x h x d.
    enum { HEADS_INTERLEAVED = (FLAGS & 0x20u) == 0u };

    // Keep full K matrix in registers.
    enum { K_IN_REGS = (FLAGS & 0x40) == 0u };

    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    enum { CTAS_PER_HEAD = CTAS_PER_HEAD_ };

    // The global memory tile to load Q.
    using Gmem_tile_q = Gmem_tile_q_<Traits_p, 
                                     Cta_tile_p, 
                                     Traits_p::BITS_PER_ELEMENT_A, 
                                     STEP, 
                                     D, 
                                     USE_LDGSTS_Q,
                                     HEADS_INTERLEAVED,
                                     1 //NUM_MATS
                                     >;

    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_a<Traits_p,
                                          Cta_tile_p,
                                          fmha::Row,
                                          Gmem_tile_q::BYTES_PER_LDG,
                                          USE_LDGSTS_Q ? 2 : 1>;

    // The global memory tile to load K.
    using Gmem_tile_k = Gmem_tile_q_<Traits_p, 
                                      Cta_tile_p, 
                                      Traits_p::BITS_PER_ELEMENT_B, 
                                      S_KV, 
                                      D, 
                                      USE_LDGSTS_K,
                                      HEADS_INTERLEAVED,
                                      2 // NUM_MATS
                                      >;

    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_b<Traits_p, Cta_tile_p, fmha::Col>;

    // The global memory tile to load V.
    using Gmem_tile_v = Gmem_tile_q_<Traits_o, 
                                      Cta_tile_o, 
                                      Traits_o::BITS_PER_ELEMENT_B, 
                                      S_KV, 
                                      D, 
                                      USE_LDGSTS_V,
                                      HEADS_INTERLEAVED,
                                      2 // NUM_MATS
                                      >;

    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Traits_o, Cta_tile_o>;

    // The global memory tile to store O.
    using Gmem_tile_o = Gmem_tile_o_<Traits_e, Cta_tile_o, CTAS_PER_HEAD>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o<Traits_e, Cta_tile_o>;

    // Make sure the number of threads match.
    static_assert((int) Gmem_tile_o::THREADS_PER_ROW == (int) Smem_tile_o::THREADS_PER_ROW, "");

    // The number of threads.
    enum { THREADS = Cta_tile_p::THREADS_PER_CTA };

    // Make sure the number of threads matches both CTAs.
    static_assert((int) THREADS == (int) Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    enum { BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE };
    // The extra amount of shared memory needed to load V.
    enum { BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE };
    // The amount of shared memory needed for Q, K and V..
    enum { BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V };
    // The amount of shared memory needed to load Q and store O.
    enum { BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE };

    // The amount of shared memory needed for Q, K, V and O.
    enum { BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE };

    // Make sure we have enough shared memory.
    static_assert(Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE <= BYTES_PER_SMEM, "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits_, 
    // The sequence length.
    int S, 
    // The hidden size per head.
    int VALID_D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags.
    uint32_t FLAGS = 0x8,
    // The mask version of the kernel
    int MASK_VERSION_ = 2>
struct Kernel_traits_interleaved_v2_ {

    // The instruction traits.
    using Traits   = typename Traits_reuse<Traits_>::Traits;
    using Traits_p = Traits;
    using Traits_o = Traits;

    // The padded D dimension
    enum { D = Next_power_of_two<VALID_D>::VALUE };

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = typename Traits::template Cta_tile_extd<STEP,
                                                               S,
                                                               D,
                                                               S,
                                                               VALID_D,
                                                               WARPS_M,
                                                               WARPS_N,
                                                               1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = typename Traits::template Cta_tile_extd<STEP,
                                                               D,
                                                               S,
                                                               VALID_D,
                                                               S,
                                                               WARPS_M,
                                                               1,
                                                               WARPS_N>;

    // The version.
    enum { VERSION = 2 };
    enum { MASK_VERSION = MASK_VERSION_ };
    // Whether use causal mask or not.
    enum { CAUSAL_MASK = MASK_VERSION_ >= 3 };
    // Whether limite the past sequence length for attention.
    enum { LIMITED_PAST_SEQUENCE = MASK_VERSION_ == 4 };

    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    enum { CTAS_PER_HEAD = CTAS_PER_HEAD_ };

    // Do we use LDGSTS for Q, K or V.
    enum { USE_LDGSTS_Q = ( FLAGS & 0x1u ) != 0u };
    enum { USE_LDGSTS_K = ( FLAGS & 0x2u ) != 0u };
    enum { USE_LDGSTS_V = ( FLAGS & 0x4u ) != 0u };

    // Do we use one buffer for K and V.
    enum { SHARE_SMEM_FOR_K_AND_V = ( FLAGS & 0x8u ) != 0u };

    // Do we use the scale max trick.
    enum { USE_SCALE_MAX = ( FLAGS & 16 ) != 0u };

    // The global memory tile to load Q.
    using Gmem_tile_q = fmha::v2::Gmem_tile_qkv_interleaved<Traits,
                                                            Cta_tile_p,
                                                            Traits::BITS_PER_ELEMENT_A,
                                                            STEP,
                                                            D,
                                                            USE_LDGSTS_Q>;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_qk_interleaved_a<Traits, Cta_tile_p>;

    // The global memory tile to load K.
    using Gmem_tile_k = fmha::v2::Gmem_tile_qkv_interleaved<Traits,
                                                            Cta_tile_p,
                                                            Traits::BITS_PER_ELEMENT_B,
                                                            S,
                                                            D,
                                                            USE_LDGSTS_K>;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_qk_interleaved_b<Traits, Cta_tile_p>;

    // The global memory tile to load V.
    using Gmem_tile_v = fmha::v2::Gmem_tile_qkv_interleaved<Traits,
                                                            Cta_tile_o,
                                                            Traits::BITS_PER_ELEMENT_B,
                                                            S,
                                                            D,
                                                            USE_LDGSTS_V>;

    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v_interleaved_b<Traits, Cta_tile_o>;

    // The global memory tile to store O.
    using Gmem_tile_o = fmha::v2::Imma_gmem_tile_o_interleaved<Traits, Cta_tile_o, CTAS_PER_HEAD>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o_interleaved<Traits, Cta_tile_o>;

    // Make sure the number of threads match.
    static_assert((int) Gmem_tile_o::THREADS_PER_ROW == (int) Smem_tile_o::THREADS_PER_ROW, "");

    // The number of threads.
    enum { THREADS = Cta_tile_p::THREADS_PER_CTA };

    // Make sure the number of threads matches both CTAs.
    static_assert((int) THREADS == (int)Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    enum { BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE };
    // The extra amount of shared memory needed to load V.
    enum { BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE };
    // The amount of shared memory needed for Q, K and V..
    enum { BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V };
    // The amount of shared memory needed to load Q and store O.
    enum { BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE };

    // The amount of shared memory needed for Q, K, V and O.
    enum { BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE };

    // Make sure we have enough shared memory.
    static_assert(Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE <= BYTES_PER_SMEM, "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The sequence length.
    int S,
    // The hidden size per head.
    int VALID_D,
    // The number of timesteps per iteration of the main loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags.
    uint32_t FLAGS = 0x8,
    // The mask version of the kernel
    int MASK_VERSION_ = 2>
using Kernel_traits_interleaved_v2 = Kernel_traits_interleaved_v2_<Traits_,
                                                                   S,
                                                                   VALID_D,
                                                                   STEP,
                                                                   WARPS_M,
                                                                   WARPS_N,
                                                                   CTAS_PER_HEAD_,
                                                                   FLAGS,
                                                                   2>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The sequence length.
    int S,
    // The hidden size per head.
    int VALID_D,
    // The number of timesteps per iteration of the main loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags.
    uint32_t FLAGS = 0x8,
    // The mask version of the kernel
    int MASK_VERSION_ = 2>
using Kernel_traits_interleaved_v2_causal_mask = Kernel_traits_interleaved_v2_<Traits_,
                                                                               S,
                                                                               VALID_D,
                                                                               STEP,
                                                                               WARPS_M,
                                                                               WARPS_N,
                                                                               CTAS_PER_HEAD_,
                                                                               FLAGS,
                                                                               3>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The sequence length.
    int S, 
    // The hidden size per head.
    int D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8 
>
using Kernel_traits_v1 = Kernel_traits_<Traits,
                                        fmha::v1::Gmem_tile_qkv,
                                        fmha::v1::Gmem_tile_o,
                                        S,
                                        D,
                                        STEP,
                                        WARPS_M,
                                        WARPS_N,
                                        CTAS_PER_HEAD,
                                        FLAGS,
                                        1,
                                        1>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The sequence length.
    int S,
    // The hidden size per head.
    int D,
    // The number of timesteps per iteration of the main loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8>
using Kernel_traits_v1_causal_mask = Kernel_traits_<Traits,
                                                    fmha::v1::Gmem_tile_qkv,
                                                    fmha::v1::Gmem_tile_o,
                                                    S,
                                                    D,
                                                    STEP,
                                                    WARPS_M,
                                                    WARPS_N,
                                                    CTAS_PER_HEAD,
                                                    FLAGS,
                                                    1,   // VERSION_
                                                    3>;  // MASK_VERSION_

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The sequence length.
    int S, 
    // The hidden size per head.
    int D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true
>
using Kernel_traits_v2 = Kernel_traits_<Traits,
                                        fmha::v2::Gmem_tile_qkv,
                                        fmha::v2::Gmem_tile_o,
                                        S,
                                        D,
                                        STEP,
                                        WARPS_M,
                                        WARPS_N,
                                        CTAS_PER_HEAD,
                                        FLAGS,
                                        2,
                                        2,
                                        BMM2_FP16_EPILOGUE>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The sequence length.
    int S, 
    // The hidden size per head.
    int D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true
>
using Kernel_traits_v2_causal_mask = Kernel_traits_<Traits,
                                                    fmha::v2::Gmem_tile_qkv,
                                                    fmha::v2::Gmem_tile_o,
                                                    S,
                                                    D,
                                                    STEP,
                                                    WARPS_M,
                                                    WARPS_N,
                                                    CTAS_PER_HEAD,
                                                    FLAGS,
                                                    2,
                                                    3,
                                                    BMM2_FP16_EPILOGUE>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The sequence length.
    int S, 
    // The hidden size per head.
    int D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true
>
using Kernel_traits_v2_limited_length_causal_mask = Kernel_traits_<Traits,
                                                                   fmha::v2::Gmem_tile_qkv,
                                                                   fmha::v2::Gmem_tile_o,
                                                                   S,
                                                                   D,
                                                                   STEP,
                                                                   WARPS_M,
                                                                   WARPS_N,
                                                                   CTAS_PER_HEAD,
                                                                   FLAGS,
                                                                   2,
                                                                   4,
                                                                   BMM2_FP16_EPILOGUE>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The sequence length for K and V.
    int S_KV, 
    // The hidden size per head.
    int D, 
    // The number of timesteps per iteration of the main loop.
    int STEP, 
    // The number of vertical warps.
    int WARPS_M, 
    // The number of horizontal warps.
    int WARPS_N, 
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8 
>
using Kernel_traits_fmhca = Kernel_traits_fmhca_<Traits,
                                                 fmha::v2::Gmem_tile_q_kv,
                                                 fmha::v2::Gmem_tile_o,
                                                 S_KV,
                                                 D,
                                                 STEP,
                                                 WARPS_M,
                                                 WARPS_N,
                                                 CTAS_PER_HEAD,
                                                 FLAGS,
                                                 2>;

////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace fmha 

