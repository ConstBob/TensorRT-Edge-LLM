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
#include <fmha/fragment.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/traits.h>

namespace fmha {
  
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// F R A G M E N T  (A)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template< int M, int N, int K, bool A_RF, bool B_RF, typename Layout >
struct Fragment_a<Hopper_hgmma_fp16_traits<M, N, K, A_RF, B_RF>, Layout> 
  : public Fragment<uint16_t, (M*K) / (Hopper::WARPS_PER_WARP_GROUP*Hopper::THREADS_PER_WARP) > {
    // A should be coming from RF.
    static_assert(A_RF, "A_RF must be true to allocate RF for Operand A.\n");
};
////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template< int M, int N, int K, bool A_RF, bool B_RF, typename Layout >
struct Fragment_a<Hopper_hgmma_bf16_traits<M, N, K, A_RF, B_RF>, Layout> 
  : public Fragment<uint16_t, (M*K) / (Hopper::WARPS_PER_WARP_GROUP*Hopper::THREADS_PER_WARP) > {
    // A should be coming from RF.
    static_assert(A_RF, "A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template<int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout> 
 : public Fragment<uint16_t, 
                   (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP) > {
   // A should be coming from RF.
   static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template<int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout> 
  : public Fragment<int8_t, 
                    (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP) > {
    // A should be coming from RF.
    static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int GMMA_M, 
         int GMMA_N, 
         int GMMA_K, 
         typename Input_type_A, 
         typename Input_type_B, 
         typename Output_type,
         typename Layout>
struct Fragment_a<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M,
                                 GMMA_N,
                                 GMMA_K,
                                 true,
                                 false,
                                 Input_type_A,
                                 Input_type_B,
                                 Output_type
                                 >,
    Layout> 
    // TODO: Do we need the * 4 or not?
    : public Fragment<Input_type_A, 
                    (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>{
    static_assert(sizeof(Input_type_A) == 1);
    static_assert(sizeof(Input_type_B) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM

template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, false, false> > 
  : public Fragment<uint16_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<uint16_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg( ii ) = hadd2( this->reg( ii ), other.reg( ii ) );
        }
    }


    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Gmma_single_desc_a &single_desc_a,
                                const Gmma_single_desc_b &single_desc_b ) {
        // call hgmma
        fmha::hgmma_fp16<
            Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N,
            INCREMENT_SCORE_BOARD>( single_desc_a.get(), single_desc_b.get(), this->regs_ );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM

template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, false, false> > 
  : public Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }


    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Gmma_single_desc_a &single_desc_a,
                                const Gmma_single_desc_b &single_desc_b ) {
        // call hgmma
        fmha::hgmma_bf16<
            Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N,
            INCREMENT_SCORE_BOARD>( single_desc_a.get(), single_desc_b.get(), this->regs_ );
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false> > 
  : public Fragment<uint16_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<uint16_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;
    
    // The Traits
    using Traits = Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg( ii ) = hadd2( this->reg( ii ), other.reg( ii ) );
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Fragment_a<Traits, Layout_a> &a,
                                const Gmma_single_desc_b &single_desc_b ) {
        // call hgmma
        fmha::hgmma_rfa_fp16<
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N,
            INCREMENT_SCORE_BOARD>( a.regs_, single_desc_b.get(), this->regs_ );
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, true, false> > 
  : public Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;
    
    // The Traits
    using Traits = Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Fragment_a<Traits, Layout_a> &a,
                                const Gmma_single_desc_b &single_desc_b ) {
        // call hgmma
        fmha::hgmma_rfa_bf16<
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N,
            INCREMENT_SCORE_BOARD>( a.regs_, single_desc_b.get(), this->regs_ );
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 3 2
//
//////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, false, false> >   
: public Fragment<float, 
   (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

   // The base class.
   using Base = Fragment<float,
   (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

   // Add two fragments.
   template< typename Other_fragment_ >
   inline __device__ void add(const Other_fragment_ &other) {
       for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
           this->elt( ii ) = this->elt( ii ) + other.elt( ii );
       }
   }

   
   // Do the GMMA.
   template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
   inline __device__ void mma( const Gmma_single_desc_a &single_desc_a,
                               const Gmma_single_desc_b &single_desc_b ) {
       // call hgmma
       fmha::hgmma_fp32<
           Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
           Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
           GMMA_N,
           INCREMENT_SCORE_BOARD>( single_desc_a.get(), single_desc_b.get(), this->regs_ );
   }
};
//
////////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false> >   
: public Fragment<float, 
   (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

   // The base class.
   using Base = Fragment<float, 
   (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;
   
   // The Traits
   using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;


   // Add two fragments.
   template< typename Other_fragment_ >
   inline __device__ void add(const Other_fragment_ &other) {
       for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
           this->elt( ii ) = this->elt( ii ) + other.elt( ii );
       }
   }

   
   // Do the GMMA.
   template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
   inline __device__ void mma( const Fragment_a<Traits, Layout_a> &a,
                               const Gmma_single_desc_b &single_desc_b ) {
       // call hgmma
       fmha::hgmma_rfa_fp32<
           Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
           GMMA_N,
           INCREMENT_SCORE_BOARD>( a.regs_, single_desc_b.get(), this->regs_ );
   }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Q G M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I G M M A . I N T 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Both operands are coming from SMEM.
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, false, false>>
    : public Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Gmma_single_desc_a &single_desc_a,
                                const Gmma_single_desc_b &single_desc_b ) {
        fmha::igmma_int8_int32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(),
                                                              single_desc_b.get(), 
                                                              this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>>
    : public Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // The Traits.
    using Traits = Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Fragment_a<Traits, Layout_a> &a,
                                const Gmma_single_desc_b &single_desc_b ) {

        fmha::igmma_rfa_int8_int32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_,
                                                                  single_desc_b.get(),
                                                                  this->regs_ );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fp32 Accumulator A operand from RF and B operand from SMEM
template<int GMMA_M, 
         int GMMA_N, 
         int GMMA_K, 
         typename Input_type_A, 
         typename Input_type_B, 
         typename Output_type
         >
struct Fragment_accumulator<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M,
                                 GMMA_N,
                                 GMMA_K,
                                 true,
                                 false,
                                 Input_type_A,
                                 Input_type_B,
                                 Output_type>
                           >
: public Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<float,
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }

    // The Traits
    using Traits = Hopper_qgmma_fp8_fp32_traits<GMMA_M,
                                                GMMA_N,
                                                GMMA_K,
                                                true,
                                                false,
                                                Input_type_A,
                                                Input_type_B,
                                                Output_type
                                                >;
    
    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Fragment_a<Traits, Layout_a> &a,
                                const Gmma_single_desc_b &single_desc_b ) {

        // call hgmma
        if( std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e4m3_t> ) {
            qgmma_rfa_e4m3_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e4m3_t> ) {
            qgmma_rfa_e5m2_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e5m2_t> ) {
            qgmma_rfa_e4m3_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e5m2_t> ) {
            qgmma_rfa_e5m2_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        } else {
            assert(false && "unsupported");
        }
    }
}; 

////////////////////////////////////////////////////////////////////////////////////////////////////

// fp32 accumulator
// Both operands are coming from SMEM.
template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          typename Input_type_A,
          typename Input_type_B,
          typename Output_type
          >
struct Fragment_accumulator<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M,
                                 GMMA_N,
                                 GMMA_K,
                                 false,
                                 false,
                                 Input_type_A,
                                 Input_type_B,
                                 Output_type
                                 >
                           >
    : public Fragment<float,
                      ( GMMA_M * GMMA_N ) /
                          ( Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP )> {

    // The base class.
    using Base =
        Fragment<float,
                 ( GMMA_M * GMMA_N ) / ( Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP )>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma( const Gmma_single_desc_a &single_desc_a,
                                const Gmma_single_desc_b &single_desc_b ) {
        if( std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e4m3_t> ) {
            qgmma_e4m3_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e4m3_t> ) {
            qgmma_e5m2_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e5m2_t> ) {
            qgmma_e4m3_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else if( std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e5m2_t> ) {
            qgmma_e5m2_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else {
            assert(false && "unsupported");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

