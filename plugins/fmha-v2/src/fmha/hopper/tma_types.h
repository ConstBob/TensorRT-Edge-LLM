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
////////////////////////////////////////////////////////////////////////////////////////////////////
// TMA ptx instructions are documented here: 
// https://docs.google.com/document/d/1D2rlIVr3wroYbcUpBfXcbd1MlM14kXpDr2IALenAkI8/edit#
// TMA desc documentation
// https://p4viewer/get/hw/doc/gpu/hopper/hopper/design/Functional_Descriptions/Hopper-189-190_TMA_tensor_descriptor.xlsx
// and https://p4viewer.nvidia.com/get///hw/nvgpu/class/mfs/class/compute/hopper_compute.mfs
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha {

// TMA desc type.
typedef enum { TILED = 0, IM2COL } cudaTmaDescType;

// TMA swizzle type.
typedef enum { SWIZZLE_DISABLED, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B, SWIZZLE_MAX } cudaTmaDescSwizzle;

typedef enum { BARRIER64, BARRIER128 } cudaTmaDescBarrier;

// TMA interleave type.
typedef enum {
  INTERLEAVE_DISABLED,
  INTERLEAVE_16B,
  INTERLEAVE_32B,
  INTERLEAVE_MAX
} cudaTmaDescInterleave;

// TMA L2 sector promotion.
typedef enum {
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
} cudaTmaDescPromotion;

// TMA data type. 
typedef enum {
    U8 = 0,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16_RN,
    F32_RN,
    F32_FTZ_RN,
    F64_RN,
    BF16_RN,
    FORMAT_MAX
} cudaTmaDescFormat;

// TMA cache control. 
typedef enum {
  PREFETCH,      // Prefetch tma descriptor using global memory address
  INVALIDATE,    // Invalidate tma descriptor in l2 cache
  INVALIDATE_ALL // Invalidate tma descriptor and all elements in l2 cache line
} cudaTmaDescCacheCtrl;

// TMA OOB fill modes. 
typedef enum { TENSOR_ZFILL, TENSOR_CFILL } cudaTmaDescOobFillMode;

constexpr uint64_t k_max_tensor_size = (1llu << 36);
constexpr uint64_t k_max_tensor_stride = (1llu << 36);
constexpr uint64_t k_max_block_size = 256llu;
constexpr uint64_t k_max_traversal_stride = (1llu << 3);

constexpr uint64_t k_min_tensor_size = 1llu;
constexpr uint64_t k_min_tensor_stride = 0llu;
constexpr uint64_t k_min_block_size = 1llu;
constexpr uint64_t k_min_traversal_stride = 1llu;

constexpr uint32_t k_max_cta_id = (1 << 6) - 1;

// The 512 bit of descriptor for tiled mode.
typedef struct {
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];  //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];          //< value -1
    uint32_t traversal_stride_box_0;  //< packed 3b (-1)

    uint32_t box_size_end;
} cudaTmaDescTiled;

// The 512 bit of descritptro for im2col mode.
typedef struct {
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
} cudaTmaDescIm2Col;

// TMA desc size
constexpr uint32_t TMA_DESC_SIZE_IN_BYTE = 64;

// TMA desc
typedef struct alignas(64) {
    uint64_t data[8];
} cudaTmaDesc;

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace fmha
