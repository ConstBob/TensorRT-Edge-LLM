/***************************************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <fmha/hopper/tma_types.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int DIM, cudaTmaDescType DESC_TYPE, bool USE_TMA_MULTICAST >
inline __device__ void utmaldg(const cudaTmaDesc *p_desc,            // TMA desc
                               uint32_t smem_ptr,                    // desc smem address
                               uint32_t smem_barrier,                // smem_barrier
                               const int32_t (&coord)[DIM]) {        // coord
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// UTMALDG TILED WITHOUT MULTICAST
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ void utmaldg<2, fmha::cudaTmaDescType::TILED, false>(const cudaTmaDesc *p_desc, 
                                                                       uint32_t smem_ptr,
                                                                       uint32_t smem_barrier, 
                                                                       const int32_t (&coord)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile( 
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes " \
            "[%0], [%1, {%2, %3}], [%4];\n"
                :
                : "r"(smem_ptr)
                , "l"(reinterpret_cast<uint64_t>(p_desc))
                , "r"(coord[0])
                , "r"(coord[1])
                , "r"(smem_barrier)
                : "memory");
#endif                                                                   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ void utmaldg<3, fmha::cudaTmaDescType::TILED, false>(const cudaTmaDesc *p_desc, 
                                                                       uint32_t smem_ptr,
                                                                       uint32_t smem_barrier, 
                                                                       const int32_t (&coord)[3]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile( 
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes " \
            "[%0], [%1, {%2, %3, %4}], [%5];\n"
                :
                : "r"(smem_ptr)
                , "l"(reinterpret_cast<uint64_t>(p_desc))
                , "r"(coord[0])
                , "r"(coord[1])
                , "r"(coord[2])
                , "r"(smem_barrier)
                : "memory");
#endif                                                                          
}

// 4D, TILED, without Multicast
template<>
inline __device__ void utmaldg<4, fmha::cudaTmaDescType::TILED, false>(const cudaTmaDesc *p_desc, 
                                                                       uint32_t smem_ptr,
                                                                       uint32_t smem_barrier, 
                                                                       const int32_t (&coord)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile( 
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes " \
            "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
                :
                : "r"(smem_ptr)
                , "l"(reinterpret_cast<uint64_t>(p_desc))
                , "r"(coord[0])
                , "r"(coord[1])
                , "r"(coord[2])
                , "r"(coord[3])
                , "r"(smem_barrier)
                : "memory");
#endif                                                                          
}
                                                                       

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace fmha

