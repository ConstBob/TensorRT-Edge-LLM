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


//We can disable the FADD trick for archs with F2IP
#if 0 // disable_fadd_trick
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif // USE_I2F_EMULATION_TRICK

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif // USE_F2I_EMULATION_TRICK
#endif // disable_fadd_trick

#include <cuda.h>

#if CUDA_VERSION >= 11000

#include <fused_multihead_flash_attention_kernel_noloop.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>
#include <fused_multihead_flash_attention_kernel.h>

using Launch_params = bert::Fused_multihead_attention_launch_params;

#if 0 // has_noloop (unconditionally disabled since not maintained & not actively used)
using Kernel_traits = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u>;

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention<Kernel_traits>(params);
}

void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {
    FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }
  dim3 grid(params.h, params.b);
  fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
}

#endif // has_noloop

#if 1 && !1 // has_noloop && !tiled
using Kernel_traits_nl = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

using Kernel_traits_nl_causal = fmha::Kernel_traits_v2_causal_mask<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

using Kernel_traits_nl_limited_length_causal = fmha::Kernel_traits_v2_limited_length_causal_mask<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl>(params);
}

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_causal>(params);
}

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_limited_length_causal>(params);
}

void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  // runtime q_loop_iters
  int loop_iters = ( params.s + 128 - 1 )  / 128;
  // dim3 grid(params.h, params.b, loop_iters);
  dim3 grid(loop_iters, params.h, params.b); // better locality
  constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
  } else if( launch_params.attention_mask_type == Attention_mask_type::LIMITED_LENGTH_CAUSAL ) {
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
  } else {
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
  }
}

#endif // has_noloop && !tiled

#if 1 // tiled

using Kernel_traits_nl_tiled = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

using Kernel_traits_nl_tiled_causal = fmha::Kernel_traits_v2_causal_mask<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

using Kernel_traits_nl_tiled_limited_length_causal = fmha::Kernel_traits_v2_limited_length_causal_mask<
    fmha::Ampere_hmma_fp16_traits,
    128,
    64,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */>;

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled>(params);
}

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_causal>(params);
}

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_limited_length_causal>(params);
}

// Granular tiling
void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl_tiled(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){
  // runtime q_loop_iters
  using Cta_tile_o = typename Kernel_traits_nl_tiled::Cta_tile_o;
  int ctas_per_o_row = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
  int loop_iters = ( params.s + 128 - 1 )  / 128;
  dim3 grid(loop_iters * ctas_per_o_row, params.h, params.b);
  constexpr int smem_size = Kernel_traits_nl_tiled::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
  } else if( launch_params.attention_mask_type == Attention_mask_type::LIMITED_LENGTH_CAUSAL ) {
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl_tiled,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
  } else {
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
  }
}

#endif // tiled

#else // CUDA_VERSION >= 11000

void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_nl_tiled(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

#endif // CUDA_VERSION >= 11000
