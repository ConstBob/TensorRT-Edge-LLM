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


#include <fused_multihead_attention_utils.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/hopper/smem_tile.h>
#include <fmha/utils.h>
#include <fmha/hopper/compute_tile.h>

#include <fmha/warpspec/kernel_traits.h>
#include <fmha/warpspec/dma.h>
#include <fmha/warpspec/compute.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#if CUDA_VERSION >= 11000

static constexpr int DMA2COMPUTE_DEPTH = 1;
static constexpr int NUM_COMPUTE_GROUPS = 2;

using Launch_params = bert::Fused_multihead_attention_launch_params;

using Ktraits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                        64,
                                        64,
                                        256,
                                        1,
                                        2,
                                        NUM_COMPUTE_GROUPS,
                                        DMA2COMPUTE_DEPTH,
                                        0,
                                        true>;

using Ktraits_causal = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                        64,
                                        64,
                                        256,
                                        1,
                                        2,
                                        NUM_COMPUTE_GROUPS,
                                        DMA2COMPUTE_DEPTH,
                                        1,
                                        true>;

using Ktraits_limited_length_causal = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                            64,
                                                            64,
                                                            256,
                                                            1,
                                                            2,
                                                            NUM_COMPUTE_GROUPS,
                                                            DMA2COMPUTE_DEPTH,
                                                            2,
                                                            true>;

////////////////////////////////////////////////////////////////////////////////////////////////////

using Shared = typename Ktraits::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits::THREADS, 1)
void fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared *shared = reinterpret_cast<Shared *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched

        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));

        if( tidx == 0 ) {  // Scheduler.

            fmha::ws::DMA<Ktraits>::Device dma_device;
            dma_device.run(params, shared);
        }

    } else {  // math

        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32  %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));

        fmha::ws::Compute<fmha::Hopper_hgmma_fp16_traits, Ktraits> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

using Shared_causal = typename Ktraits_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_causal::THREADS, 1)
void fmha_v2_flash_attention_fp16_64_64_S_256_causal_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_causal *shared = reinterpret_cast<Shared_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched

        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));

        if( tidx == 0 ) {  // Scheduler.

            fmha::ws::DMA<Ktraits_causal>::Device dma_device;
            dma_device.run(params, shared);
        }

    } else {  // math

        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));

        fmha::ws::Compute<fmha::Hopper_hgmma_fp16_traits, Ktraits_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

using Shared_limited_length_causal = typename Ktraits_limited_length_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_limited_length_causal::THREADS, 1)
void fmha_v2_flash_attention_fp16_64_64_S_256_limited_length_causal_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_limited_length_causal *shared =
        reinterpret_cast<Shared_limited_length_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched

        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));

        if( tidx == 0 ) {  // Scheduler.

            fmha::ws::DMA<Ktraits_limited_length_causal>::Device dma_device;
            dma_device.run(params, shared);
        }

    } else {  // math

        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));

        fmha::ws::Compute<fmha::Hopper_hgmma_fp16_traits, Ktraits_limited_length_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90(
    bert::Fused_multihead_attention_params_v2 &params, 
    const Launch_params &launch_params, cudaStream_t stream){

    // TMA configuration
    // Note that this may only need to init once during inference (for different layers)
    // Reuse the same traits for initializing tma descriptors.
    fmha::ws::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, stream);

    dim3 block_size(1, std::min(params.b * params.h, launch_params.multi_processor_count));

    // distrbute m steps to multiple blocks (fully utilize SMs)
    // block.x = blocks that handle single head, block.y = blocks that handle different heads
    size_t sms_per_head = (launch_params.multi_processor_count) / block_size.y;
    size_t m_steps = size_t((params.s + 64 - 1) / 64);
    m_steps = size_t((m_steps + NUM_COMPUTE_GROUPS - 1) / NUM_COMPUTE_GROUPS) * NUM_COMPUTE_GROUPS;

    size_t size_in_bytes = params.b * params.s * params.qkv_stride_in_bytes;
    if( launch_params.attention_mask_type == Attention_mask_type::PADDING &&
        size_in_bytes <= launch_params.device_l2_cache_size ) {
        // strategy 1: limit to only 1 wave
        block_size.x = std::min(m_steps / NUM_COMPUTE_GROUPS, sms_per_head);
    } else {
        // strategy 2: fully unroll the q loops (contiguous blocks handle all q loops)
        block_size.x = m_steps / NUM_COMPUTE_GROUPS;
    }

    // Reuse the same bytes_per_smem for launching kernels.
    constexpr int SMEM_BYTES = Ktraits::BYTES_PER_SMEM;
    if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_64_64_S_256_causal_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_fp16_64_64_S_256_causal_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(params);
    } else if( launch_params.attention_mask_type == Attention_mask_type::LIMITED_LENGTH_CAUSAL ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_64_64_S_256_limited_length_causal_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_fp16_64_64_S_256_limited_length_causal_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(params);
    } else {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(params);
    }

}

#endif
