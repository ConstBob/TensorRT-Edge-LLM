
#include "fused_multihead_attention_kernel.h"
#include "fmha/kernel_traits.h"
#include "fmha/hopper/kernel_traits.h"
#include <fmha/warpspec/kernel_traits.h>

using namespace fmha;

int main(){
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_32_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_32_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                32,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_32_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                64,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_64_64_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_64_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 128, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                128,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_128_64_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_128_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                256,
                64,
                64,
                4,
                1,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_256_64_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_256_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_32_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 32, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                32,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_32_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_32_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 192, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                384,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_384_64_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_384_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_limited_length_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_causal_ldgsts_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
    {
            using Traits_p = fmha::Hopper_hgmma_fp16_traits<64, 256, 16, false, false>;
            using Traits_o = fmha::Hopper_hgmma_fp16_traits<64, 64, 16, true, false>;

            using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                Traits_p,
                Traits_o,
                512,
                64,
                64,
                4,
                2,
                2,
                0x07u>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_fp16_512_64_limited_length_causal_ldgsts_sm90_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_fp16_512_64_ldgsts_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          32,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          0>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          32,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          1>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_32_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          32,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          2>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_32_limited_length_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          64,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          0>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          64,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          1>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_64_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          256,
                                                          64,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          2>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_256_S_64_limited_length_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          128,
                                                          128,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          0>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          128,
                                                          128,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          1>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          128,
                                                          128,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          2>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_limited_length_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          64,
                                                          256,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          0>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          64,
                                                          256,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          1>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_256_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90.cu",
                64,
                1);
        }
 {
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_fp16_traits,
                                                          64,
                                                          64,
                                                          256,
                                                          1,
                                                          2,
                                                          NUM_COMPUTE_GROUPS,
                                                          DMA2COMPUTE_DEPTH,
                                                          2>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_256_limited_length_causal_tma_ws_sm90_kernel",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                16,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_16_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_16_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                16,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_16_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                16,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_16_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_16_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                32,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_32_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_32_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                32,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_32_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                32,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_32_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_32_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                40,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_40_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_40_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                40,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_40_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                40,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_40_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_40_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                64,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_64_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                64,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_64_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                64,
                128,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_128_128_S_64_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_128_128_S_64_sm80.cu",
                128,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                80,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_80_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                80,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_80_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                80,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_80_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                128,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                128,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                128,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_128_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                160,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_160_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                160,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                160,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_160_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                256,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_256_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_256_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                256,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_256_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                128,
                256,
                64,
                4,
                1,
                1,
                0x1007u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_128_S_256_limited_length_causal_sm80_kernel_nl_tiled",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_128_S_256_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                16,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_16_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_16_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                16,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_16_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                16,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_16_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_16_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                32,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_32_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_32_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                32,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_32_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                64,
                32,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_64_S_32_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_64_S_32_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                40,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_40_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                40,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_40_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                40,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_40_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_40_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                64,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_64_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                64,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_64_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                64,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_64_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_64_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                80,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                80,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                80,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_80_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_80_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                128,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                128,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                32,
                128,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_32_S_128_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_32_S_128_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                160,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                160,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                160,
                64,
                4,
                1,
                1,
                0x07u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_160_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_160_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                256,
                64,
                4,
                1,
                1,
                0x181u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_256_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                256,
                64,
                4,
                1,
                1,
                0x181u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_256_sm80.cu",
                64,
                1);
        }
    {
            using Kernel_traits = Kernel_traits_v2<
                fmha::Ampere_hmma_fp16_traits,
                16,
                256,
                64,
                4,
                1,
                1,
                0x181u | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\n",
                "fmha_v2_flash_attention_fp16_64_16_S_256_limited_length_causal_sm80_kernel_nl",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                "fmha_v2_flash_attention_fp16_64_16_S_256_sm80.cu",
                64,
                1);
        }
}
