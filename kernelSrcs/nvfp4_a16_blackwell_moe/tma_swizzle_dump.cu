// Standalone evidence tool for the TMA swizzle image of the Thor W4A16 MoE layout (not built by CMake): nvcc -O2 -std=c++17 -arch=sm_110a tma_swizzle_dump.cu
// Thor 2026-09-05: SWIZZLE_32B == CuTe Swizzle<1,4,3> (rows 4-7 of every 8 swap 16-byte halves), 64B == <2,4,3>, 128B
// == <3,4,3>. Dump the byte permutation TMA applies for a (32 B x 128 rows) box with SWIZZLE_32B (and 64B/128B) on this
// GPU: gmem tile holds byte value = (index & 0xFF) pattern via 16-bit chunk ids; we load one tile into SMEM and write
// it back.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                  \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)
__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__global__ void dumpKernel(__grid_constant__ const CUtensorMap map, uint8_t* out)
{
    __shared__ __align__(1024) uint8_t tile[4096];
    __shared__ __align__(8) uint64_t bar;
    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemAddr(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;" ::"r"(smemAddr(&bar)) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::
                "r"(smemAddr(tile)),
            "l"(reinterpret_cast<uint64_t>(&map)), "r"(smemAddr(&bar)), "r"(0), "r"(0)
            : "memory");
        uint32_t done = 0;
        while (!done)
        {
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                         : "=r"(done)
                         : "r"(smemAddr(&bar))
                         : "memory");
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        out[i] = tile[i];
}
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, cuuint64_t const*,
    cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
int main()
{
    uint8_t* g = nullptr;
    uint8_t* o = nullptr;
    CK(cudaMalloc(&g, 4096));
    CK(cudaMalloc(&o, 4096));
    uint8_t h[4096];
    for (int i = 0; i < 4096; ++i)
        h[i] = static_cast<uint8_t>(i / 16); // 16-byte chunk id (0..255)
    CK(cudaMemcpy(g, h, 4096, cudaMemcpyHostToDevice));
    EncodeFn encode = nullptr;
    cudaDriverEntryPointQueryResult q{};
    CK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), cudaEnableDefault, &q));
    struct Cfg
    {
        char const* name;
        uint32_t rowBytes;
        CUtensorMapSwizzle sw;
    } cfgs[] = {
        {"32B rows SW32", 32, CU_TENSOR_MAP_SWIZZLE_32B},
        {"64B rows SW64", 64, CU_TENSOR_MAP_SWIZZLE_64B},
        {"128B rows SW128", 128, CU_TENSOR_MAP_SWIZZLE_128B},
    };
    for (auto const& c : cfgs)
    {
        CUtensorMap map{};
        cuuint64_t dims[2]{c.rowBytes, 4096 / c.rowBytes};
        cuuint64_t strides[1]{c.rowBytes};
        cuuint32_t box[2]{c.rowBytes, 4096 / c.rowBytes};
        cuuint32_t estr[2]{1, 1};
        CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, g, dims, strides, box, estr,
            CU_TENSOR_MAP_INTERLEAVE_NONE, c.sw, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS)
        {
            std::printf("%s: encode failed %d\n", c.name, (int) r);
            continue;
        }
        dumpKernel<<<1, 128>>>(map, o);
        CK(cudaDeviceSynchronize());
        uint8_t d[4096];
        CK(cudaMemcpy(d, o, 4096, cudaMemcpyDeviceToHost));
        // For each smem 16-byte chunk s, which gmem chunk landed there? Print the first 64 chunks and infer the XOR
        // rule.
        std::printf("%s: smem_chunk -> gmem_chunk (first 64):", c.name);
        for (int s = 0; s < 64; ++s)
            std::printf("%s%d", s % 16 == 0 ? "\n  " : " ", d[s * 16]);
        std::printf("\n");
        // infer xor: chunk index bits; find mask m s.t. gmem = s ^ f(s)
        int consistent = 1;
        int xorBits[8] = {0};
        for (int s = 0; s < 256; ++s)
        {
            int x = s ^ d[s * 16];
            for (int b = 0; b < 8; ++b)
                if ((x >> b) & 1)
                    xorBits[b]++;
        }
        std::printf("  chunk-index bits flipped (count over 256 chunks): ");
        for (int b = 0; b < 8; ++b)
            std::printf("b%d=%d ", b, xorBits[b]);
        // check the CuTe Swizzle<B,4,3>-style rule: byte address bits [4..4+B) ^= bits [7..7+B)
        for (int B = 1; B <= 3; ++B)
        {
            int ok = 1;
            for (int s = 0; s < 256; ++s)
            {
                int addr = s * 16;
                int sw = addr ^ (((addr >> 7) & ((1 << B) - 1)) << 4);
                if (d[s * 16] != (sw / 16))
                {
                    ok = 0;
                    break;
                }
            }
            std::printf(" | Swizzle<%d,4,3> match=%d", B, ok);
        }
        std::printf("\n");
        (void) consistent;
    }
    return 0;
}
