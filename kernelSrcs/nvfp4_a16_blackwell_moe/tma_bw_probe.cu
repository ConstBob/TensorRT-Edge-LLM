// TMA streaming-bandwidth probe for Thor (SM110): does the TMA box row size
// (32 B rows as in the NVFP4 weight tile vs 128 B / 2 KB rows vs a 1-D bulk
// copy) change the achievable DRAM->SMEM bandwidth at equal bytes per
// instruction (4 KB) and equal in-flight depth?
//
// One CTA per SM streams a large buffer tile by tile through a STAGES-deep SMEM
// ring. A single thread issues cp.async.bulk(.tensor) loads and waits on the
// completing mbarrier of the oldest stage before re-issuing into it, so exactly
// STAGES loads (STAGES * 4 KB) are in flight per SM. Nothing consumes the data:
// the measurement is the memory system alone.
//
// Standalone evidence tool for the Thor W4A16 MoE TMA streaming rate (not built by CMake).
// Build (board): nvcc -O3 -std=c++17 -arch=sm_110a tma_bw_probe.cu -o tma_bw_probe
// Run:           ./tma_bw_probe [bytes_mb=512] [reps=5]
// Thor 2026-09-05: 32 B rows 228-236 GB/s at every depth; 64 B+ rows, 2 KB rows and 1-D bulk 256-270 GB/s.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

constexpr uint32_t kTileBytes = 4096;

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count));
}

__device__ __forceinline__ void fenceBarrierInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarArriveExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n"
            " .reg .pred p;\n"
            " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(smemAddr(bar)), "r"(phase)
            : "memory");
    }
}

__device__ __forceinline__ void tmaLoad2d(void* smem, CUtensorMap const* map, int32_t c0, int32_t c1, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
            smemAddr(smem)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(smemAddr(bar)), "r"(c0), "r"(c1)
        : "memory");
}

__device__ __forceinline__ void bulkLoad1d(void* smem, void const* gmem, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr(smem)),
        "l"(gmem), "r"(bytes), "r"(smemAddr(bar))
        : "memory");
}

//! mode 0: 2-D TMA with box (rowBytes, kTileBytes/rowBytes) rows; mode 1: 1-D bulk copy.
__global__ void __launch_bounds__(32) streamKernel(__grid_constant__ const CUtensorMap map, uint8_t const* base,
    int32_t numTiles, int32_t stages, int32_t mode, int32_t boxRows)
{
    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + static_cast<size_t>(stages) * kTileBytes);
    if (threadIdx.x != 0)
    {
        return;
    }
    for (int32_t s = 0; s < stages; ++s)
    {
        mbarInit(&bars[s], 1);
    }
    fenceBarrierInit();
    if (mode == 0)
    {
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&map)) : "memory");
    }
    int32_t const first = static_cast<int32_t>(blockIdx.x);
    int32_t const stride = static_cast<int32_t>(gridDim.x);
    int32_t issued = 0; // loads issued by this CTA
    int32_t retired = 0;
    auto issue = [&](int32_t tile, int32_t stage) {
        uint8_t* dst = smem + static_cast<size_t>(stage) * kTileBytes;
        mbarArriveExpectTx(&bars[stage], kTileBytes);
        if (mode == 0)
        {
            tmaLoad2d(dst, &map, 0, tile * boxRows, &bars[stage]);
        }
        else
        {
            bulkLoad1d(dst, base + static_cast<size_t>(tile) * kTileBytes, kTileBytes, &bars[stage]);
        }
    };
    // Prologue: fill the ring.
    for (int32_t s = 0; s < stages; ++s)
    {
        int32_t const tile = first + issued * stride;
        if (tile >= numTiles)
        {
            break;
        }
        issue(tile, s);
        ++issued;
    }
    // Steady state: retire the oldest stage, refill it.
    while (retired < issued)
    {
        int32_t const stage = retired % stages;
        uint32_t const phase = static_cast<uint32_t>((retired / stages) & 1);
        mbarWait(&bars[stage], phase);
        ++retired;
        int32_t const tile = first + issued * stride;
        if (tile < numTiles)
        {
            issue(tile, stage);
            ++issued;
        }
    }
}

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, cuuint64_t const*,
    cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

struct Config
{
    char const* name;
    int32_t mode;      // 0 tensor, 1 bulk
    uint32_t rowBytes; // inner box bytes (tensor mode)
    uint32_t elemBytes;
    CUtensorMapSwizzle swizzle;
    CUtensorMapL2promotion promo;
};

int main(int argc, char** argv)
{
    size_t const totalBytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 512ULL) << 20;
    int32_t const reps = argc > 2 ? std::atoi(argv[2]) : 5;
    int32_t numSms = 0;
    CK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0));
    int32_t maxSmemOptin = 0;
    CK(cudaDeviceGetAttribute(&maxSmemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    uint8_t* buf = nullptr;
    CK(cudaMalloc(&buf, totalBytes));
    CK(cudaMemset(buf, 0x5A, totalBytes));
    int32_t const numTiles = static_cast<int32_t>(totalBytes / kTileBytes);

    EncodeFn encode = nullptr;
    cudaDriverEntryPointQueryResult q{};
    CK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), cudaEnableDefault, &q));
    if (encode == nullptr || q != cudaDriverEntryPointSuccess)
    {
        std::fprintf(stderr, "cuTensorMapEncodeTiled unavailable\n");
        return 1;
    }

    std::vector<Config> configs{
        {"tma 32B rows x128 SW32  (weights tile)", 0, 32, 1, CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"tma 32B rows x128 SW32 L2::128B", 0, 32, 1, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B},
        {"tma 32B rows x128 SW32 L2::256B", 0, 32, 1, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B},
        {"tma 32B rows x128 none", 0, 32, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"tma 64B rows x64 SW64", 0, 64, 1, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"tma 128B rows x32 SW128", 0, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"tma 128B rows x32 SW128 L2::256B", 0, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B},
        {"tma 256B rows x16 none", 0, 256, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"tma 2KB rows x2 none (u64)", 0, 2048, 8, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE},
        {"bulk 1-D 4KB", 1, 0, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE},
    };
    std::vector<int32_t> stageList{8, 16, 32, 48};

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::printf("device SMs=%d smem_optin=%d KB buffer=%zu MB tiles=%d reps=%d\n", numSms, maxSmemOptin / 1024,
        totalBytes >> 20, numTiles, reps);
    std::printf("%-40s", "config");
    for (int32_t st : stageList)
    {
        std::printf(" | %2d stages (%3d KB)", st, st * 4);
    }
    std::printf("\n");
    for (Config const& c : configs)
    {
        CUtensorMap map{};
        int32_t boxRows = 1;
        if (c.mode == 0)
        {
            uint32_t const elems = c.rowBytes / c.elemBytes;
            boxRows = static_cast<int32_t>(kTileBytes / c.rowBytes);
            cuuint64_t dims[2]{elems, static_cast<cuuint64_t>(totalBytes / c.rowBytes)};
            cuuint64_t strides[1]{c.rowBytes};
            cuuint32_t box[2]{elems, static_cast<cuuint32_t>(boxRows)};
            cuuint32_t estr[2]{1, 1};
            CUtensorMapDataType const dt
                = c.elemBytes == 8 ? CU_TENSOR_MAP_DATA_TYPE_UINT64 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
            CUresult const r = encode(&map, dt, 2, buf, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                c.swizzle, c.promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r != CUDA_SUCCESS)
            {
                std::printf("%-40s | encode failed (%d)\n", c.name, static_cast<int>(r));
                continue;
            }
        }
        std::printf("%-40s", c.name);
        for (int32_t stages : stageList)
        {
            size_t const smemBytes = static_cast<size_t>(stages) * kTileBytes + static_cast<size_t>(stages) * 8 + 1024;
            if (smemBytes > static_cast<size_t>(maxSmemOptin))
            {
                std::printf(" | %18s", "n/a");
                continue;
            }
            CK(cudaFuncSetAttribute(
                streamKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes)));
            // warmup
            streamKernel<<<numSms, 32, smemBytes>>>(map, buf, numTiles, stages, c.mode, boxRows);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (int32_t r = 0; r < reps; ++r)
            {
                CK(cudaEventRecord(e0));
                streamKernel<<<numSms, 32, smemBytes>>>(map, buf, numTiles, stages, c.mode, boxRows);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = ms < best ? ms : best;
            }
            std::printf(" | %8.1f GB/s %5.2fms", static_cast<double>(totalBytes) / (best * 1e-3) / 1e9, best);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    return 0;
}
