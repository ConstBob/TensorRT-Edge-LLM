#include "initializeCosSinCache.h"
#include "common/common.h"

#include <cuda_runtime.h>

namespace drivellm
{
namespace kernel
{

template <int32_t RotaryDim>
__global__ void initializeNormalRopeCosSinKernel(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale,
    int32_t rotaryEmbeddingMaxPositions)
{
    // In this kernel, each warp compute one "position" of the cos/sin cache, and loop until max position.
    // Each CTA will be assigned 4 warps so it proceeds 4 positions in an iteration.
    static_assert(RotaryDim % 64 == 0, "rotaryDim must be multiple of 64");

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;

    float ropeConstants[RotaryDim / 64];

    #pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 64; ++i)
    {
        uint32_t zid = tIdx + i * 32;
        ropeConstants[i] = pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = posIdx * RotaryDim;

        #pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 64; ++i)
        {
            float invFreq = posIdx * rotaryScale / ropeConstants[i];
            float cosVal = cos(invFreq);
            float sinVal = sin(invFreq);

            uint32_t zid = tIdx + i * 32;
            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream)
{
    // Each CTA get assigned 128 threads.
    dim3 block(32, 4);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
        case 64:
            kernelPtr = (void*) initializeNormalRopeCosSinKernel<64>;
            break;
        case 128:
            kernelPtr = (void*) initializeNormalRopeCosSinKernel<128>;
            break;
        default:
            throw std::runtime_error("Un-implemented rotaryDim for initializeNormalRopeCosSin: " + std::to_string(rotaryDim));
    }
    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));

    int32_t const numBlocks = std::min(maxBlockPerSM * numSMs, rotaryEmbeddingMaxPositions / 4);
    dim3 grid(numBlocks);

    void* kernelArgs[] = {
        reinterpret_cast<void*>(&cosSinCache),
        reinterpret_cast<void*>(&rotaryBaseFrequency),
        reinterpret_cast<void*>(&rotaryScale),
        reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions)
    };
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, grid, block, kernelArgs, 0, stream));
}



} // namespace kernel
} // namespace drivellm