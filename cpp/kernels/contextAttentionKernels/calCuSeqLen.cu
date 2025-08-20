#include "calCuSeqLen.h"

namespace drivellm
{
namespace kernel
{

__global__ void calCuSeqLenskernel(int32_t const* seqLen, int32_t* cu, int32_t B)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cu[0] = 0;
        int32_t running = 0;
        for (int32_t i = 0; i < B; ++i)
        {
            running += seqLen[i];
            cu[i + 1] = running;
        }
    }
}

void calCuSeqLens(int32_t const* seqLenDev, int32_t* cuSeqLensDev, int32_t B, cudaStream_t stream)
{
    calCuSeqLenskernel<<<1, 1, 0, stream>>>(seqLenDev, cuSeqLensDev, B);
}

} // namespace kernel
} // namespace drivellm