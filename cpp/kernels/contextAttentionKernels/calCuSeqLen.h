#pragma once

#include <cuda_runtime_api.h>

namespace drivellm
{
namespace kernel
{

// Host launcher to build cumulative sequence lengths array (prefix sums) on device.
// seqLenDevice : pointer to device int32_t array of sequence lengths for each batch element (size B)
// cuSeqLensDevice : pointer to device int32_t array where the cumulative sequence lengths (prefix sums)
//                   will be written. Must have size B + 1. The first element is always 0.
// B : batch size (number of sequences)
// stream : CUDA stream to execute the kernel on, shall be the same stream to launch FMHA-v2 kernel.
void calCuSeqLens(int32_t const* seqLenDevice, int32_t* cuSeqLensDevice, int32_t B, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm