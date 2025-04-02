#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(SanityCheck, InitializeCUDA) 
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    // Ensure the function executes successfully
    ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
    // We assume at least one GPU is present in the system
    ASSERT_GT(device_count, 0) << "CUDA device is not available";

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        // Check if properties were fetched successfully
        ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceProperties failed for device " << i 
                                    << ": " << cudaGetErrorString(err);
    }
}