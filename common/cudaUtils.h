#pragma once

#include "common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

static char const* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

static char const* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

inline int getDevice()
{
    int current_dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

template <typename T1, typename T2>
inline size_t divUp(const T1& a, const T2& n)
{
    size_t tmp_a = static_cast<size_t>(a);
    size_t tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

inline bool isCudaLaunchBlocking()
{
    static bool firstCall = true;
    static bool result = false;

    if (firstCall)
    {
        char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
        result = env != nullptr && std::string(env) == "1";
        firstCall = false;
    }

    return result;
}

/// Get the memory info
/// \return The free and total amount of memory in bytes
inline std::tuple<size_t, size_t> getDeviceMemoryInfo()
{
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return {free, total};
}

#ifdef NDEBUG
#define sync_check_cuda_error()                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t result = cudaDeviceSynchronize();                                                                  \
        CUDA_CHECK(result);                                                                                            \
    } while (0)
#else
#define sync_check_cuda_error()
#endif