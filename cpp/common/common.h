/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <NvInferRuntime.h>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>

namespace drivellm
{

inline void check(bool condition, std::string errorMsg)
{
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

inline std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
    {
        return "";
    }

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    check(size2 == size, std::string(std::strerror(errno)));

    return stringBuf;
}

inline std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
};

inline void _checkCuda(cudaError_t result, char const* const func, [[maybe_unused]] char const* const file,
    [[maybe_unused]] int const line)
{
    if (result)
    {
        throw std::runtime_error(fmtstr("CUDA runtime error in %s: %s", func, cudaGetErrorString(result)));
    }
}

inline void _checkCudaDriver(
    CUresult result, char const* const func, [[maybe_unused]] char const* const file, [[maybe_unused]] int const line)
{
    if (result)
    {
        char const* errorName = nullptr;
        if (cuGetErrorName(result, &errorName) != CUDA_SUCCESS)
        {
            errorName = "CUDA driver API error happened, but we failed to get error name.";
        }
        throw std::runtime_error(fmtstr("CUDA driver API error in %s: %s", func, errorName));
    }
}
/*
 * Macros compliant with TensorRT coding conventions
 */
#define CUDA_CHECK(stat)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        _checkCuda((stat), #stat, __FILE__, __LINE__);                                                                 \
    } while (0)

#define CUDA_DRIVER_CHECK(stat)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        _checkCudaDriver((stat), #stat, __FILE__, __LINE__);                                                           \
    } while (0)

inline std::string extractFolderName(std::string const& path)
{
    size_t found = path.find_last_of("/\\");
    if (found != std::string::npos)
    {
        return path.substr(0, found);
    }
    return "";
}

inline int copyFile(std::string const& srcPath, std::string const& dstPath)
{
    if (srcPath == dstPath)
    {
        printf("Source and target file path are same, skip copying.");
    }
    else
    {
        std::ifstream source(srcPath, std::ios::in | std::ios::binary);
        if (!source)
        {
            printf("Failed to open file for reading: %s", srcPath.c_str());
            return EXIT_FAILURE;
        }

        std::ofstream dest(dstPath, std::ios::out | std::ios::binary);
        if (dest)
        {
            dest << source.rdbuf();
            printf("Successfully copied file to %s.\n", dstPath.c_str());
        }
        else
        {
            printf("Failed to copy file to %s", dstPath.c_str());
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

} // namespace drivellm
