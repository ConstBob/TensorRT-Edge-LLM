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

#include "stringUtils.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace drivellm
{
namespace format
{
namespace
{
/**
 * @brief Format a string using va_list arguments
 * @param fmt Format string
 * @param args Variable arguments list
 * @return Formatted string
 */
std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
    {
        return "";
    }

    // Ensure that the underlying string buffer is large enough, even for
    // the null terminator
    std::string stringBuf(size + 1, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    if (size2 != size)
    {
        throw std::runtime_error(std::string(std::strerror(errno)));
    }
    stringBuf.resize(size);

    return stringBuf;
}
} // anonymous namespace
std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
}

} // namespace format
} // namespace drivellm
