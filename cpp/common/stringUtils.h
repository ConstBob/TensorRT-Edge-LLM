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

#include <cstdarg>
#include <string>

namespace drivellm
{
namespace format
{

/**
 * @brief Format a string using variable arguments
 * @param format Format string
 * @param ... Variable arguments
 * @return Formatted string
 */
std::string fmtstr(char const* format, ...);

} // namespace format
} // namespace drivellm
