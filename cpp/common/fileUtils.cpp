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

#include "fileUtils.h"
#include "logger.h"

#include <cstdlib>
#include <filesystem>

namespace drivellm
{
namespace file_io
{

bool copyFile(std::string const& srcPath, std::string const& dstPath)
{
    std::filesystem::path const src{srcPath};
    if (!std::filesystem::exists(src))
    {
        LOG_INFO("Failed to open file for reading: %s", srcPath.c_str());
        return false;
    }
    std::filesystem::path const dst{dstPath};
    if (std::filesystem::exists(dst) && std::filesystem::equivalent(src, dst))
    {
        LOG_INFO("Source and target file path are same, skip copying.");
    }
    else
    {
        try
        {
            auto const options = std::filesystem::copy_options::overwrite_existing;
            std::filesystem::copy(src, dst, options);
            LOG_INFO("Successfully copied %s to %s", srcPath.c_str(), dstPath.c_str());
        }
        catch (std::filesystem::filesystem_error& e)
        {
            LOG_ERROR("Error copying %s to %s - %s", srcPath.c_str(), dstPath.c_str(), e.what());
            return false;
        }
    }
    return true;
}

} // namespace file_io
} // namespace drivellm
