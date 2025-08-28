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

#include "mmapReader.h"

#include "common.h"
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace drivellm
{

namespace file_io
{

MmapReader::MmapReader()
    : mData(nullptr)
    , mBytes(0)
{
}

MmapReader::MmapReader(std::filesystem::path const& fp)
    : mData(nullptr)
    , mBytes(0)
{
    if (!loadFile(fp))
    {
        throw std::runtime_error("Failed to load file in MmapReader constructor");
    }
}

MmapReader::~MmapReader()
{
    release();
}

void MmapReader::release()
{
    if (mData != nullptr && mBytes > 0)
    {
        munmap(mData, mBytes);
        mData = nullptr;
        mBytes = 0;
    }
}

bool MmapReader::loadFile(std::filesystem::path const& fp)
{
    // Release any existing memory
    release();

    std::string const filePath = fp.string();
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd <= 0)
    {
        std::string errorMsg = fmtstr("MmapReader: Cannot open file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }

    struct stat status;
    if (fstat(fd, &status) != 0)
    {
        close(fd);
        std::string errorMsg = fmtstr("MmapReader: fstat failed for file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    mBytes = status.st_size;
    if (mBytes == 0)
    {
        close(fd);
        std::string errorMsg = fmtstr("MmapReader: File %s is empty.", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    mData = mmap(nullptr, mBytes, PROT_READ, MAP_SHARED, fd, 0);
    if (mData == MAP_FAILED)
    {
        mData = nullptr;
        mBytes = 0;
        close(fd);
        std::string errorMsg = fmtstr("MmapReader: mmap failed for file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    close(fd);
    return true;
}

int8_t const* MmapReader::getByteData() const noexcept
{
    return reinterpret_cast<int8_t const*>(mData);
}

void const* MmapReader::getData() const noexcept
{
    return mData;
}

size_t MmapReader::getSize() const noexcept
{
    return mBytes;
}

} // namespace file_io
} // namespace drivellm
