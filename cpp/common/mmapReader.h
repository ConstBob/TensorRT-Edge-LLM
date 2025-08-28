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

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace drivellm
{
namespace file_io
{

class MmapReader
{
public:
    MmapReader();

    explicit MmapReader(std::filesystem::path const& fp);

    MmapReader(MmapReader const&) = delete;
    MmapReader& operator=(MmapReader const&) = delete;

    ~MmapReader();

    void release();

    bool loadFile(std::filesystem::path const& fp);

    int8_t const* getByteData() const noexcept;

    void const* getData() const noexcept;

    size_t getSize() const noexcept;

private:
    void* mData;
    size_t mBytes;
};

} // namespace file_io
} // namespace drivellm
