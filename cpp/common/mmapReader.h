/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
