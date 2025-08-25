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

#include <memory>
#include <string>

namespace drivellm
{
namespace rt
{
namespace imageUtils
{

// Image data structure for managing loaded images
class ImageData
{
public:
    std::shared_ptr<unsigned char[]> buffer;
    int width;
    int height;
    int channels;
    bool isThumbnail; // TODO: Clean old API

    ImageData(unsigned char* data, int w, int h, int c, bool thumbnail = false)
        : buffer(data)
        , width(w)
        , height(h)
        , channels(c)
        , isThumbnail(thumbnail)
    {
    }

    unsigned char* data() const
    {
        return buffer.get();
    }
};

ImageData loadImageFromFile(std::string const& path);

ImageData loadImageFromMemory(unsigned char const* data, size_t size);

ImageData resizeImage(ImageData const& image, int newWidth, int newHeight, bool isThumbnail = false);

} // namespace imageUtils
} // namespace rt
} // namespace drivellm