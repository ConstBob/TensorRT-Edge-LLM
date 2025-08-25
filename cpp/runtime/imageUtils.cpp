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

#include "runtime/imageUtils.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

namespace drivellm
{
namespace rt
{
namespace imageUtils
{

ImageData loadImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (data == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));
    }
    return ImageData(data, width, height, channels);
}

ImageData loadImageFromMemory(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    if (imageData == nullptr)
    {
        throw std::runtime_error("Failed to load image from memory: " + std::string(stbi_failure_reason()));
    }
    return ImageData(imageData, width, height, desiredChannels);
}

ImageData resizeImage(ImageData const& image, int newWidth, int newHeight, bool isThumbnail)
{
    if (newWidth <= 0 || newHeight <= 0)
    {
        throw std::invalid_argument("New dimensions must be positive");
    }

    // Allocate memory for resized image
    unsigned char* resizedData = (unsigned char*) malloc(newWidth * newHeight * image.channels);
    if (resizedData == nullptr)
    {
        throw std::runtime_error("Failed to allocate memory for resized image");
    }

    // Resize the image
    stbir_resize_uint8_linear(
        image.data(), image.width, image.height, 0, resizedData, newWidth, newHeight, 0, STBIR_RGB);

    return ImageData(resizedData, newWidth, newHeight, image.channels, isThumbnail);
}

} // namespace imageUtils
} // namespace rt
} // namespace drivellm