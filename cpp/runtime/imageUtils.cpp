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