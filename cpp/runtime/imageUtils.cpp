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
#include "common/checkMacros.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

ImageData::ImageData(rt::Tensor&& data, bool thumbnail)
{
    check::check(data.getDataType() == nvinfer1::DataType::kUINT8, "Image data must be UINT8");
    check::check(data.getShape().getNumDims() == 3, "Image data must have 3 dimensions");
    check::check(data.getShape()[2] == 3, "Image data must have 3 channels");

    // Image data is expected to have shape [height, width, channels]
    height = data.getShape()[0];
    width = data.getShape()[1];
    channels = data.getShape()[2];
    buffer = std::make_shared<rt::Tensor>(std::move(data));
}

unsigned char* ImageData::data() const
{
    return buffer->dataPointer<unsigned char>();
}

ImageData loadImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load(path.c_str(), &width, &height, &channels, desiredChannels);
    if (imageData == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));
    }

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({height, width, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * channels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData loadImageFromMemory(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    if (imageData == nullptr)
    {
        throw std::runtime_error("Failed to load image from memory: " + std::string(stbi_failure_reason()));
    }

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({height, width, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * channels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData resizeImage(ImageData const& image, int64_t newWidth, int64_t newHeight)
{
    if (newWidth <= 0 || newHeight <= 0)
    {
        throw std::invalid_argument("New dimensions must be positive");
    }

    // Allocate memory for resized image
    rt::Tensor resizedTensor
        = rt::Tensor({newHeight, newWidth, image.channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);

    // Resize the image, default strides for src/dst images to zero.
    constexpr int32_t kINPUT_STRIDE_BYTES{0};
    constexpr int32_t kOUTPUT_STRIDE_BYTES{0};
    stbir_resize_uint8_linear(image.data(), image.width, image.height, kINPUT_STRIDE_BYTES,
        resizedTensor.dataPointer<unsigned char>(), newWidth, newHeight, kOUTPUT_STRIDE_BYTES, STBIR_RGB);
    return ImageData(std::move(resizedTensor));
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm