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

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <memory>
#include <string>

namespace drivellm
{
namespace rt
{
namespace imageUtils
{

// ImageData is a shared pointer to a rt::Tensor, used to manage stbi loaded images
// The tensor is expected to have shape [height, width, channels] and channels is 3 for RGB image
class ImageData
{
public:
    std::shared_ptr<rt::Tensor> buffer;
    int32_t width;
    int32_t height;
    int32_t channels;

    ImageData(rt::Tensor&& data, bool thumbnail = false);

    unsigned char* data() const;
};

ImageData loadImageFromFile(std::string const& path);

ImageData loadImageFromMemory(unsigned char const* data, size_t size);

ImageData resizeImage(ImageData const& image, int newWidth, int newHeight);

} // namespace imageUtils
} // namespace rt
} // namespace drivellm