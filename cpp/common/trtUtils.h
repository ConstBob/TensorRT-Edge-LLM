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

#include "logger.h"
#include "stringUtils.h"
#include <NvInfer.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace drivellm
{

inline std::int64_t volume(nvinfer1::Dims const& dims)
{

    return dims.nbDims < 0 ? -1
        : dims.nbDims == 0 ? 0
                           : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1}, std::multiplies<>{});
}

struct EngineInputDesc
{
    std::string name;
    void* deviceBufferForContext;
    void* deviceBufferForDecode;
    nvinfer1::Dims contextDims;
    nvinfer1::Dims generationDims;
    EngineInputDesc(std::string const name, void* deviceBufferForContext, void* deviceBufferForDecode,
        nvinfer1::Dims const contextDims, nvinfer1::Dims const generationDims)
        : name(name)
        , deviceBufferForContext(deviceBufferForContext)
        , deviceBufferForDecode(deviceBufferForDecode)
        , contextDims(contextDims)
        , generationDims(generationDims)
    {
    }
};

// Define a custom deleter type to handle the noexcept attribute
struct DlDeleter
{
    void operator()(void* handle) const noexcept
    {
        if (handle)
        {
            dlclose(handle);
        }
    }
};

inline std::unique_ptr<void, DlDeleter> loadEdgellmPluginLib(void)
{
    char const* pluginPath = std::getenv("EDGELLM_PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("EDGELLM_PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        LOG_INFO("EDGELLM_PLUGIN_PATH variable is not set. Default to build/libNvInfer_edgellm_plugin.so");
        pluginPath = "build/libNvInfer_edgellm_plugin.so";
    }

    auto handle = std::unique_ptr<void, DlDeleter>(dlopen(pluginPath, RTLD_LAZY));
    if (!handle)
    {
        LOG_ERROR("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, DlDeleter>(nullptr);
    }
    return handle;
}

struct TensorInfo
{
    void* data;
    nvinfer1::Dims dims;
    TensorInfo(void* data, nvinfer1::Dims const dims)
        : data(data)
        , dims(dims)
    {
    }
};

} // namespace drivellm
