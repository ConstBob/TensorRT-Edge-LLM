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
