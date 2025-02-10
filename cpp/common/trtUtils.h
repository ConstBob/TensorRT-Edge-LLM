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

#include "common.h"
#include "logger.h"
#include <NvInfer.h>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <vector>

inline nvinfer1::Dims createDims(std::vector<int64_t> const& shape)
{
    nvinfer1::Dims dims{static_cast<int32_t>(shape.size()), {}};
    for (int i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}

struct EngineInputDesc
{
    std::string name;
    void* deviceBuffer;
    nvinfer1::Dims contextDims;
    nvinfer1::Dims generationDims;
    EngineInputDesc(const std::string name, void* deviceBuffer, const nvinfer1::Dims contextDims, 
        const nvinfer1::Dims generationDims)
        : name(name)
        , deviceBuffer(deviceBuffer)
        , contextDims(contextDims)
        , generationDims(generationDims)
    {
    }
};

inline bool setOptimizationProfile(nvinfer1::IOptimizationProfile* profile, char const* inputName,
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims)
{
    return profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
}

inline std::unique_ptr<void, decltype(&dlclose)> loadAttentionPlugin(void)
{
    char const* pluginPath = std::getenv("ATTENTION_PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("ATTENTION_PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        LOG_INFO("ATTENTION_PLUGIN_PATH variable is not set. Default to build/libAttentionPlugin.so");
        pluginPath = "build/libAttentionPlugin.so";
    }

    auto handle = std::unique_ptr<void, decltype(&dlclose)>(dlopen(pluginPath, RTLD_LAZY), &dlclose);
    if (!handle)
    {
        LOG_ERROR("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, decltype(&dlclose)>(nullptr, &dlclose);
    }
    return handle;
}

inline std::unique_ptr<void, decltype(&dlclose)> loadInt4GemmPlugin(void)
{
    char const* pluginPath = std::getenv("INT4_GEMM_PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("INT4_GEMM_PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        LOG_INFO("INT4_GEMM_PLUGIN_PATH variable is not set. Default to build/libInt4GemmPlugin.so");
        pluginPath = "build/libInt4GemmPlugin.so";
    }

    auto handle = std::unique_ptr<void, decltype(&dlclose)>(dlopen(pluginPath, RTLD_LAZY), &dlclose);
    if (!handle)
    {
        LOG_WARNING("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, decltype(&dlclose)>(nullptr, &dlclose);
    }
    return handle;
}

inline std::vector<std::unique_ptr<void, decltype(&dlclose)>> loadPlugins(bool int4GemmPlugin = true)
{
    std::vector<std::unique_ptr<void, decltype(&dlclose)>> handles;
    handles.push_back(loadAttentionPlugin());
    if (int4GemmPlugin)
    {
        handles.push_back(loadInt4GemmPlugin());
    }
    return handles;
}

// StreamReader ported from TRT-LLM to read from engine file.
class StreamReader final : public nvinfer1::IStreamReader
{
public:
    StreamReader(std::filesystem::path fp)
    {
        mFile.open(fp.string(), std::ios::binary | std::ios::in);
        if (!mFile.good())
        {
            throw std::runtime_error(fmtstr("Cannot open engine file: %s", fp.string()));
        };
    }

    virtual ~StreamReader()
    {
        if (mFile.is_open())
        {
            mFile.close();
        }
    }

    int64_t read(void* destination, int64_t nbBytes) final
    {
        if (!mFile.good())
        {
            return -1;
        }
        mFile.read(static_cast<char*>(destination), nbBytes);
        return mFile.gcount();
    }
    std::ifstream mFile;
};

struct TensorInfo
{
    void* data;
    nvinfer1::Dims dims;
    TensorInfo(void* data, const nvinfer1::Dims dims)
        : data(data)
        , dims(dims)
    {
    }
};