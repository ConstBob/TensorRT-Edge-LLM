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
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

inline nvinfer1::Dims createDims(std::vector<int64_t> const& shape)
{
    nvinfer1::Dims dims{static_cast<int32_t>(shape.size()), {}};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}

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

inline bool checkOptimizationProfileDims(
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims)
{
    if (minDims.nbDims != optDims.nbDims || optDims.nbDims != maxDims.nbDims)
    {
        LOG_ERROR("Dimension count mismatch: minDims.nbDims=%d, optDims.nbDims=%d, maxDims.nbDims=%d", minDims.nbDims,
            optDims.nbDims, maxDims.nbDims);
        return false;
    }
    for (int i = 0; i < minDims.nbDims; ++i)
    {
        if (minDims.d[i] > optDims.d[i] || optDims.d[i] > maxDims.d[i])
        {
            LOG_ERROR("Dimension value mismatch at index %d: min=%d, opt=%d, max=%d", i, minDims.d[i], optDims.d[i],
                maxDims.d[i]);
            return false;
        }
    }
    return true;
}

inline bool setOptimizationProfile(nvinfer1::IOptimizationProfile* profile, char const* inputName,
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims)
{
    if (!checkOptimizationProfileDims(minDims, optDims, maxDims))
    {
        LOG_INFO("setOptimizationProfile: %s is not valid", inputName);
        return false;
    }
    return profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
}

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

inline std::unique_ptr<void, DlDeleter> loadAttentionPlugin(void)
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

    auto handle = std::unique_ptr<void, DlDeleter>(dlopen(pluginPath, RTLD_LAZY));
    if (!handle)
    {
        LOG_ERROR("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, DlDeleter>(nullptr);
    }
    return handle;
}

inline std::unique_ptr<void, DlDeleter> loadInt4GemmPlugin(void)
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

    auto handle = std::unique_ptr<void, DlDeleter>(dlopen(pluginPath, RTLD_LAZY));
    if (!handle)
    {
        LOG_WARNING("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, DlDeleter>(nullptr);
    }
    return handle;
}

inline std::vector<std::unique_ptr<void, DlDeleter>> loadPlugins(bool int4GemmPlugin = true)
{
    std::vector<std::unique_ptr<void, DlDeleter>> handles;
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

class MmapReader
{
public:
    MmapReader(std::filesystem::path fp)
    {
        std::string const enginePath = fp.string();
        int fd = open(enginePath.c_str(), O_RDONLY);
        if (fd <= 0)
        {
            throw std::runtime_error(fmtstr("Cannot open engine file: %s", enginePath));
        }
        try
        {
            struct stat status;
            if (fstat(fd, &status) != 0)
            {
                throw std::runtime_error(fmtstr("Engine file %s fstat failed.", enginePath));
            }
            mBytes = status.st_size;
            if (mBytes == 0)
            {
                throw std::runtime_error(fmtstr("Engine file %s is empty.", enginePath));
            }
            mData = mmap(nullptr, mBytes, PROT_READ, MAP_SHARED, fd, 0);
            if (mData == MAP_FAILED)
            {
                mData = nullptr;
                throw std::runtime_error(fmtstr("Engine file %s mmap failed.", enginePath));
            }
        }
        catch (...)
        {
            close(fd);
            throw;
        }
        close(fd);
    }
    MmapReader()
        : mData(nullptr)
        , mBytes(0)
    {
    }
    ~MmapReader()
    {
        if (mData != nullptr && mData != MAP_FAILED)
        {
            munmap(mData, mBytes);
            mData = nullptr;
            mBytes = 0;
        }
    }
    void const* getData() const
    {
        return mData;
    }
    size_t getSize() const
    {
        return mBytes;
    }

private:
    void* mData;
    size_t mBytes;
};

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