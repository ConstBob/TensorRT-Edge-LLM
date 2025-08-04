#include "safetensorsLoader.h"

#include "common/common.h"
#include "common/logger.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace drivellm
{

SafeTensorsLoader::SafeTensorsLoader(std::string const& filePath)
    : mFilePath(filePath)
{
}

SafeTensorsLoader::~SafeTensorsLoader()
{
    // Free all GPU memory
    for (auto& [name, info] : mTensorInfo)
    {
        if (info.gpuPtr)
        {
            cudaFree(info.gpuPtr);
            info.gpuPtr = nullptr;
        }
    }
}

bool SafeTensorsLoader::loadFromFileToGPU()
{
    // Read the file into memory
    MmapReader mmapReader(mFilePath);
    if (!mmapReader.loadFile(mFilePath))
    {
        LOG_ERROR("Failed to use MMap to read safetensors file from path: %s", mFilePath.c_str());
        return false;
    }

    // Read the header size (8 bytes)
    uint64_t headerSize = *reinterpret_cast<uint64_t const*>(mmapReader.getByteData());

    // Read the metadata JSON
    std::string metadataStr(reinterpret_cast<char const*>(mmapReader.getByteData() + sizeof(headerSize)), headerSize);

    // Parse the metadata
    if (!parseJsonHeader(metadataStr))
    {
        LOG_ERROR("Failed to parse metadata");
        return false;
    }

    // Load each tensor directly to GPU
    size_t tensorDataStart = sizeof(headerSize) + headerSize;
    for (auto& [name, info] : mTensorInfo)
    {
        int8_t const* tensorData = mmapReader.getByteData() + tensorDataStart + info.dataOffsets[0];

        if (!loadTensorToGPU(info, tensorData))
        {
            LOG_ERROR("Failed to load tensor %s to GPU", name.c_str());
            return false;
        }
    }

    return true;
}

bool SafeTensorsLoader::loadTensorToGPU(SafeTensorsInfo& info, int8_t const* data)
{
    // Calculate total elements
    size_t totalElements = 1;
    for (size_t dim : info.shape)
    {
        totalElements *= dim;
    }

    // Only support F16 and BF16 for now
    if (info.dtype != "F16" && info.dtype != "BF16")
    {
        LOG_ERROR("Unsupported data type: %s. Only F16 and BF16 are supported", info.dtype.c_str());
        return false;
    }

    // Allocate GPU memory
    cudaMalloc(&info.gpuPtr, totalElements * sizeof(uint16_t));
    if (!info.gpuPtr)
    {
        LOG_ERROR("Failed to allocate GPU memory");
        return false;
    }

    cudaMemcpy(info.gpuPtr, data, totalElements * sizeof(uint16_t), cudaMemcpyHostToDevice);

    return true;
}

bool SafeTensorsLoader::parseJsonHeader(std::string const& metadataStr)
{
    Json header;
    try
    {
        header = Json::parse(metadataStr);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse JSON metadata: %s", e.what());
        LOG_ERROR("Detailed Json parsing error: %s", e.what());
        return false;
    }

    auto validateTensorEntry = [](Json const& value) {
        return value.is_object() && value.contains("dtype") && value["dtype"].is_string() && value.contains("shape")
            && value["shape"].is_array() && value.contains("data_offsets") && value["data_offsets"].is_array()
            && value["data_offsets"].size() == 2;
    };

    for (auto const& [key, value] : header.items())
    {
        if (key == "__metadata__")
        {
            LOG_DEBUG("Loading SafeTensor Header, Metadata: %s", value.dump().c_str());
            continue;
        }

        if (validateTensorEntry(value))
        {
            SafeTensorsInfo info;
            info.shape = value["shape"].get<std::vector<size_t>>();
            info.dtype = value["dtype"].get<std::string>();
            info.dataOffsets[0] = value["data_offsets"][0].get<size_t>();
            info.dataOffsets[1] = value["data_offsets"][1].get<size_t>();
            mTensorInfo[key] = info;
        }
        else
        {
            LOG_ERROR("Malformed tensor entry of SafeTensor object: %s : %s", key.c_str(), value.dump().c_str());
            return false;
        }
    }
    return true;
}

std::unordered_map<std::string, SafeTensorsInfo> const& SafeTensorsLoader::getSafeTensorsInfo() const
{
    return mTensorInfo;
}

} // namespace drivellm
