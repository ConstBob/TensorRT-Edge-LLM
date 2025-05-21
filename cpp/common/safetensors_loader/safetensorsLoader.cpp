#include "safetensorsLoader.h"
#include "common/logger.h"
#include <cmath>
#include <cstring>
#include <fstream>

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

    // Clear the file buffer
    mFileBuffer.clear();
    mFileBuffer.shrink_to_fit();
}

bool SafeTensorsLoader::loadFromFileToGPU()
{
    // Read the file into memory
    if (!readTensorData())
    {
        LOG_ERROR("Failed to read tensor data from file");
        return false;
    }

    // Read the header size (8 bytes)
    uint64_t headerSize = *reinterpret_cast<uint64_t*>(mFileBuffer.data());

    // Read the metadata JSON
    std::string metadataStr(reinterpret_cast<char*>(mFileBuffer.data() + sizeof(headerSize)), headerSize);

    // Parse the metadata
    if (!parseMetadata(metadataStr))
    {
        LOG_ERROR("Failed to parse metadata");
        return false;
    }

    // Load each tensor directly to GPU
    size_t tensorDataStart = sizeof(headerSize) + headerSize;
    for (auto& [name, info] : mTensorInfo)
    {
        uint8_t const* tensorData = mFileBuffer.data() + tensorDataStart + info.dataOffsets[0];

        if (!loadTensorToGPU(info, tensorData))
        {
            LOG_ERROR("Failed to load tensor %s to GPU", name.c_str());
            return false;
        }
    }

    // Clear the file buffer after loading to GPU
    mFileBuffer.clear();
    mFileBuffer.shrink_to_fit();

    return true;
}

bool SafeTensorsLoader::readTensorData()
{
    // Open the file
    // TODO: use mmap to read the file
    std::ifstream file(mFilePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open file: %s", mFilePath.c_str());
        return false;
    }

    // Get file size
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the entire file into memory
    mFileBuffer.resize(fileSize);
    file.read(reinterpret_cast<char*>(mFileBuffer.data()), fileSize);

    if (file.fail())
    {
        LOG_ERROR("Failed to read file");
        return false;
    }

    return true;
}

bool SafeTensorsLoader::loadTensorToGPU(SafeTensorsInfo& info, uint8_t const* data)
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

bool SafeTensorsLoader::parseMetadata(std::string const& metadataStr)
{
    mMetadata = std::make_unique<JsonRoot>();
    if (!mMetadata->parse(metadataStr))
    {
        LOG_ERROR("Failed to parse JSON metadata");
        return false;
    }

    auto root = mMetadata->getRoot();
    if (!root.isObject())
    {
        LOG_ERROR("Root is not an object");
        return false;
    }

    // Parse each tensor's metadata
    for (size_t i = 0; i < root.size(); ++i)
    {
        auto tensorNode = root[i];
        std::string tensorName = tensorNode.getName();
        // skip __metadata__
        if (tensorName == "__metadata__")
        {
            continue;
        }
        if (!tensorNode.isObject())
        {
            LOG_ERROR("Tensor %s is not an object", tensorName.c_str());
            return false;
        }

        SafeTensorsInfo info;
        if (!parseSafeTensorsInfo(tensorNode, info))
        {
            LOG_ERROR("Failed to parse tensor info for %s", tensorName.c_str());
            return false;
        }

        mTensorInfo[tensorName] = info;
    }

    return true;
}

bool SafeTensorsLoader::parseSafeTensorsInfo(JsonNode& node, SafeTensorsInfo& info)
{
    // Parse shape
    auto shapeNode = node["shape"];
    if (!shapeNode.isArray())
    {
        LOG_ERROR("Error parsing tensor info: shape is not an array");
        return false;
    }

    info.shape.clear();
    for (size_t i = 0; i < shapeNode.size(); ++i)
    {
        if (!shapeNode[i].isInteger())
        {
            LOG_ERROR("Error parsing tensor info: shape element is not an integer");
            return false;
        }
        info.shape.push_back(shapeNode[i].getInteger());
    }

    // Parse dtype
    auto dtypeNode = node["dtype"];
    if (!dtypeNode.isString())
    {
        LOG_ERROR("Error parsing tensor info: dtype is not a string");
        return false;
    }
    info.dtype = dtypeNode.getString();

    // Parse data_offsets
    auto offsetsNode = node["data_offsets"];
    if (!offsetsNode.isArray() || offsetsNode.size() != 2)
    {
        LOG_ERROR("Error parsing tensor info: data offsets is not an array of size 2");
        return false;
    }

    for (size_t i = 0; i < 2; ++i)
    {
        if (!offsetsNode[i].isInteger())
        {
            LOG_ERROR("Error parsing tensor info: data offset is not an integer");
            return false;
        }
        info.dataOffsets[i] = offsetsNode[i].getInteger();
    }

    return true;
}

std::unordered_map<std::string, SafeTensorsInfo> const& SafeTensorsLoader::getSafeTensorsInfo() const
{
    return mTensorInfo;
}

} // namespace drivellm
