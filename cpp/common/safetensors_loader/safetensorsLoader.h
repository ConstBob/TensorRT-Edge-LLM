#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace drivellm
{

struct SafeTensorsInfo
{
    std::vector<size_t> shape;
    std::string dtype;
    size_t dataOffsets[2];
    void* gpuPtr = nullptr; // GPU memory location for this tensor
};

class SafeTensorsLoader
{
public:
    explicit SafeTensorsLoader(std::string const& filePath);
    ~SafeTensorsLoader();

    // Load a safetensors file from disk directly to GPU
    bool loadFromFileToGPU();

    // Get tensor information including GPU pointers
    std::unordered_map<std::string, SafeTensorsInfo> const& getSafeTensorsInfo() const;

private:
    bool parseJsonHeader(std::string const& metadataStr);
    bool loadTensorToGPU(SafeTensorsInfo& info, int8_t const* data);

    std::string mFilePath;
    std::unordered_map<std::string, SafeTensorsInfo> mTensorInfo;
};

} // namespace drivellm