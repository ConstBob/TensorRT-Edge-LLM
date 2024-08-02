#pragma once
#ifndef DECODER_H
#define DECODER_H
#include <NvInferRuntime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <map>

std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));

void check(cudaError_t result, char const* const func, char const* const file, int const line)
{
    if (result)
    {
        throw std::runtime_error(fmtstr("CUDA runtime error in %s: %s", func, cudaGetErrorString(result)));
    }
}
/*
 * Macros compliant with TensorRT coding conventions
 */
#define CUDA_CHECK(stat)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        check((stat), #stat, __FILE__, __LINE__);                                                \
    } while (0)

// StreamReader ported from TRT-LLM to read from engine file.
class StreamReader final : public nvinfer1::IStreamReader
{
public:
    StreamReader(std::filesystem::path fp)
    {
        mFile.open(fp.string(), std::ios::binary | std::ios::in);
        if (!mFile.good()){
            throw std::string("Error opening engine file: " + fp.string());
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

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kERROR)
            std::cerr << "[ERROR]: " << msg << std::endl;
        else if (severity == nvinfer1::ILogger::Severity::kWARNING)
            std::cerr << "[WARNING]: " << msg << std::endl;
        else
            std::cout << "[LOG]: " << msg << std::endl;
    }
};

struct ModelConfig
// This is the model config inferred from optimization profiles
{
    int64_t batchSize;
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
    int64_t maxLength; // Equivalent to maxOutputLength;
    int64_t numLayers;
}


class Decoder
{

public:
    Decoder():
        mLogger{nullptr},
        mStream{nullptr},
        mEngine{nullptr},
        mContextExecutionContext{nullptr},
        mGenerationExecutionContext{nullptr},
        stream_{0},
        isSetup{false},
        mConfig{0,0,0,0,0,0,0},
        mDeviceBuffer{},
        mHostBuffer{}
    {}
    bool setup(std::filesystem::path& const fp, cudaStream_t& const stream);
    std::vector<std::vector<int64_t>> generate(std::vector<std::vector<int64_t>>& input_ids, std::vector<std::vector<int64_t>>& attention_mask, std::vector<std::vector<int64_t>>& position_ids);

private:
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContextExecutionContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<Logger> mLogger;
    cudaStream_t mStream;
    bool isSetup;
    ModelConfig mConfig;
    std::map<std::string, void*> mDeviceBuffer;
    std::map<std::string, void*> mHostBuffer;
    bool validateAndFillConfig();
    bool checkStaticShape(const string& name);
    bool allocateBuffer();


}