#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <NvInferRuntime.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <cstdlib>
#include <sstream>
#include <cerrno>
#include <cstring>

using namespace std;

inline void check(bool condition, std::string errorMsg) {
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

inline std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
        return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    check(size2 == size, std::string(std::strerror(errno)));

    return stringBuf;
}

inline std::string fmtstr(char const* format, ...){
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
};


inline void _checkCuda(cudaError_t result, char const* const func, char const* const file, int const line)
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
        _checkCuda((stat), #stat, __FILE__, __LINE__);                                                \
    } while (0)


// int constexpr VOID_PTR_SZ = 2 + sizeof(void *) * 2;

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
    Logger(){};
    ~Logger(){};
    void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
    {
        if (severity > _minSeverity){
            switch(severity){
                case nvinfer1::ILogger::Severity::kVERBOSE: {
                    verbose(msg);
                    break;
                }
                case nvinfer1::ILogger::Severity::kERROR: {
                    error(msg);
                    break;
                }
                case nvinfer1::ILogger::Severity::kWARNING:
                {
                    warning(msg);
                    break;
                }
                case nvinfer1::ILogger::Severity::kINFO:
                {
                    info(msg);
                    break;
                }
                default:
                {
                    error(msg);
                    break;
                }
            }
        }
    }

    void verbose(char const* msg)
    {
        std::cout << "[VERBOSE]: " << msg << std::endl;
    }

    void warning(char const* msg)
    {
        std::cerr << "[WARNING]: " << msg << std::endl;
    }

    void error(char const* msg)
    {
        std::cerr << "[ERROR]: " << msg << std::endl;
    }

    void info(char const* msg)
    {
        std::cout << "[INFO]: " << msg << std::endl;
    }

    void setLevel(nvinfer1::ILogger::Severity minSeverity){
        _minSeverity = minSeverity;
    }
private:
    nvinfer1::ILogger::Severity _minSeverity = nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
};

#endif