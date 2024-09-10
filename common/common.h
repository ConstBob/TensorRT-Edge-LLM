#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <NvInferRuntime.h>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

inline void check(bool condition, std::string errorMsg)
{
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

inline std::string fmtstr(char const* format, ...)
{
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
#define CUDA_CHECK(stat)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        _checkCuda((stat), #stat, __FILE__, __LINE__);                                                                 \
    } while (0)

// StreamReader ported from TRT-LLM to read from engine file.
class StreamReader final : public nvinfer1::IStreamReader
{
public:
    StreamReader(std::filesystem::path fp)
    {
        mFile.open(fp.string(), std::ios::binary | std::ios::in);
        if (!mFile.good())
        {
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
        std::string strMsg(msg);
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kVERBOSE:
        {
            debug(msg);
            break;
        }
        case nvinfer1::ILogger::Severity::kERROR:
        {
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

    void debug(std::string const& msg)
    {
        if (_minSeverity >= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << "[DEBUG]: " << msg << std::endl;
        }
    }

    void warning(std::string const& msg)
    {
        if (_minSeverity >= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cerr << "[WARNING]: " << msg << std::endl;
        }
    }

    void error(std::string const& msg)
    {
        if (_minSeverity >= nvinfer1::ILogger::Severity::kERROR)
        {
            std::cerr << "[ERROR]: " << msg << std::endl;
        }
    }

    void info(std::string const& msg)
    {
        if (_minSeverity >= nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << "[INFO]: " << msg << std::endl;
        }
    }

    void setLevel(nvinfer1::ILogger::Severity minSeverity)
    {
        _minSeverity = minSeverity;
    }

    nvinfer1::ILogger::Severity getLevel()
    {
        return _minSeverity;
    }

private:
    nvinfer1::ILogger::Severity _minSeverity = nvinfer1::ILogger::Severity::kVERBOSE;
};

inline Logger gLogger{};

#define LOG_DEBUG(message)                                                                                             \
    if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kVERBOSE)                                                   \
    {                                                                                                                  \
        gLogger.debug(message);                                                                                        \
    }
#define LOG_INFO(message)                                                                                              \
    if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kINFO)                                                      \
    {                                                                                                                  \
        gLogger.info(message);                                                                                         \
    }
#define LOG_ERROR(message)                                                                                             \
    if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kERROR)                                                     \
    {                                                                                                                  \
        gLogger.error(message);                                                                                        \
    }
#define LOG_WARNING(message)                                                                                           \
    if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kWARNING)                                                   \
    {                                                                                                                  \
        gLogger.warning(message);                                                                                      \
    }

#endif