#pragma once

#include "common.h"
#include <NvInferRuntime.h>
#include <string>

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

#define LOG_DEBUG(...)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kVERBOSE)                                               \
        {                                                                                                              \
            gLogger.debug(fmtstr(__VA_ARGS__));                                                                        \
        }                                                                                                              \
    } while (0)
#define LOG_INFO(...)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kINFO)                                                  \
        {                                                                                                              \
            gLogger.info(fmtstr(__VA_ARGS__));                                                                         \
        }                                                                                                              \
    } while (0)
#define LOG_ERROR(...)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kERROR)                                                 \
        {                                                                                                              \
            gLogger.error(fmtstr(__VA_ARGS__));                                                                        \
        }                                                                                                              \
    } while (0)
#define LOG_WARNING(...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kWARNING)                                               \
        {                                                                                                              \
            gLogger.warning(fmtstr(__VA_ARGS__));                                                                      \
        }                                                                                                              \
    } while (0)
