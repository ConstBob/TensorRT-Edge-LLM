/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "stringUtils.h"
#include <NvInferRuntime.h>
#include <iostream>
#include <string>

namespace drivellm
{

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    Logger() {};
    ~Logger() {};
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
            gLogger.debug(format::fmtstr(__VA_ARGS__));                                                                \
        }                                                                                                              \
    } while (0)
#define LOG_INFO(...)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kINFO)                                                  \
        {                                                                                                              \
            gLogger.info(format::fmtstr(__VA_ARGS__));                                                                 \
        }                                                                                                              \
    } while (0)
#define LOG_ERROR(...)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kERROR)                                                 \
        {                                                                                                              \
            gLogger.error(format::fmtstr(__VA_ARGS__));                                                                \
        }                                                                                                              \
    } while (0)
#define LOG_WARNING(...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (gLogger.getLevel() >= nvinfer1::ILogger::Severity::kWARNING)                                               \
        {                                                                                                              \
            gLogger.warning(format::fmtstr(__VA_ARGS__));                                                              \
        }                                                                                                              \
    } while (0)

} // namespace drivellm
