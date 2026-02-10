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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Audio encoder builder for Qwen3-Omni
 * Builds TensorRT engine from ONNX audio encoder model
 */

#include "builder/audioBuilder.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

struct AudioBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};
    int64_t minTimeSteps{100};
    int64_t maxTimeSteps{6000};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--onnxDir str> <--engineDir str> [--debug] "
                 "[--minTimeSteps int] [--maxTimeSteps int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr << "  --onnxDir            Directory containing audio encoder ONNX file (model.onnx). Required."
              << std::endl;
    std::cerr << "  --engineDir          Base output directory for audio encoder. Required." << std::endl;
    std::cerr << "                       Note: Engine will be saved to <engineDir>/audio/" << std::endl;
    std::cerr << "  --debug              Use debug mode with verbose output" << std::endl;
    std::cerr << "  --minTimeSteps       Minimum audio time steps. Default = 100 (~0.64s audio)" << std::endl;
    std::cerr << "  --maxTimeSteps       Maximum audio time steps. Default = 6000 (~38.4s audio)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Time steps calculation (with hop_length=160, sample_rate=16000):" << std::endl;
    std::cerr << "  Audio duration (seconds) = (time_steps * 160) / 16000" << std::endl;
    std::cerr << "  Example: 290 steps = 1.86s, 1000 steps = 6.4s, 6000 steps = 38.4s" << std::endl;
}

bool parseAudioBuildArgs(AudioBuildArgs& args, int argc, char* argv[])
{
    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
        {"onnxDir", required_argument, nullptr, 'o'}, {"engineDir", required_argument, nullptr, 'e'},
        {"debug", no_argument, nullptr, 'd'}, {"minTimeSteps", required_argument, nullptr, 'm'},
        {"maxTimeSteps", required_argument, nullptr, 'M'}, {nullptr, 0, nullptr, 0}};

    int optionIndex = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "ho:e:dm:M:", longOptions, &optionIndex)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'o': args.onnxDir = optarg; break;
        case 'e': args.engineDir = optarg; break;
        case 'd': args.debug = true; break;
        case 'm': args.minTimeSteps = std::stoll(optarg); break;
        case 'M': args.maxTimeSteps = std::stoll(optarg); break;
        default:
            std::cerr << "Error: Invalid argument" << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }

    if (args.onnxDir.empty() || args.engineDir.empty())
    {
        std::cerr << "Error: --onnxDir and --engineDir are required" << std::endl;
        printUsage(argv[0]);
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    AudioBuildArgs args;
    if ((argc < 2) || (!parseAudioBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    // Validate input directory and required files
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_WARNING("config.json not found in onnx directory: %s. Using default parameters.", args.onnxDir.c_str());
    }
    configFile.close();

    std::string actualEngineDir = args.engineDir + "/audio";

    LOG_INFO("Building audio encoder for model in: %s", args.onnxDir.c_str());
    LOG_INFO("Output engine directory: %s", actualEngineDir.c_str());
    LOG_INFO("Time steps range: [%ld, %ld]", args.minTimeSteps, args.maxTimeSteps);

    // Create AudioBuilderConfig from args
    builder::AudioBuilderConfig config;
    config.minTimeSteps = args.minTimeSteps;
    config.maxTimeSteps = args.maxTimeSteps;

    // Create and run the builder
    builder::AudioBuilder audioBuilder(args.onnxDir, actualEngineDir, config);
    if (!audioBuilder.build())
    {
        LOG_ERROR("Failed to build Audio engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("Audio engine built successfully.");
    return EXIT_SUCCESS;
}
