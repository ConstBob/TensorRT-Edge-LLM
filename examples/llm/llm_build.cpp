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

#include "builder/builder.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace drivellm;

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    int64_t maxInputLen{128};
    int64_t maxSeqLen{4096};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // Default to 0 means no LoRA
    bool eagleDraft{false};
    bool eagleBase{false};
    bool eagle2{false};
    int64_t maxDecodingTokens{60};
    int64_t maxDraftTokensPerStep{60};
    bool isVlm{false};
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
    bool enableReuseKVCache{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnxDir <dir> --engineDir <dir> [--maxInputLen <int>] "
                 "[--maxSeqLen <int>] [--maxBatchSize <int>] [--debug] [--maxLoraRank <int>]"
                 "[--eagleDraft] [--eagleBase] [--eagle2] [--maxDecodingTokens <int>] "
                 "[--maxDraftTokensPerStep <int>] [--vlm] [--minImageTokens <int>] [--maxImageTokens <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --onnxDir                 Provide the input ONNX directory path. Required. " << std::endl;
    std::cerr << "  --engineDir               Provide the output TensorRT engine directory path. Required. "
              << std::endl;
    std::cerr << "  --maxInputLen             Provide the maximum input length for the model. Default = 128"
              << std::endl;
    std::cerr << "  --maxSeqLen               Provide the maximum output length for the model (including the input). "
                 "Default = 4096"
              << std::endl;
    std::cerr << "  --maxBatchSize            Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --debug                   Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank             Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    std::cerr << "  --eagleDraft              Enable Eagle draft mode" << std::endl;
    std::cerr << "  --eagleBase               Enable Eagle base mode" << std::endl;
    std::cerr << "  --eagle2                  Enable Eagle2 mode" << std::endl;
    std::cerr << "  --maxDecodingTokens       Maximum decoding tokens for Eagle. Default = 60" << std::endl;
    std::cerr << "  --maxDraftTokensPerStep   Maximum draft tokens per step for Eagle. Default = 60" << std::endl;
    std::cerr << "  --vlm                     Enable VLM mode" << std::endl;
    std::cerr << "  --minImageTokens          Minimum image tokens for VLM. Default = 4" << std::endl;
    std::cerr << "  --maxImageTokens          Maximum image tokens for VLM. Default = 1024" << std::endl;
    std::cerr << "  --enableReuseKVCache      Enable KVCache reuse to speed up prefill execution. Default = false"
              << std::endl;
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, 701}, {"onnxDir", required_argument, 0, 702},
        {"engineDir", required_argument, 0, 703}, {"maxInputLen", required_argument, 0, 704},
        {"maxSeqLen", required_argument, 0, 705}, {"debug", no_argument, 0, 706},
        {"maxBatchSize", required_argument, 0, 707}, {"maxLoraRank", required_argument, 0, 708},
        {"eagleDraft", no_argument, 0, 709}, {"eagleBase", no_argument, 0, 710}, {"eagle2", no_argument, 0, 711},
        {"maxDecodingTokens", required_argument, 0, 712}, {"maxDraftTokensPerStep", required_argument, 0, 713},
        {"vlm", no_argument, 0, 714}, {"minImageTokens", required_argument, 0, 715},
        {"maxImageTokens", required_argument, 0, 716}, {"enableReuseKVCache", no_argument, 0, 717}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", buildOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case 701: args.help = true; return true;
        case 702:
            if (optarg)
            {
                args.onnxDir = optarg;
            }
            else
            {
                LOG_ERROR("--onnxDir requires option argument.");
                return false;
            }
            break;
        case 703:
            if (optarg)
            {
                args.engineDir = optarg;
            }
            else
            {
                LOG_ERROR("--engineDir requires option argument.");
                return false;
            }
            break;
        case 704:
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case 705:
            if (optarg)
            {
                args.maxSeqLen = std::stoi(optarg);
            }
            break;
        case 706: args.debug = true; break;
        case 707:
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case 708:
            if (optarg)
            {
                args.maxLoraRank = std::stoi(optarg);
            }
            break;
        case 709: args.eagleDraft = true; break;
        case 710: args.eagleBase = true; break;
        case 711: args.eagle2 = true; break;
        case 712:
            if (optarg)
            {
                args.maxDecodingTokens = std::stoi(optarg);
            }
            break;
        case 713:
            if (optarg)
            {
                args.maxDraftTokensPerStep = std::stoi(optarg);
            }
            break;
        case 714: args.isVlm = true; break;
        case 715:
            if (optarg)
            {
                args.minImageTokens = std::stoi(optarg);
            }
            break;
        case 716:
            if (optarg)
            {
                args.maxImageTokens = std::stoi(optarg);
            }
            break;
        case 717: args.enableReuseKVCache = true; break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    LLMBuildArgs args;
    if ((argc < 2) || (!parseLLMBuildArgs(args, argc, argv)))
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
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    // Create LLMBuilderConfig from args
    builder::LLMBuilderConfig config;
    config.maxInputLen = args.maxInputLen;
    config.maxSeqLen = args.maxSeqLen;
    config.maxBatchSize = args.maxBatchSize;
    config.maxLoraRank = args.maxLoraRank;
    config.enableReuseKVCache = args.enableReuseKVCache;
    config.eagleDraft = args.eagleDraft;
    config.eagleBase = args.eagleBase;
    config.eagle2 = args.eagle2;
    config.maxDecodingTokens = args.maxDecodingTokens;
    config.maxDraftTokensPerStep = args.maxDraftTokensPerStep;
    config.isVlm = args.isVlm;
    config.minImageTokens = args.minImageTokens;
    config.maxImageTokens = args.maxImageTokens;

    // Create and run the builder
    builder::LLMBuilder llmBuilder(args.onnxDir, args.engineDir, config);
    if (!llmBuilder.build())
    {
        LOG_ERROR("Failed to build LLM engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("LLM engine built successfully.");
    return EXIT_SUCCESS;
}
