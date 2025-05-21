/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// #include "common/trtUtils.h"
#include "decoder/decoder.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <dlfcn.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

struct LLMChatArgs
{
    bool help{false};
    bool interactive{false};
    std::vector<std::string> inputStrings;
    std::string enginePath;
    std::string tokenizerPath;
    int maxLength{256};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-i or --interactive] [-e or --enginePath=<path to TensorRT engine>] [-s or "
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>] [--inputString=<input string for "
                 "one batch>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --interactive    Interactive chat mode. " << std::endl;
    std::cerr << "  --inputString    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 256"
              << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more information." << std::endl;
};

bool parseLLMChatArgs(LLMChatArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"interactive", no_argument, 0, 'i'},
        {"inputString", required_argument, 0, 'c'}, {"enginePath", required_argument, 0, 'e'},
        {"tokenizerPath", required_argument, 0, 't'}, {"maxLength", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "he:t:s:di", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i': args.interactive = true; break;
        case 'c':
            if (optarg)
            {
                args.inputStrings.emplace_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                return false;
            }
            break;
        case 'e':
            if (optarg)
            {
                args.enginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --enginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 't':
            if (optarg)
            {
                args.tokenizerPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --tokenizerPath requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            break;
        case 'd': args.debug = true; break;
        default: return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    LLMChatArgs args;
    if ((argc < 2) || (!parseLLMChatArgs(args, argc, argv)))
    {
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

    auto pluginHandles = loadPlugins();

    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.tokenizerPath);
    auto decoder = std::make_unique<Decoder<half>>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    decoder->setup(args.enginePath, stream);
    int32_t maxContextLength = static_cast<int32_t>(decoder->getMaxContextLength());
    int64_t batchSize = decoder->getModelBatchSize();
    // int64_t batchCount = 0; UNUSED
    int64_t padId = tokenizer->getPadId();
    std::vector<int64_t> inputIds(batchSize * maxContextLength, padId);
    std::vector<int32_t> contextLengths(batchSize, 0);
    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};
    std::string quitString = "quit";
    std::cout << "Welcome to NVIDIA DriveOS LLM SDK! Please enter your prompts. Enter quit to exit the program."
              << std::endl;

    if (args.interactive)
    { // interactive mode
        while (true)
        {
            for (int64_t i = 0; i < batchSize; ++i)
            {
                std::string inputString;
                std::cout << "Prompt for batch " << i << ": ";
                std::getline(std::cin, inputString);
                if (inputString == quitString)
                {
                    std::cout << "Exit. Thanks for using DriveOS LLM SDK!" << std::endl;
                    return EXIT_SUCCESS;
                }
                std::vector<int64_t> batchInputIds = tokenizer->encode(inputString, true);
                int32_t inputSize = static_cast<int32_t>(batchInputIds.size());
                if (inputSize > maxContextLength)
                {
                    std::cout << "Warning: input length > max context length. The last tokens will be truncated."
                              << std::endl;
                }
                contextLengths[i] = std::min(inputSize, maxContextLength);
                batchInputIds.resize(maxContextLength, padId);
                std::copy(batchInputIds.begin(), batchInputIds.end(), inputIds.begin() + i * maxContextLength);
            }
            std::vector<std::vector<int64_t>> outputIds(batchSize);
            for (int i = 0; i < batchSize; ++i)
            {
                outputIds[i].reserve(generationConfig.maxLength);
            }
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());
            for (int i = 0; i < batchSize; ++i)
            {
                std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
            }
            // Reset the values
            std::fill(inputIds.begin(), inputIds.end(), padId);
            std::fill(contextLengths.begin(), contextLengths.end(), 0);
        }
        return EXIT_FAILURE;
    }
    // non-interactive mode
    if (args.inputStrings.size() != static_cast<size_t>(batchSize))
    {
        std::cerr << "Error: Number of input strings (" << args.inputStrings.size()
                  << ") must match the model's batch size (by --batchSize)" << batchSize << "). Please provide exactly "
                  << batchSize << " input string(s) using --inputString flag." << std::endl;
        return EXIT_FAILURE;
    }

    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::string inputString = args.inputStrings[i];
        std::cout << "Input string for batch: " << i << ": " << inputString << std::endl;

        std::vector<int64_t> batchInputIds = tokenizer->encode(inputString, true);
        int32_t inputSize = static_cast<int32_t>(batchInputIds.size());
        if (inputSize > maxContextLength)
        {
            std::cout << "Warning: input length > max context length. The last tokens will be truncated." << std::endl;
        }
        contextLengths[i] = std::min(inputSize, maxContextLength);
        batchInputIds.resize(maxContextLength, padId);
        std::copy(batchInputIds.begin(), batchInputIds.end(), inputIds.begin() + i * maxContextLength);
    }
    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());
    for (int i = 0; i < batchSize; ++i)
    {
        std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
    }
    // Reset the values
    std::fill(inputIds.begin(), inputIds.end(), padId);
    std::fill(contextLengths.begin(), contextLengths.end(), 0);
    return EXIT_SUCCESS;
};