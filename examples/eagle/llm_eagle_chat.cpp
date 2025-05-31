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
#include "common/json.h"
#include "decoder/decoder.h"
#include "eagle/eagle.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

struct LLMEagleChatArgs
{
    bool help{false};
    std::string enginePath;
    std::string eagleEnginePath;
    std::string tokenizerPath;
    int maxLength{256};
    bool debug{false};
    bool isEagle3{false};
    std::string inputString;
    int32_t topK{10};
    int32_t maxPathLen{6};
    int32_t maxDecodingTokens{60};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-e or --enginePath=<path to TensorRT engine>] [-g or --eagleEnginePath=<path to Eagle "
                 "TensorRT engine>] [-s or "
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
              << " [-i or --inputString=<input string>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --eagleEnginePath Provide the input Eagle engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 256"
              << std::endl;
    std::cerr << "  --inputString          Provide the input string for generation. If not provided, will enter "
                 "interactive mode."
              << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more information." << std::endl;
    std::cerr << "  --isEagle3       Use Eagle3 algorithm. Default is Eagle2." << std::endl;
    std::cerr << "  --maxDecodingTokens Provide the maximum decoding tokens for target model, the number provided must "
                 "be aligned with building phase. Default = 60"
              << std::endl;
    std::cerr << "  --topK           Provide the topK for draft model to select the topK candidates. the number "
                 "provided must be aligned with building phase. Default is 10."
              << std::endl;
    std::cerr << "  --maxPathLen     Provide the max stack layers for draft model to construct the max tree path "
                 "length. the number provided must be aligned with building phase. Default is 6."
              << std::endl;
};

bool parseLLMEagleChatArgs(LLMEagleChatArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"enginePath", required_argument, 0, 'e'},
        {"tokenizerPath", required_argument, 0, 't'}, {"maxLength", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {"eagleEnginePath", required_argument, 0, 'g'},
        {"isEagle3", no_argument, 0, 'a'}, {"inputString", required_argument, 0, 'i'},
        {"topK", required_argument, 0, 'k'}, {"maxPathLen", required_argument, 0, 'p'},
        {"maxDecodingTokens", required_argument, 0, 'm'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "he:t:s:dga:i:k:p:m:", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
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
        case 'g':
            if (optarg)
            {
                args.eagleEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --eagleEnginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 'd': args.debug = true; break;
        case 'a': args.isEagle3 = true; break;
        case 'i':
            if (optarg)
            {
                args.inputString = optarg;
            }
            break;
        case 'k':
            if (optarg)
            {
                args.topK = std::stoi(optarg);
            }
            break;
        case 'p':
            if (optarg)
            {
                args.maxPathLen = std::stoi(optarg);
            }
            break;
        case 'm':
            if (optarg)
            {
                args.maxDecodingTokens = std::stoi(optarg);
            }
            break;
        default: return false;
        }
    }
    return true;
}

void inference(std::vector<int64_t>& inputIds, Eagle<half>* eagle, std::vector<int32_t>& contextLengths,
    GenerationConfig& generationConfig, Tokenizer* tokenizer, bool isEagle3, int32_t batchSize,
    std::vector<int32_t>* newTokensNumbers = nullptr, std::vector<int32_t>* iterNumbers = nullptr)
{
    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    eagle->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId(), isEagle3, nullptr,
        newTokensNumbers, iterNumbers);
    for (int i = 0; i < batchSize; ++i)
    {
        std::cout << "Output for batch " << i << ": " << tokenizer->decode(outputIds[i]) << std::endl;
    }
    // Reset the values
    std::fill(inputIds.begin(), inputIds.end(), 0);
    std::fill(contextLengths.begin(), contextLengths.end(), 0);
}

int main(int argc, char* argv[])
{
    LLMEagleChatArgs args;
    if ((argc < 2) || (!parseLLMEagleChatArgs(args, argc, argv)))
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
    auto baseDecoder = std::make_unique<Decoder<half>>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    baseDecoder->setup(args.enginePath, stream, true, 1, true);

    auto draftDecoder = std::make_unique<Decoder<half>>();
    draftDecoder->setup(args.eagleEnginePath, stream, true, 1, true);

    int32_t maxContextLength = static_cast<int32_t>(baseDecoder->getMaxContextLength()) + 1; // for eagle
    int64_t batchSize = baseDecoder->getModelBatchSize();
    if (batchSize != 1)
    {
        printf("Eagle only supports batch size 1 currently!\n");
        return EXIT_FAILURE;
    }

    auto eagle = new Eagle<half>(std::move(baseDecoder), std::move(draftDecoder), stream, args.eagleEnginePath,
        args.maxPathLen, args.topK, args.isEagle3, args.maxDecodingTokens);

    std::vector<int64_t> inputIds(batchSize);
    std::vector<int32_t> contextLengths(batchSize, 0);
    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};

    bool isInteractive = args.inputString.empty();
    if (isInteractive)
    {
        std::string quitString = "quit";
        std::cout << "Welcome to NVIDIA DriveOS LLM SDK! Please enter your prompts. Enter quit to exit the program."
                  << std::endl;
        while (true)
        {
            std::string inputString;
            std::getline(std::cin, inputString);
            if (inputString == quitString)
            {
                std::cout << "Exit. Thanks for using DriveOS LLM SDK!" << std::endl;
                return EXIT_SUCCESS;
            }
            inputIds = tokenizer->encode(inputString, true);
            int32_t inputSize = static_cast<int32_t>(inputIds.size());
            if (inputSize > maxContextLength)
            {
                std::cout << "Warning: input length > max context length. The last tokens will be truncated."
                          << std::endl;
            }
            contextLengths[0] = inputSize;

            inference(inputIds, eagle, contextLengths, generationConfig, tokenizer.get(), args.isEagle3, batchSize);
        }
    }
    else
    {
        inputIds = tokenizer->encode(args.inputString, true);
        int32_t inputSize = static_cast<int32_t>(inputIds.size());
        if (inputSize > maxContextLength)
        {
            std::cout << "Warning: input length > max context length. The last tokens will be truncated." << std::endl;
        }
        contextLengths[0] = inputSize;
        inference(inputIds, eagle, contextLengths, generationConfig, tokenizer.get(), args.isEagle3, batchSize);
    }

    return 0;
};