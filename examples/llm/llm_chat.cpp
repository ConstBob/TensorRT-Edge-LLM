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

#include "common/common.h"
#include "common/trtUtils.h"
#include "decoder/decoder.h"
#include "eagle/eagle.h"
#include "engine/llm_engine.h"
#include "llm_param.h"
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
    bool interactive{false};
    std::vector<std::string> inputStrings;
    BaseParams baseParams;
    EagleParams eagleParams;
    int maxLength{256};
    LoraWeights loraWeights;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--interactive] [--engineDir=<path to TensorRT engine directory>] "
                 "[--maxLength=<int>] [--inputString=<input string for "
                 "one batch>] [--loraWeights=<name:path>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --interactive    Interactive chat mode. " << std::endl;
    std::cerr << "  --inputString    Provide the input string to the runtime. Required in non-interactive mode. "
              << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 256"
              << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printLoraOptions();
};

bool parseLLMChatArgs(LLMChatArgs& args, int argc, char* argv[])
{
    static struct option chatOptions[]
        = {{"interactive", no_argument, 0, 901}, {"inputString", required_argument, 0, 902},
            {"maxLength", required_argument, 0, 903}, {"loraWeights", required_argument, 0, 904}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::eagleOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleOptions[i];
    for (int i = 0; chatOptions[i].name != 0; ++i)
        long_options[idx++] = chatOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1)
    {
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg))
        {
            continue;
        }

        if (CommonOptions::parseEagleOptions(args.eagleParams, opt, optarg))
        {
            continue;
        }

        switch (opt)
        {
        case 901: args.interactive = true; break;
        case 902:
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
        case 903:
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            break;
        case 904:
            if (optarg)
            {
                if (LoraWeights::validateFormat(optarg))
                {
                    auto loraPair = LoraWeights::parse(optarg);
                    args.loraWeights.add(loraPair.first, loraPair.second);
                }
                else
                {
                    return false;
                }
            }
            break;
        default: return false;
        }
    }

    // Validate LoRA weights in non-interactive mode
    if (!args.interactive && args.loraWeights.weights.size() > 1)
    {
        std::cerr << "ERROR: Only one LoRA weight is allowed in non-interactive mode" << std::endl;
        return false;
    }

    return true;
}

void interactiveLoraSelection(std::unique_ptr<Decoder>& decoder)
{
    if (decoder->getLoraNames().size() > 1)
    {
        std::cout << "\nAvailable LoRA weights:" << std::endl;
        for (auto const& name : decoder->getLoraNames())
        {
            std::cout << "- " << name << std::endl;
        }
        std::cout << "Do you want to switch LoRA weights? (yes/no): ";
        std::string answer;
        std::getline(std::cin, answer);
        if (answer == "yes")
        {
            bool validChoice = false;
            while (!validChoice)
            {
                std::cout << "Enter LoRA name to switch to: ";
                std::string loraName;
                std::getline(std::cin, loraName);
                if (std::find(decoder->getLoraNames().begin(), decoder->getLoraNames().end(), loraName)
                    != decoder->getLoraNames().end())
                {
                    if (decoder->switchLora(loraName))
                    {
                        std::cout << "Switched to LoRA: " << loraName << std::endl;
                        validChoice = true;
                    }
                    else
                    {
                        std::cout << "Failed to switch to LoRA: " << loraName << std::endl;
                    }
                }
                else
                {
                    std::cout << "Invalid LoRA name. Please choose from the available options." << std::endl;
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    LLMChatArgs args;
    if ((argc < 2) || (!parseLLMChatArgs(args, argc, argv)))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.baseParams.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.baseParams.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    auto pluginHandles = loadEdgellmPluginLib();

    auto tokenizer = std::make_unique<Tokenizer>();
    // For EAGLE mode, load tokenizer from baseModelDir, otherwise from engineDir
    if (args.eagleParams.baseModelDir.empty() && args.eagleParams.draftModelDir.empty())
    {
        tokenizer->loadFromHF(args.baseParams.engineDir);
    }
    else
    {
        tokenizer->loadFromHF(args.eagleParams.baseModelDir);
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    EngineConfig engineConfig;
    if (args.eagleParams.baseModelDir.empty() && args.eagleParams.draftModelDir.empty())
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = EngineConfig(args.baseParams.engineDir, !args.baseParams.noCudaGraph);
    }
    else
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = EngineConfig(args.baseParams.engineDir, args.eagleParams.baseModelDir,
            args.eagleParams.draftModelDir, args.eagleParams.maxPathLen, args.eagleParams.topK,
            args.eagleParams.isEagle3, args.eagleParams.maxDecodingTokens, !args.baseParams.noCudaGraph);
    }
    auto llmEngine = std::make_unique<LLMEngine>(engineConfig, stream);
    auto const batchSize = llmEngine->getBatchSize();
    bool const eagleMode = llmEngine->isEagleModel();

    if (eagleMode && batchSize != 1)
    {
        printf("Eagle only supports batch size 1 currently!\n");
        return EXIT_FAILURE;
    }

    std::vector<int32_t> inputIds;
    std::vector<int32_t> contextLengths(batchSize, 0);

    int32_t padId = tokenizer->getPadId();
    GenerationConfig generationConfig{args.maxLength, 0, 1, 1};
    std::string quitString = "quit";
    std::cout << "Welcome to NVIDIA DriveOS LLM SDK! Please enter your prompts. Enter quit to exit the program."
              << std::endl;

    llmEngine->setupRopeCosSin();

    // Load LoRA weights
    if (!eagleMode)
    {
        auto& decoderPtr = llmEngine->getDecoder();
        // Load LoRA weights
        for (auto const& [name, path] : args.loraWeights.weights)
        {
            if (!decoderPtr->addLora(name, path))
            {
                std::cerr << "Failed to load LoRA weights: " << name << " from " << path << std::endl;
                if (!args.interactive)
                {
                    return EXIT_FAILURE;
                }
            }
        }
        // In non-interactive mode, switch to the specified LoRA if provided
        if (!args.interactive && !args.loraWeights.weights.empty())
        {
            if (!decoderPtr->switchLora(args.loraWeights.weights.begin()->first))
            {
                std::cerr << "Failed to switch to LoRA: " << args.loraWeights.weights.begin()->first << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    if (args.interactive)
    { // interactive mode
        while (true)
        {
            if (!eagleMode)
            {
                // Check if we should switch LoRA weights
                auto& decoderPtr = llmEngine->getDecoder();
                interactiveLoraSelection(decoderPtr);
            }
            std::vector<std::string> inputStrings;
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
                inputStrings.emplace_back(inputString);
            }
            inputIds = llmEngine->processInputSequence(inputStrings, tokenizer.get(), contextLengths, padId);
            std::vector<std::vector<int32_t>> outputIds(batchSize);
            for (int i = 0; i < batchSize; ++i)
            {
                outputIds[i].reserve(generationConfig.maxLength);
            }
            llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr,
                tokenizer.get(), true);
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
                  << ") must match the model's batch size (" << batchSize << "). Please provide exactly " << batchSize
                  << " input string(s) using --inputString flag." << std::endl;
        return EXIT_FAILURE;
    }

    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::string inputString = args.inputStrings[i];
        std::cout << "Input string for batch: " << i << ": " << inputString << std::endl;
    }

    inputIds = llmEngine->processInputSequence(args.inputStrings, tokenizer.get(), contextLengths, padId);
    std::vector<std::vector<int32_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    llmEngine->generate(
        inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer.get(), true);
    // Reset the values
    std::fill(inputIds.begin(), inputIds.end(), padId);
    std::fill(contextLengths.begin(), contextLengths.end(), 0);
    return EXIT_SUCCESS;
};