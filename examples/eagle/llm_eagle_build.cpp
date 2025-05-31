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

#include "NvOnnxParser.h"
#include "common/common.h"
#include "common/cudaUtils.h"
#include "common/json.h"
#include "common/trtUtils.h"
#include <NvInfer.h>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

struct LLMEagleBuildArgs
{
    bool help{false};
    std::string onnxPath;
    std::string enginePath;
    int64_t batchSize{1};
    int64_t maxInputLen{128};
    int64_t maxSeqLen{4096};
    bool dynamicShape{true};
    bool debug{false};
    int64_t maxBatchSize{1};
    bool isEagleBase{false};
    bool isEagleDraft{false};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--onnxPath str> <--enginePath str> [-b or "
                 "--batchSize int] [-c or --maxInputLen int] [-s or --maxSeqLen int] [--dynamicShape] [--maxBatchSize "
                 "int] [--debug] [-e or --eagleBase] [-g or --eagleDraft] [-a or --eagle3] [-m or --mMaxDecodingTokens]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --onnxPath       Provide the input onnx file path. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the output TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize      Provide the desired batch_size for builder. Default = 1" << std::endl;
    std::cerr << "  --maxBatchSize   Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --maxInputLen    Provide the maximum input length for the model. Default = 128" << std::endl;
    std::cerr
        << "  --maxSeqLen      Provide the maximum output length for the model (including the input). Default = 4096"
        << std::endl;
    std::cerr << "  --maxDecodingTokens Provide the maximum decoding tokens for target model. Default = 60"
              << std::endl;
    std::cerr << "  --dynamicShape   Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more logs." << std::endl;
}

bool parseLLMEagleBuildArgs(LLMEagleBuildArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"onnxPath", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'o'}, {"batchSize", required_argument, 0, 'b'},
        {"maxInputLen", required_argument, 0, 'c'}, {"maxSeqLen", required_argument, 0, 's'},
        {"eagleDraft", no_argument, 0, 'g'}, {"eagleBase", no_argument, 0, 'e'}, {"debug", no_argument, 0, 'd'},
        {"eagle3", no_argument, 0, 'a'}, {"dynamicShape", no_argument, 0, 'y'},
        {"maxBatchSize", required_argument, 0, 'B'}, {"imageTokens", required_argument, 0, 0},
        {"minImageTokens", required_argument, 0, 0}, {"maxImageTokens", required_argument, 0, 0},
        {"modelType", required_argument, 0, 0}, {"maxDecodingTokens", required_argument, 0, 0}, {0, 0, 0, 0}};

    int opt;
    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "hi:o:B:b:c:s:dygeam:", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.onnxPath = optarg;
            }
            else
            {
                LOG_ERROR("--onnxPath requires option argument.");
                return false;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.enginePath = optarg;
            }
            else
            {
                LOG_ERROR("--enginePath requires option argument.");
                return false;
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batchSize = std::stoi(optarg);
            }
            break;
        case 'B':
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case 'c':
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case 's':
            if (optarg)
            {
                args.maxSeqLen = std::stoi(optarg);
            }
            break;
        case 'm':
            if (optarg)
            {
                args.maxDecodingTokens = std::stoi(optarg);
            }
            break;
        case 'g': args.isEagleDraft = true; break;
        case 'e': args.isEagleBase = true; break;
        case 'a': args.isEagle3 = true; break;
        case 'd': args.debug = true; break;
        case 'y': args.dynamicShape = true; break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

std::string generateTRTExecCommand(
    LLMEagleBuildArgs const& args, int64_t const& numKVHeads, int64_t const& hiddenSizePerHead)
{
    int64_t minInputLen = args.maxInputLen;
    int64_t optInputLen = args.maxInputLen;
    int64_t maxInputLen = args.maxInputLen;
    int64_t optBatchSize = args.batchSize;
    int64_t minBatchSize, maxBatchSize;
    if (args.dynamicShape)
    {
        minInputLen = 1;
        optInputLen = maxInputLen / 2;
        minBatchSize = 1;
        maxBatchSize = args.maxBatchSize;
    }
    else
    {
        minBatchSize = args.batchSize;
        maxBatchSize = args.batchSize;
    }

    int64_t maxLength = args.maxSeqLen;

    std::string trtExecCommand = fmtstr(
        "Equivalent trtexec command: trtexec --onnx=%s --saveEngine=%s --staticPlugins=${ATTENTION_PLUGIN_PATH} "
        "--stronglyTyped "
        "--verbose "
        "--profile=0 "
        "--minShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--optShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--maxShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--profile=1 "
        "--minShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx%ldx%ld "
        "--optShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx%ldx%ld "
        "--maxShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx%ldx%ld",
        args.onnxPath.c_str(), args.enginePath.c_str(), minBatchSize, minInputLen, minBatchSize, minBatchSize,
        minBatchSize, numKVHeads, hiddenSizePerHead, optBatchSize, optInputLen, optBatchSize, optBatchSize,
        optBatchSize, numKVHeads, hiddenSizePerHead, maxBatchSize, maxInputLen, maxBatchSize, maxBatchSize,
        maxBatchSize, numKVHeads, hiddenSizePerHead, minBatchSize, minBatchSize, minBatchSize, minBatchSize, numKVHeads,
        maxLength, hiddenSizePerHead, optBatchSize, optBatchSize, optBatchSize, optBatchSize, numKVHeads, maxLength,
        hiddenSizePerHead, maxBatchSize, maxBatchSize, maxBatchSize, maxBatchSize, numKVHeads, maxLength,
        hiddenSizePerHead);
    return trtExecCommand;
}

int main(int argc, char** argv)
{
    LLMEagleBuildArgs args;
    if ((argc < 2) || (!parseLLMEagleBuildArgs(args, argc, argv)))
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

    auto pluginHandles = loadPlugins();

    // Create the builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        LOG_ERROR("Failed to create builder.");
        return EXIT_FAILURE;
    }

    // Create the network definition
    auto const stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network)
    {
        LOG_ERROR("Failed to create network.");
        return EXIT_FAILURE;
    }

    // Create the ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        LOG_ERROR("Failed to create parser.");
        return EXIT_FAILURE;
    }

    // Parse the ONNX model
    if (!parser->parseFromFile(args.onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", args.onnxPath.c_str());
        return EXIT_FAILURE;
    }

    // Build the engine
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config.");
        return EXIT_FAILURE;
    }
    if (!(args.isEagleDraft || args.isEagleBase))
    {
        LOG_ERROR("Model is not Eagle model");
        return EXIT_FAILURE;
    }

    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    drivellm::JsonRoot root;
    std::string onnxFolderPath = extractFolderName(args.onnxPath);
    std::string json_path = onnxFolderPath + "/config.json";
    root.parseFromPath(json_path);
    auto rootNode = root.getRoot();
    auto hiddenSizeDim = rootNode["hidden_size"].getInteger();
    auto targetModelOutputHiddenDim = args.isEagle3 ? hiddenSizeDim * 3 : hiddenSizeDim;
    int64_t numKVHeads = rootNode["num_key_value_heads"].getInteger();
    int64_t numAttentionHeads = rootNode["num_attention_heads"].getInteger();
    int64_t hiddenSizePerHead = hiddenSizeDim / numAttentionHeads;
    int32_t nbKVCacheInputs = rootNode["num_hidden_layers"].getInteger();
    bool result = true;
    int64_t optBatchSize = args.batchSize;
    int64_t minBatchSize, maxBatchSize;
    if (args.dynamicShape)
    {
        minBatchSize = 1;
        maxBatchSize = args.maxBatchSize;
    }
    else
    {
        minBatchSize = args.batchSize;
        maxBatchSize = args.batchSize;
    }

    int const mMaxDecodingTokens = args.maxDecodingTokens;
    if (args.isEagleDraft or args.isEagleBase)
    {
        result &= setOptimizationProfile(contextProfile, "input_ids", createDims({minBatchSize, 1}),
            createDims({optBatchSize, args.maxInputLen / 2}), createDims({maxBatchSize, args.maxInputLen}));

        result &= setOptimizationProfile(generationProfile, "input_ids", createDims({minBatchSize, 1}),
            createDims({optBatchSize, mMaxDecodingTokens / 2}), createDims({maxBatchSize, mMaxDecodingTokens}));
    }

    result &= setOptimizationProfile(contextProfile, "context_lengths", createDims({minBatchSize}),
        createDims({optBatchSize}), createDims({maxBatchSize}));

    result &= setOptimizationProfile(generationProfile, "context_lengths", createDims({minBatchSize}),
        createDims({optBatchSize}), createDims({maxBatchSize}));

    if (args.isEagleDraft)
    {
        if (args.isEagle3)
        {
            result &= setOptimizationProfile(contextProfile, "hidden_states_from_draft",
                createDims({minBatchSize, 1, hiddenSizeDim}),
                createDims({optBatchSize, args.maxInputLen / 2, hiddenSizeDim}),
                createDims({maxBatchSize, args.maxInputLen, hiddenSizeDim}));
            result &= setOptimizationProfile(generationProfile, "hidden_states_from_draft",
                createDims({minBatchSize, 1, hiddenSizeDim}),
                createDims({optBatchSize, mMaxDecodingTokens / 2, hiddenSizeDim}),
                createDims({maxBatchSize, mMaxDecodingTokens, hiddenSizeDim}));

            result &= setOptimizationProfile(contextProfile, "hidden_states_input",
                createDims({minBatchSize, 1, targetModelOutputHiddenDim}),
                createDims({optBatchSize, args.maxInputLen / 2, targetModelOutputHiddenDim}),
                createDims({maxBatchSize, args.maxInputLen, targetModelOutputHiddenDim}));
            result &= setOptimizationProfile(generationProfile, "hidden_states_input",
                createDims({minBatchSize, 1, targetModelOutputHiddenDim}),
                createDims({optBatchSize, mMaxDecodingTokens / 2, targetModelOutputHiddenDim}),
                createDims({maxBatchSize, mMaxDecodingTokens, targetModelOutputHiddenDim}));
        }
        else
        {

            result &= setOptimizationProfile(contextProfile, "hidden_states_input",
                createDims({minBatchSize, 1, hiddenSizeDim}),
                createDims({optBatchSize, args.maxInputLen / 2, hiddenSizeDim}),
                createDims({maxBatchSize, args.maxInputLen, hiddenSizeDim}));
            result &= setOptimizationProfile(generationProfile, "hidden_states_input",
                createDims({minBatchSize, 1, hiddenSizeDim}),
                createDims({optBatchSize, mMaxDecodingTokens / 2, hiddenSizeDim}),
                createDims({maxBatchSize, mMaxDecodingTokens, hiddenSizeDim}));
        }

        result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({1}),
            createDims({args.maxInputLen / 2}), createDims({args.maxInputLen}));
        result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({1}),
            createDims({mMaxDecodingTokens / 2}), createDims({mMaxDecodingTokens}));
    }
    else if (args.isEagleBase)
    {
        result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({1}),
            createDims({args.maxInputLen / 2}), createDims({args.maxInputLen}));
        result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({1}),
            createDims({mMaxDecodingTokens / 2}), createDims({mMaxDecodingTokens}));
    }

    if (args.isEagleDraft or args.isEagleBase)
    {
        const int32_t attn_mask_align_size = 32;
        result &= setOptimizationProfile(contextProfile, "attention_mask", createDims({minBatchSize, 1, 1}),
            createDims({optBatchSize, 1, 1}), createDims({maxBatchSize, 1, 1}));
        result &= setOptimizationProfile(generationProfile, "attention_mask", createDims({minBatchSize, 1, 1}),
            createDims({optBatchSize, mMaxDecodingTokens / 2,
                divUp(mMaxDecodingTokens / 2, attn_mask_align_size) * attn_mask_align_size}),
            createDims({maxBatchSize, mMaxDecodingTokens,
                divUp(mMaxDecodingTokens, attn_mask_align_size) * attn_mask_align_size}));
        result &= setOptimizationProfile(contextProfile, "attention_pos_id", createDims({minBatchSize, 1}),
            createDims({optBatchSize, 1}), createDims({maxBatchSize, 1}));
        result &= setOptimizationProfile(generationProfile, "attention_pos_id", createDims({minBatchSize, 1}),
            createDims({optBatchSize, mMaxDecodingTokens / 2}), createDims({maxBatchSize, mMaxDecodingTokens}));
    }

    nvinfer1::Dims minKVContextShape = createDims({minBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    nvinfer1::Dims optKVContextShape = createDims({optBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    nvinfer1::Dims maxKVContextShape = createDims({maxBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    nvinfer1::Dims minKVGenerationShape = createDims({minBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});
    nvinfer1::Dims optKVGenerationShape = createDims({optBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});
    nvinfer1::Dims maxKVGenerationShape = createDims({maxBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});

    for (int i = 0; i < nbKVCacheInputs; ++i)
    {
        result &= setOptimizationProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), minKVContextShape,
            optKVContextShape, maxKVContextShape);
        result &= setOptimizationProfile(generationProfile, fmtstr("past_key_values.%d", i).c_str(),
            minKVGenerationShape, optKVGenerationShape, maxKVGenerationShape);
    }

    if (!result)
    {
        LOG_ERROR("Issues setting up optimization profile");
        return EXIT_FAILURE;
    }

    config->addOptimizationProfile(contextProfile);
    config->addOptimizationProfile(generationProfile);
    config->setFlag(nvinfer1::BuilderFlag::kMONITOR_MEMORY);
    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine)
    {
        LOG_ERROR("Failed to build engine.");
        return EXIT_FAILURE;
    }
    std::string folderPath = extractFolderName(args.enginePath);
    if (folderPath != "")
    {
        if (!std::filesystem::exists(folderPath))
        {
            if (std::filesystem::create_directories(folderPath))
            {
                LOG_INFO("Created directory %s for saving LLM engine.", folderPath.c_str());
            }
            else
            {
                LOG_INFO("Failed to create directory %s for saving LLM engine.", folderPath.c_str());
            }
        }
    }
    std::ofstream ofs(args.enginePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", args.enginePath.c_str());
        return EXIT_FAILURE;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", args.enginePath.c_str());
    // Copy config.json to engine path
    auto configPath = extractFolderName(args.onnxPath) + "/config.json";
    std::string targetConfigPath = extractFolderName(args.enginePath) + "/config.json";
    copyFile(configPath, targetConfigPath);

    if (args.isEagle3 && args.isEagleDraft)
    {
        // Copy d2t.bin to enginePath if it exists
        std::string d2tPath = extractFolderName(args.onnxPath) + "/d2t.bin";
        std::string targettD2tPath = extractFolderName(args.enginePath) + "/d2t.bin";
        copyFile(d2tPath, targettD2tPath);
    }
    LOG_INFO(generateTRTExecCommand(args, numKVHeads, hiddenSizePerHead).c_str());
    return EXIT_SUCCESS;
}