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
#include "common/logger.h"
#include "common/trtUtils.h"
#include "llm_param.h"
#include <NvInfer.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

struct ViTBuildArgs
{
    std::string visualOnnxPath;
    std::string visualEnginePath;
    bool dynamicShape{false};
    bool help{false};
    bool debug{false};
    VLMBuildParams vlmBuildParams;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--visualOnnxPath str> <--visualEnginePath str> [--dynamicShape]"
                 "[--imageTokens int] [--minImageTokens int] [--maxImageTokens int] [--modelType str] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr << "  --visualOnnxPath   Provide the input onnx file path for visual encoder. Required. " << std::endl;
    std::cerr << "  --visualEnginePath Provide the output TensorRT engine file path for visual encoder. Required. "
              << std::endl;
    std::cerr << "  --dynamicShape     Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug            Use debug mode, which outputs tensors." << std::endl;
    CommonUsage::printVLMBuildOptions();
}

bool parseViTBuildArgs(ViTBuildArgs& args, int argc, char* argv[])
{
    static struct option vitOptions[] = {{"help", no_argument, 0, 601}, {"visualOnnxPath", required_argument, 0, 602},
        {"visualEnginePath", required_argument, 0, 603}, {"dynamicShape", no_argument, 0, 604},
        {"debug", no_argument, 0, 605}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::vlmBuildOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::vlmBuildOptions[i];
    for (int i = 0; vitOptions[i].name != 0; ++i)
        long_options[idx++] = vitOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1)
    {
        if (CommonOptions::parseVLMBuildOptions(args.vlmBuildParams, opt, optarg))
        {
            continue;
        }

        switch (opt)
        {
        case 601: args.help = true; return true;
        case 602:
            if (optarg)
            {
                args.visualOnnxPath = optarg;
            }
            else
            {
                LOG_ERROR("--visualOnnxPath requires option argument.");
                return false;
            }
            break;
        case 603:
            if (optarg)
            {
                args.visualEnginePath = optarg;
            }
            else
            {
                LOG_ERROR("--visualEnginePath requires option argument.");
                return false;
            }
            break;
        case 604: args.dynamicShape = true; break;
        case 605: args.debug = true; break;
        default: LOG_ERROR("ERROR: Invalid Argument %c is %s", opt, optarg); return false;
        }
    }
    return true;
}

std::string generateViTTRTExecCommand(ViTBuildArgs const& args, int64_t const& patchSize, int64_t const& ropeEmbdSize,
    int64_t const& minHW, int64_t const& optHW, int64_t const& maxHW)
{
    std::string trtExecCommand
        = fmtstr("Equivalent ViT trtexec command: trtexec --onnx=%s --saveEngine=%s --stronglyTyped --verbose ",
            args.visualOnnxPath.c_str(), args.visualEnginePath.c_str());

    if (args.vlmBuildParams.modelType == "qwen2_vl")
    {
        trtExecCommand += fmtstr(
            "--minShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld "
            "--optShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld "
            "--maxShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld ",
            minHW, patchSize, minHW, ropeEmbdSize, minHW, minHW, optHW, patchSize, optHW, ropeEmbdSize, optHW, optHW,
            maxHW, patchSize, maxHW, ropeEmbdSize, maxHW, maxHW);
    }
    else if (args.vlmBuildParams.modelType == "qwen2_5_vl")
    {
        trtExecCommand += fmtstr(
            "--minShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld,window_attention_mask:1x%ldx%ld,"
            "window_index:%ld,reverse_window_index:%ld "
            "--optShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld,window_attention_mask:1x%ldx%ld,"
            "window_index:%ld,reverse_window_index:%ld "
            "--maxShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld,window_attention_mask:1x%ldx%ld,"
            "window_index:%ld,reverse_window_index:%ld ",
            minHW, patchSize, minHW, ropeEmbdSize, minHW, minHW, minHW, minHW, minHW / 4, minHW / 4, optHW, patchSize,
            optHW, ropeEmbdSize, optHW, optHW, optHW, optHW, optHW / 4, optHW / 4, maxHW, patchSize, maxHW,
            ropeEmbdSize, maxHW, maxHW, maxHW, maxHW, maxHW / 4, maxHW / 4);
    }
    else if (args.vlmBuildParams.modelType == "internvl3")
    {
        trtExecCommand += fmtstr(
            "--minShapes=input:%ldx%ld "
            "--optShapes=input:%ldx%ld "
            "--maxShapes=input:%ldx%ld ",
            minHW, patchSize, optHW, patchSize, maxHW, patchSize);
    }

    return trtExecCommand;
}

int buildViT(ViTBuildArgs const& args)
{
    int64_t minHW, optHW, maxHW;
    if (args.dynamicShape)
    {
        // In Qwen2-VL, HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
        minHW = args.vlmBuildParams.minImageTokens * 4;
        maxHW = args.vlmBuildParams.maxImageTokens * 4;
        // InternVL ViT model has a reshape which requires HW to be divisible by 1024
        if (args.vlmBuildParams.modelType == "internvl3")
        {
            optHW = (minHW / 1024 + maxHW / 1024) / 2 * 1024;
        }
        else
        {
            optHW = (minHW + maxHW) / 2;
        }
    }
    else
    {
        minHW = args.vlmBuildParams.imageTokens * 4;
        optHW = minHW;
        maxHW = minHW;
    }

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
    if (!parser->parseFromFile(args.visualOnnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", args.visualOnnxPath.c_str());
        return EXIT_FAILURE;
    }

    // Add optimization profile
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config.");
        return EXIT_FAILURE;
    }

    int32_t const nbInputs = network->getNbInputs();
    auto* visualProfile = builder->createOptimizationProfile();
    int64_t patchSize = 0;
    int64_t ropeEmbdSize = 0;
    for (int32_t i = 0; i < nbInputs; ++i)
    {
        if (strcmp(network->getInput(i)->getName(), "input") == 0)
        {
            patchSize = network->getInput(i)->getDimensions().d[1];
        }
        else if (strcmp(network->getInput(i)->getName(), "rotary_pos_emb") == 0)
        {
            ropeEmbdSize = network->getInput(i)->getDimensions().d[1];
        }
    }

    if ((patchSize == 0))
    {
        LOG_ERROR("Cannot infer patchSize. Do you have proper ONNX input: input?");
        return EXIT_FAILURE;
    }

    if ((args.vlmBuildParams.modelType == "qwen2_5_vl" || args.vlmBuildParams.modelType == "qwen2_vl")
        && (ropeEmbdSize == 0))
    {
        LOG_ERROR("Cannot infer ropeEmbdSize. Do you have proper ONNX input: rotary_pos_emb?");
        return EXIT_FAILURE;
    }

    // set dimensions for input tensors. Only context phase input can possibly be dynamic
    bool result = true;

    result &= setOptimizationProfile(visualProfile, "input", createDims({minHW, patchSize}),
        createDims({optHW, patchSize}), createDims({maxHW, patchSize}));

    if (args.vlmBuildParams.modelType == "qwen2_vl" || args.vlmBuildParams.modelType == "qwen2_5_vl")
    {
        result &= setOptimizationProfile(visualProfile, "rotary_pos_emb", createDims({minHW, ropeEmbdSize}),
            createDims({optHW, ropeEmbdSize}), createDims({maxHW, ropeEmbdSize}));
        result &= setOptimizationProfile(visualProfile, "attention_mask", createDims({1, minHW, minHW}),
            createDims({1, optHW, optHW}), createDims({1, maxHW, maxHW}));
    }

    if (args.vlmBuildParams.modelType == "qwen2_5_vl")
    {
        result &= setOptimizationProfile(visualProfile, "window_attention_mask", createDims({1, minHW, minHW}),
            createDims({1, optHW, optHW}), createDims({1, maxHW, maxHW}));
        result &= setOptimizationProfile(
            visualProfile, "window_index", createDims({minHW / 4}), createDims({optHW / 4}), createDims({maxHW / 4}));
        result &= setOptimizationProfile(visualProfile, "reverse_window_index", createDims({minHW / 4}),
            createDims({optHW / 4}), createDims({maxHW / 4}));
    }

    if (!result)
    {
        LOG_ERROR("Issues setting up optimization profile");
        return EXIT_FAILURE;
    }

    config->addOptimizationProfile(visualProfile);
    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine)
    {
        LOG_ERROR("Failed to build engine.");
        return EXIT_FAILURE;
    }
    std::string folderPath = extractFolderName(args.visualEnginePath);
    if (folderPath != "")
    {
        if (!std::filesystem::exists(folderPath))
        {
            if (std::filesystem::create_directories(folderPath))
            {
                LOG_INFO("Created directory %s for saving ViT engine.", folderPath.c_str());
            }
            else
            {
                LOG_INFO("Failed to create directory %s for saving ViT engine.", folderPath.c_str());
            }
        }
    }
    std::ofstream ofs(args.visualEnginePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", args.visualEnginePath.c_str());
        return EXIT_FAILURE;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", args.visualEnginePath.c_str());
    LOG_INFO(generateViTTRTExecCommand(args, patchSize, ropeEmbdSize, minHW, optHW, maxHW).c_str());
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    ViTBuildArgs args;
    if ((argc < 2) || (!parseViTBuildArgs(args, argc, argv)))
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

    if (args.vlmBuildParams.modelType != "qwen2_vl" && args.vlmBuildParams.modelType != "qwen2_5_vl"
        && args.vlmBuildParams.modelType != "internvl3")
    {
        LOG_ERROR("Currently only QWen2-VL, Qwen2.5-VL and InternVL3 are supported for VLM.");
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;
    bool isVisualBuild = true;
    status = buildViT(args);
    if (status != EXIT_SUCCESS)
    {
        LOG_ERROR("Cannot build Visual Engine. Please check logs for detailed error message.");
        return status;
    }

    return status;
}