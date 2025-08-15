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
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

using Json = nlohmann::json;

struct ViTBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool dynamicShape{false};
    bool help{false};
    bool debug{false};
    VLMBuildParams vlmBuildParams;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--onnxDir str> <--engineDir str> [--dynamicShape]"
                 "[--imageTokens int] [--minImageTokens int] [--maxImageTokens int] [--modelType str] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr
        << "  --onnxDir          Provide the directory containing the input onnx file for visual encoder. Required. "
        << std::endl;
    std::cerr << "  --engineDir        Provide the output TensorRT engine directory path for visual encoder. Required. "
              << std::endl;
    std::cerr << "  --dynamicShape     Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug            Use debug mode, which outputs tensors." << std::endl;
    CommonUsage::printVLMBuildOptions();
}

bool parseViTBuildArgs(ViTBuildArgs& args, int argc, char* argv[])
{
    static struct option vitOptions[] = {{"help", no_argument, 0, 601}, {"onnxDir", required_argument, 0, 602},
        {"engineDir", required_argument, 0, 603}, {"dynamicShape", no_argument, 0, 604}, {"debug", no_argument, 0, 605},
        {0, 0, 0, 0}};

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
                args.onnxDir = optarg;
            }
            else
            {
                LOG_ERROR("--onnxDir requires option argument.");
                return false;
            }
            break;
        case 603:
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
        case 604: args.dynamicShape = true; break;
        case 605: args.debug = true; break;
        default: LOG_ERROR("ERROR: Invalid Argument %c is %s", opt, optarg); return false;
        }
    }
    return true;
}

std::string generateViTTRTExecCommand(ViTBuildArgs const& args, std::string const& modelType, int64_t const& patchSize,
    int64_t const& ropeEmbdSize, int64_t const& minHW, int64_t const& optHW, int64_t const& maxHW)
{
    std::string onnxPath = args.onnxDir + "/model.onnx";
    std::string enginePath = args.engineDir + "/visual.engine";
    std::string trtExecCommand
        = fmtstr("Equivalent ViT trtexec command: trtexec --onnx=%s --saveEngine=%s --stronglyTyped --verbose ",
            onnxPath.c_str(), enginePath.c_str());

    if (modelType == "qwen2_vl")
    {
        trtExecCommand += fmtstr(
            "--minShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld "
            "--optShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld "
            "--maxShapes=input:%ldx%ld,rotary_pos_emb:%ldx%ld,attention_mask:1x%ldx%ld ",
            minHW, patchSize, minHW, ropeEmbdSize, minHW, minHW, optHW, patchSize, optHW, ropeEmbdSize, optHW, optHW,
            maxHW, patchSize, maxHW, ropeEmbdSize, maxHW, maxHW);
    }
    else if (modelType == "qwen2_5_vl")
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
    else if (modelType == "internvl")
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
    // Read config.json to get model type and patch size
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return EXIT_FAILURE;
    }

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(configFileStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return EXIT_FAILURE;
    }

    // Read model type from vision_config.model_type
    if (!jsonConfig.contains("vision_config") || !jsonConfig["vision_config"].contains("model_type"))
    {
        LOG_ERROR("vision_config.model_type not found in config.json");
        return EXIT_FAILURE;
    }

    // TODO: Unify the model_type at runtime and here.
    std::string modelType = jsonConfig["vision_config"]["model_type"].get<std::string>();

    // Validate model type
    if (modelType != "qwen2_vl" && modelType != "qwen2_5_vl" && modelType != "internvl_vision")
    {
        LOG_ERROR("Currently only Qwen2-VL, Qwen2.5-VL and InternVL are supported for VLM. You provided: %s",
            modelType.c_str());
        return EXIT_FAILURE;
    }

    int64_t minHW, optHW, maxHW;
    if (args.dynamicShape)
    {
        // In Qwen2-VL, HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
        minHW = args.vlmBuildParams.minImageTokens * 4;
        maxHW = args.vlmBuildParams.maxImageTokens * 4;
        // InternVL ViT model has a reshape which requires HW to be divisible by 1024
        if (modelType == "internvl_vision")
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
    std::string onnxPath = args.onnxDir + "/model.onnx";
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", onnxPath.c_str());
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
    // TODO: add support to read ropeEmbdSize and patchSize from config file
    // Currently the definition is different from the config, so we need to infer it from the ONNX model.
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

    if ((modelType == "qwen2_5_vl" || modelType == "qwen2_vl") && (ropeEmbdSize == 0))
    {
        LOG_ERROR("Cannot infer ropeEmbdSize. Do you have proper ONNX input: rotary_pos_emb?");
        return EXIT_FAILURE;
    }

    // set dimensions for input tensors. Only context phase input can possibly be dynamic
    bool result = true;

    result &= setOptimizationProfile(visualProfile, "input", createDims({minHW, patchSize}),
        createDims({optHW, patchSize}), createDims({maxHW, patchSize}));

    if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
    {
        result &= setOptimizationProfile(visualProfile, "rotary_pos_emb", createDims({minHW, ropeEmbdSize}),
            createDims({optHW, ropeEmbdSize}), createDims({maxHW, ropeEmbdSize}));
        result &= setOptimizationProfile(visualProfile, "attention_mask", createDims({1, minHW, minHW}),
            createDims({1, optHW, optHW}), createDims({1, maxHW, maxHW}));
    }

    if (modelType == "qwen2_5_vl")
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
    // Create engine directory if it doesn't exist
    if (!std::filesystem::exists(args.engineDir))
    {
        if (std::filesystem::create_directories(args.engineDir))
        {
            LOG_INFO("Created directory %s for saving ViT engine.", args.engineDir.c_str());
        }
        else
        {
            LOG_ERROR("Failed to create directory %s for saving ViT engine.", args.engineDir.c_str());
            return EXIT_FAILURE;
        }
    }
    else
    {
        LOG_INFO("Engine directory %s already exists, the ViT engine will be overwritten.", args.engineDir.c_str());
    }

    // Save engine with appropriate name
    std::string engineFilePath = args.engineDir + "/visual.engine";
    std::ofstream ofs(engineFilePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", engineFilePath.c_str());
        return EXIT_FAILURE;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", engineFilePath.c_str());

    // Copy config.json to engine directory
    std::string targetConfigPath = args.engineDir + "/config.json";
    int configResult = copyFile(configPath, targetConfigPath);
    if (configResult == EXIT_SUCCESS)
    {
        LOG_INFO("Copied config.json to %s", targetConfigPath.c_str());
    }
    else
    {
        LOG_WARNING("Failed to copy config.json to %s", targetConfigPath.c_str());
    }

    LOG_INFO(generateViTTRTExecCommand(args, modelType, patchSize, ropeEmbdSize, minHW, optHW, maxHW).c_str());
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

    auto pluginHandles = loadEdgellmPluginLib();

    int status = buildViT(args);
    if (status != EXIT_SUCCESS)
    {
        LOG_ERROR("Cannot build Visual Engine. Please check logs for detailed error message.");
        return status;
    }

    return status;
}