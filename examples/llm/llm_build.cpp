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

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxPath;
    std::string enginePath;
    int64_t batchSize{1};
    int64_t maxInputLen{128};
    int64_t maxSeqLen{4096};
    bool dynamicShape{false};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // Default to 0 means no LoRA
    EagleBuildParams eagleBuildParams;
};

class LLMEngineProfileBuilder
{
public:
    LLMEngineProfileBuilder(LLMBuildArgs& args, nvinfer1::IOptimizationProfile* contextProfile,
        nvinfer1::IOptimizationProfile* generationProfile, bool& result, nvinfer1::INetworkDefinition* network)
        : args(args)
        , contextProfile(contextProfile)
        , generationProfile(generationProfile)
        , result(result)
        , network(network)
    {

        initializeModelDimensions();
    }

    void setupAllProfiles()
    {

        setupCommonProfiles();
        if (args.eagleBuildParams.isEagleBase || args.eagleBuildParams.isEagleDraft)
        {
            setupEagleProfiles();
        }
        else
        {
            setupVanillaProfiles();
        }
        setupProfilesForLora();
        // TODO: add other profiles here
        // setupProfilesForVLM();
    }
    std::string generateTRTExecCommand()
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
            maxBatchSize, numKVHeads, hiddenSizePerHead, minBatchSize, minBatchSize, minBatchSize, minBatchSize,
            numKVHeads, maxLength, hiddenSizePerHead, optBatchSize, optBatchSize, optBatchSize, optBatchSize,
            numKVHeads, maxLength, hiddenSizePerHead, maxBatchSize, maxBatchSize, maxBatchSize, maxBatchSize,
            numKVHeads, maxLength, hiddenSizePerHead);
        return trtExecCommand;
    }

    void saveConfigJson()
    {
        // Copy config.json to engine path
        auto configPath = extractFolderName(args.onnxPath) + "/config.json";
        std::string targetConfigPath = extractFolderName(args.enginePath) + "/config.json";
        copyFile(configPath, targetConfigPath);

        if (args.eagleBuildParams.isEagle3 && args.eagleBuildParams.isEagleDraft)
        {
            // Copy d2t.bin to enginePath if it exists
            std::string d2tPath = extractFolderName(args.onnxPath) + "/d2t.bin";
            std::string targettD2tPath = extractFolderName(args.enginePath) + "/d2t.bin";
            copyFile(d2tPath, targettD2tPath);
        }
    }

private:
    LLMBuildArgs& args;
    nvinfer1::IOptimizationProfile* contextProfile;
    nvinfer1::IOptimizationProfile* generationProfile;
    nvinfer1::INetworkDefinition* network;
    bool& result;

    int64_t optBatchSize;
    int64_t minBatchSize;
    int64_t maxBatchSize;
    int64_t numKVHeads;
    int64_t hiddenSizePerHead;
    int32_t nbKVCacheInputs;
    int32_t hiddenSizeDim;
    int32_t targetModelOutputHiddenDim;

    void initializeModelDimensions()
    {
        optBatchSize = args.batchSize;
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
        if (args.eagleBuildParams.isEagleBase || args.eagleBuildParams.isEagleDraft)
        {
            drivellm::JsonRoot root;
            std::string onnxFolderPath = extractFolderName(args.onnxPath);
            std::string json_path = onnxFolderPath + "/config.json";
            root.parseFromPath(json_path);
            auto rootNode = root.getRoot();
            hiddenSizeDim = rootNode["hidden_size"].getInteger();
            targetModelOutputHiddenDim = args.eagleBuildParams.isEagle3 ? hiddenSizeDim * 3 : hiddenSizeDim;
            numKVHeads = rootNode["num_key_value_heads"].getInteger();
            auto numAttentionHeads = rootNode["num_attention_heads"].getInteger();
            hiddenSizePerHead = hiddenSizeDim / numAttentionHeads;
            nbKVCacheInputs = rootNode["num_hidden_layers"].getInteger();
        }
        else
        {
            // TODO: get them from config.json directly
            nbKVCacheInputs = 0;
            nvinfer1::Dims kvDims;
            bool getKvDims = false;
            for (int32_t i = 0; i < network->getNbInputs(); ++i)
            {
                if (network->getInput(i)->getDimensions().nbDims == 5)
                {
                    ++nbKVCacheInputs;
                    if (!getKvDims)
                    {
                        kvDims = network->getInput(i)->getDimensions();
                        getKvDims = true;
                    }
                }
            }
            if (nbKVCacheInputs == 0)
            {
                LOG_ERROR("Cannot find KV Cache input. KV Cache is guaranteed to have 5 dimensions in Plugin network!");
                result = false;
                return;
            }
            numKVHeads = kvDims.d[2];
            hiddenSizePerHead = kvDims.d[4];
        }
    }

    void setupCommonProfiles()
    {
        result &= setOptimizationProfile(contextProfile, "context_lengths", createDims({minBatchSize}),
            createDims({optBatchSize}), createDims({maxBatchSize}));
        result &= setOptimizationProfile(generationProfile, "context_lengths", createDims({minBatchSize}),
            createDims({optBatchSize}), createDims({maxBatchSize}));

        setupKVCacheProfiles();
    }

    void setupKVCacheProfiles()
    {
        // Plugin does not have any dynamic shape.
        nvinfer1::Dims minKVContextShape = createDims({minBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
        nvinfer1::Dims optKVContextShape = createDims({optBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
        nvinfer1::Dims maxKVContextShape = createDims({maxBatchSize, 2, numKVHeads, 0, hiddenSizePerHead});
        nvinfer1::Dims minKVGenerationShape
            = createDims({minBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});
        nvinfer1::Dims optKVGenerationShape
            = createDims({optBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});
        nvinfer1::Dims maxKVGenerationShape
            = createDims({maxBatchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});

        for (int i = 0; i < nbKVCacheInputs; ++i)
        {
            result &= setOptimizationProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), minKVContextShape,
                optKVContextShape, maxKVContextShape);
            result &= setOptimizationProfile(generationProfile, fmtstr("past_key_values.%d", i).c_str(),
                minKVGenerationShape, optKVGenerationShape, maxKVGenerationShape);
        }
    }

    void setupVanillaProfiles()
    {
        if (args.dynamicShape)
        {
            result &= setOptimizationProfile(contextProfile, "input_ids", createDims({minBatchSize, 1}),
                createDims({optBatchSize, args.maxInputLen / 2}), createDims({maxBatchSize, args.maxInputLen}));
        }
        else
        {
            // Set static shape
            result &= setOptimizationProfile(contextProfile, "input_ids", createDims({optBatchSize, args.maxInputLen}),
                createDims({optBatchSize, args.maxInputLen}), createDims({optBatchSize, args.maxInputLen}));
        }

        result &= setOptimizationProfile(generationProfile, "input_ids", createDims({minBatchSize, 1}),
            createDims({optBatchSize, 1}), createDims({maxBatchSize, 1}));

        result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({minBatchSize, 1}),
            createDims({optBatchSize, 1}), createDims({maxBatchSize, 1}));
        result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({minBatchSize, 1}),
            createDims({optBatchSize, 1}), createDims({maxBatchSize, 1}));
    }

    void setupEagleProfiles()
    {
        // Eagle-specific profile setup logic
        int const mMaxDecodingTokens = args.eagleBuildParams.maxDecodingTokens;
        int const mMaxDraftTokensPerStep = args.eagleBuildParams.mMaxDraftTokensPerStep;
        int const mMaxTokens = args.eagleBuildParams.isEagleDraft ? mMaxDraftTokensPerStep : mMaxDecodingTokens;
        if (args.eagleBuildParams.isEagleDraft || args.eagleBuildParams.isEagleBase)
        {
            result &= setOptimizationProfile(contextProfile, "input_ids", createDims({minBatchSize, 1}),
                createDims({optBatchSize, args.maxInputLen / 2}), createDims({maxBatchSize, args.maxInputLen}));

            result &= setOptimizationProfile(generationProfile, "input_ids", createDims({minBatchSize, 1}),
                createDims({optBatchSize, mMaxTokens / 2}), createDims({maxBatchSize, mMaxTokens}));
        }

        if (args.eagleBuildParams.isEagleDraft)
        {

            result &= setOptimizationProfile(contextProfile, "hidden_states_from_draft",
                createDims({minBatchSize, 1, hiddenSizeDim}),
                createDims({optBatchSize, args.maxInputLen / 2, hiddenSizeDim}),
                createDims({maxBatchSize, args.maxInputLen, hiddenSizeDim}));
            result &= setOptimizationProfile(generationProfile, "hidden_states_from_draft",
                createDims({minBatchSize, 1, hiddenSizeDim}), createDims({optBatchSize, mMaxTokens / 2, hiddenSizeDim}),
                createDims({maxBatchSize, mMaxTokens, hiddenSizeDim}));

            result &= setOptimizationProfile(contextProfile, "hidden_states_input",
                createDims({minBatchSize, 1, targetModelOutputHiddenDim}),
                createDims({optBatchSize, args.maxInputLen / 2, targetModelOutputHiddenDim}),
                createDims({maxBatchSize, args.maxInputLen, targetModelOutputHiddenDim}));
            result &= setOptimizationProfile(generationProfile, "hidden_states_input",
                createDims({minBatchSize, 1, targetModelOutputHiddenDim}),
                createDims({optBatchSize, mMaxTokens / 2, targetModelOutputHiddenDim}),
                createDims({maxBatchSize, mMaxTokens, targetModelOutputHiddenDim}));

            result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({1}),
                createDims({args.maxInputLen / 2}), createDims({args.maxInputLen}));
            result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({1}),
                createDims({mMaxTokens / 2}), createDims({mMaxTokens}));
        }
        else if (args.eagleBuildParams.isEagleBase)
        {
            result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({1}),
                createDims({args.maxInputLen / 2}), createDims({args.maxInputLen}));
            result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({1}),
                createDims({mMaxTokens / 2}), createDims({mMaxTokens}));
        }

        if (args.eagleBuildParams.isEagleDraft || args.eagleBuildParams.isEagleBase)
        {
            const int32_t attnMaskAlignSize = 32;
            result &= setOptimizationProfile(contextProfile, "attention_mask", createDims({minBatchSize, 1, 1}),
                createDims({optBatchSize, 1, 1}), createDims({maxBatchSize, 1, 1}));
            result &= setOptimizationProfile(generationProfile, "attention_mask", createDims({minBatchSize, 1, 1}),
                createDims(
                    {optBatchSize, mMaxTokens / 2, divUp(mMaxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize}),
                createDims({maxBatchSize, mMaxTokens, divUp(mMaxTokens, attnMaskAlignSize) * attnMaskAlignSize}));
            result &= setOptimizationProfile(contextProfile, "attention_pos_id", createDims({minBatchSize, 1}),
                createDims({optBatchSize, 1}), createDims({maxBatchSize, 1}));
            result &= setOptimizationProfile(generationProfile, "attention_pos_id", createDims({minBatchSize, 1}),
                createDims({optBatchSize, mMaxTokens / 2}), createDims({maxBatchSize, mMaxTokens}));
        }
    }
    void setupProfilesForLora()
    {
        // Add LoRA optimization profiles if maxLoraRank > 0
        if (args.maxLoraRank > 0)
        {
            for (int i = 0; i < network->getNbInputs(); ++i)
            {
                auto* input = network->getInput(i);
                std::string inputName = input->getName();

                if (inputName.find("lora_A") != std::string::npos)
                {
                    // For lora_A, the shape is [gemm_k, lora_rank]
                    auto dims = input->getDimensions();
                    if (dims.nbDims == 2)
                    {
                        int64_t gemm_k = dims.d[0];
                        result &= setOptimizationProfile(contextProfile, inputName.c_str(),
                            createDims({gemm_k, 0}),                    // min shape
                            createDims({gemm_k, args.maxLoraRank / 2}), // opt shape
                            createDims({gemm_k, args.maxLoraRank}));    // max shape
                        result &= setOptimizationProfile(generationProfile, inputName.c_str(),
                            createDims({gemm_k, 0}),                    // min shape
                            createDims({gemm_k, args.maxLoraRank / 2}), // opt shape
                            createDims({gemm_k, args.maxLoraRank}));    // max shape
                    }
                }
                else if (inputName.find("lora_B") != std::string::npos)
                {
                    // For lora_B, the shape is [lora_rank, gemm_n]
                    auto dims = input->getDimensions();
                    if (dims.nbDims == 2)
                    {
                        int64_t gemm_n = dims.d[1];
                        result &= setOptimizationProfile(contextProfile, inputName.c_str(),
                            createDims({0, gemm_n}),                    // min shape
                            createDims({args.maxLoraRank / 2, gemm_n}), // opt shape
                            createDims({args.maxLoraRank, gemm_n}));    // max shape
                        result &= setOptimizationProfile(generationProfile, inputName.c_str(),
                            createDims({0, gemm_n}),                    // min shape
                            createDims({args.maxLoraRank / 2, gemm_n}), // opt shape
                            createDims({args.maxLoraRank, gemm_n}));    // max shape
                    }
                }
            }
        }
    }
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--onnxPath str> <--enginePath str> [-b or "
                 "--batchSize int] [-c or --maxInputLen int] [-s or --maxSeqLen int] [--dynamicShape] [--maxBatchSize "
                 "int] [--debug] [--maxLoraRank int]"
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
    std::cerr << "  --dynamicShape   Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank    Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)" << std::endl;
    CommonUsage::printEagleBuildOptions();
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, 'h'}, {"onnxPath", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'o'}, {"batchSize", required_argument, 0, 'b'},
        {"maxInputLen", required_argument, 0, 'c'}, {"maxSeqLen", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {"dynamicShape", no_argument, 0, 'y'},
        {"maxBatchSize", required_argument, 0, 'B'}, {"maxLoraRank", required_argument, 0, 'L'},
        {"imageTokens", required_argument, 0, 0}, {"minImageTokens", required_argument, 0, 0},
        {"maxImageTokens", required_argument, 0, 0}, {"modelType", required_argument, 0, 0}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::eagleBuildOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleBuildOptions[i];
    for (int i = 0; buildOptions[i].name != 0; ++i)
        long_options[idx++] = buildOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "hi:o:b:c:s:dyB:L:ega:m:", long_options, &option_index)) != -1)
    {
        if (CommonOptions::parseEagleBuildOptions(args.eagleBuildParams, opt, optarg))
        {
            continue;
        }

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
        case 'd': args.debug = true; break;
        case 'y': args.dynamicShape = true; break;
        case 'L':
            if (optarg)
            {
                args.maxLoraRank = std::stoi(optarg);
            }
            break;
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

    int32_t const nbInputs = network->getNbInputs();
    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    // set dimensions for input tensors. Only context phase input can possibly be dynamic
    bool result = true;
    LLMEngineProfileBuilder profileManager(args, contextProfile, generationProfile, result, network.get());
    profileManager.setupAllProfiles();
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
    profileManager.saveConfigJson();
    if (!args.eagleBuildParams.isEagleBase && !args.eagleBuildParams.isEagleDraft)
    {
        LOG_INFO(profileManager.generateTRTExecCommand().c_str());
    }
    return EXIT_SUCCESS;
}