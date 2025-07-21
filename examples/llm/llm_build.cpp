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
    VLMBuildParams vlmBuildParams;
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
        if (args.vlmBuildParams.usePromptTuning)
        {
            setupExtraProfilesForVLM();
        }
        setupProfilesForLora();
        // TODO: add other profiles here
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
            "--minShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld "
            "--optShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld "
            "--maxShapes=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld "
            "--profile=1 "
            "--minShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld "
            "--optShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld "
            "--maxShapes=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ldx1,past_key_values.*:%ldx2x%ldx0x%ld,"
            "rope_rotary_cos_sin:%ldx%ldx%ld",
            args.onnxPath.c_str(), args.enginePath.c_str(), minBatchSize, minInputLen, minBatchSize, minBatchSize,
            minBatchSize, numKVHeads, headSize, minBatchSize, maxLength, rotaryDim, optBatchSize, optInputLen,
            optBatchSize, optBatchSize, optBatchSize, numKVHeads, headSize, optBatchSize, maxLength, rotaryDim,
            maxBatchSize, maxInputLen, maxBatchSize, maxBatchSize, maxBatchSize, numKVHeads, headSize, maxBatchSize,
            maxPositionEmbeddings, rotaryDim, minBatchSize, minBatchSize, minBatchSize, minBatchSize, numKVHeads,
            maxLength, headSize, minBatchSize, maxLength, rotaryDim, optBatchSize, optBatchSize, optBatchSize,
            optBatchSize, numKVHeads, maxLength, headSize, optBatchSize, maxLength, rotaryDim, maxBatchSize,
            maxBatchSize, maxBatchSize, maxBatchSize, numKVHeads, maxLength, headSize, maxBatchSize,
            maxPositionEmbeddings, rotaryDim);
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
    int64_t headSize;
    int64_t rotaryDim;
    int32_t nbKVCacheInputs;
    int32_t maxPositionEmbeddings;

    // for eagle
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
            args.vlmBuildParams.maxImageTokens = args.vlmBuildParams.imageTokens;
            args.vlmBuildParams.minImageTokens = args.vlmBuildParams.imageTokens;
        }

        drivellm::JsonRoot root;
        std::string onnxFolderPath = extractFolderName(args.onnxPath);
        std::string json_path = onnxFolderPath + "/config.json";
        root.parseFromPath(json_path);
        auto rootNode = root.getRoot();
        hiddenSizeDim = rootNode["hidden_size"].getInteger();
        targetModelOutputHiddenDim = args.eagleBuildParams.isEagle3 ? hiddenSizeDim * 3 : hiddenSizeDim;
        numKVHeads = rootNode["num_key_value_heads"].getInteger();
        auto numAttentionHeads = rootNode["num_attention_heads"].getInteger();
        if (rootNode.hasMember("head_dim"))
        {
            headSize = rootNode["head_dim"].getInteger();
        }
        else
        {
            headSize = hiddenSizeDim / numAttentionHeads;
        }
        if (rootNode.hasMember("partial_rotary_factor"))
        {
            rotaryDim = (int64_t) (rootNode["partial_rotary_factor"].getFloat() * headSize);
        }
        else
        {
            rotaryDim = headSize;
        }
        maxPositionEmbeddings = rootNode["max_position_embeddings"].getInteger();
        nbKVCacheInputs = rootNode["num_hidden_layers"].getInteger();
    }

    void setupExtraProfilesForVLM()
    {
        int32_t imageHiddenSize = 0;
        for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
        {
            if (strcmp(network->getInput(idx)->getName(), "image_embeds") == 0)
            {
                imageHiddenSize = network->getInput(idx)->getDimensions().d[1];
            }
        }
        if (imageHiddenSize == 0)
        {
            LOG_ERROR("Please add image_embeds as inputs for VLM.");
        }
        int64_t optImageTokens = (args.vlmBuildParams.maxImageTokens + args.vlmBuildParams.minImageTokens) / 2;

        result &= setOptimizationProfile(contextProfile, "image_embeds",
            createDims({args.vlmBuildParams.minImageTokens, imageHiddenSize}),
            createDims({optImageTokens, imageHiddenSize}),
            createDims({args.vlmBuildParams.maxImageTokens, imageHiddenSize}));
        result &= setOptimizationProfile(generationProfile, "image_embeds", createDims({1, imageHiddenSize}),
            createDims({1, imageHiddenSize}), createDims({1, imageHiddenSize}));

        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupExtraProfilesForVLMCommon().");
        }
    }

    void setupCommonProfiles()
    {
        result &= setOptimizationProfile(contextProfile, "context_lengths", createDims({minBatchSize}),
            createDims({optBatchSize}), createDims({maxBatchSize}));
        result &= setOptimizationProfile(generationProfile, "context_lengths", createDims({minBatchSize}),
            createDims({optBatchSize}), createDims({maxBatchSize}));

        result &= setOptimizationProfile(contextProfile, "rope_rotary_cos_sin",
            createDims({minBatchSize, args.maxSeqLen, rotaryDim}),
            createDims({optBatchSize, args.maxSeqLen, rotaryDim}),
            createDims({maxBatchSize, maxPositionEmbeddings, rotaryDim}));
        result &= setOptimizationProfile(generationProfile, "rope_rotary_cos_sin",
            createDims({minBatchSize, args.maxSeqLen, rotaryDim}),
            createDims({optBatchSize, args.maxSeqLen, rotaryDim}),
            createDims({maxBatchSize, maxPositionEmbeddings, rotaryDim}));

        setupKVCacheProfiles();

        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupCommonProfiles().");
        }
    }

    void setupKVCacheProfiles()
    {
        // Plugin does not have any dynamic shape.
        nvinfer1::Dims minKVContextShape = createDims({minBatchSize, 2, numKVHeads, 0, headSize});
        nvinfer1::Dims optKVContextShape = createDims({optBatchSize, 2, numKVHeads, 0, headSize});
        nvinfer1::Dims maxKVContextShape = createDims({maxBatchSize, 2, numKVHeads, 0, headSize});
        nvinfer1::Dims minKVGenerationShape = createDims({minBatchSize, 2, numKVHeads, args.maxSeqLen, headSize});
        nvinfer1::Dims optKVGenerationShape = createDims({optBatchSize, 2, numKVHeads, args.maxSeqLen, headSize});
        nvinfer1::Dims maxKVGenerationShape = createDims({maxBatchSize, 2, numKVHeads, args.maxSeqLen, headSize});

        for (int i = 0; i < nbKVCacheInputs; ++i)
        {
            result &= setOptimizationProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), minKVContextShape,
                optKVContextShape, maxKVContextShape);
            result &= setOptimizationProfile(generationProfile, fmtstr("past_key_values.%d", i).c_str(),
                minKVGenerationShape, optKVGenerationShape, maxKVGenerationShape);
        }
        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupKVCacheProfiles().");
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

        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupVanillaProfiles().");
        }
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
            int32_t const attnMaskAlignSize = 32;
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

        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupEagleProfiles().");
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

        if (result == false)
        {
            LOG_ERROR("Failed to setup optimization profiles at setupProfilesForLora().");
        }
    }
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnxPath <path> --enginePath <path> [--batchSize <int>] [--maxInputLen <int>] "
                 "[--maxSeqLen <int>] [--dynamicShape] [--maxBatchSize <int>] [--debug] [--maxLoraRank <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr << "  --onnxPath           Provide the input onnx file path. Required. " << std::endl;
    std::cerr << "  --enginePath         Provide the output TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize          Provide the desired batch_size for builder. Default = 1" << std::endl;
    std::cerr << "  --maxBatchSize       Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --maxInputLen        Provide the maximum input length for the model. Default = 128" << std::endl;
    std::cerr
        << "  --maxSeqLen         Provide the maximum output length for the model (including the input). Default = 4096"
        << std::endl;
    std::cerr << "  --dynamicShape      Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug             Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank       Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    CommonUsage::printEagleBuildOptions();
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, 701}, {"onnxPath", required_argument, 0, 702},
        {"enginePath", required_argument, 0, 703}, {"batchSize", required_argument, 0, 704},
        {"maxInputLen", required_argument, 0, 705}, {"maxSeqLen", required_argument, 0, 706},
        {"debug", no_argument, 0, 707}, {"dynamicShape", no_argument, 0, 708},
        {"maxBatchSize", required_argument, 0, 709}, {"maxLoraRank", required_argument, 0, 710}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::eagleBuildOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleBuildOptions[i];
    for (int i = 0; CommonOptions::vlmBuildOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::vlmBuildOptions[i];
    for (int i = 0; buildOptions[i].name != 0; ++i)
        long_options[idx++] = buildOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1)
    {
        if (CommonOptions::parseEagleBuildOptions(args.eagleBuildParams, opt, optarg))
        {
            continue;
        }
        if (CommonOptions::parseVLMBuildOptions(args.vlmBuildParams, opt, optarg))
        {
            continue;
        }
        switch (opt)
        {
        case 701: args.help = true; return true;
        case 702:
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
        case 703:
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
        case 704:
            if (optarg)
            {
                args.batchSize = std::stoi(optarg);
            }
            break;
        case 705:
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case 706:
            if (optarg)
            {
                args.maxSeqLen = std::stoi(optarg);
            }
            break;
        case 707: args.debug = true; break;
        case 708: args.dynamicShape = true; break;
        case 709:
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case 710:
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

    auto pluginHandles = loadEdgellmPluginLib();

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