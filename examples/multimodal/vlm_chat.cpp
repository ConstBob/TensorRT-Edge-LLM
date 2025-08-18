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
#include "decoder/decoder.h"
#include "engine/llm_engine.h"
#include "exampleUtils.h"
#include "multimodal/internViTRunner.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "tokenizer/tokenizer.h"
#include <dlfcn.h>
#include <getopt.h>

using namespace drivellm;
using namespace drivellm::rt;
using namespace drivellm::tokenizer;

struct VlmChatArgs
{
    BaseParams baseParams;
    EagleParams eagleParams;
    VLMRunParams vlmRunParams;
    LoraWeights loraWeights;
    std::vector<std::string> inputStrings;
    std::vector<std::vector<std::string>> imagePaths;
    int maxLength{1024};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to LLM engine directory>] [--visualEngineDir=<path to visual engine "
                 "directory>]"
                 " [--inputString=<input string>] [--imagePaths=<image paths>]"
                 " [--maxLength=<int>] [--loraWeights=<name:path>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printVLMRunOptions();
    CommonUsage::printLoraOptions();
    std::cerr << "  --inputString    Provide the input string to the runtime. " << std::endl;
    std::cerr << "  --imagePaths     Provide the image paths separated by comma. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 1024"
              << std::endl;
};

bool parseVlmChatArgs(VlmChatArgs& args, int argc, char* argv[])
{
    static struct option chatOptions[]
        = {{"inputString", required_argument, 0, 801}, {"imagePaths", required_argument, 0, 802},
            {"maxLength", required_argument, 0, 803}, {"loraWeights", required_argument, 0, 804}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::eagleOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleOptions[i];
    for (int i = 0; CommonOptions::vlmRunOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::vlmRunOptions[i];
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

        if (CommonOptions::parseVLMRunOptions(args.vlmRunParams, opt, optarg))
        {
            continue;
        }

        switch (opt)
        {
        case 801:
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
        case 802:
            if (optarg)
            {
                std::vector<std::string> paths;
                std::stringstream ss(optarg);
                std::string path;
                while (std::getline(ss, path, ','))
                {
                    // Trim spaces
                    path = regex_replace(path, std::regex("(^[ ]+)|([ ]+$)"), "");
                    paths.push_back(path);
                }
                args.imagePaths.emplace_back(paths);
            }
            else
            {
                std::cerr << "ERROR: --imagePaths requires option argument" << std::endl;
                return false;
            }
            break;
        case 803:
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --maxLength requires option argument" << std::endl;
                return false;
            }
            break;
        case 804:
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
            else
            {
                std::cerr << "ERROR: --loraWeights requires option argument" << std::endl;
                return false;
            }
            break;
        default: return false;
        }
    }
    return true;
}

void loadImages(std::vector<std::vector<std::string>> const& imagePaths,
    std::vector<std::vector<ImageData>>& imageBuffers, std::string const& modelType,
    std::unique_ptr<MultimodalRunner> const& multimodalRunner, bool const useThumbnail = true)
{
    for (size_t b = 0; b < imagePaths.size(); ++b)
    {
        std::vector<ImageData> imageBuffer;
        for (size_t i = 0; i < imagePaths[b].size(); ++i)
        {
            // Loaded pixels
            ImageData image = loadImageFromFile(imagePaths[b][i]);
            int width = image.width;
            int height = image.height;
            int resizedHeight, resizedWidth;
            int thumbnailW, thumbnailH;

            // Resize image using multimodal runner's resize method
            if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
            {
                // User should set appropriate minPixels and maxPixels
                // For details please refer to examples/multimodal/README.md
                QwenViTConfig* config = static_cast<QwenViTConfig*>(multimodalRunner->getConfig());
                int factor = config->patchSize * config->mergeSize;
                auto [h, w] = QwenViTRunner::resizeImage(height, width, factor, 128 * 28 * 28, 512 * 28 * 28);
                resizedHeight = h;
                resizedWidth = w;
            }
            else if (modelType == "internvl")
            {
                // User should set appropriate max number of patches
                // Downsize to max number of 6 patches to limit the number of image tokens.
                // Original preprocessing on huggingface allows for max 12 patches.
                // For details please refer to examples/multimodal/README.md
                InternViTConfig* config = static_cast<InternViTConfig*>(multimodalRunner->getConfig());
                auto [h, w] = InternViTRunner::resizeImage(
                    height, width, config->blockImageSizeH, config->blockImageSizeW, 1, 6);
                resizedHeight = h;
                resizedWidth = w;
                thumbnailH = config->blockImageSizeH;
                thumbnailW = config->blockImageSizeW;
            }
            else
            {
                // Default keep original size
                resizedHeight = height;
                resizedWidth = width;
            }

            ImageData resizedImage = resizeImage(image, resizedWidth, resizedHeight);
            imageBuffer.emplace_back(resizedImage);

            // Insert thumbnail image for some models
            if (useThumbnail && modelType == "internvl")
            {
                ImageData thumbnailImage = resizeImage(image, thumbnailW, thumbnailH, true);
                imageBuffer.emplace_back(thumbnailImage);
            }
        }

        imageBuffers.emplace_back(imageBuffer);
    }
}

int decodeMultimodal(BaseParams const& baseParams, EagleParams const& eagleParams, VLMRunParams const& vlmRunParams,
    LoraWeights const& loraWeights, std::vector<std::string> const& inputStrings,
    std::vector<std::vector<std::string>> const& imagePaths, Tokenizer* tokenizer,
    GenerationConfig const& generationConfig, int32_t const batchSize, std::vector<std::vector<int32_t>>& outputIds)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto multimodalRunner = getMultimodalRunner(vlmRunParams, stream);
    auto llmEngine = getLLMEngine(batchSize, baseParams, eagleParams, loraWeights, stream);
    llmEngine->setupExtraInputs(multimodalRunner->getComputedEmbeddings());

    auto modelConfig = llmEngine->getBaseModelConfig();
    int const maxSupportedInputLength = modelConfig.maxSupportedInputLength;
    bool const enableDynamicShape = modelConfig.minSupportedInputLength != maxSupportedInputLength;
    int const maxLength = modelConfig.maxLength;
    int const rotaryDim = modelConfig.rotaryDim;

    // Load images
    std::vector<std::vector<ImageData>> imageBuffers;
    loadImages(imagePaths, imageBuffers, multimodalRunner->getModelType(), multimodalRunner, true);

    // Preprocess
    std::vector<int32_t> inputIds;
    std::vector<int32_t> contextLengths;
    void* ropeRotaryCosSinDevice = llmEngine->getDecoder()->getDeviceBuffer("rope_rotary_cos_sin");
    multimodalRunner->preprocess(inputStrings, imageBuffers, inputIds, contextLengths, tokenizer,
        maxSupportedInputLength, enableDynamicShape, ropeRotaryCosSinDevice, maxLength, rotaryDim, stream);

    // Infer
    multimodalRunner->infer(stream);
    llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}

std::vector<std::string> decode(VlmChatArgs args)
{
    GenerationConfig generationConfig{args.maxLength, 0, 1, 1};
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

    // Set default inputString and imagePaths to batchSize
    int32_t batchSize = std::max(args.inputStrings.size(), args.imagePaths.size());
    for (int i = args.inputStrings.size(); i < batchSize; ++i)
    {
        args.inputStrings.emplace_back("Describe this image.");
    }
    for (int i = args.imagePaths.size(); i < batchSize; ++i)
    {
        args.imagePaths.emplace_back(std::vector<std::string>{});
    }

    std::vector<std::vector<int32_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(args.maxLength);
    }

    decodeMultimodal(args.baseParams, args.eagleParams, args.vlmRunParams, args.loraWeights, args.inputStrings,
        args.imagePaths, tokenizer.get(), generationConfig, batchSize, outputIds);

    std::vector<std::string> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        LOG_DEBUG("Output%d length is %d", i, outputIds[i].size());
        output[i] = tokenizer->decode(outputIds[i], true);
        LOG_INFO("Input%d is: %s", i, args.inputStrings[i].c_str());
        LOG_INFO("Output%d is: %s", i, output[i].c_str());
    }
    return output;
}

int main(int argc, char* argv[])
{
    VlmChatArgs args;
    if ((argc < 2) || (!parseVlmChatArgs(args, argc, argv)))
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

    auto output = decode(args);

    return EXIT_SUCCESS;
};