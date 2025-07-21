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
#include "internvl3/vit_runner.h"
#include "llm_param.h"
#include "qwen2vl/vit_runner.h"
#include "tokenizer/tokenizer.h"
#include <dlfcn.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

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
              << " [--help] [--enginePath=<path to LLM engine>] [--visualEnginePath=<path to visual engine>]"
                 " [--tokenizerPath=<path to HF tokenizer>] [--inputString=<input string for one batch>]"
                 " [--imagePaths=<image paths for one batch>] [--maxLength=<int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --inputString       Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --imagePaths        Provide the input image paths to the runtime. Required. " << std::endl;
    std::cerr << "  --maxLength         Provide the maximum output length for the generation session (including the "
                 "input). Default = 1024."
              << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printVLMRunOptions();
    CommonUsage::printLoraOptions();
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
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg, true))
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

template <typename ViTRunnerType>
std::unique_ptr<LLMEngineHalf> getLLMEngine(BaseParams const& baseParams, EagleParams const& eagleParams,
    LoraWeights const& loraWeights, cudaStream_t stream, ViTRunnerType* vitrunner)
{
    EngineConfig engineConfig;
    bool eagleMode = !eagleParams.eagleEnginePath.empty();
    if (eagleMode)
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = EngineConfig(baseParams.enginePath, eagleParams.eagleEnginePath, eagleParams.maxPathLen,
            eagleParams.topK, eagleParams.isEagle3, eagleParams.maxDecodingTokens, !baseParams.noCudaGraph);
    }
    else
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = EngineConfig(baseParams.enginePath);
    }
    auto llmEngine = std::make_unique<LLMEngineHalf>(engineConfig, stream);
    llmEngine->setupExtraInputs(vitrunner->getExtraLLMInputs());

    // Load and switch to LoRA weights if provided
    if (loraWeights.hasWeights() && !eagleMode)
    {
        auto& decoderPtr = llmEngine->getDecoder();
        auto loraPair = loraWeights.getFirst();
        if (!decoderPtr->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return nullptr;
        }
        if (!decoderPtr->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return nullptr;
        }
    }

    return llmEngine;
}

void decodeQwen2VL(BaseParams const& baseParams, EagleParams const& eagleParams, VLMRunParams const& vlmRunParams,
    std::vector<std::string>& inputStrings, std::vector<std::vector<std::string>> const& imagePaths,
    Tokenizer* tokenizer, GenerationConfig const& generationConfig, int32_t const batchSize,
    std::vector<std::vector<int64_t>>& outputIds, LoraWeights const& loraWeights)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner(vlmRunParams.modelType);
    vitrunner->setup(vlmRunParams.visualEnginePath, stream, batchSize);

    auto llmEngine = getLLMEngine<Qwen2ViTRunner>(baseParams, eagleParams, loraWeights, stream, vitrunner);

    // Preprocess
    std::vector<half> visualInput;
    std::vector<half> visualAttentionMask;
    std::vector<float> visualRotaryPosEmb;
    std::vector<std::vector<int64_t>> visualGridTHWs;
    std::vector<int64_t> inputIds;
    std::vector<int32_t> contextLengths;

    // Load images
    std::vector<unsigned char*> imageBuffers;
    std::vector<std::vector<int>> imageSizes;
    std::vector<int> numImages;
    for (size_t b = 0; b < imagePaths.size(); ++b)
    {
        numImages.emplace_back(imagePaths[b].size());
        for (size_t i = 0; i < imagePaths[b].size(); ++i)
        {
            int width{0}, height{0}, channels{0};
            int desiredChannels = 3;
            // Loaded pixels in hwc, rgb order
            unsigned char* image = stbi_load(imagePaths[b][i].c_str(), &width, &height, &channels, desiredChannels);
            if (image == nullptr)
            {
                LOG_ERROR("Failed to load image: %s", stbi_failure_reason());
                return;
            }

            // Adjust image size to limit number of tokens generated in desired range
            // User should set appropriate minPixels and maxPixels according to their use case and match engine build
            // config. For details please refer to README.md#image-preprocess-and-number-of-image-tokens
            auto [resizedHeight, resizedWidth]
                = vitrunner->adjustImageSize(height, width, 128 * 28 * 28, 512 * 28 * 28);
            unsigned char* resizedImage = (unsigned char*) malloc(resizedHeight * resizedWidth * desiredChannels);

            stbir_resize_uint8_linear(
                image, width, height, 0, resizedImage, resizedWidth, resizedHeight, 0, stbir_pixel_layout::STBIR_RGB);

            imageBuffers.emplace_back(resizedImage);
            imageSizes.emplace_back(std::vector<int>{resizedWidth, resizedHeight, desiredChannels});

            stbi_image_free(image);
        }
    }

    vitrunner->visualPreprocess(
        imageBuffers, imageSizes, visualInput, visualAttentionMask, visualRotaryPosEmb, visualGridTHWs);
    vitrunner->textPreprocess(
        inputStrings, numImages, visualGridTHWs, tokenizer, inputIds, contextLengths, llmEngine->getMaxContextLength());

    // Infer
    if (vlmRunParams.modelType == "qwen2_vl")
    {
        vitrunner->qwen2ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
    }
    else
    {
        std::vector<half> visualWindowAttentionMask;
        std::vector<int64_t> visualWindowIndex;
        std::vector<int64_t> reverseWindowIndex;
        vitrunner->getWindowIndex(visualGridTHWs, visualWindowAttentionMask, visualWindowIndex, reverseWindowIndex);
        vitrunner->qwen2_5ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb, visualWindowAttentionMask,
            visualWindowIndex, reverseWindowIndex);
    }

    llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);

    for (auto& buffer : imageBuffers)
    {
        free(buffer);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete vitrunner;
}

void decodeInternVL3(BaseParams const& baseParams, EagleParams const& eagleParams, VLMRunParams const& vlmRunParams,
    std::vector<std::string>& inputStrings, std::vector<std::vector<std::string>> const& imagePaths,
    Tokenizer* tokenizer, GenerationConfig const& generationConfig, int32_t const batchSize,
    std::vector<std::vector<int64_t>>& outputIds, LoraWeights const& loraWeights, bool const useThumbnail)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new InternVLViTRunner(vlmRunParams.modelType);
    vitrunner->setup(vlmRunParams.visualEnginePath, stream, batchSize);

    auto llmEngine = getLLMEngine<InternVLViTRunner>(baseParams, eagleParams, loraWeights, stream, vitrunner);

    // Preprocess
    std::vector<half> visualInput;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> inputIds;
    std::vector<int32_t> contextLengths;

    // Load images
    std::vector<unsigned char*> imageBuffers;
    std::vector<unsigned char*> thumbnailImageBuffers;
    std::vector<std::vector<int>> imageSizes;
    std::vector<int> numImages;
    for (size_t b = 0; b < imagePaths.size(); ++b)
    {
        numImages.emplace_back(imagePaths[b].size());
        for (size_t i = 0; i < imagePaths[b].size(); ++i)
        {
            int width{0}, height{0}, channels{0};
            int desiredChannels = 3;
            // Loaded pixels in hwc, rgb order
            unsigned char* image = stbi_load(imagePaths[b][i].c_str(), &width, &height, &channels, desiredChannels);
            if (image == nullptr)
            {
                LOG_ERROR("Failed to load image: %s", stbi_failure_reason());
                return;
            }

            // Adjust image size to the nearest target ratio
            // User should set appropriate imageTokens value according to their use case and match engine build
            // config. For details please refer to README.md#image-preprocess-and-number-of-image-tokens

            // Downsized to max 6 448x448 blocks. The preprocessing on hf allows for max 12 448x448 blocks.
            // This was done to reduce the number of image tokens since engine build with a longer output sequence
            // can be supported if configured during onnx export.
            std::vector<std::pair<int, int>> targetRatios = {{1, 1}, {1, 2}, {2, 1}, {3, 1}, {1, 3}, {2, 2}, {4, 1},
                {1, 4}, {5, 1}, {1, 5}, {1, 6}, {6, 1}, {3, 2}, {2, 3}};
            auto [resizedHeight, resizedWidth] = vitrunner->adjustImageSize(height, width, targetRatios);
            unsigned char* resizedImage = (unsigned char*) malloc(resizedHeight * resizedWidth * desiredChannels);

            stbir_resize_uint8_linear(
                image, width, height, 0, resizedImage, resizedWidth, resizedHeight, 0, stbir_pixel_layout::STBIR_RGB);

            imageBuffers.emplace_back(resizedImage);
            imageSizes.emplace_back(std::vector<int>{resizedWidth, resizedHeight, desiredChannels});

            if (useThumbnail)
            {
                int thumbnailImageSize = 448;
                unsigned char* thumbnailImage
                    = (unsigned char*) malloc(thumbnailImageSize * thumbnailImageSize * desiredChannels);
                stbir_resize_uint8_linear(image, width, height, 0, thumbnailImage, thumbnailImageSize,
                    thumbnailImageSize, 0, stbir_pixel_layout::STBIR_RGB);
                thumbnailImageBuffers.emplace_back(thumbnailImage);
            }
            else
            {
                thumbnailImageBuffers.emplace_back(nullptr);
            }

            stbi_image_free(image);
        }
    }
    vitrunner->visualPreprocess(
        imageBuffers, thumbnailImageBuffers, imageSizes, visualInput, imageTokenLengths, useThumbnail);
    vitrunner->textPreprocess(inputStrings, numImages, imageTokenLengths, tokenizer, inputIds, contextLengths,
        llmEngine->getMaxContextLength());

    // Infer
    vitrunner->internVLViTInfer(visualInput);
    llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);

    for (auto& buffer : imageBuffers)
    {
        free(buffer);
    }
    for (auto& buffer : thumbnailImageBuffers)
    {
        if (buffer)
        {
            free(buffer);
        }
    }
}

std::vector<std::string> decode(BaseParams const& baseParams, EagleParams const& eagleParams,
    VLMRunParams const& vlmRunParams, std::vector<std::string>& inputStrings,
    std::vector<std::vector<std::string>>& imagePaths, Tokenizer* tokenizer, GenerationConfig const& generationConfig,
    LoraWeights const& loraWeights)
{
    int32_t batchSize = std::max(inputStrings.size(), imagePaths.size());

    // Set default inputString and imagePaths to batchSize
    for (int i = inputStrings.size(); i < batchSize; ++i)
    {
        inputStrings.emplace_back("Describe this image.");
    }
    for (int i = imagePaths.size(); i < batchSize; ++i)
    {
        imagePaths.emplace_back(std::vector<std::string>{});
    }

    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }

    if (vlmRunParams.modelType == "qwen2_vl" || vlmRunParams.modelType == "qwen2_5_vl")
    {
        decodeQwen2VL(baseParams, eagleParams, vlmRunParams, inputStrings, imagePaths, tokenizer, generationConfig,
            batchSize, outputIds, loraWeights);
    }
    else if (vlmRunParams.modelType == "internvl3")
    {
        decodeInternVL3(baseParams, eagleParams, vlmRunParams, inputStrings, imagePaths, tokenizer, generationConfig,
            batchSize, outputIds, loraWeights, true);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL and InternVL3 models for Multimodal models.");
    }

    std::vector<std::string> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        LOG_DEBUG("Output%d length is %d", i, outputIds[i].size());
        output[i] = tokenizer->decode(outputIds[i], true);
        LOG_INFO("Input%d is: %s", i, inputStrings[i].c_str());
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

    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};
    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.baseParams.tokenizerPath);
    auto output = decode(args.baseParams, args.eagleParams, args.vlmRunParams, args.inputStrings, args.imagePaths,
        tokenizer.get(), generationConfig, args.loraWeights);

    return EXIT_SUCCESS;
};