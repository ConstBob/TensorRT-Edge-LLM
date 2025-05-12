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
#include "qwen2vl/vit_runner.h"
#include "tokenizer/tokenizer.h"
#include <dlfcn.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

struct RuntimeArgs
{
    bool help{false};
    std::vector<std::string> inputStrings;
    std::vector<std::vector<std::string>> imagePaths;
    std::string llmEnginePath;
    std::string visualEnginePath;
    std::string tokenizerPath;
    int maxLength{1024};
    bool debug{false};
    std::string modelType{"qwen2_vl"};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-e or --llmEnginePath=<path to LLM engine>] [-v or --visualEnginePath=<path to visual engine>]"
                 " [-s or --maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
                 " [--inputString=<input string for one batch>] [--imagePaths=<image paths for one batch>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h                  Display this help message" << std::endl;
    std::cerr << "  --inputString       Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --imagePaths        Provide the input image paths to the runtime. Required. " << std::endl;
    std::cerr << "  --llmEnginePath     Provide the Qwen TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath     Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength         Provide the maximum output length for the generation session (including the "
                 "input). Default = 1024."
              << std::endl;
    std::cerr << "  --modelType         Provide the model type. Default = qwen2_vl." << std::endl;
    std::cerr << "  --debug             Use debug mode, which outputs tensors." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"inputString", required_argument, 0, 'i'},
        {"imagePaths", required_argument, 0, 'p'}, {"llmEnginePath", required_argument, 0, 'e'},
        {"visualEnginePath", required_argument, 0, 'v'}, {"tokenizerPath", required_argument, 0, 't'},
        {"maxLength", required_argument, 0, 's'}, {"debug", no_argument, 0, 'd'},
        {"modelType", required_argument, 0, 0}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "h:iest", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
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
        case 'p':
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
        case 'e':
            if (optarg)
            {
                args.llmEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --llmEnginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 'v':
            if (optarg)
            {
                args.visualEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --visualEnginePath requires option argument" << std::endl;
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
        case 0:
            if (strcmp(long_options[option_index].name, "modelType") == 0)
            {
                {
                    if (optarg)
                    {
                        args.modelType = optarg;
                    }
                    else
                    {
                        std::cerr << "ERROR: model type requires option argument,support only qwen2_vl currently"
                                  << std::endl;
                        return false;
                    }
                }
            }
            break;
        default: return false;
        }
    }
    return true;
}

void decodeQwen2VL(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    std::vector<std::string>& inputStrings, std::vector<std::vector<std::string>> const& imagePaths,
    Tokenizer* tokenizer, GenerationConfig const& generationConfig, int32_t const batchSize,
    std::vector<std::vector<int64_t>>& outputIds, std::string modelType)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner(modelType);
    vitrunner->setup(visualEnginePath, stream, batchSize);
    auto decoder = new Decoder<half>();
    decoder->setup(llmEnginePath, stream, false, batchSize);
    decoder->setupExtraInputs(vitrunner->getExtraLLMInputs());

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
    for (int b = 0; b < imagePaths.size(); ++b)
    {
        numImages.emplace_back(imagePaths[b].size());
        for (int i = 0; i < imagePaths[b].size(); ++i)
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
        inputStrings, numImages, visualGridTHWs, tokenizer, inputIds, contextLengths, decoder->getMaxContextLength());

    // Infer
    if (modelType == "qwen2_vl")
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

    decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());

    for (auto& buffer : imageBuffers)
    {
        free(buffer);
    }
}

std::vector<std::string> decode(std::filesystem::path const& llmEnginePath,
    std::filesystem::path const& visualEnginePath, std::vector<std::string>& inputStrings,
    std::vector<std::vector<std::string>>& imagePaths, Tokenizer* tokenizer, GenerationConfig const& generationConfig,
    std::string modelType)
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

    if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
    {
        decodeQwen2VL(llmEnginePath, visualEnginePath, inputStrings, imagePaths, tokenizer, generationConfig, batchSize,
            outputIds, modelType);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL model for Multimodal models.");
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
    RuntimeArgs args;
    if ((argc < 2) || (!parseRuntimeArgs(args, argc, argv)))
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

    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};
    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.tokenizerPath);
    auto output = decode(args.llmEnginePath, args.visualEnginePath, args.inputStrings, args.imagePaths, tokenizer.get(),
        generationConfig, args.modelType);

    return EXIT_SUCCESS;
};