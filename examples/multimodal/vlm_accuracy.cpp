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
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

using namespace nvinfer1;

struct MultimodalAccuracyArgs
{
    BaseParams baseParams;
    LoraWeights loraWeights;
    EagleParams eagleParams;
    VLMRunParams vlmRunParams;
    std::string datasetPath;
    std::string outputPath;
};

struct MMMUTestData
{
    std::string id;
    std::string split;
    std::string question;
    std::vector<std::string> options;
    std::vector<std::vector<unsigned char>> images;
    std::string answer;
    std::string pred;
    std::string questionType;

    std::string format() const
    {
        std::string prompt = question;

        if (questionType == "multiple-choice")
        {
            for (size_t i = 0; i < options.size(); ++i)
            {
                char letter = 'A' + i;
                prompt += "\n" + std::string(1, letter) + ". " + options[i];
            }
            prompt += "\n\nAnswer with the option's letter from the given choices directly.";
        }
        else
        {
            prompt += "\n\nAnswer the question using a single word or phrase.";
        }

        return prompt;
    }
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--enginePath=<path to LLM engine>] [--visualEnginePath=<path to visual engine>]"
                 " [--tokenizerPath=<path to HF tokenizer>] [--datasetPath=<path to dataset>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printVLMRunOptions();
    CommonUsage::printLoraOptions();
    std::cerr << "  --datasetPath       Provide the dataset path for evaluation." << std::endl;
    std::cerr << "  --outputPath        Provide the path to save output." << std::endl;
};

bool parseMultimodalAccuracyArgs(MultimodalAccuracyArgs& args, int argc, char* argv[])
{
    static struct option accuracyOptions[] = {{"datasetPath", required_argument, 0, 501},
        {"outputPath", required_argument, 0, 502}, {"loraWeights", required_argument, 0, 503}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::eagleOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleOptions[i];
    for (int i = 0; CommonOptions::vlmRunOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::vlmRunOptions[i];
    for (int i = 0; accuracyOptions[i].name != 0; ++i)
        long_options[idx++] = accuracyOptions[i];
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
        case 501: // datasetPath
            if (optarg)
            {
                args.datasetPath = optarg;
            }
            break;
        case 502: // outputPath
            if (optarg)
            {
                args.outputPath = optarg;
            }
            break;
        case 503: // loraWeights
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
    return true;
}

std::vector<unsigned char> base64Decode(std::string const& encoded)
{
    static const std::string BASE64_CHARS
        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz"
          "0123456789+/";

    std::vector<unsigned char> decoded;
    if (encoded.empty())
    {
        return decoded;
    }

    int encodedLength = encoded.size();
    if (encodedLength % 4)
    {
        throw std::runtime_error("Invalid base64 length, must be multiple of 4!");
    }

    for (int i = 0; i < encodedLength; i += 4)
    {
        std::uint32_t bits = 0;
        int nBits = 0;

        for (int j = 0; j < 4; ++j)
        {
            char c = encoded[i + j];

            if (c != '=')
            {
                bits <<= 6;
                nBits += 6;
                bits |= BASE64_CHARS.find(c);
                ;
            }
            else if (j == 2)
            {
                assert(encoded[i + j + 1] == '=');
                bits >>= 4;
                nBits -= 4;
                break;
            }
            else if (j == 3)
            {
                bits >>= 2;
                nBits -= 2;
            }
            else
            {
                throw std::runtime_error("Invalid base64-encoded data.");
            }
        }

        while (nBits > 0)
        {
            nBits -= 8;
            decoded.emplace_back(char((bits >> nBits) & 0xFF));
        }
    }

    return decoded;
}

std::vector<MMMUTestData*> parseTSVFile(std::filesystem::path const& csvPath)
{
    std::ifstream file(csvPath);
    std::vector<MMMUTestData*> res;
    LOG_DEBUG(csvPath.c_str());

    // Skip header
    std::string dummy;
    std::getline(file, dummy);

    while (!file.eof())
    {
        auto data = new MMMUTestData{};

        auto getItem = [&file, &csvPath](char const delim = '\t') -> std::string {
            // Handling this according to http://super-csv.github.io/super-csv/csv_specification.html
            std::string item;
            auto c = file.get();
            if (file.eof() || c == delim)
            {
                return item;
            }

            std::string tmp;
            if (c == '\"')
            {
                while (1)
                {
                    if (!std::getline(file, tmp, '\"'))
                    {
                        throw std::runtime_error(
                            "File " + csvPath.string() + " EOF unexpectedly. Current buffer: " + item);
                    }
                    item += tmp;
                    c = file.get();
                    if (file.eof())
                    {
                        throw std::runtime_error(
                            "File " + csvPath.string() + " EOF unexpectedly. Current buffer: " + item);
                    }
                    if (c == '\"')
                    {
                        item += c;
                    }
                    else if (c == delim)
                    {
                        break;
                    }
                    else
                    {
                        throw std::runtime_error("Error parsing line: " + item);
                    }
                }
            }
            else
            {
                item = c;
                std::getline(file, tmp, delim);
                item += tmp;
            }
            return item;
        };

        auto parseImgBytes = [](std::vector<std::vector<unsigned char>>& images, std::string const& imgStr) {
            if (imgStr.size() < 2)
            {
                return;
            }

            size_t start = 1;
            size_t end = imgStr.find(", ");
            while (end != std::string::npos)
            {
                // remove quotes
                std::string base64Str = imgStr.substr(start + 1, end - start - 2);
                images.emplace_back(base64Decode(base64Str));
                start = end + 2;
                end = imgStr.find(", ", start);
            }

            // last image
            if (imgStr[start] == '\'')
            {
                std::string base64Str = imgStr.substr(start + 1, imgStr.size() - start - 3);
                images.emplace_back(base64Decode(base64Str));
            }
        };

        // Each record items has the order of:
        //     id, index, question, split, A, B, C, D, answer, topic_difficulty, subfield,
        //     image_type, question_type, explanation, image, image_path, E, F, G, H, I, category, l2-category"

        std::string id = getItem();
        if (id.empty())
        {
            break;
        }
        data->id = id;
        getItem(); // index
        data->question = getItem();
        data->split = getItem();

        // A, B, C, D
        for (int i = 0; i < 4; ++i)
        {
            std::string opt = getItem();
            if (!opt.empty())
            {
                data->options.emplace_back(opt);
            }
        }

        data->answer = getItem();
        getItem();                      // difficulty
        getItem();                      // subfield
        getItem();                      // imgType
        data->questionType = getItem();
        getItem();                      // imgType

        std::string imgStr = getItem(); // image str
        parseImgBytes(data->images, imgStr);
        getItem();                      // image_path

        // E, F, G, H, I,
        for (int i = 0; i < 5; ++i)
        {
            std::string opt = getItem();
            if (!opt.empty())
            {
                data->options.emplace_back(opt);
            }
        }

        getItem();     // category
        getItem('\n'); // l2-category
        res.emplace_back(data);
    }

    file.close();
    return res;
}

void saveResult(std::filesystem::path const& outputPath, std::vector<MMMUTestData*> const& dataset)
{
    LOG_INFO("Saving results to %s", outputPath.c_str());
    std::ofstream outFile(outputPath);

    // Header
    outFile << "\"data_id\",\"question_type\",\"pred\",\"answer\","
            << "\"A\",\"B\",\"C\",\"D\",\"E\",\"F\",\"G\",\"H\",\"I\"\n";

    // Escape double quotes and enclose fields in quotes
    auto escapeAndQuote = [](std::string const& field) {
        std::string escapedField = "\"";
        for (char c : field)
        {
            if (c == '"')
            {
                escapedField += "\"\""; // Escape double quotes
            }
            else if (c == '\r')
            {
                escapedField += 'r'; // Escape '\r'
            }
            else if (c == '\n')
            {
                escapedField += 'n'; // Escape '\n'
            }
            else
            {
                escapedField += c;
            }
        }
        escapedField += "\"";
        return escapedField;
    };

    for (auto const& data : dataset)
    {
        if (data->split != "validation")
        {
            continue;
        }
        // Store id, question type, pred, answer
        outFile << escapeAndQuote(data->id) << ",";
        outFile << escapeAndQuote(data->questionType) << ",";
        outFile << escapeAndQuote(data->pred) << ",";
        outFile << escapeAndQuote(data->answer) << ",";

        // Store options in list of strings
        // outFile << "\"[";
        for (size_t i = 0; i < 9; ++i)
        {
            if (i < data->options.size())
            {
                auto opt = data->options[i];
                outFile << escapeAndQuote(data->options[i]);
            }
            if (i < 8)
            {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
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

void evalQwen2VL(std::vector<MMMUTestData*> const& dataset, Tokenizer* tokenizer, VLMRunParams const& vlmRunParams,
    LoraWeights const& loraWeights, EagleParams const& eagleParams, BaseParams const& baseParams)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner(vlmRunParams.modelType);
    vitrunner->setup(vlmRunParams.visualEnginePath, stream, 1);

    auto llmEngine = getLLMEngine<Qwen2ViTRunner>(baseParams, eagleParams, loraWeights, stream, vitrunner);
    if (!llmEngine)
    {
        LOG_ERROR("Failed to create LLM engine");
        return;
    }

    int maxInputLength = llmEngine->getMaxContextLength();

    GenerationConfig generationConfig{maxInputLength + 256, 0, 1, 0};

    int i = 0;
    LOG_INFO("Starting running tests.");

    for (auto data : dataset)
    {
        if (data->split != "validation")
        {
            continue;
        }
        LOG_INFO(data->id.c_str());

        std::vector<half> visualInput;
        std::vector<half> visualAttentionMask;
        std::vector<float> visualRotaryPosEmb;
        std::vector<std::vector<int64_t>> visualGridTHWs;
        std::vector<int64_t> inputIds;
        std::vector<int32_t> contextLengths;
        std::vector<std::vector<int64_t>> outputIds(1);

        // Preprocess
        std::vector<unsigned char*> imageBuffers;
        std::vector<std::vector<int>> imageSizes;
        for (auto& buffer : data->images)
        {
            int width{0}, height{0}, channels{0};
            int desiredChannels = 3;
            unsigned char* image
                = stbi_load_from_memory(buffer.data(), buffer.size(), &width, &height, &channels, desiredChannels);
            if (image == nullptr)
            {
                LOG_ERROR("Failed to load image: %s", stbi_failure_reason());
                continue;
            }

            // Diviving by 2 to deal with a few images with large size.
            // Otherwise, it requires larger dynamic shape range, which is not supported by TensorRT.
            auto [resizedHeight, resizedWidth]
                = vitrunner->adjustImageSize(height / 2, width / 2, 1280 * 28 * 28, 6620 * 28 * 28);
            unsigned char* resizedImage = (unsigned char*) malloc(resizedHeight * resizedWidth * desiredChannels);
            stbir_resize_uint8_linear(
                image, width, height, 0, resizedImage, resizedWidth, resizedHeight, 0, stbir_pixel_layout::STBIR_RGB);

            imageBuffers.emplace_back(resizedImage);
            imageSizes.emplace_back(std::vector<int>{resizedWidth, resizedHeight, desiredChannels});

            stbi_image_free(image);
        }

        vitrunner->visualPreprocess(
            imageBuffers, imageSizes, visualInput, visualAttentionMask, visualRotaryPosEmb, visualGridTHWs);

        std::string prompt = data->format();
        int numImage = data->images.size();
        vitrunner->textPreprocess(
            {prompt}, {numImage}, visualGridTHWs, tokenizer, inputIds, contextLengths, maxInputLength);

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
        llmEngine->generate(
            inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer, false);

        std::string pred = tokenizer->decode(outputIds[0], true);
        data->pred = pred;

        ++i;
        for (auto& buffer : imageBuffers)
        {
            free(buffer);
        }
    }

    LOG_INFO("Collected results on %d questions.", i);
}

void evalInternVL3(std::vector<MMMUTestData*> const& dataset, Tokenizer* tokenizer, VLMRunParams const& vlmRunParams,
    LoraWeights const& loraWeights, EagleParams const& eagleParams, BaseParams const& baseParams)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new InternVLViTRunner(vlmRunParams.modelType);
    vitrunner->setup(vlmRunParams.visualEnginePath, stream, 1);

    auto llmEngine = getLLMEngine<InternVLViTRunner>(baseParams, eagleParams, loraWeights, stream, vitrunner);
    if (!llmEngine)
    {
        LOG_ERROR("Failed to create LLM engine");
        return;
    }

    int maxInputLength = llmEngine->getMaxContextLength();
    GenerationConfig generationConfig{maxInputLength + 256, 0, 1, 0};

    int i = 0;
    LOG_INFO("Starting running tests.");

    for (auto data : dataset)
    {
        if (data->split != "validation")
        {
            continue;
        }
        LOG_INFO(data->id.c_str());

        bool useThumbnail = true;
        std::vector<half> visualInput;
        std::vector<int64_t> inputIds;
        std::vector<int64_t> imageTokenLengths;
        std::vector<int32_t> contextLengths;
        std::vector<std::vector<int64_t>> outputIds(1);

        // Preprocess
        std::vector<unsigned char*> imageBuffers;
        std::vector<unsigned char*> thumbnailImageBuffers;
        std::vector<std::vector<int>> imageSizes;
        for (auto& buffer : data->images)
        {
            int width{0}, height{0}, channels{0};
            int desiredChannels = 3;
            unsigned char* image
                = stbi_load_from_memory(buffer.data(), buffer.size(), &width, &height, &channels, desiredChannels);
            if (image == nullptr)
            {
                LOG_ERROR("Failed to load image: %s", stbi_failure_reason());
                continue;
            }

            // Restricting to max 4 448x448 blocks. Some questions have more than 4 images.
            // Otherwise, it requires larger dynamic shape range, which is not supported by TensorRT.
            std::vector<std::pair<int, int>> targetRatios = {{1, 1}, {1, 2}, {2, 1}, {3, 1}, {1, 3}, {2, 2}, {4, 1},
                {1, 4}, {5, 1}, {1, 5}, {1, 6}, {6, 1}, {3, 2}, {2, 3}};
            // {{1, 1}, {1, 2}, {2, 1}, {3, 1}, {1, 3}, {2, 2}, {4, 1}, {1, 4}};
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

        vitrunner->visualPreprocess(
            imageBuffers, thumbnailImageBuffers, imageSizes, visualInput, imageTokenLengths, useThumbnail);

        std::string prompt = data->format();
        int numImage = data->images.size();
        vitrunner->textPreprocess(
            {prompt}, {numImage}, imageTokenLengths, tokenizer, inputIds, contextLengths, maxInputLength);

        // Infer
        vitrunner->internVLViTInfer(visualInput);
        llmEngine->generate(
            inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer, false);

        std::string pred = tokenizer->decode(outputIds[0], true);
        data->pred = pred;

        ++i;
        for (auto& buffer : imageBuffers)
        {
            free(buffer);
        }
        for (auto& buffer : thumbnailImageBuffers)
        {
            free(buffer);
        }
    }

    LOG_INFO("Collected results on %d questions.", i);
}

void mmmuAccuracy(MultimodalAccuracyArgs const& args, Tokenizer* tokenizer)
{
    std::vector<MMMUTestData*> dataset;
    try
    {
        assert(std::filesystem::exists(args.datasetPath));
        dataset = parseTSVFile(args.datasetPath);
        LOG_DEBUG("Loaded dataset size: %d", dataset.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load dataset: %s", e.what());
        return;
    }

    if (args.vlmRunParams.modelType == "qwen2_vl" || args.vlmRunParams.modelType == "qwen2_5_vl")
    {
        evalQwen2VL(dataset, tokenizer, args.vlmRunParams, args.loraWeights, args.eagleParams, args.baseParams);
    }
    else if (args.vlmRunParams.modelType == "internvl3")
    {
        evalInternVL3(dataset, tokenizer, args.vlmRunParams, args.loraWeights, args.eagleParams, args.baseParams);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL and InternVL3 models for Multimodal models.");
    }

    saveResult(args.outputPath, dataset);
}

int main(int argc, char* argv[])
{
    MultimodalAccuracyArgs args;
    if ((argc < 2) || (!parseMultimodalAccuracyArgs(args, argc, argv)))
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

    auto pluginHandles = loadPlugins();

    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.baseParams.tokenizerPath);
    mmmuAccuracy(args, tokenizer.get());

    return EXIT_SUCCESS;
};