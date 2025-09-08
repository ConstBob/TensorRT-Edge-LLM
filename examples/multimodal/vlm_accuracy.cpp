/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
              << " [--help] [--engineDir=<path to LLM engine directory>] [--visualEngineDir=<path to visual engine "
                 "directory>]"
                 " [--datasetPath=<path to dataset>]"
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
    static std::string const BASE64_CHARS
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
        throw std::runtime_error(
            "Invalid base64 length: " + std::to_string(encodedLength) + ", must be multiple of 4!");
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
        getItem(); // difficulty
        getItem(); // subfield
        getItem(); // imgType
        data->questionType = getItem();
        getItem(); // imgType

        std::string imgStr = getItem(); // image str
        parseImgBytes(data->images, imgStr);
        getItem(); // image_path

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

void loadImages(std::vector<std::vector<unsigned char>> const& rawBuffers,
    std::vector<rt::imageUtils::ImageData>& imageBuffer, std::string const& modelType,
    std::unique_ptr<MultimodalRunner> const& multimodalRunner, bool const useThumbnail = true)
{
    for (auto& buffer : rawBuffers)
    {
        // Load image from memory
        auto image = rt::imageUtils::loadImageFromMemory(buffer.data(), buffer.size());
        int width = image.width;
        int height = image.height;
        int resizedHeight, resizedWidth;
        int thumbnailW, thumbnailH;

        // Resize image using multimodal runner's resize method
        if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
        {
            // Restrict each image to [1280, 6620] image tokens.
            // Diviving original height and width by 2 to deal with a few images with large size.
            QwenViTConfig* config = static_cast<QwenViTConfig*>(multimodalRunner->getConfig());
            int factor = config->patchSize * config->mergeSize;
            auto [h, w]
                = QwenViTRunner::getResizedImageSize(height / 2, width / 2, factor, 1280 * 28 * 28, 6620 * 28 * 28);
            resizedHeight = h;
            resizedWidth = w;
        }
        else if (modelType == "internvl")
        {
            // Downsize to max number of 6 patches to limit the number of image tokens.
            InternViTConfig* config = static_cast<InternViTConfig*>(multimodalRunner->getConfig());
            auto [h, w] = InternViTRunner::getResizedImageSize(
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

        auto resizedImage = rt::imageUtils::resizeImage(image, resizedWidth, resizedHeight);
        imageBuffer.emplace_back(resizedImage);

        // Insert thumbnail image for some models
        if (useThumbnail && modelType == "internvl")
        {
            auto thumbnailImage = rt::imageUtils::resizeImage(image, thumbnailW, thumbnailH, true);
            imageBuffer.emplace_back(thumbnailImage);
        }
    }
}

void evalMultimodal(BaseParams const& baseParams, EagleParams const& eagleParams, VLMRunParams const& vlmRunParams,
    LoraWeights const& loraWeights, std::vector<MMMUTestData*> const& dataset, Tokenizer* tokenizer)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto multimodalRunner = MultimodalRunner::create(vlmRunParams.visualEngineDir, stream);
    // Batch size = 1
    auto llmEngine = getLLMEngine(1, baseParams, eagleParams, loraWeights, stream);
    llmEngine->setupExtraInputs(multimodalRunner->getComputedEmbeddings());

    auto modelConfig = llmEngine->getBaseModelConfig();
    int const maxSupportedInputLength = modelConfig.maxSupportedInputLength;
    bool const enableDynamicShape = modelConfig.minSupportedInputLength != maxSupportedInputLength;
    int const maxLength = modelConfig.maxLength;
    int const rotaryDim = modelConfig.rotaryDim;

    GenerationConfig generationConfig{maxSupportedInputLength + 256, 0, 1, 1};

    int i = 0;
    LOG_INFO("Starting running tests.");
    std::vector<std::vector<int32_t>> outputIds(1);

    for (auto data : dataset)
    {
        if (data->split != "validation")
        {
            continue;
        }
        LOG_INFO(data->id.c_str());
        outputIds[0].clear();

        std::vector<rt::imageUtils::ImageData> imageBuffer;
        std::vector<int32_t> inputIds;
        std::vector<int32_t> contextLengths;

        try
        {
            // Load images
            loadImages(data->images, imageBuffer, multimodalRunner->getModelType(), multimodalRunner, true);

            // Preprocess
            std::string prompt = data->format();
            void* ropeRotaryCosSinDevice = llmEngine->getDecoder()->getDeviceBuffer("rope_rotary_cos_sin");
            multimodalRunner->preprocess({prompt}, {imageBuffer}, inputIds, contextLengths, tokenizer,
                maxSupportedInputLength, enableDynamicShape, ropeRotaryCosSinDevice, maxLength, rotaryDim, stream);

            // Infer
            multimodalRunner->infer(stream);
            llmEngine->generate(
                inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);

            std::string pred = tokenizer->decode(outputIds[0], true);
            data->pred = pred;

            ++i;
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to run inference: %s", e.what());
            continue;
        }
    }

    LOG_INFO("Collected results on %d questions.", i);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return;
}

void accuracy(MultimodalAccuracyArgs const& args)
{
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

    // Ensure output directory exists
    std::filesystem::path outputPath(args.outputPath);
    std::filesystem::path dir = outputPath.parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir))
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec) && ec)
        {
            throw std::runtime_error("Failed to create output directory: " + dir.string());
        }
    }

    evalMultimodal(args.baseParams, args.eagleParams, args.vlmRunParams, args.loraWeights, dataset, tokenizer.get());

    saveResult(outputPath, dataset);
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

    auto pluginHandles = loadEdgellmPluginLib();

    accuracy(args);

    return EXIT_SUCCESS;
};
