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
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace nvinfer1;

struct MultimodalAccuracyArgs
{
    bool help{false};
    std::string llmEnginePath;
    std::string visualEnginePath;
    std::string tokenizerPath;
    std::string datasetPath;
    std::string outputPath;
    std::string modelType{"qwen2_vl"};
    bool debug{false};
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
            for (int i = 0; i < options.size(); ++i)
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
              << " [-h] [-e or --llmEnginePath=<path to LLM engine>] [-v or --visualEnginePath=<path to visual engine>]"
                 " [-t or --tokenizerPath=<path to HF tokenizer>] [--datasetPath=<path to dataset>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h                  Display this help message" << std::endl;
    std::cerr << "  --llmEnginePath     Provide the Qwen TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath     Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --datasetPath       Provide the dataset path for evaluation." << std::endl;
    std::cerr << "  --outputPath        Provide the path to save output." << std::endl;
    std::cerr << "  --modelType         Provide the model type. Default = qwen2_vl." << std::endl;
    std::cerr << "  --debug             Use debug mode, which outputs tensors." << std::endl;
};

bool parseMultimodalAccuracyArgs(MultimodalAccuracyArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"llmEnginePath", required_argument, 0, 'e'},
        {"visualEnginePath", required_argument, 0, 'v'}, {"tokenizerPath", required_argument, 0, 't'},
        {"datasetPath", required_argument, 0, 'D'}, {"outputPath", required_argument, 0, 'o'},
        {"modelType", required_argument, 0, 0}, {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "h:evtDod", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
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
        case 'D':
            if (optarg)
            {
                args.datasetPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --datasetPath requires option argument" << std::endl;
                return false;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.outputPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --outputPath requires option argument" << std::endl;
                return false;
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
        for (int i = 0; i < 9; ++i)
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

void evalQwen2VL(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    std::vector<MMMUTestData*> const& dataset, Tokenizer* tokenizer)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner();
    // Set minPixels according to: https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py
    vitrunner->setup(visualEnginePath, stream, 1, 1280 * 28 * 28);

    auto decoder = new Decoder<half>();
    decoder->setup(llmEnginePath, stream);

    int maxInputLength = decoder->getMaxContextLength();
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
            assert(image != NULL && "Failed to load image.");
            imageBuffers.emplace_back(image);
            imageSizes.emplace_back(std::vector<int>{width, height, desiredChannels});
        }

        vitrunner->visualPreprocess(
            imageBuffers, imageSizes, visualInput, visualAttentionMask, visualRotaryPosEmb, visualGridTHWs);
        vitrunner->allocateBuffer();

        std::string prompt = data->format();
        int numImage = data->images.size();
        vitrunner->textPreprocess(
            {prompt}, {numImage}, visualGridTHWs, tokenizer, inputIds, contextLengths, maxInputLength);

        vitrunner->visualInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        decoder->setupExtraInputs(vitrunner->getExtraLLMInputs());
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());

        std::string pred = tokenizer->decode(outputIds[0], true);
        data->pred = pred;

        ++i;
        vitrunner->freeBuffer();
    }

    LOG_INFO("Collected results on %d questions.", i);
}

void mmmuAccuracy(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    std::filesystem::path const& datasetPath, std::filesystem::path const& outputPath, Tokenizer* tokenizer,
    std::string modelType)
{
    std::vector<MMMUTestData*> dataset;
    try
    {
        assert(std::filesystem::exists(datasetPath));
        dataset = parseTSVFile(datasetPath);
        LOG_DEBUG("Loaded dataset size: %d", dataset.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load dataset: %s", e.what());
        return;
    }

    if (modelType == "qwen2_vl")
    {
        evalQwen2VL(llmEnginePath, visualEnginePath, dataset, tokenizer);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL model for Multimodal models.");
    }

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

    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.tokenizerPath);
    mmmuAccuracy(
        args.llmEnginePath, args.visualEnginePath, args.datasetPath, args.outputPath, tokenizer.get(), args.modelType);

    return EXIT_SUCCESS;
};