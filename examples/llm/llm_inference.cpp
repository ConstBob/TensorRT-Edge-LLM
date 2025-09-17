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

#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <nlohmann/json.hpp>

using namespace drivellm;
using Json = nlohmann::json;

struct LLMInferenceArgs
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string inputFile;
    std::string outputFile;
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --inputFile     " << std::endl;
    std::cerr << "  --engineDir     " << std::endl;
    std::cerr << "  --multimodalEngineDir     " << std::endl;
    std::cerr << "  --outputFile     " << std::endl;
    std::cerr << "  --debug     " << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"inputFile", required_argument, 0, 901},
        {"engineDir", required_argument, 0, 902}, {"multimodalEngineDir", optional_argument, 0, 903},
        {"outputFile", optional_argument, 0, 904}, {"debug", no_argument, 0, 905}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case 901: args.inputFile = optarg; break;
        case 902: args.engineDir = optarg; break;
        case 903: args.multimodalEngineDir = optarg ? optarg : ""; break;
        case 904: args.outputFile = optarg ? optarg : ""; break;
        case 905: args.debug = true; break;
        default: return false;
        }
    }

    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (!args.outputFile.empty())
    {
        LOG_INFO("args.outputFile: %s", args.outputFile.c_str());
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

std::vector<rt::LLMGenerationRequest> parseInputFile(std::filesystem::path const& inputFilePath)
{
    std::vector<rt::LLMGenerationRequest> requests;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    if (!inputFileStream.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFilePath.string().c_str());
        throw std::runtime_error("Failed to open input file: " + inputFilePath.string());
    }
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse input file with error: %s", e.what());
        throw std::runtime_error("Failed to parse input file: " + inputFilePath.string());
    }

    // Extract global parameters
    int batchSize = inputData.value("batch_size", 1);
    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength = inputData.value("max_generate_length", 256);
    std::string defaultSystemPrompt = inputData.value("default_system_prompt", "");

    // Parse messages
    if (inputData.contains("messages") && inputData["messages"].is_array())
    {
        auto& messages = inputData["messages"];
        size_t numMessages = messages.size();

        // Process messages in batches according to batchSize
        for (size_t i = 0; i < numMessages; i += batchSize)
        {
            rt::LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = topP;
            request.topK = topK;
            request.maxGenerateLength = maxGenerateLength;

            // Add messages to this batch (up to batchSize messages)
            size_t endIdx = std::min(i + batchSize, numMessages);
            for (size_t j = i; j < endIdx; ++j)
            {
                auto const& message = messages[j];

                // Parse system prompt (use message-specific or default)
                std::string systemPrompt = message.value("system", defaultSystemPrompt);

                // Parse user prompt
                if (!message.contains("user"))
                {
                    LOG_ERROR("user prompt is not present");
                    throw std::runtime_error("user prompt is not present");
                }
                std::string userPrompt = message["user"];

                // Create prompt
                rt::LLMGenerationRequest::Prompt prompt;
                prompt.systemPrompt = systemPrompt;
                prompt.userPrompt = userPrompt;

                // Parse images if present
                if (message.contains("images") && message["images"].is_array())
                {
                    std::vector<rt::imageUtils::ImageData> imageBuffer;

                    for (auto const& imagePath : message["images"])
                    {
                        auto image = rt::imageUtils::loadImageFromFile(imagePath.get<std::string>());
                        imageBuffer.push_back(image);
                    }

                    if (!imageBuffer.empty())
                    {
                        prompt.imageBuffers = imageBuffer;
                    }
                }
                request.prompts.push_back(prompt);
            }

            requests.push_back(request);
        }
    }
    else
    {
        LOG_ERROR("messages is not an array");
        throw std::runtime_error("messages is not an array");
    }

    return requests;
}

int main(int argc, char* argv[])
{
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return 1;
    }

    auto pluginHandles = loadEdgellmPluginLib();
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    try
    {
        llmInferenceRuntime
            = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: {}", e.what());
        return EXIT_FAILURE;
    }

    // Capture CUDA graph and execute the graph for text only input.
    // TODO: Enable CUDA graph capture for multimodal inputs.
    if (args.multimodalEngineDir.empty())
    {
        bool const captureStatus = llmInferenceRuntime->captureDecodingCUDAGraph(stream);
        if (!captureStatus)
        {
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
    }

    // load input file and parse to requests
    std::vector<rt::LLMGenerationRequest> requests;
    try
    {
        requests = parseInputFile(args.inputFile);
        LOG_INFO("Successfully parsed %zu requests from input file.", requests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: {}", e.what());
        return EXIT_FAILURE;
    }

    if (requests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    // Structure to collect all responses for JSON export
    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    // Process each request
    for (size_t batchIdx = 0; batchIdx < requests.size(); ++batchIdx)
    {
        auto& request = requests[batchIdx];
        rt::LLMGenerationResponse response;

        if (llmInferenceRuntime->handleRequest(request, response, stream))
        {
            LOG_INFO("Generation finished for batch %zu.", batchIdx);

            // Display responses for each prompt in the batch
            for (size_t i = 0; i < response.outputTexts.size(); ++i)
            {
                LOG_INFO("Response for prompt %zu:\n%s", i, response.outputTexts[i].c_str());

                nlohmann::json responseJson;
                responseJson["system_prompt"] = request.prompts[i].systemPrompt;
                responseJson["user_prompt"] = request.prompts[i].userPrompt;
                responseJson["output_text"] = response.outputTexts[i];
                outputData["responses"].push_back(responseJson);
            }
        }
        else
        {
            LOG_ERROR("Generation failed for batch %zu.", batchIdx);
            return EXIT_FAILURE;
        }
    }

    // Export to JSON file if outputFile is provided
    if (!args.outputFile.empty())
    {
        try
        {
            std::ofstream outputFile(args.outputFile);
            if (outputFile.is_open())
            {
                outputFile << outputData.dump(4); // Pretty print with 4 spaces indentation
                outputFile.close();
                LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open output file: {}", args.outputFile);
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: {}", e.what());
            return EXIT_FAILURE;
        }
    }

    return 0;
}