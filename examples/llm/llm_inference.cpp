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
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace drivellm;
using Json = nlohmann::json;

struct LLMInferenceArgs
{
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"inputFile", required_argument, 0, 901},
        {"engineDir", required_argument, 0, 902}, {"multimodalEngineDir", required_argument, 0, 903},
        {"outputFile", required_argument, 0, 904}, {"debug", no_argument, 0, 905}, {"dumpProfile", no_argument, 0, 906},
        {"profileOutputFile", required_argument, 0, 907}, {"warmup", required_argument, 0, 908},
        {"dumpOutput", no_argument, 0, 909}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case 901: args.inputFile = optarg; break;
        case 902: args.engineDir = optarg; break;
        case 903: args.multimodalEngineDir = optarg; break;
        case 904: args.outputFile = optarg; break;
        case 905: args.debug = true; break;
        case 906: args.dumpProfile = true; break;
        case 907: args.profileOutputFile = optarg; break;
        case 908:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case 909: args.dumpOutput = true; break;
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

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
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

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath)
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
    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraWeightsName, loraWeightsPath] : availableLoraWeights.items())
        {
            if (loraWeightsMap.find(loraWeightsName) != loraWeightsMap.end())
            {
                LOG_ERROR("LoRA weights %s already exists", loraWeightsName.c_str());
                throw std::runtime_error("LoRA weights " + loraWeightsName + " already exists");
            }
            loraWeightsMap[loraWeightsName] = loraWeightsPath.get<std::string>();
        }
    }

    // Parse messages
    if (inputData.contains("messages") && inputData["messages"].is_array())
    {
        auto& messages = inputData["messages"];
        size_t numMessages = messages.size();

        // Process messages in batches according to batchSize
        for (size_t startIdx = 0; startIdx < numMessages; startIdx += batchSize)
        {
            rt::LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = topP;
            request.topK = topK;
            request.maxGenerateLength = maxGenerateLength;

            // Add messages to this batch (up to batchSize messages)
            size_t endIdx = std::min(startIdx + batchSize, numMessages);
            for (size_t messageIdx = startIdx; messageIdx < endIdx; ++messageIdx)
            {
                auto const& message = messages[messageIdx];

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
                if (message.contains("lora_weights"))
                {
                    if (messageIdx == startIdx)
                    {
                        request.loraWeightsName = message["lora_weights"].get<std::string>();
                    }
                    else
                    {
                        if (request.loraWeightsName != message["lora_weights"].get<std::string>())
                        {
                            LOG_ERROR(
                                "Multi-LoRA for the same batch is not supported. Please use the same LoRA weights for "
                                "all messages in the same batch.");
                            throw std::runtime_error(
                                "Multi-LoRA for the same batch is not supported. Please use the same LoRA weights for "
                                "all messages in the same batch.");
                        }
                    }
                }

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

    return std::make_pair(loraWeightsMap, requests);
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
    // load input file and parse to requests
    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> requests;
    try
    {
        std::tie(loraWeightsMap, requests) = parseInputFile(args.inputFile);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu requests from input file.", requests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (requests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }
    bool profilerEnabled = args.dumpProfile;
    MemoryMonitor memoryMonitor;

    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    try
    {
        llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
            args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
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

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        // Stop profiling for warmup runs
        gTimer.stopTiming();
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = requests[0];

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            if (!llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream))
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        // Start profiling for actual runs
        gTimer.startTiming();
        // Start memory monitoring for examples
        memoryMonitor.start();
    }

    // Structure to collect all responses for JSON export
    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;

    // Process each request with progress indication
    LOG_INFO("Processing %zu requests...", requests.size());
    for (size_t requestIdx = 0; requestIdx < requests.size(); ++requestIdx)
    {
        auto& request = requests[requestIdx];
        rt::LLMGenerationResponse response;

        // Show progress every 10% or every 100 requests, whichever is smaller
        size_t progressInterval = std::max(size_t(1), std::min(requests.size() / 10, size_t(100)));
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == requests.size() - 1)
        {
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, requests.size(),
                100.0 * (requestIdx + 1) / requests.size());
        }

        bool requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);

        if (requestStatus)
        {
            // Display inference output to console if --dumpOutput is enabled
            if (args.dumpOutput)
            {
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                        response.outputTexts[batchIdx].c_str());
                }
            }
        }
        else
        {
            // Handle failed request - highlight failures
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // Add to JSON output
        for (size_t batchIdx = 0; batchIdx < request.prompts.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            if (requestStatus)
            {
                responseJson["output_text"] = response.outputTexts[batchIdx];
            }
            else
            {
                responseJson["output_text"] = errorMessage;
            }
            responseJson["request_idx"] = requestIdx;
            responseJson["system_prompt"] = request.prompts[batchIdx].systemPrompt;
            responseJson["user_prompt"] = request.prompts[batchIdx].userPrompt;
            outputData["responses"].push_back(responseJson);
        }
    }

    // Final processing summary
    LOG_INFO("Processing complete: %zu/%zu requests successful", requests.size() - failedCount, requests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu REQUESTS FAILED ***", failedCount);
    }

    // Stop timing after all benchmark runs complete
    // Pending timings are automatically calculated when stopTiming() is called
    if (profilerEnabled)
    {
        gTimer.stopTiming();
        // Stop memory monitoring for examples
        memoryMonitor.stop();
    }

    // Dump profile summary to console
    size_t peakMemoryBytes = profilerEnabled ? memoryMonitor.getPeakMemory() : 0;
    if (args.dumpProfile)
    {
        auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
        printSummary(llmInferenceRuntime->getPrefillMetrics(), llmInferenceRuntime->getGenerationMetrics(),
            multimodalMetrics, peakMemoryBytes);
    }

    // Export profile to JSON file
    if (!args.profileOutputFile.empty())
    {
        try
        {
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
            std::string profileJson = getJsonSummary(llmInferenceRuntime->getPrefillMetrics(),
                llmInferenceRuntime->getGenerationMetrics(), multimodalMetrics, peakMemoryBytes);
            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson;
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    // Export to JSON file
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
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    // Return false if any request failed
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}