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
#include "common/json.h"
#include "common/trtUtils.h"
#include "decoder/decoder.h"
#include "eagle/eagle.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>


struct LLMEagleBenchmarkArgs
{
    bool help{false};
    std::string enginePath;
    std::string eagleEnginePath;
    std::string tokenizerPath;
    int maxLength{0};
    int64_t numRuns{10};
    int64_t warmUp{2};
    bool debug{false};
    bool noCudaGraph{true};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
    int32_t topK{10};
    int32_t maxPathLen{6};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--enginePath str> <--inputLength int> <--maxLength int> [--warmUp int] [--numRuns int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --eagleEnginePath     Provide the input Eagle engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Required."
              << std::endl;
    std::cerr << "  --warmUp         Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns        Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --noCudaGraph    Cuda graph is default enabled. Use this flag to disable cuda graph." << std::endl;
    std::cerr << "  --isEagle3       Use Eagle3 mode. Default is Eagle2." << std::endl;
    std::cerr << "  --maxDecodingTokens Provide the maximum decoding tokens for target model, the number provided must be aligned with building phase. Default = 60"
              << std::endl;
    std::cerr << "  --topK           Provide the topK for draft model to select the topK candidates. the number provided must be aligned with building phase. Default is 10." << std::endl;
    std::cerr << "  --maxPathLen     Provide the max stack layers for draft model to constrcut the max tree path length. the number provided must be aligned with building phase. Default is 6." << std::endl;
};

bool parseLLMEagleBenchmarkArgs(LLMEagleBenchmarkArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"enginePath", required_argument, 0, 'e'},
        {"maxLength", required_argument, 0, 's'}, {"inputLength", required_argument, 0, 'c'},
        {"warmUp", required_argument, 0, 'w'}, {"numRuns", required_argument, 0, 'r'},
        {"eagleEnginePath", required_argument, 0, 'l'}, {"tokenizerPath", required_argument, 0, 't'},
        {"noCudaGraph", no_argument, 0, 'g'}, {"debug", no_argument, 0, 'd'}, {"isEagle3", no_argument, 0, 'i'},
        {"maxDecodingTokens", required_argument, 0, 'm'}, {"topK", required_argument, 0, 'k'}, {"maxPathLen", required_argument, 0, 'p'},
        {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "he:s:c:w:r:d:g:i:l", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'e':
            if (optarg)
            {
                args.enginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --enginePath requires option argument" << std::endl;
                return false;
            }
            break;

        case 's':
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            break;
        case 'l':
            if (optarg)
            {
                args.eagleEnginePath = optarg;
            }
            break;
        case 'w':
            if (optarg)
            {
                args.warmUp = std::stoi(optarg);
            }
            break;
        case 'r':
            if (optarg)
            {
                args.numRuns = std::stoi(optarg);
            }
            break;
        case 'm':
            if (optarg)
            {
                args.maxDecodingTokens = std::stoi(optarg);
            }
            break;
        case 'k':
            if (optarg)
            {
                args.topK = std::stoi(optarg);
            }
            break;
        case 'p':
            if (optarg)
            {
                args.maxPathLen = std::stoi(optarg);
            }
            break;
        case 'g': args.noCudaGraph = true; break;
        case 'd': args.debug = true; break;
        case 'i': args.isEagle3 = true; break;
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
        default: return false;
        }
    }
    return true;
}

void replace_all_substrings_inplace(std::string& subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    if (search.empty()) {
        return;
    }
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        // Advance past the replaced segment to avoid issues if 'replace' contains 'search'
        pos += replace.length();
    }
}

std::vector<std::string> extract_question_contents_with_json_parser(const std::string& filename) {
    std::vector<std::string> all_question_contents;
    std::ifstream file(filename);

    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(file, line)) {
        drivellm::JsonRoot json_root;
        if (!json_root.parse(line)) {
            fprintf(stderr, "Error: Failed to parse JSON line: %s\n", line.c_str());
            continue;
        }

        drivellm::JsonNode root_node = json_root.getRoot();
        if (root_node.isObject() && root_node.hasMember("question")) {
            drivellm::JsonNode question_node = root_node["question"];
            if (question_node.isArray()) {
                std::string combined_content;
                for (size_t i = 0; i < question_node.size(); ++i) {
                    drivellm::JsonNode element_node = question_node[i];
                    if (element_node.isString()) {
                        std::string segment = element_node.getString();
                        // Apply JSON unescaping
                        // Order matters: \\ must be replaced first, then specific escapes like \", \/
                        // then simple char escapes like \n, \t.
                        replace_all_substrings_inplace(segment, "\\\\", "\\"); // Unescape \\ to \
                        replace_all_substrings_inplace(segment, "\\\"", "\""); // Unescape \" to "
                        replace_all_substrings_inplace(segment, "\\/", "/");   // Unescape \/ to /
                        replace_all_substrings_inplace(segment, "\\n", "\n");   // Unescape \n to newline
                        replace_all_substrings_inplace(segment, "\\r", "\r");   // Unescape \r to carriage return
                        replace_all_substrings_inplace(segment, "\\t", "\t");   // Unescape \t to tab
                        replace_all_substrings_inplace(segment, "\\b", "\b");   // Unescape \b to backspace
                        replace_all_substrings_inplace(segment, "\\f", "\f");   // Unescape \f to form feed
                        // Note: Unicode escapes \uXXXX are not handled by this simple replacement.
                        combined_content += segment;
                    } else {
                        
                        printf("Warning: Non-string element found in 'question' array on line: %s\n", line.c_str());
                    }
                }
                all_question_contents.push_back(combined_content);
            }
        } else {
        
            printf("Warning: 'question' field not found or root is not an object on line: %s\n", line.c_str());
        }
        
    }

    file.close();
    return all_question_contents;
}

template <typename T>
float calculateAverage(std::vector<T> const& vec)
{
    if (vec.empty())
        return 0.0f;
    return static_cast<float>(std::accumulate(vec.begin(), vec.end(), static_cast<T>(0))) / vec.size();
}

void benchmarkLLM(std::string& enginePath, std::string& eagleEnginePath, std::string& tokenizerPath, int64_t warmUp,
    int64_t numRuns, GenerationConfig const& generationConfig, bool useCudaGraph, bool isEagle3, int32_t maxDecodingTokens, int32_t topK, int32_t maxPathLen)
{
    auto profiler = std::make_shared<BenchmarkProfiler>();
    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();

    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(tokenizerPath);
    auto baseDecoder = std::make_unique<Decoder<half>>();
    auto draftDecoder = std::make_unique<Decoder<half>>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    profiler->recordHostStart("decoder setup");
    baseDecoder->setup(enginePath, stream, useCudaGraph, 1, true);
    draftDecoder->setup(eagleEnginePath, stream, useCudaGraph, 1, true);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();
    int32_t maxContextLength = static_cast<int32_t>(baseDecoder->getMaxContextLength()) + 1; // for eagle
    int64_t batchSize = baseDecoder->getModelBatchSize();
    if (batchSize != 1)
    {
        printf("Eagle only supports batch size 1 currently!\n");
        assert(false);
    }
    
    auto eagle = new Eagle<half>(std::move(baseDecoder), std::move(draftDecoder), stream, eagleEnginePath, maxPathLen, topK, isEagle3,
        maxDecodingTokens);
    std::vector<int64_t> inputIds(batchSize);
    std::vector<int32_t> contextLengths(batchSize, 0);

    std::vector<std::vector<int64_t>> outputIds(batchSize);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
    std::string filepath = "examples/eagle/mt_bench_data/mt_dataset.json";
    auto questions = extract_question_contents_with_json_parser(filepath);
    // bs = 0
    inputIds = tokenizer->encode(questions[0], false);
    contextLengths[0] = inputIds.size();
    outputIds[0].reserve(generationConfig.maxLength);
    
    for (int64_t i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("seq latency");
        eagle->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId(), isEagle3,
            profiler, nullptr, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaDeviceSynchronize();

    std::vector<int32_t> allNewTokensNumbers;
    std::vector<int32_t> allIterNumbers;
    std::vector<float> allAcceptanceRates;
    std::vector<float> allSeqLatencies;
    std::vector<float> allFirstTokenLatencies;
    std::vector<float> allGenerationTimes;
    std::vector<float> allDecoderSetupTimes;
    std::vector<float> allTokensPerSec;
    std::vector<float> allGenerationTokensPerSec;
    std::vector<int32_t> allInputLengths;
    std::vector<int32_t> allOutputLengths;

    for (size_t sampleIdx = 0; sampleIdx < questions.size(); ++sampleIdx)
    {
        
        std::vector<int32_t> newTokensNumbers;
        std::vector<int32_t> iterNumbers;
        inputIds = tokenizer->encode(questions[sampleIdx], false);
        contextLengths[0] = inputIds.size();
        allInputLengths.push_back(inputIds.size());
        profiler->startTiming();
        cudaProfilerStart();
        for (int64_t i = 0; i < numRuns; i++)
        {
            profiler->recordHostStart("seq latency");
            profiler->recordHostStart("first token latency");
            eagle->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId(), isEagle3,
                profiler, &newTokensNumbers, &iterNumbers);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            profiler->recordHostEnd("seq latency");

            for (int i = 0; i < batchSize; ++i)
            {
                outputIds[i].resize(0);
            }
        }

        cudaProfilerStop();
        profiler->stopTiming();

        auto totalNewTokens = std::accumulate(newTokensNumbers.begin(), newTokensNumbers.end(), 0);
        auto totalIter = std::accumulate(iterNumbers.begin(), iterNumbers.end(), 0);
        auto maxNewTokens = totalNewTokens / numRuns;
        auto [averageSeqLatency, duration, seqLatencies] = profiler->getHostElapsedTimeMs("seq latency");
        auto [averageFirstTokenLatency, totalFirstTokenLatency, firstTokenLatencies]
            = profiler->getHostElapsedTimeMs("first token latency");
        auto [averageGenerationTime, totalGenerationTime, generationTimes]
            = profiler->getDeviceElapsedTimeMs("generation");
        auto decoderSetupLatency = std::get<1>(profiler->getHostElapsedTimeMs("decoder setup"));
        auto tokensPerSec = maxNewTokens / (averageSeqLatency / 1000);
        auto generationTokensPerSec = maxNewTokens / (averageGenerationTime / 1000);
        auto averageAcceptanceRate = static_cast<float>(totalNewTokens) / totalIter;

        allAcceptanceRates.push_back(averageAcceptanceRate);
        allSeqLatencies.push_back(averageSeqLatency);
        allFirstTokenLatencies.push_back(averageFirstTokenLatency);
        allGenerationTimes.push_back(averageGenerationTime);
        allDecoderSetupTimes.push_back(decoderSetupLatency);
        allTokensPerSec.push_back(tokensPerSec);
        allGenerationTokensPerSec.push_back(generationTokensPerSec);
        allNewTokensNumbers.push_back(maxNewTokens);
        allIterNumbers.push_back(totalIter / numRuns);

        int32_t avgOutputLen = std::accumulate(newTokensNumbers.begin(), newTokensNumbers.end(), 0) / numRuns;
        allOutputLengths.push_back(avgOutputLen);
    }

    auto peakDeviceMem = profiler->recordDeviceMemEnd();
    auto peakHostMem = profiler->recordHostMemEnd();

    // Calculate averages using template function
    float avgInputLength = calculateAverage(allInputLengths);
    float avgOutputLength = calculateAverage(allOutputLengths);
    float avgAcceptanceRate = calculateAverage(allAcceptanceRates);
    float avgSeqLatency = calculateAverage(allSeqLatencies);
    float avgFirstTokenLatency = calculateAverage(allFirstTokenLatencies);
    float avgGenerationTime = calculateAverage(allGenerationTimes);
    float avgDecoderSetupTime = calculateAverage(allDecoderSetupTimes);
    float avgTokensPerSec = calculateAverage(allTokensPerSec);
    float avgGenerationTokensPerSec = calculateAverage(allGenerationTokensPerSec);
    float avgNewTokens = calculateAverage(allNewTokensNumbers);
    float avgIterNumbers = calculateAverage(allIterNumbers);

    // Print statistics
    LOG_INFO("========================================================");
    LOG_INFO("Average Statistics Across All %d MT Samples:", questions.size());
    LOG_INFO("Average Input Length: %.2f tokens", avgInputLength);
    LOG_INFO("Average Output Length: %.2f tokens", avgOutputLength);
    LOG_INFO("Average Acceptance Rate: %.2f", avgAcceptanceRate);
    LOG_INFO("Average Sequence Latency (ms): %.2f", avgSeqLatency);
    LOG_INFO("Average First Token Latency (ms): %.2f", avgFirstTokenLatency);
    LOG_INFO("Average Generation Time (ms): %.2f", avgGenerationTime);
    LOG_INFO("Average Decoder Setup Time (ms): %.2f", avgDecoderSetupTime);
    LOG_INFO("Average Tokens per Second: %.2f", avgTokensPerSec);
    LOG_INFO("Average Generation Tokens per Second: %.2f", avgGenerationTokensPerSec);
    LOG_INFO("cpu_peak_mem(GiB): %.2f", peakHostMem / 1048576.0);
    LOG_INFO("gpu_peak_mem(GiB): %.2f", peakDeviceMem / 1073741824.0);
    LOG_INFO("Average Iterations per Sample: %.2f", avgIterNumbers);
    LOG_INFO("execution_context_mem(GiB): %.2f", eagle->getDeviceMemorySize() / 1073741824.0);
    LOG_INFO("========================================================");
}

int main(int argc, char* argv[])
{
    LLMEagleBenchmarkArgs args;
    if ((argc < 2) || (!parseLLMEagleBenchmarkArgs(args, argc, argv)))
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

    GenerationConfig generationConfig{args.maxLength, args.maxLength, 1, 0};

    if ((args.maxLength < 1))
    {
        LOG_ERROR("Please specify --maxLength for benchmark.");
        return EXIT_FAILURE;
    }
    benchmarkLLM(args.enginePath, args.eagleEnginePath, args.tokenizerPath, args.warmUp, args.numRuns, generationConfig,
        !args.noCudaGraph, args.isEagle3, args.maxDecodingTokens, args.topK, args.maxPathLen);
    return EXIT_SUCCESS;
};