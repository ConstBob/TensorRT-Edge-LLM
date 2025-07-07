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
#include "common/trtUtils.h"
#include "decoder/decoder.h"
#include "engine/llm_engine.h"
#include "llm_param.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct LLMBenchmarkArgs
{
    BaseParams baseParams;
    EagleParams eagleParams;
    int maxLength{0};
    int inputLength{0};
    int64_t numRuns{10};
    int64_t warmUp{2};
    LoraWeights loraWeights;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--enginePath str> <--inputLength int> <--maxLength int> [--warmUp int] [--numRuns int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    CommonUsage::printBaseOptions();
    std::cerr << "  --inputLength    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Required."
              << std::endl;
    std::cerr << "  --warmUp         Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns        Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    CommonUsage::printBenchmarkOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printLoraOptions();
};

void warmupRun(std::unique_ptr<LLMEngineHalf>& llmEngine, std::vector<int64_t>& inputIds,
    std::vector<int32_t>& contextLengths, std::vector<std::vector<int64_t>>& outputIds,
    GenerationConfig const& generationConfig, int64_t warmUp, cudaStream_t stream, Tokenizer* tokenizer = nullptr)
{
    for (int64_t i = 0; i < warmUp; i++)
    {

        llmEngine->generate(
            inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < outputIds.size(); ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaDeviceSynchronize();
}

void benchmarkRun(std::unique_ptr<LLMEngineHalf>& llmEngine, std::vector<int64_t>& inputIds,
    std::vector<int32_t>& contextLengths, std::vector<std::vector<int64_t>>& outputIds,
    GenerationConfig const& generationConfig, std::shared_ptr<BenchmarkProfiler> const profiler, int64_t numRuns,
    cudaStream_t stream, std::vector<int32_t>* newTokensNumbers, std::vector<int32_t>* iterNumbers,
    Tokenizer* tokenizer = nullptr)
{
    profiler->startTiming();
    cudaProfilerStart();
    for (int64_t i = 0; i < numRuns; i++)
    {
        profiler->recordHostStart("seq latency");
        profiler->recordHostStart("first token latency");

        llmEngine->generate(
            inputIds, contextLengths, outputIds, generationConfig, newTokensNumbers, iterNumbers, profiler, tokenizer);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (size_t i = 0; i < outputIds.size(); ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaProfilerStop();
    profiler->stopTiming();
}

void replace_all_substrings_inplace(std::string& subject, std::string const& search, std::string const& replace)
{
    size_t pos = 0;
    if (search.empty())
    {
        return;
    }
    while ((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        // Advance past the replaced segment to avoid issues if 'replace' contains 'search'
        pos += replace.length();
    }
}

std::vector<std::string> extract_question_contents_with_json_parser(std::string const& filename)
{
    std::vector<std::string> all_question_contents;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(file, line))
    {
        drivellm::JsonRoot json_root;
        if (!json_root.parse(line))
        {
            fprintf(stderr, "Error: Failed to parse JSON line: %s\n", line.c_str());
            continue;
        }

        drivellm::JsonNode root_node = json_root.getRoot();
        if (root_node.isObject() && root_node.hasMember("question"))
        {
            drivellm::JsonNode question_node = root_node["question"];
            if (question_node.isArray())
            {
                std::string combined_content;
                for (size_t i = 0; i < question_node.size(); ++i)
                {
                    drivellm::JsonNode element_node = question_node[i];
                    if (element_node.isString())
                    {
                        std::string segment = element_node.getString();
                        // Apply JSON unescaping
                        // Order matters: \\ must be replaced first, then specific escapes like \", \/
                        // then simple char escapes like \n, \t.
                        replace_all_substrings_inplace(segment, "\\\\", "\\");
                        replace_all_substrings_inplace(segment, "\\/", "/");  // Unescape \/ to /
                        replace_all_substrings_inplace(segment, "\\n", "\n"); // Unescape \n to newline
                        replace_all_substrings_inplace(segment, "\\r", "\r"); // Unescape \r to carriage return
                        replace_all_substrings_inplace(segment, "\\t", "\t"); // Unescape \t to tab
                        replace_all_substrings_inplace(segment, "\\b", "\b"); // Unescape \b to backspace
                        replace_all_substrings_inplace(segment, "\\f", "\f"); // Unescape \f to form feed
                        // Note: Unicode escapes \uXXXX are not handled by this simple replacement.
                        combined_content += segment;
                    }
                    else
                    {

                        printf("Warning: Non-string element found in 'question' array on line: %s\n", line.c_str());
                    }
                }
                all_question_contents.push_back(combined_content);
            }
        }
        else
        {

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

bool parseLLMBenchmarkArgs(LLMBenchmarkArgs& args, int argc, char* argv[])
{
    static struct option benchmarkOptions[] = {{"inputLength", required_argument, 0, 'c'},
        {"maxLength", required_argument, 0, 's'}, {"warmUp", required_argument, 0, 'w'},
        {"numRuns", required_argument, 0, 'r'}, {"loraWeights", required_argument, 0, 'o'}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::eagleOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleOptions[i];
    for (int i = 0; benchmarkOptions[i].name != 0; ++i)
        long_options[idx++] = benchmarkOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "s:c:w:r:o:e:t:hdga:m:k:p:E:", long_options, nullptr)) != -1)
    {
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg))
        {
            continue;
        }

        if (CommonOptions::parseEagleOptions(args.eagleParams, opt, optarg))
        {
            continue;
        }
        switch (opt)
        {
        case 's':
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            break;
        case 'c':
            if (optarg)
            {
                args.inputLength = std::stoi(optarg);
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
        case 'o':
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

void benchmarkLLM(LLMBenchmarkArgs const& args, GenerationConfig const& generationConfig)
{
    auto profiler = std::make_shared<BenchmarkProfiler>();
    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    EngineConfig engineConfig;
    if (args.eagleParams.eagleEnginePath.empty())
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = EngineConfig(args.baseParams.enginePath);
    }
    else
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = EngineConfig(args.baseParams.enginePath, args.eagleParams.eagleEnginePath,
            args.eagleParams.maxPathLen, args.eagleParams.topK, args.eagleParams.isEagle3,
            args.eagleParams.maxDecodingTokens, !args.baseParams.noCudaGraph);
    }
    profiler->recordHostStart("decoder setup");
    auto llmEngine = std::make_unique<LLMEngineHalf>(engineConfig, stream);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();
    bool const eagleMode = llmEngine->isEagleModel();

    if (args.loraWeights.hasWeights() && !eagleMode)
    {
        auto loraPair = args.loraWeights.getFirst();
        auto& decoder = llmEngine->getDecoder();
        if (!decoder->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return;
        }
        if (!decoder->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return;
        }
    }

    auto const batchSize = llmEngine->getBatchSize();
    auto const maxContextLength = llmEngine->getMaxContextLength();
    std::vector<std::vector<int64_t>> outputIds(batchSize);

    if (eagleMode)
    {

        auto tokenizer = args.baseParams.tokenizerPath.empty() ? nullptr : std::make_unique<Tokenizer>();
        tokenizer->loadFromHF(args.baseParams.tokenizerPath);
        if (!tokenizer)
        {
            LOG_ERROR(
                "Failed to load tokenizer: %s, please provide the tokenizer path with --tokenizerPath for Eagle mode "
                "benchmark",
                args.baseParams.tokenizerPath.c_str());
            return;
        }

        std::vector<int64_t> inputIds(batchSize);
        std::vector<int32_t> contextLengths(batchSize, 0);
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].reserve(generationConfig.maxLength);
        }

        std::string filepath = "examples/llm/mt_bench_data/mt_dataset.json";
        auto questions = extract_question_contents_with_json_parser(filepath);
        inputIds = tokenizer->encode(questions[0], false);
        contextLengths[0] = inputIds.size();
        warmupRun(
            llmEngine, inputIds, contextLengths, outputIds, generationConfig, args.warmUp, stream, tokenizer.get());

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
            benchmarkRun(llmEngine, inputIds, contextLengths, outputIds, generationConfig, profiler, args.numRuns,
                stream, &newTokensNumbers, &iterNumbers, tokenizer.get());

            auto totalNewTokens = std::accumulate(newTokensNumbers.begin(), newTokensNumbers.end(), 0);
            auto totalIter = std::accumulate(iterNumbers.begin(), iterNumbers.end(), 0);
            auto maxNewTokens = totalNewTokens / args.numRuns;
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
            allIterNumbers.push_back(totalIter / args.numRuns);

            int32_t avgOutputLen = std::accumulate(newTokensNumbers.begin(), newTokensNumbers.end(), 0) / args.numRuns;
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
        LOG_INFO("execution_context_mem(GiB): %.2f", llmEngine->getDeviceMemorySize() / 1073741824.0);
        LOG_INFO("========================================================");
    }
    else
    {
        std::vector<int64_t> inputIds(batchSize * maxContextLength, -1);
        std::vector<int32_t> contextLengths(batchSize, args.inputLength);
        std::vector<int64_t> lastTokenIds(batchSize, args.inputLength - 1);
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
        for (int i = 0; i < batchSize; ++i)
        {
            auto beginIter = inputIds.begin() + i * maxContextLength;
            std::generate(beginIter, beginIter + args.inputLength, [&rng, &dist]() { return dist(rng); });
            outputIds[i].reserve(generationConfig.maxLength);
        }
        warmupRun(llmEngine, inputIds, contextLengths, outputIds, generationConfig, args.warmUp, stream);

        benchmarkRun(llmEngine, inputIds, contextLengths, outputIds, generationConfig, profiler, args.numRuns, stream,
            nullptr, nullptr);

        auto peakDeviceMem = profiler->recordDeviceMemEnd();
        auto peakHostMem = profiler->recordHostMemEnd();
        auto maxNewTokens = batchSize * (generationConfig.maxLength - args.inputLength);
        auto [averageSeqLatency, duration, seqLatencies] = profiler->getHostElapsedTimeMs("seq latency");
        auto [averageFirstTokenLatency, totalFirstTokenLatency, firstTokenLatencies]
            = profiler->getHostElapsedTimeMs("first token latency");
        auto [averageGenerationTime, totalGenerationTime, generationTimes]
            = profiler->getDeviceElapsedTimeMs("generation");
        auto decoderSetupLatency = std::get<1>(profiler->getHostElapsedTimeMs("decoder setup"));
        auto tokensPerSec = maxNewTokens / (averageSeqLatency / 1000);
        auto generationTokensPerSec = maxNewTokens / (averageGenerationTime / 1000);
        LOG_INFO("========================================================");
        LOG_INFO("Benchmarking done. Decoder setup: %.2fs, Iteration: %d, GPU Time: %.2fs.", decoderSetupLatency / 1000,
            seqLatencies.size(), duration / 1000);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "Sequence Latencies(ms): [";
        constexpr int maxPrintedLatencies{20};
        for (size_t i = 0; i < seqLatencies.size(); ++i)
        {
            oss << seqLatencies[i];
            if (i == seqLatencies.size() - 1)
            {
                oss << "]";
            }
            else if (seqLatencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
            {
                oss << " ... ";
                i = seqLatencies.size() - maxPrintedLatencies / 2;
            }
            else
            {
                oss << ", ";
            }
        }

        LOG_INFO(oss.str().c_str());

        oss.str("");
        oss.clear();
        oss << "First Token Latencies(ms): [";
        for (size_t i = 0; i < firstTokenLatencies.size(); ++i)
        {
            oss << firstTokenLatencies[i];
            if (i == firstTokenLatencies.size() - 1)
            {
                oss << "]";
            }
            else if (firstTokenLatencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
            {
                oss << " ... ";
                i = firstTokenLatencies.size() - maxPrintedLatencies / 2;
            }
            else
            {
                oss << ", ";
            }
        }
        LOG_INFO(oss.str().c_str());
        LOG_INFO("batch_size: %d", batchSize);
        LOG_INFO("input_length per batch: %d", args.inputLength);
        LOG_INFO("output_length per batch: %d", maxNewTokens);
        LOG_INFO("seq_latency(ms): %.2f", averageSeqLatency);
        LOG_INFO("first_token_latency(ms): %.2f", averageFirstTokenLatency);
        LOG_INFO("tokens_per_sec: %.2f", tokensPerSec);
        LOG_INFO("generation_time(ms): %.2f", averageGenerationTime);
        LOG_INFO("generation_tokens_per_sec: %.2f", generationTokensPerSec);
        LOG_INFO("cpu_peak_mem(GiB): %.2f", peakHostMem / 1048576.0);
        LOG_INFO("gpu_peak_mem(GiB): %.2f", peakDeviceMem / 1073741824.0);
        LOG_INFO("execution_context_mem(GiB): %.2f", llmEngine->getDeviceMemorySize() / 1073741824.0);
        LOG_INFO("========================================================");
    }
}

int main(int argc, char* argv[])
{
    LLMBenchmarkArgs args;
    if ((argc < 2) || (!parseLLMBenchmarkArgs(args, argc, argv)))
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

    GenerationConfig generationConfig{args.maxLength, args.maxLength, 1, 0};

    if (args.eagleParams.eagleEnginePath.empty())
    {
        if ((args.inputLength < 1) || (args.maxLength < 1))
        {
            LOG_ERROR("Please specify --inputLength and --maxLength for benchmark.");
            return EXIT_FAILURE;
        }
    }
    benchmarkLLM(args, generationConfig);
    return EXIT_SUCCESS;
};