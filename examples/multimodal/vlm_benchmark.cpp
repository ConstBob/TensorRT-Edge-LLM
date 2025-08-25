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
#include "multimodal/multimodalRunner.h"
#include "tokenizer/tokenizer.h"
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <getopt.h>

using namespace drivellm;
using namespace drivellm::rt;

struct VlmBenchmarkArgs
{
    BaseParams baseParams;
    VLMRunParams vlmRunParams;
    LoraWeights loraWeights;
    int textTokenLength{0};
    int imageTokenLength{0};
    int outputLength{0};
    int numRuns{10};
    int warmUp{2};
    int batchSize{1};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to LLM engine directory>] [--visualEngineDir=<path to visual engine "
                 "directory>]"
                 " [--textTokenLength=<int>] [--imageTokenLength=<int>]"
                 " [--outputLength=<int>] [--warmUp=<int>] [--numRuns=<int>] [--batchSize=<int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printVLMRunOptions();
    CommonUsage::printLoraOptions();
    std::cerr << "  --textTokenLength   Provide the text token length. Required. " << std::endl;
    std::cerr << "  --imageTokenLength  Provide the image token length. Required. " << std::endl;
    std::cerr << "  --outputLength      Provide the output length. Required. " << std::endl;
    std::cerr << "  --warmUp            Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns           Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    std::cerr << "  --batchSize         Provide the batch size. Default = 1." << std::endl;
};

bool parseVlmBenchmarkArgs(VlmBenchmarkArgs& args, int argc, char* argv[])
{
    static struct option benchmarkOptions[] = {{"textTokenLength", required_argument, 0, 1201},
        {"imageTokenLength", required_argument, 0, 1202}, {"outputLength", required_argument, 0, 1203},
        {"warmUp", required_argument, 0, 1204}, {"numRuns", required_argument, 0, 1205},
        {"loraWeights", required_argument, 0, 1206}, {"batchSize", required_argument, 0, 1207}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::vlmRunOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::vlmRunOptions[i];
    for (int i = 0; benchmarkOptions[i].name != 0; ++i)
        long_options[idx++] = benchmarkOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1)
    {
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg))
        {
            continue;
        }

        if (CommonOptions::parseVLMRunOptions(args.vlmRunParams, opt, optarg))
        {
            continue;
        }

        switch (opt)
        {
        case 1201:
            if (optarg)
            {
                args.textTokenLength = std::stoi(optarg);
            }
            break;
        case 1202:
            if (optarg)
            {
                args.imageTokenLength = std::stoi(optarg);
            }
            break;
        case 1203:
            if (optarg)
            {
                args.outputLength = std::stoi(optarg);
            }
            break;
        case 1204:
            if (optarg)
            {
                args.warmUp = std::stoi(optarg);
            }
            break;
        case 1205:
            if (optarg)
            {
                args.numRuns = std::stoi(optarg);
            }
            break;
        case 1206:
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
        case 1207:
            if (optarg)
            {
                args.batchSize = std::stoi(optarg);
            }
            break;
        default: return false;
        }
    }
    return true;
}

void printBenchmarkResult(std::shared_ptr<BenchmarkProfiler> const profiler, int const batchSize,
    int const textTokenLength, int const imageTokenLength, int const outputLength, size_t const deviceMemorySize)
{
    profiler->stopTiming();
    auto peakDeviceMem = profiler->recordDeviceMemEnd();
    auto peakHostMem = profiler->recordHostMemEnd();

    auto [averagePipeLatency, totalPipeLatency, pipeLatencies] = profiler->getHostElapsedTimeMs("pipeline latency");
    auto [averageVisualLatency, totalVisualLatency, VisualLatencies]
        = profiler->getHostElapsedTimeMs("visual encoder latency");
    auto [averageSeqLatency, duration, seqLatencies] = profiler->getHostElapsedTimeMs("seq latency");
    auto [averageFirstTokenLatency, totalFirstTokenLatency, firstTokenLatencies]
        = profiler->getHostElapsedTimeMs("first token latency");
    auto [averageGenerationTime, totalGenerationTime, generationTimes] = profiler->getDeviceElapsedTimeMs("generation");
    auto decoderSetupLatency = std::get<1>(profiler->getHostElapsedTimeMs("decoder setup"));
    auto seqTokensPerSec = (batchSize * outputLength) / (averageSeqLatency / 1000);
    auto generationTokensPerSec = (batchSize * outputLength) / (averageGenerationTime / 1000);

    LOG_INFO("========================================================");
    LOG_INFO("Benchmarking done. Decoder setup: %.2fs, Iteration: %d, GPU Time: %.2fs.", decoderSetupLatency / 1000,
        seqLatencies.size(), duration / 1000);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Sequence Latencies(ms): [";
    constexpr int maxPrintedLatencies{20};
    for (size_t i = 0; i < pipeLatencies.size(); ++i)
    {
        oss << pipeLatencies[i];
        if (i == pipeLatencies.size() - 1)
        {
            oss << "]";
        }
        else if (pipeLatencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
        {
            oss << " ... ";
            i = pipeLatencies.size() - maxPrintedLatencies / 2;
        }
        else
        {
            oss << ", ";
        }
    }

    LOG_INFO("batch_size: %d", batchSize);
    LOG_INFO("text_token_length per batch: %d", textTokenLength);
    LOG_INFO("visual_token_length per batch: %d", imageTokenLength);
    LOG_INFO("input_length per batch: %d", textTokenLength + imageTokenLength);
    LOG_INFO("output_length per batch: %d", outputLength);
    LOG_INFO("pipeline_latency(ms): %.2f", averagePipeLatency);
    LOG_INFO("visual_encoder_latency(ms): %.2f", averageVisualLatency);
    LOG_INFO("llm_latency(ms): %.2f", averageSeqLatency);
    LOG_INFO("llm_first_token_latency(ms): %.2f", averageFirstTokenLatency);
    LOG_INFO("llm_tokens_per_sec: %.2f", seqTokensPerSec);
    LOG_INFO("llm_generation_latency(ms): %.2f", averageGenerationTime);
    LOG_INFO("llm_generation_tokens_per_sec: %.2f", generationTokensPerSec);
    LOG_INFO("cpu_peak_mem(GiB): %.2f", peakHostMem / 1048576.0);
    LOG_INFO("gpu_peak_mem(GiB): %.2f", peakDeviceMem / 1073741824.0);
    LOG_INFO("execution_context_mem(GiB): %.2f", deviceMemorySize / 1073741824.0);
    LOG_INFO("========================================================");
}

size_t benchmarkMultimodal(BaseParams const& baseParams, VLMRunParams const& vlmRunParams,
    LoraWeights const& loraWeights, GenerationConfig const& generationConfig, int const batchSize,
    int const textTokenLength, int const imageTokenLength, std::vector<std::vector<int32_t>>& outputIds,
    std::shared_ptr<BenchmarkProfiler> const profiler, int const warmUp, int const numRuns)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    profiler->recordHostStart("decoder setup");

    auto multimodalRunner = MultimodalRunner::create(vlmRunParams.visualEngineDir, stream);
    // This script does not support benchmarking VLM Eagle models, passing default EagleParams
    auto llmEngine = getLLMEngine(batchSize, baseParams, EagleParams(), loraWeights, stream);
    llmEngine->setupExtraInputs(multimodalRunner->getComputedEmbeddings());

    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    // Preprocess
    auto modelConfig = llmEngine->getBaseModelConfig();
    int const maxSupportedInputLength = modelConfig.maxSupportedInputLength;
    int const minSupportedInputLength = modelConfig.minSupportedInputLength;
    int const inputLength = textTokenLength + imageTokenLength;
    if (inputLength > maxSupportedInputLength || inputLength < minSupportedInputLength)
    {
        throw std::runtime_error("Input length is out of range: " + std::to_string(inputLength) + " (min: "
            + std::to_string(minSupportedInputLength) + ", max: " + std::to_string(maxSupportedInputLength) + ")");
    }

    std::vector<int32_t> inputIds(batchSize * inputLength, -1);
    std::vector<int32_t> contextLengths(batchSize, inputLength);
    multimodalRunner->initRandomInputs(inputIds, batchSize, imageTokenLength, inputLength, stream);

    // Warmup for profiler
    for (int i = 0; i < warmUp; i++)
    {
        profiler->recordHostStart("pipeline latency");
        multimodalRunner->infer(stream);
        llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("pipeline latency");
    }
    cudaDeviceSynchronize();

    // Benchmark
    profiler->startTiming();
    cudaProfilerStart();
    for (int i = 0; i < numRuns; i++)
    {
        profiler->recordHostStart("pipeline latency");
        profiler->recordHostStart("visual encoder latency");
        multimodalRunner->infer(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("visual encoder latency");

        profiler->recordHostStart("seq latency");
        profiler->recordHostStart("first token latency");
        llmEngine->generate(inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        profiler->recordHostEnd("pipeline latency");
    }
    cudaProfilerStop();
    profiler->stopTiming();

    // Cleanup
    size_t deviceMemorySize = llmEngine->getDeviceMemorySize();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return deviceMemorySize;
}

void benchmark(VlmBenchmarkArgs const& args)
{
    int totalSeqLength = args.textTokenLength + args.imageTokenLength + args.outputLength;
    GenerationConfig generationConfig{totalSeqLength, totalSeqLength, 1, 1};

    std::vector<std::vector<int32_t>> outputIds(args.batchSize);
    for (int i = 0; i < args.batchSize; ++i)
    {
        outputIds[i].reserve(args.outputLength);
    }

    auto profiler = std::make_shared<BenchmarkProfiler>();
    size_t deviceMemorySize;

    deviceMemorySize = benchmarkMultimodal(args.baseParams, args.vlmRunParams, args.loraWeights, generationConfig,
        args.batchSize, args.textTokenLength, args.imageTokenLength, outputIds, profiler, args.warmUp, args.numRuns);

    printBenchmarkResult(
        profiler, args.batchSize, args.textTokenLength, args.imageTokenLength, args.outputLength, deviceMemorySize);
}

int main(int argc, char* argv[])
{
    VlmBenchmarkArgs args;
    if ((argc < 2) || (!parseVlmBenchmarkArgs(args, argc, argv)))
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

    if ((args.textTokenLength < 1) || (args.imageTokenLength < 1) || (args.outputLength < 1))
    {
        LOG_ERROR("Please specify --textTokenLength, --imageTokenLength and --outputLength for benchmark.");
        return EXIT_FAILURE;
    }

    benchmark(args);

    return EXIT_SUCCESS;
};