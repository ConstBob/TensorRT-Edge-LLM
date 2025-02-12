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

struct RuntimeArgs
{
    bool help{false};
    std::string llmEnginePath;
    std::string visualEnginePath;
    int batchSize{1};
    int textTokenLength{0};
    int imageTokenLength{0};
    int outputLength{0};
    std::string modelType{"qwen2_vl"};
    int numRuns{10};
    int warmUp{2};
    bool noCudaGraph{false};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-e or --llmEnginePath=<path to LLM engine>] [-v or --visualEnginePath=<path to visual engine>]"
                 " <--imageTokenLength int> <--textTokenLength int> <--outputLength int> [--warmUp int] [--numRuns int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h                  Display this help message" << std::endl;
    std::cerr << "  --llmEnginePath     Provide the Qwen TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize         Provide the batch size to benchmark. Default is 1. " << std::endl;
    std::cerr << "  --textTokenLength   Provide the number of text tokens to the runtime. Required. " << std::endl;
    std::cerr << "  --imageTokenLength  Provide the number of image tokens to the runtime. Required. " << std::endl;
    std::cerr << "  --outputLength      Provide the output token length for the generation session (NOT including the "
                 "input). Required."
              << std::endl;
    std::cerr << "  --modelType         Provide the model type. Default = qwen2_vl." << std::endl;
    std::cerr << "  --warmUp            Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns           Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    std::cerr << "  --noCudaGraph       Cuda graph is default enabled. Use this flag to disable cuda graph."
              << std::endl;
    std::cerr << "  --debug             Use debug mode, which outputs tensors." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"llmEnginePath", required_argument, 0, 'e'},
        {"visualEnginePath", required_argument, 0, 'v'}, {"textTokenLength", required_argument, 0, 't'},
        {"imageTokenLength", required_argument, 0, 'i'}, {"outputLength", required_argument, 0, 'o'},
        {"modelType", required_argument, 0, 0}, {"warmUp", required_argument, 0, 'w'},
        {"batchSize", required_argument, 0, 'b'}, {"numRuns", required_argument, 0, 'r'},
        {"noCudaGraph", no_argument, 0, 'g'}, {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "he:v:t:i:o:w:r:d", long_options, &option_index)) != -1)
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
        case 'b':
            if (optarg)
            {
                args.batchSize = std::stoi(optarg);
            }
            break;
        case 'o':
            if (optarg)
            {
                args.outputLength = std::stoi(optarg);
            }
            break;
        case 't':
            if (optarg)
            {
                args.textTokenLength = std::stoi(optarg);
            }
            break;
        case 'i':
            if (optarg)
            {
                args.imageTokenLength = std::stoi(optarg);
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
        case 'g': args.noCudaGraph = true; break;
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
    for (int i = 0; i < pipeLatencies.size(); ++i)
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

size_t benchmarkQwen2VL(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    GenerationConfig const& generationConfig, int const batchSize, int const textTokenLength,
    int const imageTokenLength, std::vector<std::vector<int64_t>>& outputIds,
    std::shared_ptr<BenchmarkProfiler> const profiler, int const warmUp, int const numRuns, bool useCudaGraph)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner();
    auto decoder = new Decoder<half>();

    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    profiler->recordHostStart("decoder setup");
    vitrunner->setup(visualEnginePath, stream, batchSize);
    decoder->setup(llmEnginePath, stream, useCudaGraph, batchSize);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    // Preprocess
    std::vector<half> visualInput;
    std::vector<half> visualAttentionMask;
    std::vector<float> visualRotaryPosEmb;
    std::vector<int64_t> inputIds(batchSize * decoder->getMaxContextLength(), -1);
    std::vector<int32_t> contextLengths(batchSize, textTokenLength + imageTokenLength);

    vitrunner->initRandomInputs(visualInput, visualAttentionMask, visualRotaryPosEmb, inputIds, textTokenLength,
        imageTokenLength, decoder->getMaxContextLength());
    decoder->setupExtraInputs(vitrunner->getExtraLLMInputs());

    for (int i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("pipeline latency");
        vitrunner->visualInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("pipeline latency");
    }
    cudaDeviceSynchronize();

    profiler->startTiming();
    cudaProfilerStart();
    for (int i = 0; i < numRuns; i++)
    {
        profiler->recordHostStart("pipeline latency");
        profiler->recordHostStart("visual encoder latency");
        vitrunner->visualInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("visual encoder latency");

        profiler->recordHostStart("seq latency");
        profiler->recordHostStart("first token latency");
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        profiler->recordHostEnd("pipeline latency");
    }
    cudaProfilerStop();
    profiler->stopTiming();

    return decoder->getDeviceMemorySize();
}

void benchmarkVLM(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    int const batchSize, int const textTokenLength, int const imageTokenLength, int const outputLength,
    std::string modelType, int const warmUp, int const numRuns, bool useCudaGraph)
{
    int totalSeqLength = textTokenLength + imageTokenLength + outputLength;
    GenerationConfig generationConfig{totalSeqLength, totalSeqLength, 1, 0};

    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(outputLength);
    }

    auto profiler = std::make_shared<BenchmarkProfiler>();
    size_t deviceMemorySize;

    if (modelType == "qwen2_vl")
    {
        deviceMemorySize = benchmarkQwen2VL(llmEnginePath, visualEnginePath, generationConfig, batchSize,
            textTokenLength, imageTokenLength, outputIds, profiler, warmUp, numRuns, useCudaGraph);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL model for Multimodal models.");
    }

    printBenchmarkResult(profiler, batchSize, textTokenLength, imageTokenLength, outputLength, deviceMemorySize);
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

    if ((args.textTokenLength < 1) || (args.imageTokenLength < 1) || (args.outputLength < 1))
    {
        LOG_ERROR("Please specify --textTokenLength, --imageTokenLength and --outputLength for benchmark.");
        return EXIT_FAILURE;
    }
    benchmarkVLM(args.llmEnginePath, args.visualEnginePath, args.batchSize, args.textTokenLength, args.imageTokenLength,
        args.outputLength, args.modelType, args.warmUp, args.numRuns, !args.noCudaGraph);

    return EXIT_SUCCESS;
};