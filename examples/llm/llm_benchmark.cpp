
#include "common/common.h"
#include "common/trtUtils.h"
#include "decoder/decoder.h"
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
    bool help{false};
    std::string enginePath;
    int maxLength{0};
    int inputLength{0};
    int64_t numRuns{10};
    int64_t warmUp{2};
    bool debug{false};
    bool noCudaGraph{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--enginePath str> <--inputLength int> <--maxLength int> [--warmUp int] [--numRuns int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --inputLength    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Required."
              << std::endl;
    std::cerr << "  --warmUp         Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns        Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --noCudaGraph    Cuda graph is default enabled. Use this flag to disable cuda graph." << std::endl;
};

bool parseLLMBenchmarkArgs(LLMBenchmarkArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"enginePath", required_argument, 0, 'e'},
        {"maxLength", required_argument, 0, 's'}, {"inputLength", required_argument, 0, 'c'},
        {"warmUp", required_argument, 0, 'w'}, {"numRuns", required_argument, 0, 'r'},
        {"noCudaGraph", no_argument, 0, 'g'}, {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "he:s:c:w:r:d", long_options, nullptr)) != -1)
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
        case 'g': args.noCudaGraph = true; break;
        case 'd': args.debug = true; break;
        default: return false;
        }
    }
    return true;
}

void benchmarkLLM(std::string& enginePath, int const inputLength, int64_t warmUp, int64_t numRuns,
    GenerationConfig const& generationConfig, bool useCudaGraph)
{
    auto profiler = std::make_shared<BenchmarkProfiler>();
    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    auto decoder = std::make_unique<Decoder<half>>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    profiler->recordHostStart("decoder setup");
    decoder->setup(enginePath, stream, useCudaGraph);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    int64_t batchSize = decoder->getModelBatchSize();
    std::vector<int64_t> inputIds(batchSize * decoder->getMaxContextLength(), -1);
    std::vector<int32_t> contextLengths(batchSize, inputLength);
    std::vector<int64_t> lastTokenIds(batchSize, inputLength - 1);
    std::vector<std::vector<int64_t>> outputIds(batchSize);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
    for (int i = 0; i < batchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * decoder->getMaxContextLength();
        std::generate(beginIter, beginIter + inputLength, [&rng, &dist]() { return dist(rng); });
        outputIds[i].reserve(generationConfig.maxLength);
    }

    for (int64_t i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("seq latency");
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaDeviceSynchronize();

    profiler->startTiming();
    cudaProfilerStart();
    for (int64_t i = 0; i < numRuns; i++)
    {
        profiler->recordHostStart("seq latency");
        profiler->recordHostStart("first token latency");
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaProfilerStop();
    profiler->stopTiming();
    auto peakDeviceMem = profiler->recordDeviceMemEnd();
    auto peakHostMem = profiler->recordHostMemEnd();

    auto maxNewTokens = batchSize * (generationConfig.maxLength - inputLength);
    auto [averageSeqLatency, duration, seqLatencies] = profiler->getHostElapsedTimeMs("seq latency");
    auto [averageFirstTokenLatency, totalFirstTokenLatency, firstTokenLatencies]
        = profiler->getHostElapsedTimeMs("first token latency");
    auto [averageGenerationTime, totalGenerationTime, generationTimes] = profiler->getDeviceElapsedTimeMs("generation");
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
    for (int i = 0; i < seqLatencies.size(); ++i)
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
    for (int i = 0; i < firstTokenLatencies.size(); ++i)
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
    LOG_INFO("input_length per batch: %d", inputLength);
    LOG_INFO("output_length per batch: %d", maxNewTokens);
    LOG_INFO("seq_latency(ms): %.2f", averageSeqLatency);
    LOG_INFO("first_token_latency(ms): %.2f", averageFirstTokenLatency);
    LOG_INFO("tokens_per_sec: %.2f", tokensPerSec);
    LOG_INFO("generation_time(ms): %.2f", averageGenerationTime);
    LOG_INFO("generation_tokens_per_sec: %.2f", generationTokensPerSec);
    LOG_INFO("cpu_peak_mem(GiB): %.2f", peakHostMem / 1048576.0);
    LOG_INFO("gpu_peak_mem(GiB): %.2f", peakDeviceMem / 1073741824.0);
    LOG_INFO("execution_context_mem(GiB): %.2f", decoder->getDeviceMemorySize() / 1073741824.0);
    LOG_INFO("========================================================");
}

int main(int argc, char* argv[])
{
    LLMBenchmarkArgs args;
    if ((argc < 2) || (!parseLLMBenchmarkArgs(args, argc, argv)))
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

    if ((args.inputLength < 1) || (args.maxLength < 1))
    {
        LOG_ERROR("Please specify --inputLength and --maxLength for benchmark.");
        return EXIT_FAILURE;
    }
    benchmarkLLM(args.enginePath, args.inputLength, args.warmUp, args.numRuns, generationConfig, !args.noCudaGraph);
    return EXIT_SUCCESS;
};
