
#include "common/common.h"
#include "decoder.h"
#include "tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <bits/getopt_core.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace nvinfer1;

enum class MODE
{
    kInference,
    kBenchmark
};

struct RuntimeArgs
{
    bool help{false};
    std::vector<std::string> inputString;
    std::string enginePath;
    std::string tokenizerPath;
    int maxLength{40};
    int inputLength;
    MODE mode{MODE::kInference};
    int64_t numRuns{10};
    int64_t warmUp{2};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-i or --inputString=<input>] [-e or --enginePath=<path to TensorRT engine>] [-s or "
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --inputString    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --inputLength    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 40"
              << std::endl;
    std::cerr << "  --mode           Provide the mode. " << std::endl;
    std::cerr << "  --warm_up        [Benchmark] Provide warm up iterations before benchmark starts." << std::endl;
    std::cerr << "  --num_runs       [Benchmark] Minimal number of iterations to run during benchmarking." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"inputString", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'e'}, {"tokenizerPath", required_argument, 0, 't'},
        {"maxLength", required_argument, 0, 's'}, {"inputLength", required_argument, 0, 'c'},
        {"mode", required_argument, 0, 'm'}, {"warm_up", required_argument, 0, 'w'},
        {"num_runs", required_argument, 0, 'r'}, {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "h:iest", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.inputString.emplace_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                return false;
            }
            break;
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
        case 'm':
            if (optarg)
            {
                if (strcmp(optarg, "benchmark") == 0)
                {
                    args.mode = MODE::kBenchmark;
                }
                else
                {
                    args.mode = MODE::kInference;
                }
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
        case 'd': args.debug = true; break;
        default: return false;
        }
    }
    return true;
}

void prepareBatchInputIds(std::vector<std::string> const& inputString, Tokenizer* tokenizer, std::vector<int64_t>& inputIds, 
    std::vector<int32_t>& contextLengths, std::vector<int64_t>& lastTokenIds, int& maxContextLength)
{
    std::vector<std::vector<int64_t>> batchInputIds;
    int batchSize = inputString.size();
    for (int i = 0; i < batchSize; ++i)
    {
        auto ids = tokenizer->encode(inputString[i], true);
        batchInputIds.emplace_back(ids);
        contextLengths[i] = ids.size();
        lastTokenIds[i] = static_cast<int64_t>(ids.size() - 1);
    }

    // right padding
    int64_t padId = tokenizer->getPadId();
    inputIds.resize(batchSize * maxContextLength, padId);
    for (int i = 0; i < batchSize; ++i)
    {
        std::copy(batchInputIds[i].begin(), batchInputIds[i].end(), inputIds.begin() + i * maxContextLength);
    }
}

std::vector<std::string> decode(std::filesystem::path const& enginePath, std::vector<std::string> const& inputString, Tokenizer* tokenizer,
    GenerationConfig const& generationConfig, bool debug = false)
{
    int batchSize = inputString.size();

    std::vector<int64_t> inputIds;  //flattened
    std::vector<int32_t> contextLengths(batchSize);
    std::vector<int64_t> lastTokenIds(batchSize);
    int32_t maxContextLength{128};
    prepareBatchInputIds(inputString, tokenizer, inputIds, contextLengths, lastTokenIds, maxContextLength);

    if (debug)
    {
        std::ostringstream oss;
        for (int i = 0; i < batchSize; ++i)
        {
            oss << "contextLengths[" << i << "] is " << contextLengths[i] << ". inputIds[" << i << "] is: [";
            for (int j = 0; j < maxContextLength; ++j)
            {
                oss << inputIds[i * maxContextLength + j] << ",";
            }
            oss << "]\n";
        }
        LOG_DEBUG(oss.str().c_str());
    }

    Decoder* decoder = new Decoder();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    decoder->setup(enginePath, stream);
    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    decoder->generate(inputIds, contextLengths, lastTokenIds, outputIds, generationConfig, maxContextLength, tokenizer->getEosId());

    std::vector<std::string> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        if (debug)
        {
            LOG_DEBUG("Output%d length is %d", i, outputIds[i].size());
        }
        output[i] = tokenizer->decode(outputIds[i]);
        LOG_INFO("Input%d is: %s", i, inputString[i].c_str());
        LOG_INFO("Output%d is: %s", i, output[i].c_str());
    }
    return output;
}

void benchmark(std::filesystem::path const& enginePath, int const& inputLength, int64_t warmUp,
    int64_t numRuns, GenerationConfig const& generationConfig)
{
    auto profiler = std::make_shared<BenchmarkProfiler>();
    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    Decoder* decoder = new Decoder();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    profiler->recordHostStart("decoder setup");
    decoder->setup(enginePath, stream);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    int64_t batchSize = decoder->getModelBatchSize();
    int32_t maxContextLength{128};
    std::vector<int64_t> inputIds(batchSize * maxContextLength, -1);
    std::vector<int32_t> contextLengths(batchSize, inputLength);
    std::vector<int64_t> lastTokenIds(batchSize, inputLength - 1);
    std::vector<std::vector<int64_t>> outputIds(batchSize);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
    for (int i = 0; i < batchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * maxContextLength;
        std::generate(beginIter, beginIter + inputLength, [&rng, &dist]() { return dist(rng); });
        outputIds[i].reserve(generationConfig.maxLength);
    }

    for (int64_t i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("seq latency");
        decoder->generate(inputIds, contextLengths, lastTokenIds, outputIds, generationConfig, maxContextLength, -1, profiler);
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
        decoder->generate(inputIds, contextLengths, lastTokenIds, outputIds, generationConfig, maxContextLength, -1, profiler);
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

    auto maxNewTokens = batchSize * generationConfig.maxLength;
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

    void* handle = dlopen("plugins/libLLamaPlugin.so", RTLD_LAZY);
    if (!handle)
    {
        LOG_ERROR("Cannot open library: %s", dlerror());
        return EXIT_FAILURE;
    }

    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};

    switch (args.mode)
    {
    case MODE::kBenchmark:
    {
        if (args.inputLength < 1)
        {
            LOG_ERROR("Please specify the input length");
            return EXIT_FAILURE;
        }
        benchmark(args.enginePath, args.inputLength, args.warmUp, args.numRuns, generationConfig);
        break;
    }
    case MODE::kInference:
    {
        Tokenizer* tokenizer = new LlamaV3Tokenizer();
        tokenizer->loadFromHF(args.tokenizerPath);
        auto output = decode(args.enginePath, args.inputString, tokenizer, generationConfig, args.debug);
        break;
    }
    }

    dlclose(handle);

    return EXIT_SUCCESS;
};
