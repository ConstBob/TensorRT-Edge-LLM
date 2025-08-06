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
#include "internvl3/vit_runner.h"
#include "llm_param.h"
#include "qwen2vl/vit_runner.h"
#include "tokenizer/tokenizer.h"
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
              << " [--help] [--enginePath=<path to LLM engine>] [--visualEnginePath=<path to visual engine>]"
                 " <--imageTokenLength int> <--textTokenLength int> <--outputLength int> [--warmUp int] [--numRuns "
                 "int] [--batchSize int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --textTokenLength   Provide the number of text tokens to the runtime. Required. " << std::endl;
    std::cerr << "  --imageTokenLength  Provide the number of image tokens to the runtime. Required. " << std::endl;
    std::cerr << "  --outputLength      Provide the output token length for the generation session (NOT including the "
                 "input). Required."
              << std::endl;
    std::cerr << "  --warmUp            Provide warm up iterations before benchmark starts. Default = 2." << std::endl;
    std::cerr << "  --numRuns           Minimal number of iterations to run during benchmarking. Default = 10."
              << std::endl;
    std::cerr << "  --batchSize         Provide the batch size for benchmarking. Default = 1." << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printVLMRunOptions();
    CommonUsage::printLoraOptions();
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
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg, false))
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

size_t benchmarkQwen2VL(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    GenerationConfig const& generationConfig, int const batchSize, int const textTokenLength,
    int const imageTokenLength, std::vector<std::vector<int64_t>>& outputIds,
    std::shared_ptr<BenchmarkProfiler> const profiler, int const warmUp, int const numRuns, bool useCudaGraph,
    std::string const& modelType, LoraWeights const& loraWeights)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new Qwen2ViTRunner(modelType);
    auto decoder = new Decoder();

    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    profiler->recordHostStart("decoder setup");
    vitrunner->setup(visualEnginePath, stream, batchSize);
    decoder->setup(llmEnginePath, stream, useCudaGraph, batchSize);
    decoder->setupExtraInputs(vitrunner->getExtraLLMInputs());

    // Load and switch to LoRA weights if provided
    if (loraWeights.hasWeights())
    {
        auto loraPair = loraWeights.getFirst();
        if (!decoder->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return 0;
        }
        if (!decoder->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return 0;
        }
    }

    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    // Preprocess
    std::vector<half> visualInput;
    std::vector<half> visualAttentionMask;
    std::vector<float> visualRotaryPosEmb;
    std::vector<int64_t> inputIds(batchSize * decoder->getMaxSupportedInputLength(), -1);
    std::vector<int32_t> contextLengths(batchSize, textTokenLength + imageTokenLength);
    // Only initialized for qwen2_5_vl
    std::vector<half> visualWindowAttentionMask;
    std::vector<int64_t> visualWindowIndex;
    std::vector<int64_t> reverseWindowIndex;

    vitrunner->initRandomInputs(visualInput, visualAttentionMask, visualRotaryPosEmb, visualWindowAttentionMask,
        visualWindowIndex, reverseWindowIndex, inputIds, textTokenLength, imageTokenLength,
        decoder->getMaxSupportedInputLength());

    for (int i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("pipeline latency");
        if (modelType == "qwen2_vl")
        {
            vitrunner->qwen2ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        }
        else
        {
            vitrunner->qwen2_5ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb, visualWindowAttentionMask,
                visualWindowIndex, reverseWindowIndex);
        }
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
        if (modelType == "qwen2_vl")
        {
            vitrunner->qwen2ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        }
        else
        {
            vitrunner->qwen2_5ViTInfer(visualInput, visualAttentionMask, visualRotaryPosEmb, visualWindowAttentionMask,
                visualWindowIndex, reverseWindowIndex);
        }
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

    size_t deviceMemorySize = decoder->getDeviceMemorySize();

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete vitrunner;
    delete decoder;

    return deviceMemorySize;
}

size_t benchmarkInternVL3(std::filesystem::path const& llmEnginePath, std::filesystem::path const& visualEnginePath,
    GenerationConfig const& generationConfig, int const batchSize, int const textTokenLength,
    int const imageTokenLength, std::vector<std::vector<int64_t>>& outputIds,
    std::shared_ptr<BenchmarkProfiler> const profiler, int const warmUp, int const numRuns, bool useCudaGraph,
    std::string const& modelType, LoraWeights const& loraWeights)
{
    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto vitrunner = new InternVLViTRunner(modelType);
    auto decoder = new Decoder();

    // Initialize rope_rotary_cos_sin
    std::string baseFolderPath = extractFolderName(llmEnginePath);
    std::string configPath = baseFolderPath + "/config.json";

    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    profiler->recordHostStart("decoder setup");
    vitrunner->setup(visualEnginePath, stream, batchSize);
    decoder->setup(llmEnginePath, stream, useCudaGraph, batchSize);
    decoder->setupExtraInputs(vitrunner->getExtraLLMInputs());
    decoder->setupRopeCosSin(configPath);

    // Load and switch to LoRA weights if provided
    if (loraWeights.hasWeights())
    {
        auto loraPair = loraWeights.getFirst();
        if (!decoder->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return 0;
        }
        if (!decoder->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return 0;
        }
    }

    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    // Preprocess
    std::vector<half> visualInput;
    std::vector<int64_t> inputIds(batchSize * decoder->getMaxSupportedInputLength(), -1);
    std::vector<int32_t> contextLengths(batchSize, textTokenLength + imageTokenLength);

    vitrunner->initRandomInputs(
        visualInput, inputIds, textTokenLength, imageTokenLength, decoder->getMaxSupportedInputLength());
    for (int i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("pipeline latency");
        vitrunner->internVLViTInfer(visualInput);

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
        vitrunner->internVLViTInfer(visualInput);

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
    size_t deviceMemorySize = decoder->getDeviceMemorySize();

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete vitrunner;
    delete decoder;

    return deviceMemorySize;
}

void benchmarkVLM(VlmBenchmarkArgs const& args)
{
    int totalSeqLength = args.textTokenLength + args.imageTokenLength + args.outputLength;
    GenerationConfig generationConfig{totalSeqLength, totalSeqLength, 1, 1};

    std::vector<std::vector<int64_t>> outputIds(args.batchSize);
    for (int i = 0; i < args.batchSize; ++i)
    {
        outputIds[i].reserve(args.outputLength);
    }

    auto profiler = std::make_shared<BenchmarkProfiler>();
    size_t deviceMemorySize;

    if (args.vlmRunParams.modelType == "qwen2_vl" || args.vlmRunParams.modelType == "qwen2_5_vl")
    {
        deviceMemorySize = benchmarkQwen2VL(args.baseParams.enginePath, args.vlmRunParams.visualEnginePath,
            generationConfig, args.batchSize, args.textTokenLength, args.imageTokenLength, outputIds, profiler,
            args.warmUp, args.numRuns, !args.baseParams.noCudaGraph, args.vlmRunParams.modelType, args.loraWeights);
    }
    else if (args.vlmRunParams.modelType == "internvl3")
    {
        deviceMemorySize = benchmarkInternVL3(args.baseParams.enginePath, args.vlmRunParams.visualEnginePath,
            generationConfig, args.batchSize, args.textTokenLength, args.imageTokenLength, outputIds, profiler,
            args.warmUp, args.numRuns, !args.baseParams.noCudaGraph, args.vlmRunParams.modelType, args.loraWeights);
    }
    else
    {
        throw std::runtime_error("Only support Qwen2-VL and InternVL3 models for Multimodal models.");
    }

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
    benchmarkVLM(args);

    return EXIT_SUCCESS;
};