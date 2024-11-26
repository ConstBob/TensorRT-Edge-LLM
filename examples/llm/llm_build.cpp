
#include "NvOnnxParser.h"
#include "common/common.h"
#include <NvInfer.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxPath;
    std::string enginePath;
    int64_t batchSize{1};
    int64_t maxInputLen{128};
    int64_t maxSeqLen{4096};
    bool dynamicShape{false};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] <--onnxPath str> <--enginePath str> [-b or "
                 "--batchSize int] [-c or --maxInputLen int] [-s or --maxSeqLen int] [--dynamicShape] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --onnxPath       Provide the input onnx file path. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the output TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize      Provide the desired batch_size for builder. Default = 1" << std::endl;
    std::cerr << "  --maxInputLen    Provide the maximum input length for the model. Default = 128" << std::endl;
    std::cerr
        << "  --maxSeqLen      Provide the maximum output length for the model (including the input). Default = 4096"
        << std::endl;
    std::cerr << "  --dynamicShape   Use dynamic shape profiles." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more logs." << std::endl;
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"onnxPath", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'o'}, {"batchSize", required_argument, 0, 'b'},
        {"maxInputLen", required_argument, 0, 'c'}, {"maxSeqLen", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {"dynamicShape", no_argument, 0, 'y'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "hi:o:b:c:s:dy", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.onnxPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: ONNX Path requires option argument" << std::endl;
                return false;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.enginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: Output Dir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batchSize = std::stoi(optarg);
            }
            break;
        case 'c':
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case 's':
            if (optarg)
            {
                args.maxSeqLen = std::stoi(optarg);
            }
            break;
        case 'd': args.debug = true; break;
        case 'y': args.dynamicShape = true; break;
        default: std::cerr << "ERROR: Invalid Argument" << fmtstr("%c is %s", opt, optarg) << std::endl; return false;
        }
    }
    return true;
}

bool setOptimizationProfile(IOptimizationProfile* profile, char const* inputName, Dims const* dims,
    Dims const* minDims = nullptr, Dims const* optDims = nullptr, Dims const* maxDims = nullptr)
{
    if (dims != nullptr)
    {
        return profile->setDimensions(inputName, OptProfileSelector::kMIN, *dims)
            && profile->setDimensions(inputName, OptProfileSelector::kOPT, *dims)
            && profile->setDimensions(inputName, OptProfileSelector::kMAX, *dims);
    }
    else
    {
        if ((minDims == nullptr) || (optDims == nullptr) || (maxDims == nullptr))
        {
            throw std::runtime_error("Cannot set optimization profiles. One of minDims, optDims and maxDims is empty.");
        }
        return profile->setDimensions(inputName, OptProfileSelector::kMIN, *minDims)
            && profile->setDimensions(inputName, OptProfileSelector::kOPT, *optDims)
            && profile->setDimensions(inputName, OptProfileSelector::kMAX, *maxDims);
    }
}

std::string generateTRTExecCommand(
    LLMBuildArgs& args, int64_t& numKVHeads, int64_t& hiddenSizePerHead, char const* pluginPath)
{
    int64_t minInputLen = args.maxInputLen;
    int64_t optInputLen = args.maxInputLen;
    int64_t maxInputLen = args.maxInputLen;
    if (args.dynamicShape)
    {
        minInputLen = 1;
        optInputLen = maxInputLen / 2;
    }

    int64_t batchSize = args.batchSize;
    int64_t maxLength = args.maxSeqLen;

    std::string trtExecCommand = fmtstr(
        "Equivalent trtexec command: trtexec --onnx=%s --saveEngine=%s --staticPlugins=%s --stronglyTyped --verbose "
        "--profile 0 "
        "--minShape=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--optShape=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--maxShape=input_ids:%ldx%ld,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx0x%ld "
        "--profile 1 "
        "--minShape=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx%ldx%ld "
        "--optShape=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx%ldx%ld "
        "--maxShape=input_ids:%ldx1,context_lengths:%ld,last_token_ids:%ld*1,past_key_values.*:%ldx2x%ldx%ldx%ld",
        args.onnxPath.c_str(), args.enginePath.c_str(), pluginPath, batchSize, minInputLen, batchSize, batchSize,
        batchSize, numKVHeads, hiddenSizePerHead, batchSize, optInputLen, batchSize, batchSize, batchSize, numKVHeads,
        hiddenSizePerHead, batchSize, maxInputLen, batchSize, batchSize, batchSize, numKVHeads, hiddenSizePerHead,
        batchSize, batchSize, batchSize, batchSize, numKVHeads, maxLength, hiddenSizePerHead, batchSize, batchSize,
        batchSize, batchSize, numKVHeads, maxLength, hiddenSizePerHead, batchSize, batchSize, batchSize, batchSize,
        numKVHeads, maxLength, hiddenSizePerHead);
    return trtExecCommand;
}

Dims* createDims(std::vector<int64_t> const& shape)
{
    Dims* dims = new Dims();
    dims->nbDims = shape.size();
    for (int i = 0; i < shape.size(); ++i)
    {
        dims->d[i] = shape[i];
    }
    return dims;
}

int main(int argc, char** argv)
{
    LLMBuildArgs args;
    if ((argc < 2) || (!parseLLMBuildArgs(args, argc, argv)))
    {
        std::cerr << "Unable to parse builder args" << std::endl;
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        std::cout << "find help mode is True" << std::endl;
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

    char const* pluginPath = std::getenv("PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        LOG_INFO("PLUGIN_PATH variable is not set. Default to build/libAttentionPlugin.so");
        pluginPath = "build/libAttentionPlugin.so";
    }

    void* handle = dlopen(pluginPath, RTLD_LAZY);
    if (!handle)
    {
        LOG_ERROR("Cannot open library: %s", dlerror());
        return EXIT_FAILURE;
    }

    // Create the builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        LOG_ERROR("Failed to create builder.");
        return EXIT_FAILURE;
    }

    // Create the network definition
    auto const stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network)
    {
        LOG_ERROR("Failed to create network.");
        return EXIT_FAILURE;
    }

    // Create the ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        LOG_ERROR("Failed to create parser.");
        return EXIT_FAILURE;
    }

    // Parse the ONNX model
    if (!parser->parseFromFile(args.onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", args.onnxPath.c_str());
        return EXIT_FAILURE;
    }

    // Build the engine
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config.");
        return EXIT_FAILURE;
    }

    int32_t const nbInputs = network->getNbInputs();
    // Excluding input_ids, context_lengths and last_token_ids
    int32_t const nbKVCacheInputs = nbInputs - 3;

    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    // Look for KV Cache
    int64_t i = 0;
    while ((i < network->getNbInputs()) && (network->getInput(i)->getDimensions().nbDims != 5))
    {
        ++i;
    }
    if (i == network->getNbInputs())
    {
        LOG_ERROR("Cannot find KV Cache input. KV Cache is guaranteed to have 5 dimensions in Plugin network!");
        return EXIT_FAILURE;
    }
    Dims kvDims = network->getInput(i)->getDimensions();
    int64_t numKVHeads = kvDims.d[2];
    int64_t hiddenSizePerHead = kvDims.d[4];

    // set dimensions for input tensors. Only context phase input can possibly be dynamic
    bool result = true;
    if (args.dynamicShape)
    {
        result &= setOptimizationProfile(contextProfile, "input_ids", nullptr, createDims({args.batchSize, 1}),
            createDims({args.batchSize, args.maxInputLen / 2}), createDims({args.batchSize, args.maxInputLen}));
    }
    else
    {
        result &= setOptimizationProfile(contextProfile, "input_ids", createDims({args.batchSize, args.maxInputLen}));
    }
    result &= setOptimizationProfile(contextProfile, "context_lengths", createDims({args.batchSize}));
    result &= setOptimizationProfile(contextProfile, "last_token_ids", createDims({args.batchSize, 1}));
    result &= setOptimizationProfile(generationProfile, "input_ids", createDims({args.batchSize, 1}));
    result &= setOptimizationProfile(generationProfile, "context_lengths", createDims({args.batchSize}));
    result &= setOptimizationProfile(generationProfile, "last_token_ids", createDims({args.batchSize, 1}));
    // Plugin does not have any dynamic shape.
    Dims* kvCacheContextShape = createDims({args.batchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    Dims* kvCacheGenerationShape = createDims({args.batchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});

    for (int i = 0; i < nbKVCacheInputs; ++i)
    {
        result &= setOptimizationProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheContextShape);
        result &= setOptimizationProfile(
            generationProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheGenerationShape);
    }

    if (!result)
    {
        LOG_ERROR("Issues setting up optimization profile");
        return EXIT_FAILURE;
    }

    config->addOptimizationProfile(contextProfile);
    config->addOptimizationProfile(generationProfile);
    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine)
    {
        LOG_ERROR("Failed to build engine.");
        return EXIT_FAILURE;
    }

    std::ofstream ofs(args.enginePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", args.enginePath.c_str());
        return EXIT_FAILURE;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", args.enginePath.c_str());
    LOG_INFO(generateTRTExecCommand(args, numKVHeads, hiddenSizePerHead, pluginPath).c_str());
    dlclose(handle);
    return EXIT_SUCCESS;
}