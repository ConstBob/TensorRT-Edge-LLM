
#include "NvOnnxParser.h"
#include "common.h"
#include <NvInfer.h>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace std;
using namespace nvinfer1;

struct BuilderArgs
{
    bool help{false};
    std::string onnxPath;
    std::string enginePath;
    int64_t batchSize{1};
    int64_t maxInputLen{128};
    int64_t maxSeqLen{256};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-i or --onnxPath=<path to onnx file>] [-o or --enginePath=<path to TensorRT engine] [-b or "
                 "--batchSize=<int>] [-c or --maxInputLen=<int>] [-s or --maxSeqLen=<int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --onnxPath       Provide the input onnx file path. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the output TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize      Provide the desired batch_size for builder. Default = 1" << std::endl;
    std::cerr << "  --maxInputLen    Provide the maximum input length for the model. Default = 128" << std::endl;
    std::cerr
        << "  --maxSeqLen      Provide the maximum output length for the model (including the input). Default = 256"
        << std::endl;
    std::cerr
        << "  --maxSeqLen      Provide the maximum output length for the model (including the input). Default = 256"
        << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs more logs." << std::endl;
}

bool parseBuilderArgs(BuilderArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"onnxPath", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'o'}, {"batchSize", required_argument, 0, 'b'},
        {"maxInputLen", required_argument, 0, 'c'}, {"maxSeqLen", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "h:iobcs", long_options, nullptr)) != -1)
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
        default:
            std::cerr << "ERROR: Output Dir requires option argument" << fmtstr("%c is %s", opt, optarg) << std::endl;
            return false;
        }
    }
    return true;
}

bool setStaticProfile(IOptimizationProfile* profile, char const* inputName, Dims const& dims)
{

    return profile->setDimensions(inputName, OptProfileSelector::kMIN, dims)
        && profile->setDimensions(inputName, OptProfileSelector::kOPT, dims)
        && profile->setDimensions(inputName, OptProfileSelector::kMAX, dims);
    return profile->setDimensions(inputName, OptProfileSelector::kMIN, dims)
        && profile->setDimensions(inputName, OptProfileSelector::kOPT, dims)
        && profile->setDimensions(inputName, OptProfileSelector::kMAX, dims);
}

Dims createDims(std::vector<int64_t> const& shape)
{
    Dims dims;
    dims.nbDims = shape.size();
    for (int i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}

int main(int argc, char** argv)
{
    BuilderArgs args;
    if ((argc < 2) || (!parseBuilderArgs(args, argc, argv)))
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

    void* handle = dlopen("../plugins/build/libLLamaPlugin.so", RTLD_LAZY);
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

    int32_t nbInputs = network->getNbInputs();
    // Excluding input_ids and context length
    int32_t nbLayers = nbInputs - 2;

    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    // Location 3 is guaranteed to be one of the KV Cache inputs
    Dims kvDims = network->getInput(3)->getDimensions();
    int64_t numKVHeads = kvDims.d[2];
    int64_t hiddenSizePerHead = kvDims.d[4];

    Dims inputIdsContextShape = createDims({args.batchSize, args.maxInputLen});
    Dims kvCacheContextShape = createDims({args.batchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    Dims inputIdsGenerationShape = createDims({args.batchSize, 1});
    Dims kvCacheGenerationShape = createDims({args.batchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});

    std::vector<int32_t> const minContextShape = {0};
    std::vector<int32_t> const optContextShape = {static_cast<int>(args.maxInputLen / 2 - 1)};
    std::vector<int32_t> const maxContextShape = {static_cast<int>(args.maxInputLen - 1)};
    // Generation Phase has to be in s = 1
    std::vector<int32_t> const minGenerationShape = {0};
    std::vector<int32_t> const optGenerationShape = {0};
    std::vector<int32_t> const maxGenerationShape = {0};
    contextProfile->setShapeValues("last_token_ids", OptProfileSelector::kMIN, minContextShape.data(), 1);
    contextProfile->setShapeValues("last_token_ids", OptProfileSelector::kOPT, optContextShape.data(), 1);
    contextProfile->setShapeValues("last_token_ids", OptProfileSelector::kMAX, maxContextShape.data(), 1);
    generationProfile->setShapeValues("last_token_ids", OptProfileSelector::kMIN, minGenerationShape.data(), 1);
    generationProfile->setShapeValues("last_token_ids", OptProfileSelector::kOPT, optGenerationShape.data(), 1);
    generationProfile->setShapeValues("last_token_ids", OptProfileSelector::kMAX, maxGenerationShape.data(), 1);

    setStaticProfile(contextProfile, "input_ids", inputIdsContextShape);
    setStaticProfile(generationProfile, "input_ids", inputIdsGenerationShape);

    for (int i = 0; i < nbLayers; ++i)
    {
        setStaticProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheContextShape);
        // std::cout << kvCacheContextShape.d[0] << "," << kvCacheContextShape.d[1] << ", " << kvCacheContextShape.d[2]
        // << std::endl;
        setStaticProfile(generationProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheGenerationShape);
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
    dlclose(handle);
    return EXIT_SUCCESS;
}