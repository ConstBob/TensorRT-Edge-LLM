
#include <iostream>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include <string>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include "common.h"
#include <dlfcn.h>

using namespace std;
using namespace nvinfer1;

struct BuilderArgs{
    bool help;
    std::string onnxPath;
    std::string enginePath;
    int batchSize{1};
    int maxInputLen{128};
    int maxSeqLen{256};
};

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [-h] [-i or --onnxPath=<path to onnx file>] [-o or --enginePath=<path to TensorRT engine] [-b or --batchSize=<int>] [-c or --maxInputLen=<int>] [-s or --maxSeqLen=<int>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --onnxPath       Provide the input onnx file path. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the output TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --batchSize      Provide the desired batch_size for builder. Default = 1" << std::endl;
    std::cerr << "  --maxInputLen    Provide the maximum input length for the model. Default = 20" << std::endl;
    std::cerr << "  --maxSeqLen      Provide the maximum output length for the model (including the input). Default = 40" << std::endl;
};

bool parseBuilderArgs(BuilderArgs& args, int argc, char* argv[]){
    static struct option long_options[]
        = {{"help", no_argument, 0, 'h'},
        {"onnxPath", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'o'},
        {"batchSize", required_argument, 0, 'b'},
        {"maxInputLen", required_argument,0, 'c'},
        {"maxSeqLen", required_argument,0, 's'},
        {0, 0, 0, 0}
    };

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "h:iobcs", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                args.help = true;
                return true;
            case 'i':
                if (optarg){
                    args.onnxPath = optarg;
                }
                else{
                    std::cerr << "ERROR: ONNX Path requires option argument" << std::endl;
                    return false;
                }
                break;
            case 'o':
                if (optarg){
                    args.enginePath = optarg;
                }
                else{
                    std::cerr << "ERROR: Output Dir requires option argument" << std::endl;
                    return false;
                }
                break;
            case 'b':
                if (optarg){
                    args.batchSize = std::stoi(optarg);
                }
                break;
            case 'c':
                if (optarg){
                    args.maxInputLen = std::stoi(optarg);
                }
                break;
            case 's':
                if (optarg){
                    args.maxSeqLen = std::stoi(optarg);
                }
                break;
            default:
                printUsage(argv[0]);
                return false;
        }
    }
    return true;
}

bool setStaticProfile(IOptimizationProfile* profile, const char* inputName, Dims const& dims){

    return profile->setDimensions(
        inputName, OptProfileSelector::kMIN, dims
    ) &&
    profile->setDimensions(
        inputName, OptProfileSelector::kOPT, dims
    ) &&
    profile->setDimensions(
        inputName, OptProfileSelector::kMAX, dims
    );
}

Dims createDims(const std::vector<int64_t>& shape){
    Dims dims;
    dims.nbDims = shape.size();
    for (int i = 0; i < shape.size(); ++i){
        dims.d[i] = shape[i];
    }
    return dims;
}

int main(int argc, char** argv){
    BuilderArgs args;
    if ((argc < 2) || (!parseBuilderArgs(args, argc, argv))){
        printUsage(argv[0]);
        return false;
    }
    if (args.help){
        printUsage(argv[0]);
        return true;
    }
    
    Logger gLogger;
    void* handle = dlopen("/home/luxiaoz/drive-llm/plugins/build/libLLamaPlugin.so", RTLD_LAZY);
    if (!handle) {
        gLogger.error(fmtstr("Cannot open library: %s", dlerror()).c_str());
        return false;
    }

    // Create the builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        gLogger.error("Failed to create builder.");
        return false;
    }

    // Create the network definition
    const auto stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network) {
        gLogger.error("Failed to create network.");
        return false;
    }

    // Create the ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        gLogger.error("Failed to create parser.");
        return false;
    }

    // Parse the ONNX model
    if (!parser->parseFromFile(args.onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        gLogger.error("Failed to parse ONNX file.");
        return false;
    }

    // Build the engine
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        gLogger.error("Failed to create builder config.");
        return false;
    }

    int32_t nbInputs = network->getNbInputs();
    // Excluding input_ids and context length
    int32_t nbLayers = nbInputs - 2;

    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    // Location 2 is guaranteed to be one of the KV Cache inputs
    Dims kvDims = network->getInput(2)->getDimensions();
    int64_t numKVHeads = kvDims.d[2];
    int64_t hiddenSizePerHead = kvDims.d[4];

    Dims inputIdsContextShape = createDims({args.batchSize, args.maxInputLen});
    Dims kvCacheContextShape = createDims({args.batchSize, 2, numKVHeads, 0, hiddenSizePerHead});
    Dims inputIdsGenerationShape = createDims({args.batchSize, 1});
    Dims kvCacheGenerationShape = createDims({args.batchSize, 2, numKVHeads, args.maxSeqLen, hiddenSizePerHead});

    setStaticProfile(contextProfile, "input_ids", inputIdsContextShape);
    setStaticProfile(generationProfile, "input_ids", inputIdsGenerationShape);
    for (int i = 0; i< nbLayers; ++i){
        setStaticProfile(contextProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheContextShape);
        setStaticProfile(generationProfile, fmtstr("past_key_values.%d", i).c_str(), kvCacheGenerationShape);
    }

    config->addOptimizationProfile(contextProfile);
    config->addOptimizationProfile(generationProfile);

    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine) {
        std::cerr << "Failed to build engine." << std::endl;
        return false;
    }

    std::ofstream ofs(args.enginePath, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << args.enginePath << std::endl;
        return false;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    std::cout << "Engine saved to " << args.enginePath << std::endl;
    dlclose(handle);
    return true;
}