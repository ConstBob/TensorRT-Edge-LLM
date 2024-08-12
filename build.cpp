
#include <iostream>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include <string>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include "common.h"

using namespace std;
using namespace nvinfer1;

struct BuilderArgs{
    bool help;
    std::string onnxPath;
    std::string enginePath;
    int batchSize{1};
    int maxInputLen{20};
    int maxSeqLen{40};
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

int main(int argc, char** argv){
    BuilderArgs args;
    if (!parseBuilderArgs(args, argc, argv)){
        printUsage(argv[0]);
        return false;
    }
    if (args.help){
        printUsage(argv[0]);
        return true;
    }

    Logger gLogger;
    // Create the builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "Failed to create builder." << std::endl;
        return false;
    }

    // Create the network definition
    const auto stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network) {
        std::cerr << "Failed to create network." << std::endl;
        return false;
    }

    // Create the ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        std::cerr << "Failed to create parser." << std::endl;
        return false;
    }

    // Parse the ONNX model
    if (!parser->parseFromFile(args.onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return false;
    }

    // Build the engine
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config." << std::endl;
        return false;
    }

    int32_t nbInputs = network->getNbInputs();
    // Excluding input_ids and context length
    int32_t nbLayers = (nbInputs - 2) / 2;

    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    // Location 2 is guaranteed to be one of the KV Cache inputs
    Dims kvDims = network->getInput(2)->getDimensions();
    int64_t numKVHeads = kvDims.d[1];
    int64_t hiddenSizePerHead = kvDims.d[3];

    Dims inputIdsContextShape = {args.batchSize, args.maxInputLen};
    Dims kvCacheContextShape = {args.batchSize, 2, numKVHeads, 0, hiddenSizePerHead};
    Dims inputIdsGenerationShape = {args.batchSize, 1};
    Dims kvCacheGenerationShape = {args.batchSize, 2, numKVHeads, args.maxSeqLen - 1, hiddenSizePerHead};

    setStaticProfile(contextProfile, "input_ids", inputIdsContextShape);
    setStaticProfile(generationProfile, "input_ids", inputIdsGenerationShape);
    for (int i = 0; i< nbLayers; ++i){
        setStaticProfile(contextProfile, fmtstr("past_key_values.%d", i), kvCacheContextShape);
        setStaticProfile(generationProfile, fmtstr("past_key_values.%d", i), kvCacheGenerationShape);
    }

    // Set the shape inputs

    vector<int32_t> lengthContextMin = {1};
    vector<int32_t> lengthContextOpt = {args.maxInputLen / 2};
    vector<int32_t> lengthContextMax = {args.maxInputLen};

    vector<int32_t> lengthGenerationMin = {1};
    vector<int32_t> lengthGenerationOpt = {args.maxSeqLen / 2};
    vector<int32_t> lengthGenerationMax = {args.maxSeqLen - 1};

    contextProfile->setShapeValues("context_length", OptProfileSelector::kMIN, lengthContextMin.data(), lengthContextMin.size());
    contextProfile->setShapeValues("context_length", OptProfileSelector::kOPT, lengthContextOpt.data(), lengthContextOpt.size());
    contextProfile->setShapeValues("context_length", OptProfileSelector::kMAX, lengthContextMax.data(), lengthContextMax.size());

    generationProfile->setShapeValues("context_length", OptProfileSelector::kMIN, lengthGenerationMin.data(), lengthGenerationMin.size());
    generationProfile->setShapeValues("context_length", OptProfileSelector::kOPT, lengthGenerationOpt.data(), lengthGenerationOpt.size());
    generationProfile->setShapeValues("context_length", OptProfileSelector::kMAX, lengthGenerationMax.data(), lengthGenerationMax.size());

    config->addOptimizationProfile(contextProfile);
    config->addOptimizationProfile(generationProfile);

    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine) {
        std::cerr << "Failed to build engine." << std::endl;
        return;
    }

    std::ofstream ofs(args.enginePath, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << args.enginePath << std::endl;
        return false;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    std::cout << "Engine saved to " << args.enginePath << std::endl;
}