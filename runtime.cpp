
#include <NvInferRuntime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "tokenizer.h"
#include "decoder.h"
#include <getopt.h>
#include <vector>
#include <dlfcn.h>

using namespace std;
using namespace nvinfer1;

struct RuntimeArgs{
    bool help;
    std::string inputString;
    std::string enginePath;
    std::string tokenizerPath;
    int maxLength{40};
    bool debug;
};

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [-h] [-i or --inputString=<input>] [-e or --enginePath=<path to TensorRT engine>] [-s or --maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --inputString    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the input). Default = 40" << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
};


bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[]){
    static struct option long_options[]
        = {{"help", no_argument, 0, 'h'},
        {"inputString", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'e'},
        {"tokenizerPath", required_argument, 0, 't'},
        {"maxLength", required_argument,0, 's'},
        {"debug", no_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "h:iest", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                args.help = true;
                return true;
            case 'i':
                if (optarg){
                    args.inputString = optarg;
                }
                else{
                    std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                    return false;
                }
                break;
            case 'e':
                if (optarg){
                    args.enginePath = optarg;
                }
                else{
                    std::cerr << "ERROR: --enginePath requires option argument" << std::endl;
                    return false;
                }
                break;
            case 't':
                if (optarg){
                    args.tokenizerPath = optarg;
                }
                else{
                    std::cerr << "ERROR: --tokenizerPath requires option argument" << std::endl;
                    return false;
                }
                break;
            case 's':
                if (optarg){
                    args.maxLength = std::stoi(optarg);
                }
                break;
            case 'd':
                args.debug = true;
                break;
            default:
                return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    RuntimeArgs args;
    if ((argc < 2) || (!parseRuntimeArgs(args, argc, argv))){
        printUsage(argv[0]);
        return false;
    }
    if (args.help){
        printUsage(argv[0]);
        return true;
    }

    if (args.debug){
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else{
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    void* handle = dlopen("../plugins/build/libLLamaPlugin.so", RTLD_LAZY);

    Tokenizer* tokenizer = new LlamaV3Tokenizer();
    tokenizer->loadFromHF(args.tokenizerPath);
    std::vector<int32_t> inputIdsInt32 = tokenizer->encode(args.inputString, true);
    std::vector<int64_t> inputIds;
    for (int i = 0; i< inputIdsInt32.size(); ++i){
        inputIds.push_back(static_cast<int64_t>(inputIdsInt32[i]));
    }
    Decoder* decoder = new Decoder();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::filesystem::path _enginePath(args.enginePath);
    decoder->setup(_enginePath, stream);
    GenerationConfig generationConfig{40, 20, 1, 0};
    std::vector<int64_t> outputIds;
    outputIds.reserve(args.maxLength);
    decoder->generate(inputIds, outputIds, generationConfig);
    std::vector<int32_t> outputIdsInt32;
    for (int i = 0; i< outputIds.size(); ++i){
        outputIdsInt32.push_back(static_cast<int32_t>(outputIds[i]));
    }
    std::string output = tokenizer->decode(outputIdsInt32);
    LOG_INFO(fmtstr("Output is %s", output.c_str()));
    dlclose(handle);

    return true;
};
