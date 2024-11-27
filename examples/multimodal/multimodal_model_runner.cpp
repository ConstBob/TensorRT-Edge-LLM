
#include "common/common.h"
#include "decoder/decoder.h"
#include "qwen2vl/vit_runner.h"
#include "tokenizer/tokenizer.h"
#include <dlfcn.h>
#include <getopt.h>

enum class MODE
{
    kInference,
    kBenchmark,
    kEvaluate
};

struct RuntimeArgs
{
    bool help{false};
    std::vector<std::string> inputStrings;
    std::vector<std::vector<std::string>> imagePaths;
    std::string lmEnginePath;
    std::string visualEnginePath;
    std::string tokenizerPath;
    std::string datasetPath;
    int maxLength{40};
    int inputLength;
    MODE mode{MODE::kInference};
    int64_t numRuns{10};
    int64_t warmUp{2};
    bool debug{false};
    std::string modelType{"qwen2_vl"};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-i or --inputString=<input>] [-e or --enginePath=<path to TensorRT engine>] [-s or "
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h                  Display this help message" << std::endl;
    std::cerr << "  --inputString       Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --imagePaths        Provide the input image paths to the runtime. Required. " << std::endl;
    std::cerr << "  --lmEnginePath      Provide the Qwen TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath     Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength         Provide the maximum output length for the generation session (including the "
                 "input). Default = 40"
              << std::endl;
    std::cerr << "  --mode              Provide the mode. " << std::endl;
    std::cerr << "  --modelType         Provide the model type, choose from llama,qwen,qwen_vl,qwen2_vl.Required."
              << std::endl;
    std::cerr << "  --debug             Use debug mode, which outputs tensors." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"inputString", required_argument, 0, 'i'},
        {"imagePaths", required_argument, 0, 'p'}, {"lmEnginePath", required_argument, 0, 'e'},
        {"visualEnginePath", required_argument, 0, 'v'}, {"tokenizerPath", required_argument, 0, 't'},
        {"maxLength", required_argument, 0, 's'}, {"inputLength", required_argument, 0, 'c'},
        {"mode", required_argument, 0, 'm'}, {"warmUp", required_argument, 0, 'w'},
        {"numRuns", required_argument, 0, 'r'}, {"datasetPath", required_argument, 0, 'D'},
        {"debug", no_argument, 0, 'd'}, {"modelType", required_argument, 0, 0}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "h:iest", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.inputStrings.emplace_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                return false;
            }
            break;
        case 'p':
            if (optarg)
            {
                std::vector<std::string> paths;
                std::stringstream ss(optarg);
                std::string path;
                while (std::getline(ss, path, ','))
                {
                    paths.push_back(path);
                }
                args.imagePaths.emplace_back(paths);
            }
            else
            {
                std::cerr << "ERROR: --imagePaths requires option argument" << std::endl;
                return false;
            }
            break;
        case 'e':
            if (optarg)
            {
                args.lmEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --lmEnginePath requires option argument" << std::endl;
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
                else if (strcmp(optarg, "evaluate") == 0)
                {
                    args.mode = MODE::kEvaluate;
                }
                if (strcmp(optarg, "inference") == 0)
                {
                    args.mode = MODE::kInference;
                }
                else
                {
                    std::cerr << "ERROR: invalid argument for --mode. Get " << optarg
                              << ", valid options: benchmark, evaluate, inference." << std::endl;
                    return false;
                }
            }
            break;
        case 'D':
            if (optarg)
            {
                args.datasetPath = optarg;
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

std::vector<std::string> decode(std::filesystem::path const& lmEnginePath,
    std::filesystem::path const& visualEnginePath, std::vector<std::string>& inputStrings,
    std::vector<std::vector<std::string>> const& imagePaths, Tokenizer* tokenizer,
    GenerationConfig const& generationConfig, bool debug = false, std::string modelType = "qwen2_vl")
{
    // Set default input string if not given
    for (int i = inputStrings.size(); i < imagePaths.size(); ++i)
    {
        inputStrings.emplace_back("Describe this image.");
    }
    int32_t batchSize = imagePaths.size();
    auto decoder = new Decoder<half>();

    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    if (modelType == "qwen2_vl")
    {
        auto vitrunner = new Qwen2ViTRunner();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        vitrunner->setup(visualEnginePath, stream, batchSize);
        // Preprocess
        std::vector<half> visualInput;
        std::vector<half> visualAttentionMask;
        std::vector<float> visualRotaryPosEmb;
        std::vector<std::vector<int64_t>> visualGridTHWs;
        std::vector<int64_t> inputIds;
        std::vector<int32_t> contextLengths;

        vitrunner->visualPreprocess(imagePaths, visualInput, visualAttentionMask, visualRotaryPosEmb, visualGridTHWs);
        vitrunner->allocateBuffer();
        vitrunner->visualInfer(visualInput, visualAttentionMask, visualRotaryPosEmb);
        decoder->setup(lmEnginePath, stream);
        vitrunner->textPreprocess(inputStrings, imagePaths, visualGridTHWs, tokenizer, inputIds, contextLengths,
            decoder->getMaxContextLength());
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId(), nullptr,
            vitrunner->getImageEmbeds(), vitrunner->getMropeRotaryCosSin(), vitrunner->getMropePositionDeltas());
    }
    else
    {

        throw std::runtime_error("Only support Qwen2-VL model for Multimodal model.");
    }

    std::vector<std::string> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        if (debug)
        {
            LOG_DEBUG("Output%d length is %d", i, outputIds[i].size());
        }
        output[i] = tokenizer->decode(outputIds[i]);
        LOG_INFO("Input%d is: %s", i, inputStrings[i].c_str());
        LOG_INFO("Output%d is: %s", i, output[i].c_str());
    }
    return output;
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

    void* handle = dlopen("build/libAttentionPlugin.so", RTLD_LAZY);
    if (!handle)
    {
        LOG_ERROR("Cannot open library: %s", dlerror());
        return EXIT_FAILURE;
    }

    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};
    Tokenizer* tokenizer = new Tokenizer();
    tokenizer->loadFromHF(args.tokenizerPath);
    auto output = decode(args.lmEnginePath, args.visualEnginePath, args.inputStrings, args.imagePaths, tokenizer,
        generationConfig, args.debug, args.modelType);

    dlclose(handle);

    return EXIT_SUCCESS;
};
