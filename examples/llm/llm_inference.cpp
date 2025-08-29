#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <getopt.h>

using namespace drivellm;

struct LLMInferenceArgs
{
    std::string engineDir;
    std::string multimodalEngineDir;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --engineDir     " << std::endl;
    std::cerr << "  --multimodalEngineDir     " << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[]
        = {{"engineDir", required_argument, 0, 901}, {"multimodalEngineDir", required_argument, 0, 902}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case 901: args.engineDir = optarg; break;
        case 902: args.multimodalEngineDir = optarg ? optarg : ""; break;
        default: return false;
        }
    }

    std::cout << "args.engineDir: " << args.engineDir << std::endl;
    if (args.engineDir.empty())
    {
        std::cerr << "ERROR: --engineDir is required" << std::endl;
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        std::cout << "args.multimodalEngineDir: " << args.multimodalEngineDir << std::endl;
    }

    return true;
}

int main(int argc, char* argv[])
{
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return 1;
    }

    auto pluginHandles = loadEdgellmPluginLib();
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    try
    {
        llmInferenceRuntime
            = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: {}", e.what());
        return EXIT_FAILURE;
    }

    rt::LLMGenerationRequest request;
    request.temperature = 1.0f;
    request.topP = 0.9f;
    request.topK = 20;
    request.maxGenerateLength = 128;
    rt::LLMGenerationResponse response;

    if (!args.multimodalEngineDir.empty())
    {
        auto image = rt::imageUtils::loadImageFromFile("examples/multimodal/pics/demo.jpeg");
        request.imageBuffers.emplace_back(std::vector<rt::imageUtils::ImageData>{image});
        request.prompts.emplace_back(rt::LLMGenerationRequest::Prompt{"", "Describe this image."});
    }
    else
    {
        request.prompts.emplace_back(
            rt::LLMGenerationRequest::Prompt{"", "Introduce NVIDIA and introduce the CEO of this company."});
        // Capture CUDA graph and execute the graph for text only input.
        // TODO: Enable CUDA graph capture for multimodal inputs.
        bool const captureStatus = llmInferenceRuntime->captureDecodingCUDAGraph(stream);
        if (!captureStatus)
        {
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
    }

    if (llmInferenceRuntime->handleRequest(request, response, stream))
    {
        LOG_INFO("Generation finished.");
    }

    std::cout << "Response text: " << response.outputTexts[0] << std::endl;

    return 0;
}