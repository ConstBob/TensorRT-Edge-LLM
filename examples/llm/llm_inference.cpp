#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include <getopt.h>

using namespace drivellm;

struct LLMInferenceArgs
{
    std::string engineDir;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [--help] [--engineDir=<path to engine directory>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --engineDir     " << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"engineDir", required_argument, 0, 901}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case 901: args.engineDir = optarg; break;
        default: return false;
        }
    }

    std::cout << "args.engineDir: " << args.engineDir << std::endl;
    if (args.engineDir.empty())
    {
        std::cerr << "ERROR: --engineDir is required" << std::endl;
        return false;
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
        llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: {}", e.what());
        return 1;
    }

    rt::LLMGenerationRequest request;
    request.prompts.emplace_back(
        rt::LLMGenerationRequest::Prompt{"", "Introduce NVIDIA and introduce the CEO of this company."});
    request.temperature = 1.0f;
    request.topP = 0.9f;
    request.topK = 20;
    request.maxGenerateLength = 128;
    rt::LLMGenerationResponse response;
    if (llmInferenceRuntime->handleRequest(request, response, stream))
    {
        LOG_INFO("Generation finished.");
    }

    std::cout << "Response text: " << response.outputTexts[0] << std::endl;

    return 0;
}