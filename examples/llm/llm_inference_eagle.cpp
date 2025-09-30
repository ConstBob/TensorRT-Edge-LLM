#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"

using namespace drivellm;

std::string const engineDir = "llama3-8B-FP16-eagle3";
// draft topk, draft step, verify tree size
rt::EagleDraftingConfig const draftingConfig = {10, 6, 60};
int main(int argc, char* argv[])
{
    auto pluginHandles = loadEdgellmPluginLib();
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    try
    {
        eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(engineDir, draftingConfig, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
        return EXIT_FAILURE;
    }

    LOG_INFO("Successfully initialized LLMInferenceSpecDecodeRuntime.");

    rt::LLMGenerationRequest request;
    rt::LLMGenerationRequest::Prompt prompt{
        "", "Please introduce the company of NVIDIA, its CEO and main working directions."};
    request.prompts.emplace_back(prompt);
    request.maxGenerateLength = 128;

    rt::LLMGenerationResponse response;
    if (eagleInferenceRuntime->handleRequest(request, response, stream))
    {
        // Display responses for each batch in the request
        for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
        {
            std::cout << "Output ids for request:\n";
            for (size_t idIdx = 0; idIdx < response.outputIds[batchIdx].size(); ++idIdx)
            {
                std::cout << response.outputIds[batchIdx][idIdx] << ", ";
            }
            std::cout << std::endl;
            LOG_INFO("Response for request:\n%s", response.outputTexts[batchIdx].c_str());
        }
    }
    else
    {
        LOG_ERROR("Failed to handle request.");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}