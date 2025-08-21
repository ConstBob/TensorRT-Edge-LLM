#include "llmInferenceRuntime.h"

#include "common/logger.h"
#include "sampler/sampling.h"

using namespace nvinfer1;

namespace drivellm
{
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    try
    {
        mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize LLMEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("LLMEngineRunner successfully loaded and initialized llm engine.");

    mEngineConfig = mLLMEngineRunner->getEngineConfig();
    // Setup sampling workspace, use default topK=100 to reserve workspace.
    // FIXME: Find a better approach to reserve sampling workspace to handle various request configurations.
    int32_t const defaultTopK = 100;
    float const defaultTopP = 0.9F;
    drivellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(drivellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize, samplingParams));

    // Allocate workspace and activation tensors for LLM engine.
    try
    {
        // Use Int8 to indicate byte for workspace.
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8);
        mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32);
        mOutputLogits = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        mSelectedIndices = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        mHostContextLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate workspace and activation tensors for LLM Inference Runtime: %s", e.what());
        throw std::runtime_error(
            "Failed to allocate workspace and activation tensors for LLM Inference Runtime: " + std::string(e.what()));
    }

    // Setup tokenizer
    // TODO: The tokenizer loading can fail, need to improve the error handling.
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    mTokenizer->loadFromHF(engineDir);
}

bool LLMInferenceRuntime::prepareInputIds(
    LLMGenerationRequest const& request, std::vector<int32_t>& packedInputIds, std::vector<int32_t>& inputIdsLengths)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.prompts.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("LLMInferenceRuntime(): The request is empty with no request prompts supplied.");
        return false;
    }

    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("LLMInferenceRuntime(): The batched request size (%d) exceeds the max supported batch size (%d).",
            activeBatchSize, mEngineConfig.maxSupportedBatchSize);
        return false;
    }

    std::vector<std::string> inputTexts;
    std::vector<std::vector<int32_t>> inputIdsVec;
    inputIdsLengths.resize(activeBatchSize, 0);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::string inputText = request.prompts[i].systemPrompt + request.prompts[i].userPrompt;
        inputTexts.emplace_back(inputText);
        auto tokenizedInput = mTokenizer->encode(inputText, true);
        inputIdsVec.emplace_back(tokenizedInput);
        inputIdsLengths[i] = static_cast<int32_t>(tokenizedInput.size());
    }

    int32_t const minInputLength = *std::min_element(inputIdsLengths.begin(), inputIdsLengths.end());
    int32_t const maxInputLength = *std::max_element(inputIdsLengths.begin(), inputIdsLengths.end());

    if (minInputLength == 0)
    {
        LOG_ERROR("LLMInferenceRuntime(): The batch of requests contains empty prompts.");
        return false;
    }

    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("Max tokenized Input length (%d) exceeds the max supported input length of the LLM Engine (%d)",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    packedInputIds.resize(activeBatchSize * maxInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(inputIdsVec[i].begin(), inputIdsVec[i].end(), packedInputIds.begin() + i * maxInputLength);
    }

    return true;
}
bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<int32_t> packedInputIds;
    std::vector<int32_t> inputIdsLengths;
    if (!prepareInputIds(request, packedInputIds, inputIdsLengths))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(request.prompts.size());
    int32_t const maxInputIdsLength = *std::max_element(inputIdsLengths.begin(), inputIdsLengths.end());
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (maxInputIdsLength + maxGenerationLength > mEngineConfig.maxSupportedInputLength)
    {
        maxGenerationLength = mEngineConfig.maxSupportedInputLength - maxInputIdsLength;
        LOG_WARNING(
            "LLMInferenceRuntime(): With requested max generation length (%d), the total sequence length (%d) may "
            "exceed the max supported input length (%d) of the LLM Engine."
            "Reduce the generation length of this request to %d to avoid the truncation of the generated tokens.",
            request.maxGenerateLength, maxInputIdsLength + request.maxGenerateLength,
            mEngineConfig.maxSupportedInputLength, maxGenerationLength);
    }

    mInputIds.reshape({activeBatchSize, maxInputIdsLength});
    mOutputLogits.reshape({activeBatchSize, mEngineConfig.vocabSize});
    mHostContextLengths.reshape({activeBatchSize});

    int32_t unFinishedBatchNum = activeBatchSize;
    int32_t generationIter{0};

    std::vector<std::vector<int32_t>> outputIds(activeBatchSize);
    std::vector<bool> finishedStates(activeBatchSize, false);
    std::vector<int32_t> selectedTokenIdsHost(activeBatchSize, 0);
    mSelectedIndices.reshape({activeBatchSize});

    SamplingParams params(activeBatchSize, mEngineConfig.vocabSize, request.temperature, request.topK, request.topP);
    auto sampleTokens = [&]() {
        drivellm::topKtopPSamplingFromLogits(mOutputLogits.dataPointer<float>(),
            mSelectedIndices.dataPointer<int32_t>(), params, mSamplingWorkspace.rawPointer(),
            mSamplingWorkspace.getMemoryCapacity(), stream);
        CUDA_CHECK(cudaMemcpyAsync(selectedTokenIdsHost.data(), mSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!finishedStates[i])
            {
                outputIds[i].push_back(selectedTokenIdsHost[i]);
                finishedStates[i] = selectedTokenIdsHost[i] == mTokenizer->getEosId();
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
        return selectedTokenIdsHost;
    };

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), packedInputIds.data(),
        activeBatchSize * maxInputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), inputIdsLengths.data(), activeBatchSize * sizeof(int32_t));
    mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, mOutputLogits, stream);
    auto generatedToken = sampleTokens();

    mInputIds.reshape({activeBatchSize, 1});
    while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
    {
        CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), generatedToken.data(), activeBatchSize * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        mLLMEngineRunner->executeVanillaDecodingStep(mInputIds, mOutputLogits, stream);
        generatedToken = sampleTokens();
    }

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i]));
    }

    return true;
}

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    // TODO: Implement the graph capture logic.
    return false;
}

} // namespace rt
} // namespace drivellm