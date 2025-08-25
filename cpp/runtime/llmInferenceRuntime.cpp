#include "llmInferenceRuntime.h"

#include "common/logger.h"
#include "multimodal/multimodalRunner.h"
#include "sampler/sampling.h"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace nvinfer1;

namespace drivellm
{
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(
    std::string const& engineDir, std::string const& multimodalEngineDir, cudaStream_t stream)
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

    // Optional: Setup multimodal engine runner
    if (!multimodalEngineDir.empty())
    {
        try
        {
            mMultimodalRunner = MultimodalRunner::create(multimodalEngineDir, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize MultimodalRunner: %s", e.what());
            throw std::runtime_error("Failed to initialize MultimodalRunner: " + std::string(e.what()));
        }
        LOG_INFO("MultimodalRunner successfully loaded and initialized multimodal engine.");
    }
}

bool LLMInferenceRuntime::getInputTexts(LLMGenerationRequest const& request, std::vector<std::string>& inputTexts)
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

    if (mMultimodalRunner)
    {
        int32_t const imageBuffersBatchSize = static_cast<int32_t>(request.imageBuffers.size());
        if (activeBatchSize != imageBuffersBatchSize)
        {
            LOG_ERROR("LLMInferenceRuntime(): The batch size of prompts and image buffers is not the same.");
            return false;
        }
    }

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        inputTexts.emplace_back(request.prompts[i].systemPrompt + request.prompts[i].userPrompt);
    }

    return true;
}

bool LLMInferenceRuntime::prepareInputIds(
    LLMGenerationRequest const& request, std::vector<int32_t>& packedInputIds, std::vector<int32_t>& inputIdsLengths)
{
    std::vector<std::string> inputTexts;
    if (!getInputTexts(request, inputTexts))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
        return false;
    }

    std::vector<std::vector<int32_t>> inputIdsVec;
    inputIdsLengths.clear();

    for (auto& inputText : inputTexts)
    {
        auto tokenizedInput = mTokenizer->encode(inputText, true);
        inputIdsVec.emplace_back(tokenizedInput);
        inputIdsLengths.emplace_back(static_cast<int32_t>(tokenizedInput.size()));
    }

    if (!packInputIds(inputIdsVec, inputIdsLengths, packedInputIds))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
        return false;
    }

    return true;
}

bool LLMInferenceRuntime::packInputIds(std::vector<std::vector<int32_t>>& batchInputIds,
    std::vector<int32_t>& inputIdsLengths, std::vector<int32_t>& packedInputIds)
{
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

    int32_t const activeBatchSize = static_cast<int32_t>(batchInputIds.size());
    packedInputIds.resize(activeBatchSize * maxInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(batchInputIds[i].begin(), batchInputIds[i].end(), packedInputIds.begin() + i * maxInputLength);
    }

    return true;
}

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<int32_t> packedInputIds;
    std::vector<int32_t> inputIdsLengths;

    if (!mMultimodalRunner)
    {
        if (!prepareInputIds(request, packedInputIds, inputIdsLengths))
        {
            LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
            return false;
        }
    }
    else
    {
        std::vector<std::string> batchInputTexts;
        if (!getInputTexts(request, batchInputTexts))
        {
            LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
            return false;
        }

        std::vector<std::vector<int32_t>> batchInputIds;
        if (!mMultimodalRunner->preprocess(batchInputTexts, request.imageBuffers, batchInputIds, inputIdsLengths,
                mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!packInputIds(batchInputIds, inputIdsLengths, packedInputIds))
        {
            LOG_ERROR("LLMInferenceRuntime(): Input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("LLMInferenceRuntime(): Multimodal inference failed. This request cannot be handled.");
            return false;
        }
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
    // Use empty tensor for when no multimodal runner is available
    rt::Tensor emptyTensor{};
    rt::Tensor& multimodalEmbeddings = mMultimodalRunner ? mMultimodalRunner->getOutputEmbedding() : emptyTensor;
    mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings, mOutputLogits, stream);
    auto generatedToken = sampleTokens();

    mInputIds.reshape({activeBatchSize, 1});
    while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
    {
        CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), generatedToken.data(), activeBatchSize * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        mLLMEngineRunner->executeVanillaDecodingStep(mInputIds, multimodalEmbeddings, mOutputLogits, stream);
        generatedToken = sampleTokens();
    }

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
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