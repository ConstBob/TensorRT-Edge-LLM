#include "llmInferenceRuntime.h"

#include "common/logger.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "sampler/sampling.h"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace nvinfer1;

namespace
{

// Left a utility function here in case we want to move to a better hashing method.
size_t hashSystemPrompt(std::string const& systemPrompt)
{
    return std::hash<std::string>{}(systemPrompt);
}

} // namespace
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

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request)
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

    return true;
}

bool LLMInferenceRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
    std::vector<std::string> const& systemPrompts, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    std::vector<int32_t> packedInputIds;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    // Record the length of the reused KVCache for each sequence.
    rt::Tensor reuseKVCacheLengths = rt::Tensor({activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
    int32_t* reuseKVCacheLengthsData = reuseKVCacheLengths.dataPointer<int32_t>();

    // Search if the system prompt has been cached. If there are cached system prompts, insert
    // the pre-computed KVCache and remove the contents from inputIds.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPrompt(systemPrompts[i]);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptHash];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
            bool const matchIds = std::equal(precachedKVCache.tokenizedPrompt.begin(),
                precachedKVCache.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "LLMInferenceRuntime(): Though system prompt strings are matched, token_ids are not perfectly "
                    "aligned. "
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    // Pack inputIds, instantiate input data for prefill step, and reset the KVCache state.
    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The max input length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    // The LLM Engine could also have minSupportedInputLength constraint.
    int32_t const packedInputLength = std::max(maxInputLength, mEngineConfig.minSupportedInputLength);
    packedInputIds.resize(activeBatchSize * packedInputLength, mTokenizer->getPadId());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Pad each sequence to the max length of this batch.
        // TODO: Implement remove input padding for better efficiency until multi-batch.
        std::copy(
            processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIds.begin() + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(reuseKVCacheLengths, stream);
    mInputIds.reshape({activeBatchSize, packedInputLength});
    mHostContextLengths.reshape({activeBatchSize});
    mOutputLogits.reshape({activeBatchSize, mEngineConfig.vocabSize});

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), packedInputIds.data(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    return true;
}

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;

    if (!examineRequest(request))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request examination failed. This request cannot be handled.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(request.prompts.size());

    // Preprocess system prompts and save KVCache for each sequence.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (mMultimodalRunner)
        {
            batchSystemPrompts.emplace_back(mMultimodalRunner->preprocessSystemPrompt(request.prompts[i].systemPrompt,
                mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream));
        }
        else
        {
            // TODO: apply chat template for system prompt
            batchSystemPrompts.emplace_back(std::move(request.prompts[i].systemPrompt));
        }
        bool const saveCacheStatus = genAndSaveSystemPromptKVCache(batchSystemPrompts[i], stream);
        if (!saveCacheStatus)
        {
            LOG_WARNING(
                "Failed to save system prompt KVCache. May be KVCache reuse feature is not enabled in the engine.");
        }
    }

    // Preprocess user prompts and encode them.
    if (!mMultimodalRunner)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            // TODO: apply chat template for user prompt.
            std::string userPrompt = request.prompts[i].userPrompt;
            std::string inputText = batchSystemPrompts[i] + userPrompt;
            batchedInputIds.emplace_back(mTokenizer->encode(inputText, true));
        }
    }
    else
    {
        if (!mMultimodalRunner->preprocess(
                request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("LLMInferenceRuntime(): Multimodal inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(batchedInputIds, batchSystemPrompts, stream))
    {
        LOG_ERROR("LLMInferenceRuntime(): Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    int32_t const maxInputIdsLength = mInputIds.getShape()[1];
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (maxInputIdsLength + maxGenerationLength > mEngineConfig.maxSequenceLength)
    {
        maxGenerationLength = mEngineConfig.maxSequenceLength - maxInputIdsLength;
        LOG_WARNING(
            "LLMInferenceRuntime(): With requested max generation length (%d), the total sequence length (%d) may "
            "exceed the max sequence length (%d) of the LLM Engine."
            "Reduce the generation length of this request to %d to avoid the truncation of the generated tokens.",
            request.maxGenerateLength, maxInputIdsLength + request.maxGenerateLength, mEngineConfig.maxSequenceLength,
            maxGenerationLength);
    }

    // Set up data structures to store the generated results during decoding.
    // Also set up sampling parameters and sampling lambda function.
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

    // Use empty tensor for when no multimodal runner is available.
    // All other data input used by prefill step is already set up in setUpForPrefillExecution().
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
    int32_t const maxSupportedBatchSize = mEngineConfig.maxSupportedBatchSize;
    int32_t const minSupportedBatchSize = mEngineConfig.enableDynamicShape ? 1 : maxSupportedBatchSize;

    bool captureStatus{true};
    // Capture the CUDA graph for all available batch sizes.
    for (int32_t batchSize = minSupportedBatchSize; batchSize <= maxSupportedBatchSize; ++batchSize)
    {
        mInputIds.reshape({batchSize, 1});
        mOutputLogits.reshape({batchSize, mEngineConfig.vocabSize});
        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(mInputIds, mOutputLogits, stream);
    }

    if (captureStatus)
    {
        LOG_INFO("LLMInferenceRuntime(): Successfully captured the decoding CUDA graph for all execution batch sizes.");
    }
    return captureStatus;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(std::string const& prompt, cudaStream_t stream)
{
    // TODO: Enable the system prompt KVCache feature by default and remove this check.
    if (!mEngineConfig.enableReuseKVCache)
    {
        LOG_ERROR("LLMInferenceRuntime(): The system prompt KVCache feature is not enabled in the engine.");
        return false;
    }

    // hash the prompt if check if the prompt cache already exists.
    size_t const promptHash = hashSystemPrompt(prompt);
    if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
    {
        LOG_INFO(
            "LLMInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());
    int32_t const activeBatchSize = 1;

    if (promptIdsLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The prompt length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            promptIdsLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(activeBatchSize, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(activeBatchSize, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, stream))
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): Prefill execution setup failed. Cannot generate the KVCache for this prompt.");
        return false;
    }

    // Execute prefill step to initialize the KVCache data.
    rt::Tensor emptyTensor{};
    rt::Tensor& multimodalEmbeddings = mMultimodalRunner ? mMultimodalRunner->getOutputEmbedding() : emptyTensor;
    mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings, mOutputLogits, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy out the KVCache content from the prefill step.
    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numDecoderLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent
        = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT);

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);
    mSystemPromptKVCache.insert({promptHash, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

} // namespace rt
} // namespace drivellm