#include "multimodalRunner.h"
#include <algorithm>
#include <cstdlib>

namespace drivellm
{
namespace rt
{

MultimodalRunner::MultimodalRunner(std::string const& engineDir, cudaStream_t stream)
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    // Construct engine path from directory
    std::string enginePath = engineDir + "/visual.engine";

    // Load engine
    char const* disableMmapLoad = std::getenv("DISABLE_MMAP_LOAD");
    if (disableMmapLoad != nullptr)
    {
        StreamReader _sr(enginePath);
        mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(_sr));
    }
    else
    {
        auto mmapReader = std::make_unique<MmapReader>(enginePath);
        mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    }

    // Create context and set optimization profile
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mVisualEngine->createExecutionContext());
    mContext->setOptimizationProfileAsync(0, stream);
}

void MultimodalRunner::flattenBatch(std::vector<int32_t>& inputIds, std::vector<int32_t>& contextLengths,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int32_t>& batchInputLengths, int32_t const padId,
    int const maxSupportedInputLength, bool enableDynamicShape)
{
    int32_t maxContextLengthInBatch = *std::max_element(batchInputLengths.begin(), batchInputLengths.end());
    if (maxContextLengthInBatch > maxSupportedInputLength)
    {
        throw std::runtime_error("maxContextLengthInBatch" + std::to_string(maxContextLengthInBatch)
            + " exceeds the maximum supported inputLength of TensorRT Engine: "
            + std::to_string(maxSupportedInputLength));
    }

    int32_t contextLenStride = enableDynamicShape ? maxContextLengthInBatch : maxSupportedInputLength;

    for (size_t i = 0; i < batchInputIds.size(); ++i)
    {
        int32_t inputSize = batchInputLengths[i];
        contextLengths.emplace_back(inputSize);
        batchInputIds[i].resize(contextLenStride, padId);
        inputIds.insert(inputIds.end(), batchInputIds[i].begin(), batchInputIds[i].end());
    }
}

void MultimodalRunner::infer(cudaStream_t stream)
{
    mContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace rt
} // namespace drivellm