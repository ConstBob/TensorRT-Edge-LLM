/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "multimodalRunner.h"
#include "common/mmapReader.h"
#include "multimodal/internViTRunner.h"
#include "multimodal/qwenViTRunner.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

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
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    // Create context and set optimization profile
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mVisualEngine->createExecutionContext());
    mContext->setOptimizationProfileAsync(0, stream);
}

std::unique_ptr<MultimodalRunner> MultimodalRunner::create(std::string const& multimodalEngineDir, cudaStream_t stream)
{
    std::unique_ptr<MultimodalRunner> multimodalRunner;

    // Read config.json to determine model type
    std::string configPath = multimodalEngineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        throw std::runtime_error("Failed to open config file: " + configPath);
    }

    nlohmann::json jsonConfig;
    try
    {
        jsonConfig = nlohmann::json::parse(configFileStream);
        configFileStream.close();
    }
    catch (nlohmann::json::parse_error const& e)
    {
        throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
    }

    std::string modelType = jsonConfig["model_type"].get<std::string>();

    if (modelType == "qwen2_vl" || modelType == "qwen2_5_vl")
    {
        multimodalRunner = std::make_unique<QwenViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == "internvl")
    {
        multimodalRunner = std::make_unique<InternViTRunner>(multimodalEngineDir, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported model type: " + modelType);
    }

    return multimodalRunner;
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

rt::Tensor& MultimodalRunner::getOutputEmbedding()
{
    return mOutputEmbedding;
}

} // namespace rt
} // namespace drivellm
