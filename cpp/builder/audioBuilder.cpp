/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audioBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

Json AudioBuilderConfig::toJson() const noexcept
{
    Json json;
    json["min_time_steps"] = minTimeSteps;
    json["max_time_steps"] = maxTimeSteps;
    return json;
}

AudioBuilderConfig AudioBuilderConfig::fromJson(Json const& json)
{
    AudioBuilderConfig config;
    if (json.contains("min_time_steps"))
    {
        config.minTimeSteps = json["min_time_steps"];
    }
    if (json.contains("max_time_steps"))
    {
        config.maxTimeSteps = json["max_time_steps"];
    }
    return config;
}

std::string AudioBuilderConfig::toString() const
{
    std::ostringstream oss;
    oss << "AudioBuilderConfig:\n";
    oss << "  minTimeSteps: " << minTimeSteps << "\n";
    oss << "  maxTimeSteps: " << maxTimeSteps << "\n";
    return oss.str();
}

AudioBuilder::AudioBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, AudioBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool AudioBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        LOG_ERROR("Failed to parse model configuration from %s", mOnnxDir.string().c_str());
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Parse ONNX model
    std::string onnxPath = (mOnnxDir / "model.onnx").string();
    auto parser = parseOnnxModel(network.get(), onnxPath);
    if (!parser)
    {
        LOG_ERROR("Failed to parse ONNX model from %s", onnxPath.c_str());
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "Audio").c_str());

    // Create builder config
    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config");
        return false;
    }

    // Setup optimization profile
    if (!setupAudioOptimizationProfile(*builder.get(), *config.get(), *network.get()))
    {
        LOG_ERROR("Failed to setup audio optimization profile. Check input dimensions and model configuration.");
        return false;
    }

    // Create engine directory
    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving Audio engine.", mEngineDir.string().c_str());
    }

    // Build and save engine
    std::string engineFilePath = (mEngineDir / "audio_encoder.engine").string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        LOG_ERROR("Failed to build and serialize engine to %s", engineFilePath.c_str());
        return false;
    }

    // Copy config
    if (!copyConfig())
    {
        LOG_ERROR("Failed to copy config to engine directory");
        return false;
    }

    return true;
}

bool AudioBuilder::parseConfig()
{
    std::string configPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(configPath, mModelConfig))
    {
        LOG_ERROR("Failed to load config from %s", configPath.c_str());
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // Read model type
    std::string modelTypeStr;
    if (mModelConfig.contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kModelTypeKey].get<std::string>();
    }
    else
    {
        LOG_ERROR(
            "model_type not found in config.json (expected either vision_config.model_type or top-level model_type)");
        return false;
    }

    mModelType = multimodal::stringToModelType(modelTypeStr);

    if (mModelType == multimodal::ModelType::UNKNOWN)
    {
        LOG_ERROR("Unsupported model type: %s", modelTypeStr.c_str());
        return false;
    }

    // Read audio-specific configuration for Qwen3-Omni
    if (mModelType == multimodal::ModelType::QWEN3_OMNI_AUDIO_ENCODER)
    {
        if (mModelConfig.contains("audio_config"))
        {
            auto audioConfig = mModelConfig["audio_config"];
            // Read from config with fallback defaults (for Qwen3-Omni)
            mMelBins = audioConfig.value("num_mel_bins", 128); // Default: 128 mel frequency bins
            // n_window is chunk size (50), but feature tensor uses 100 (n_window * 2)
            mNWindowDim = audioConfig.value("n_window", 50) * 2;         // Default: 50 * 2 = 100
            mSubsampleFactor = audioConfig.value("subsample_factor", 2); // Default: 2x subsampling
        }
    }

    return true;
}

bool AudioBuilder::setupAudioOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* audioProfile = builder.createOptimizationProfile();
    bool result = true;

    if (mModelType == multimodal::ModelType::QWEN3_OMNI_AUDIO_ENCODER)
    {
        result = setupQwen3OmniAudioProfile(*audioProfile);
    }
    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(audioProfile, "audio_profile", &network).c_str());

    config.addOptimizationProfile(audioProfile);
    return true;
}

bool AudioBuilder::setupQwen3OmniAudioProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Calculate chunk dimensions using ceiling division
    int64_t minChunks = static_cast<int64_t>(divUp(mBuilderConfig.minTimeSteps, mNWindowDim));
    int64_t maxChunks = static_cast<int64_t>(divUp(mBuilderConfig.maxTimeSteps, mNWindowDim));
    int64_t optChunks = (minChunks + maxChunks) / 2;

    // Calculate attention elements for Qwen3-Omni audio encoder
    // Each full chunk has 100 frames -> 13 frames after CNN (3 Conv layers with stride=2).
    // But it's possible for a tail chunk in a batch to have only 1 frame after CNN.
    // In multi-batch case, minElems = minChunks since each chunk can be its own batch
    // with only 1 frame after CNN. So minElems=minChunks is the safe lower bound.
    constexpr int64_t kAttentionElementsPerChunk = 13;
    int64_t minElems = minChunks; // Safe lower bound for multi-batch case
    int64_t maxElems = maxChunks * kAttentionElementsPerChunk;
    int64_t optElems = (minElems + maxElems) / 2;

    // Setup optimization profiles for audio encoder inputs
    // Expected inputs for Qwen3-Omni audio encoder:
    //   1. padded_feature: [num_chunks, num_mel_bins, n_window_dim] - 3D
    //   2. padded_mask_after_cnn_indices: [num_attention_elems, 2] - 2D
    //   3. attention_mask: [num_attention_elems, num_attention_elems] - 2D

    // Base inputs
    result &= setOptimizationProfile(&profile, "padded_feature", createDims({minChunks, mMelBins, mNWindowDim}),
        createDims({optChunks, mMelBins, mNWindowDim}), createDims({maxChunks, mMelBins, mNWindowDim}));
    result &= setOptimizationProfile(&profile, "padded_mask_after_cnn_indices", createDims({minElems, 2}),
        createDims({optElems, 2}), createDims({maxElems, 2}));
    result &= setOptimizationProfile(&profile, "attention_mask", createDims({minElems, minElems}),
        createDims({optElems, optElems}), createDims({maxElems, maxElems}));

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile at setupQwen3OmniAudioProfile().");
    }

    return result;
}

bool AudioBuilder::copyConfig()
{
    return saveConfigWithBuilderInfo(mEngineDir, mModelConfig, mBuilderConfig.toJson());
}

} // namespace builder
} // namespace trt_edgellm
