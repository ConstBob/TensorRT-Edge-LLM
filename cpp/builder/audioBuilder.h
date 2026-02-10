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

#pragma once

#include "multimodal/modelTypes.h"
#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for audio encoder model building.
//! Contains parameters needed to configure the TensorRT engine building process
//! for audio encoders used in multimodal models.
struct AudioBuilderConfig
{
    //! Minimum audio time steps. User-configurable via --minTimeSteps command-line argument.
    //! Default: 100 (~0.64s with hop_length=160, sample_rate=16000 for Qwen3-Omni)
    //! Note: This is a hardcoded default that should ideally be read from model config if available.
    int64_t minTimeSteps{100};

    //! Maximum audio time steps. User-configurable via --maxTimeSteps command-line argument.
    //! Default: 6000 (~38.4s with hop_length=160, sample_rate=16000 for Qwen3-Omni)
    //! Note: This is a hardcoded default that should ideally be read from model config if available.
    int64_t maxTimeSteps{6000};

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const noexcept;

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return AudioBuilderConfig object with parsed parameters
    static AudioBuilderConfig fromJson(Json const& json);

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const;
};

//! Builder class for audio encoder TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for audio encoders used in multimodal models (e.g., Qwen3-Omni).
class AudioBuilder
{
public:
    //! Constructor for AudioBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    //! @throws std::filesystem::filesystem_error if path operations fail
    AudioBuilder(
        std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, AudioBuilderConfig const& config);

    //! Destructor.
    ~AudioBuilder() noexcept = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    //! @throws std::runtime_error if critical build errors occur (file I/O, TensorRT API failures)
    //! @throws nlohmann::json::exception if JSON parsing fails
    bool build();

private:
    std::filesystem::path mOnnxDir;    //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir;  //!< Directory for saving built engine
    AudioBuilderConfig mBuilderConfig; //!< Build configuration
    multimodal::ModelType mModelType;  //!< Model type inferred from config.json

    //! Parse the model configuration from config.json.
    //! Extracts model type and dimensions needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    //! @throws nlohmann::json::exception if JSON parsing fails
    bool parseConfig();

    //! Set up optimization profile for audio encoder.
    //! Creates optimization profile with appropriate dynamic shapes for audio inputs.
    //! @param builder TensorRT builder object (must not be null)
    //! @param config TensorRT builder config object (must not be null)
    //! @param network TensorRT network definition (must not be null)
    //! @return true if setup was successful, false otherwise
    bool setupAudioOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profile for Qwen3-Omni audio encoder.
    //! Configures inputs for Qwen3-Omni audio encoder.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupQwen3OmniAudioProfile(nvinfer1::IOptimizationProfile& profile);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    //! @throws std::runtime_error if file I/O operations fail
    //! @throws nlohmann::json::exception if JSON serialization fails
    bool copyConfig();

    // Audio-specific configuration read from config.json
    int32_t mMelBins{128};       //!< Number of Mel-frequency bins
    int32_t mNWindowDim{100};    //!< Window dimension for feature tensor
    int32_t mSubsampleFactor{2}; //!< Audio encoder subsample factor
    Json mModelConfig;           //!< Parsed model configuration
};

} // namespace builder
} // namespace trt_edgellm
