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

#include "llmBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <fstream>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

LLMBuilder::LLMBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool LLMBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Determine ONNX file path
    std::string onnxFilePath;
    if (mBuilderConfig.maxLoraRank > 0)
    {
        onnxFilePath = mOnnxDir.string() + "/lora_model.onnx";
        LOG_INFO("Parsing LoRA-enabled ONNX model. Please ensure %s exists.", onnxFilePath.c_str());
    }
    else
    {
        onnxFilePath = mOnnxDir.string() + "/model.onnx";
        LOG_INFO("Parsing ONNX model. Please ensure %s exists.", onnxFilePath.c_str());
    }

    // Parse ONNX model
    auto parser = parseOnnxModel(network.get(), onnxFilePath);
    if (!parser)
    {
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    // Create builder config
    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        return false;
    }

    // Setup optimization profiles
    if (!setupLLMOptimizationProfiles(*builder.get(), *config.get(), *network.get()))
    {
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
        LOG_INFO("Created directory %s for saving LLM engine.", mEngineDir.string().c_str());
    }

    // Determine engine file name
    std::string engineFileName;
    if (mBuilderConfig.eagleDraft)
    {
        engineFileName = "eagle_draft.engine";
    }
    else if (mBuilderConfig.eagleBase)
    {
        engineFileName = "eagle_base.engine";
    }
    else
    {
        engineFileName = "llm.engine";
    }

    // Build and save engine
    std::string engineFilePath = mEngineDir.string() + "/" + engineFileName;
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        return false;
    }

    // Detect number of deepstack embeds from network (for Qwen3VL models)
    mNumDeepstackFeatures = 0;
    for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
    {
        std::string const inputName = network->getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            mNumDeepstackFeatures++;
        }
    }
    if (mNumDeepstackFeatures > 0)
    {
        LOG_INFO("Detected %d deepstack embedding inputs in network (Qwen3VL model)", mNumDeepstackFeatures);
    }

    // Copy files and save builder config
    if (!copyConfig())
    {
        return false;
    }

    if (!copyTokenizerFiles())
    {
        return false;
    }

    if (!copyEagleFiles())
    {
        return false;
    }

    if (!copyVocabMappingFiles())
    {
        return false;
    }

    if (!copyEmbeddingFile())
    {
        return false;
    }

    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string jsonPath = mOnnxDir.string() + "/config.json";
    if (!loadJsonConfig(jsonPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    mTargetModelOutputHiddenDim = mHiddenSize * 3;
    mNumKVHeads = mModelConfig["num_key_value_heads"].get<int32_t>();
    auto numAttentionHeads = mModelConfig["num_attention_heads"].get<int32_t>();

    if (mModelConfig.contains("head_dim"))
    {
        mHeadSize = mModelConfig["head_dim"].get<int32_t>();
    }
    else
    {
        mHeadSize = mHiddenSize / numAttentionHeads;
    }

    if (mModelConfig.contains("partial_rotary_factor"))
    {
        mRotaryDim = static_cast<int64_t>(mModelConfig["partial_rotary_factor"].get<float>() * mHeadSize);
    }
    else
    {
        mRotaryDim = mHeadSize;
    }

    mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();

    // Read trt_native_ops flag from config if present
    if (mModelConfig.contains("trt_native_ops"))
    {
        mBuilderConfig.useTrtNativeOps = mModelConfig["trt_native_ops"].get<bool>();
    }

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* contextProfile = builder.createOptimizationProfile();
    auto* generationProfile = builder.createOptimizationProfile();

    bool result = true;

    // Setup common profiles
    result &= setupCommonProfiles(*contextProfile, *generationProfile);

    // Setup model-specific profiles
    if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
    {
        result &= setupEagleProfiles(*contextProfile, *generationProfile);
    }
    else
    {
        result &= setupVanillaProfiles(*contextProfile, *generationProfile);
    }

    // Setup Deepstack profiles for Qwen3VL models
    result &= setupDeepstackProfiles(*contextProfile, *generationProfile, network);

    if (mBuilderConfig.maxLoraRank > 0)
    {
        result &= setupLoraProfiles(*contextProfile, *generationProfile, network);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
    LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());

    config.addOptimizationProfile(contextProfile);
    config.addOptimizationProfile(generationProfile);

    return true;
}

bool LLMBuilder::setupCommonProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Context lengths
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // Rope rotary cos sin
    result &= setOptimizationProfile(&contextProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));

    // For KVCacheStartIndex, we use zero shape to indicate the kvcache is empty for all sequences in the batch.
    // This can help distinguish the normal prefill and chunked prefill execution.
    result &= setOptimizationProfile(&contextProfile, binding_names::kKVCacheStartIndex, createDims({0}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // KV cache profiles
    result &= setupKVCacheProfiles(contextProfile, generationProfile);

    return result;
}

bool LLMBuilder::setupVanillaProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Input embeddings - always dynamic
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));

    // Last token IDs
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    return result;
}

bool LLMBuilder::setupEagleProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    int const maxTokens
        = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;

    // Input embeddings
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

    // Last token IDs - 2D shape [batch_size, num_selected_tokens]
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    if (mBuilderConfig.eagleDraft)
    {
        // Hidden states from draft
        result &= setOptimizationProfile(&contextProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

        // Hidden states input
        result &= setOptimizationProfile(&contextProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mTargetModelOutputHiddenDim}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mTargetModelOutputHiddenDim}));
    }

    // Attention mask and position ID
    if (mBuilderConfig.eagleDraft || mBuilderConfig.eagleBase)
    {
        int32_t const attnMaskAlignSize = 32;
        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1, 1}), createDims({mBuilderConfig.maxBatchSize, 1, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2,
                static_cast<int64_t>(divUp(maxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize)}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens,
                static_cast<int64_t>(divUp(maxTokens, attnMaskAlignSize) * attnMaskAlignSize)}));

        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens}));
    }

    return result;
}

bool LLMBuilder::setupDeepstackProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Dynamically detect all deepstack_embeds inputs in the network
    std::vector<std::string> deepstackInputs;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string const inputName = network.getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            deepstackInputs.push_back(inputName);
        }
    }

    // If no deepstack embeds found, return early (not a Qwen3VL model)
    if (deepstackInputs.empty())
    {
        return true;
    }

    LOG_INFO("Detected %zu deepstack embedding inputs", deepstackInputs.size());

    // Setup profiles for all detected deepstack_embeds inputs
    // These have the same shape as inputs_embeds: [batch_size, seq_len, hidden_size]
    for (auto const& deepstackInputName : deepstackInputs)
    {
        LOG_INFO("Setting up optimization profile for %s", deepstackInputName.c_str());

        // Same profile as inputs_embeds
        result &= setOptimizationProfile(&contextProfile, deepstackInputName.c_str(), createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));

        if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
        {
            int const maxTokens
                = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));
        }
        else
        {
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));
        }
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupDeepstackProfiles().");
    }

    return result;
}

bool LLMBuilder::setupLoraProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    if (mBuilderConfig.maxLoraRank == 0)
    {
        LOG_WARNING(
            "Your model has dynamic LoRA, but max LoRA rank is 0. This is equivalent to no LoRA. Please set "
            "--maxLoraRank to a positive value if you want to use LoRA.");
        return true;
    }

    bool findLoraWeights = false;

    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        std::string inputName = input->getName();

        if (inputName.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_A, the shape is [gemm_k, lora_rank]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_k = dims.d[0];
                result
                    &= setOptimizationProfile(&contextProfile, inputName.c_str(), createDims({gemm_k, 0}), // min shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                              // opt shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, inputName.c_str(),
                    createDims({gemm_k, 0}),                              // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}), // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));    // max shape
            }
        }
        else if (inputName.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_B, the shape is [lora_rank, gemm_n]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_n = dims.d[1];
                result
                    &= setOptimizationProfile(&contextProfile, inputName.c_str(), createDims({0, gemm_n}), // min shape
                        createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                              // opt shape
                        createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, inputName.c_str(),
                    createDims({0, gemm_n}),                              // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}), // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));    // max shape
            }
        }
    }

    if (!findLoraWeights)
    {
        LOG_ERROR(
            "Failed to find any LoRA weights inputs in the ONNX model. Have you inserted LoRA weights using "
            "tensorrt-edgellm-insert-lora command?");
        return false;
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupLoraProfiles().");
    }

    return result;
}

bool LLMBuilder::setupKVCacheProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;
    if (mBuilderConfig.useTrtNativeOps)
    {
        // TRT attention: separate K and V caches without the "2" dimension
        // Shape: [batch, num_kv_heads, seq_len, head_dim]
        nvinfer1::Dims minKVCacheShape = createDims({1, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims optKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims maxKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});

        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            // K cache bindings
            result &= setOptimizationProfile(&contextProfile, binding_names::formatKCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatKCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);

            // V cache bindings
            result &= setOptimizationProfile(&contextProfile, binding_names::formatVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        }
    }
    else
    {
        // Plugin path: combined KV cache with "2" dimension
        // KV cache shape is [B, 2, num_kv_heads, 0 to max_kv_cache_capacity, head_dim]
        nvinfer1::Dims minKVCacheShape = createDims({1, 2, mNumKVHeads, 0, mHeadSize});
        nvinfer1::Dims optKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims maxKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});

        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            result &= setOptimizationProfile(&contextProfile, binding_names::formatKVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatKVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        }
    }

    return result;
}

bool LLMBuilder::copyConfig()
{
    // Determine config file name based on model type
    std::string targetConfigPath;
    if (mBuilderConfig.eagleDraft)
    {
        targetConfigPath = mEngineDir.string() + "/draft_config.json";
    }
    else if (mBuilderConfig.eagleBase)
    {
        targetConfigPath = mEngineDir.string() + "/base_config.json";
    }
    else
    {
        targetConfigPath = mEngineDir.string() + "/config.json";
    }

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();

    // Add detected num_deepstack_features if present (Qwen3VL models)
    configWithBuilder["num_deepstack_features"] = mNumDeepstackFeatures;

    // Write updated config
    std::ofstream targetConfigFile(targetConfigPath);
    if (!targetConfigFile.is_open())
    {
        LOG_ERROR("Failed to open target config file: %s", targetConfigPath.c_str());
        return false;
    }
    targetConfigFile << configWithBuilder.dump(2);
    targetConfigFile.close();

    LOG_INFO("Copied config.json with builder config to %s", targetConfigPath.c_str());
    return true;
}

bool LLMBuilder::copyTokenizerFiles()
{
    // Eagle3 draft model does not need tokenizer files
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    std::vector<std::string> tokenizerFiles
        = {"tokenizer_config.json", "tokenizer.json", "processed_chat_template.json"};
    bool allSuccess = true;

    for (auto const& filename : tokenizerFiles)
    {
        std::string srcPath = mOnnxDir.string() + "/" + filename;
        std::string dstPath = mEngineDir.string() + "/" + filename;

        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied tokenizer file: %s", filename.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy tokenizer file %s", filename.c_str());
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool LLMBuilder::copyEagleFiles()
{
    // Copy d2t.safetensors for Eagle3 draft models
    if (mBuilderConfig.eagleDraft)
    {
        std::string d2tPath = mOnnxDir.string() + "/d2t.safetensors";
        std::string targetD2tPath = mEngineDir.string() + "/d2t.safetensors";

        if (file_io::copyFile(d2tPath, targetD2tPath))
        {
            LOG_INFO("Copied d2t.safetensors to %s", targetD2tPath.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy d2t.safetensors to %s", targetD2tPath.c_str());
            return false;
        }
    }

    return true;
}

bool LLMBuilder::copyVocabMappingFiles()
{
    // Copy vocab_map.safetensors if reduced vocabulary is used
    if (mModelConfig.contains(binding_names::kReducedVocabSizeKey)
        && mModelConfig[binding_names::kReducedVocabSizeKey].get<int32_t>() > 0)
    {
        std::string vocabMapPath = mOnnxDir.string() + "/" + binding_names::kVocabMapFileName;
        std::string targetVocabMapPath = mEngineDir.string() + "/" + binding_names::kVocabMapFileName;

        if (file_io::copyFile(vocabMapPath, targetVocabMapPath))
        {
            LOG_INFO("Copied %s to %s", binding_names::kVocabMapFileName, targetVocabMapPath.c_str());
        }
        else
        {
            LOG_WARNING("%s not found in %s. This is expected if reduced vocabulary is not used.",
                binding_names::kVocabMapFileName, mOnnxDir.string().c_str());
        }
    }

    return true;
}

bool LLMBuilder::copyEmbeddingFile()
{
    // Eagle draft model uses shared embedding table from base model, so skip copying
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    // Copy embedding.safetensors for eagleBase and vanilla LLM models
    std::string embeddingPath = mOnnxDir.string() + "/embedding.safetensors";
    std::string targetEmbeddingPath = mEngineDir.string() + "/embedding.safetensors";

    if (file_io::copyFile(embeddingPath, targetEmbeddingPath))
    {
        LOG_INFO("Copied embedding.safetensors to %s", targetEmbeddingPath.c_str());
    }
    else
    {
        LOG_ERROR(
            "Failed to copy embedding.safetensors from %s to %s", embeddingPath.c_str(), targetEmbeddingPath.c_str());
        return false;
    }

    return true;
}

} // namespace builder
} // namespace trt_edgellm
