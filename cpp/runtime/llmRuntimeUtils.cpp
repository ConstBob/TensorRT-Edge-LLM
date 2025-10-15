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

#include "runtime/llmRuntimeUtils.h"

#include "common/logger.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include <ostream>
#include <sstream>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace rt
{

std::ostream& operator<<(std::ostream& os, RopeType const& type)
{
    switch (type)
    {
    case RopeType::kDefault: os << "Default"; break;
    case RopeType::kDynamic: os << "Dynamic"; break;
    case RopeType::kLongRope: os << "LongRope"; break;
    case RopeType::kMRope: os << "MRope"; break;
    }
    return os;
}

std::string formatRopeConfig(RopeCommonConfig const& config)
{
    std::stringstream ss;
    ss << "RopeConfig:"
       << "  type: " << config.type << "  rotaryScale: " << config.rotaryScale
       << "  rotaryTheta: " << config.rotaryTheta << "  maxPositionEmbeddings: " << config.maxPositionEmbeddings;
    return ss.str();
}

RopeCommonConfig collectBaseRopeConfig(nlohmann::json const& config)
{
    RopeCommonConfig ropeConfig{};
    auto ropeScalingIt = config.find("rope_scaling");
    if (ropeScalingIt != config.end())
    {
        auto ropeTypeIt = ropeScalingIt->find("type");
        auto mropeSectionIt = ropeScalingIt->find("mrope_section");
        if (ropeTypeIt != ropeScalingIt->end())
        {
            std::string const ropeTypeStr = ropeTypeIt->get<std::string>();
            if (ropeTypeStr == "default" && mropeSectionIt != ropeScalingIt->end())
            {
                // transformers `Qwen2_5_VLVisionConfig` change type from 'mrope' to 'default'
                ropeConfig.type = RopeType::kMRope;
            }
            else if (ropeTypeStr == "default" || ropeTypeStr == "llama3")
            {
                // Route the llama3 config to default type.
                ropeConfig.type = RopeType::kDefault;
            }
            else if (ropeTypeStr == "dynamic")
            {
                ropeConfig.type = RopeType::kDynamic;
            }
            else if (ropeTypeStr == "longrope")
            {
                ropeConfig.type = RopeType::kLongRope;
            }
        }
    }
    else
    {
        LOG_WARNING(
            "rope_scaling is not specified in the model config, using default rope type. This could misalign with the "
            "model configuration, please check the config file to ensure the correctness");
        ropeConfig.type = RopeType::kDefault;
    }

    // Detect RopeTheta
    if (config.contains("rope_theta"))
    {
        ropeConfig.rotaryTheta = config["rope_theta"].get<float>();
    }
    else
    {
        LOG_WARNING("rope_theta is not specified in the model config, using default value: %f", ropeConfig.rotaryTheta);
    }

    // Detect MaxPositionEmbeddings
    if (config.contains("max_position_embeddings"))
    {
        ropeConfig.maxPositionEmbeddings = config["max_position_embeddings"].get<int32_t>();
    }
    else
    {
        LOG_WARNING("max_position_embeddings is not specified in the model config, using default value: %d",
            ropeConfig.maxPositionEmbeddings);
    }

    LOG_INFO("Collected base rope config: %s", formatRopeConfig(ropeConfig).c_str());
    return ropeConfig;
}

bool initializeRopeCosSinCache(
    rt::Tensor& cosSinCache, RopeCommonConfig const& config, nlohmann::json const& modelConfig, cudaStream_t stream)
{
    if (config.type == RopeType::kMRope)
    {
        LOG_ERROR("MRope is context dependent rope type, which cannot be initialized with basic parameters.");
        return false;
    }

    // Tensor shape: [1, maxLength, rotaryDim]
    if (cosSinCache.getShape().getNumDims() != 3 || cosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("Persistent RopeCosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }
    int64_t ropeMaxLength = cosSinCache.getShape()[1];
    int64_t rotaryDim = cosSinCache.getShape()[2];
    if (config.type == RopeType::kDefault || config.type == RopeType::kDynamic)
    {
        if (config.type == RopeType::kDynamic && ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_ERROR("Dynamic rope type with sequence length larger than maxPositionEmbeddings is not supported.");
            return false;
        }
        if (ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_WARNING(
                "maxLength %d is greater than maxPositionEmbeddings %d indicated by model config, this could cause "
                "inaccurate generation results",
                ropeMaxLength, config.maxPositionEmbeddings);
        }

        try
        {
            kernel::initializeNormalRopeCosSin(cosSinCache.dataPointer<float>(), config.rotaryTheta, config.rotaryScale,
                rotaryDim, ropeMaxLength, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("CUDA kernel launch for initializeNormalRopeCosSin failed: %s", e.what());
            return false;
        }
    }
    else if (config.type == RopeType::kLongRope)
    {
        // TODO: Implement the initialization for longrope type here.
        LOG_ERROR("Unimplemented LongRope type initialization.");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm