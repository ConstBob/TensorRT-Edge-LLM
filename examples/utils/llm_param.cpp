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

#include "llm_param.h"

namespace CommonOptions
{

const struct option baseOptions[]
    = {{"enginePath", optional_argument, 0, 1}, {"tokenizerPath", optional_argument, 0, 2}, {"help", no_argument, 0, 3},
        {"debug", no_argument, 0, 4}, {"noCudaGraph", no_argument, 0, 5}, {0, 0, 0, 0}};

const struct option eagleBuildOptions[] = {{"isEagleBase", no_argument, 0, 101}, {"isEagleDraft", no_argument, 0, 102},
    {"isEagle3", no_argument, 0, 103}, {"maxDecodingTokens", optional_argument, 0, 104},
    {"mMaxDraftTokensPerStep", optional_argument, 0, 105}, {0, 0, 0, 0}};

const struct option eagleOptions[] = {{"eagleEnginePath", optional_argument, 0, 201}, {"isEagle3", no_argument, 0, 202},
    {"maxDecodingTokens", optional_argument, 0, 203}, {"topK", optional_argument, 0, 204},
    {"maxPathLen", optional_argument, 0, 205}, {0, 0, 0, 0}};

const struct option vlmBuildOptions[] = {{"modelType", optional_argument, 0, 301},
    {"imageTokens", optional_argument, 0, 302}, {"minImageTokens", optional_argument, 0, 303},
    {"maxImageTokens", optional_argument, 0, 304}, {"usePromptTuning", no_argument, 0, 305}, {0, 0, 0, 0}};

const struct option vlmRunOptions[]
    = {{"visualEnginePath", required_argument, 0, 401}, {"modelType", optional_argument, 0, 402}, {0, 0, 0, 0}};

bool parseBaseOptions(BaseParams& baseParams, int opt, char const* optarg, bool requireTokenizer)
{
    switch (opt)
    {
    case 1: // enginePath
        if (optarg)
        {
            baseParams.enginePath = optarg;
        }
        else
        {
            std::cerr << "ERROR: --enginePath requires option argument" << std::endl;
            return false;
        }
        break;
    case 2: // tokenizerPath
        if (optarg)
        {
            baseParams.tokenizerPath = optarg;
        }
        else
        {
            if (requireTokenizer)
            {
                std::cerr << "ERROR: --tokenizerPath requires option argument" << std::endl;
                return false;
            }
        }
        break;
    case 3: // help
        baseParams.help = true;
        break;
    case 4: // debug
        baseParams.debug = true;
        break;
    case 5: // noCudaGraph
        baseParams.noCudaGraph = true;
        break;
    default: return false;
    }
    return true;
}

bool parseEagleOptions(EagleParams& eagleParams, int opt, char const* optarg)
{
    switch (opt)
    {
    case 201: // eagleEnginePath
        if (optarg)
            eagleParams.eagleEnginePath = optarg;
        break;
    case 202: // isEagle3
        eagleParams.isEagle3 = true;
        break;
    case 203: // maxDecodingTokens
        if (optarg)
        {
            eagleParams.maxDecodingTokens = std::stoi(optarg);
        }
        break;
    case 204: // topK
        if (optarg)
        {
            eagleParams.topK = std::stoi(optarg);
        }
        break;
    case 205: // maxPathLen
        if (optarg)
        {
            eagleParams.maxPathLen = std::stoi(optarg);
        }
        break;
    default: return false;
    }
    return true;
}

bool parseEagleBuildOptions(EagleBuildParams& eagleBuildParams, int opt, char const* optarg)
{
    switch (opt)
    {
    case 101: // isEagleBase
        eagleBuildParams.isEagleBase = true;
        break;
    case 102: // isEagleDraft
        eagleBuildParams.isEagleDraft = true;
        break;
    case 103: // isEagle3
        eagleBuildParams.isEagle3 = true;
        break;
    case 104: // maxDecodingTokens
        if (optarg)
        {
            eagleBuildParams.maxDecodingTokens = std::stoi(optarg);
        }
        break;
    case 105: // mMaxDraftTokensPerStep
        if (optarg)
        {
            eagleBuildParams.mMaxDraftTokensPerStep = std::stoi(optarg);
        }
        break;
    default: return false;
    }
    return true;
}

bool parseVLMBuildOptions(VLMBuildParams& vlmBuildParams, int opt, char const* optarg)
{
    switch (opt)
    {
    case 301:
        if (optarg)
        {
            vlmBuildParams.modelType = optarg;
        }
        break;
    case 302:
        if (optarg)
        {
            vlmBuildParams.imageTokens = std::stoll(optarg);
        }
        break;
    case 303:
        if (optarg)
        {
            vlmBuildParams.minImageTokens = std::stoll(optarg);
        }
        break;
    case 304:
        if (optarg)
        {
            vlmBuildParams.maxImageTokens = std::stoll(optarg);
        }
        break;
    case 305: vlmBuildParams.usePromptTuning = true; break;
    default: return false;
    }
    return true;
}

bool parseVLMRunOptions(VLMRunParams& vlmRunParams, int opt, char const* optarg)
{
    switch (opt)
    {
    case 401:
        if (optarg)
        {
            vlmRunParams.visualEnginePath = optarg;
        }
        else
        {
            std::cerr << "ERROR: --visualEnginePath requires option argument" << std::endl;
            return false;
        }
        break;
    case 402:
        if (optarg)
        {
            vlmRunParams.modelType = optarg;
        }
        break;
    default: return false;
    }
    return true;
}
} // namespace CommonOptions

namespace CommonUsage
{
void printBaseOptions()
{
    std::cerr << "  --enginePath          Provide the path to the engine file. Required." << std::endl;
    std::cerr << "  --tokenizerPath       Provide the path to HF tokenizer. Required for chat, and accuracy modes."
              << std::endl;
    std::cerr << "  --help                Print help message." << std::endl;
    std::cerr << "  --debug               Print debug message." << std::endl;
    std::cerr << "  --noCudaGraph         Disable CUDA graph." << std::endl;
}

void printEagleOptions()
{
    std::cerr << "  --eagleEnginePath     Provide the input Eagle engine file path. Required for Eagle mode."
              << std::endl;
    std::cerr << "  --isEagle3            Use Eagle3 mode. Default is Eagle2." << std::endl;
    std::cerr
        << "  --maxDecodingTokens   Provide the maximum decoding tokens for target model, the number provided must "
           "be aligned with building phase. Optional, default = 60"
        << std::endl;
    std::cerr << "  --topK                Provide the topK for draft model to select the topK candidates. the number "
                 "provided must be aligned with building phase. Optional, default is 10."
              << std::endl;
    std::cerr << "  --maxPathLen          Provide the max stack layers for draft model to construct the max tree path "
                 "length. the number provided must be aligned with building phase. Optional, default is 6."
              << std::endl;
}

void printLoraOptions()
{
    std::cerr << "  --loraWeights         Provide LoRA weights in format name:path." << std::endl;
}

void printBenchmarkOptions()
{
    std::cerr << "  --tokenizerPath       Provide the path to HF tokenizer. Required for Eagle models, optional for "
                 "standard models."
              << std::endl;
}

void printEagleBuildOptions()
{
    std::cerr << "  --isEagleBase         Build for Eagle base model. Default is false." << std::endl;
    std::cerr << "  --isEagleDraft        Build for Eagle draft model. Default is false." << std::endl;
    std::cerr << "  --isEagle3            Build for Eagle3 model. Default is false." << std::endl;
    std::cerr
        << "  --maxDecodingTokens   Provide the maximum decoding tokens for target model, the number provided must "
           "be aligned with running phase. Optional, default = 60"
        << std::endl;
}

void printVLMBuildOptions()
{
    std::cerr << "  --modelType         Model type for VLM build. Default = qwen2_vl." << std::endl;
    std::cerr << "  --imageTokens       Number of image tokens. Default = 512." << std::endl;
    std::cerr << "  --minImageTokens    Minimum number of image tokens. Default = 4." << std::endl;
    std::cerr << "  --maxImageTokens    Maximum number of image tokens. Default = 1024." << std::endl;
    std::cerr << "  --usePromptTuning  Enable prompt tuning for VLM build. Default = false." << std::endl;
}

void printVLMRunOptions()
{
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required." << std::endl;
    std::cerr << "  --modelType         Provide the model type. Default = qwen2_vl." << std::endl;
}
} // namespace CommonUsage
