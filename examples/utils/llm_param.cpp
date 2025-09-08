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

#include "llm_param.h"

namespace CommonOptions
{

const struct option baseOptions[] = {{"engineDir", optional_argument, 0, 1}, {"help", no_argument, 0, 3},
    {"debug", no_argument, 0, 4}, {"noCudaGraph", no_argument, 0, 5}, {0, 0, 0, 0}};

const struct option eagleBuildOptions[] = {{"isEagleBase", no_argument, 0, 101}, {"isEagleDraft", no_argument, 0, 102},
    {"isEagle3", no_argument, 0, 103}, {"maxDecodingTokens", optional_argument, 0, 104},
    {"mMaxDraftTokensPerStep", optional_argument, 0, 105}, {0, 0, 0, 0}};

const struct option eagleOptions[]
    = {{"baseModelDir", optional_argument, 0, 201}, {"draftModelDir", optional_argument, 0, 202},
        {"isEagle3", no_argument, 0, 203}, {"maxDecodingTokens", optional_argument, 0, 204},
        {"topK", optional_argument, 0, 205}, {"maxPathLen", optional_argument, 0, 206}, {0, 0, 0, 0}};

const struct option vlmBuildOptions[]
    = {{"imageTokens", optional_argument, 0, 301}, {"minImageTokens", optional_argument, 0, 302},
        {"maxImageTokens", optional_argument, 0, 303}, {"usePromptTuning", no_argument, 0, 304}, {0, 0, 0, 0}};

const struct option vlmRunOptions[] = {{"visualEngineDir", required_argument, 0, 401}, {0, 0, 0, 0}};

bool parseBaseOptions(BaseParams& baseParams, int opt, char const* optarg)
{
    switch (opt)
    {
    case 1: // engineDir
        if (optarg)
        {
            baseParams.engineDir = optarg;
        }
        else
        {
            std::cerr << "ERROR: --engineDir requires option argument" << std::endl;
            return false;
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
    case 201: // baseModelDir
        if (optarg)
            eagleParams.baseModelDir = optarg;
        break;
    case 202: // draftModelDir
        if (optarg)
            eagleParams.draftModelDir = optarg;
        break;
    case 203: // isEagle3
        eagleParams.isEagle3 = true;
        break;
    case 204: // maxDecodingTokens
        if (optarg)
        {
            eagleParams.maxDecodingTokens = std::stoi(optarg);
        }
        break;
    case 205: // topK
        if (optarg)
        {
            eagleParams.topK = std::stoi(optarg);
        }
        break;
    case 206: // maxPathLen
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
            vlmBuildParams.imageTokens = std::stoll(optarg);
        }
        break;
    case 302:
        if (optarg)
        {
            vlmBuildParams.minImageTokens = std::stoll(optarg);
        }
        break;
    case 303:
        if (optarg)
        {
            vlmBuildParams.maxImageTokens = std::stoll(optarg);
        }
        break;
    case 304: vlmBuildParams.usePromptTuning = true; break;
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
            vlmRunParams.visualEngineDir = optarg;
        }
        else
        {
            std::cerr << "ERROR: --visualEngineDir requires option argument" << std::endl;
            return false;
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
    std::cerr << "  --engineDir           Provide the path to the engine directory. Required." << std::endl;
    std::cerr << "  --help                Print help message." << std::endl;
    std::cerr << "  --debug               Print debug message." << std::endl;
    std::cerr << "  --noCudaGraph         Disable CUDA graph." << std::endl;
}

void printEagleOptions()
{
    std::cerr << "  --baseModelDir        Provide the base model directory path. Required for Eagle mode." << std::endl;
    std::cerr << "  --draftModelDir       Provide the draft model directory path. Required for Eagle mode."
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
    std::cerr << "  --engineDir           Provide the path to the engine directory. Required." << std::endl;
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
    std::cerr << "  --imageTokens       Number of image tokens. Default = 512." << std::endl;
    std::cerr << "  --minImageTokens    Minimum number of image tokens. Default = 4." << std::endl;
    std::cerr << "  --maxImageTokens    Maximum number of image tokens. Default = 1024." << std::endl;
    std::cerr << "  --usePromptTuning  Enable prompt tuning for VLM build. Default = false." << std::endl;
}

void printVLMRunOptions()
{
    std::cerr << "  --visualEngineDir   Provide the visual TensorRT engine directory path. Required." << std::endl;
}
} // namespace CommonUsage
