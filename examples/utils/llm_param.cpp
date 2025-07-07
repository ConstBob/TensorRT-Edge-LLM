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

const struct option baseOptions[] = {{"enginePath", optional_argument, 0, 'e'},
    {"tokenizerPath", optional_argument, 0, 't'}, {"help", no_argument, 0, 'h'}, {"debug", no_argument, 0, 'd'},
    {"noCudaGraph", no_argument, 0, 'g'}, {0, 0, 0, 0}};

const struct option eagleBuildOptions[] = {{"isEagleBase", no_argument, 0, 'e'}, {"isEagleDraft", no_argument, 0, 'g'},
    {"isEagle3", no_argument, 0, 'a'}, {"maxDecodingTokens", optional_argument, 0, 'm'},
    {"mMaxDraftTokensPerStep", optional_argument, 0, 'p'}, {0, 0, 0, 0}};
const struct option eagleOptions[] = {{"eagleEnginePath", optional_argument, 0, 'E'}, {"isEagle3", no_argument, 0, 'a'},
    {"maxDecodingTokens", optional_argument, 0, 'm'}, {"topK", optional_argument, 0, 'k'},
    {"maxPathLen", optional_argument, 0, 'p'}, {0, 0, 0, 0}};

bool parseBaseOptions(BaseParams& baseParams, int opt, char const* optarg, bool requireTokenizer)
{
    switch (opt)
    {
    case 'e': // enginePath
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
    case 't': // tokenizerPath
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
    case 'h': // help
        baseParams.help = true;
        break;
    case 'd': // debug
        baseParams.debug = true;
        break;
    case 'g': // noCudaGraph
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
    case 'E': // eagleEnginePath
        if (optarg)
            eagleParams.eagleEnginePath = optarg;
        break;
    case 'a': // isEagle3
        eagleParams.isEagle3 = true;
        break;
    case 'm': // maxDecodingTokens
        if (optarg)
        {
            eagleParams.maxDecodingTokens = std::stoi(optarg);
        }
        break;
    case 'k': // topK
        if (optarg)
        {
            eagleParams.topK = std::stoi(optarg);
        }
        break;
    case 'p': // maxPathLen
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
    case 'e': // isEagleBase
        eagleBuildParams.isEagleBase = true;
        break;
    case 'g': // isEagleDraft
        eagleBuildParams.isEagleDraft = true;
        break;
    case 'a': // isEagle3
        eagleBuildParams.isEagle3 = true;
        break;
    case 'm': // maxDecodingTokens
        if (optarg)
        {
            eagleBuildParams.maxDecodingTokens = std::stoi(optarg);
        }
        break;
    case 'p': // mMaxDraftTokensPerStep
        if (optarg)
        {
            eagleBuildParams.mMaxDraftTokensPerStep = std::stoi(optarg);
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
} // namespace CommonUsage
