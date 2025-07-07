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

#pragma once

#include <getopt.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct BaseParams
{
    std::string enginePath;
    std::string tokenizerPath{""};
    bool help{false};
    bool debug{false};
    bool noCudaGraph{false};
};

struct EagleParams
{
    std::string eagleEnginePath{""};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
    int32_t topK{10};
    int32_t maxPathLen{6};
};

struct EagleBuildParams
{
    bool isEagleBase{false};
    bool isEagleDraft{false};
    bool isEagle3{false};
    int32_t maxDecodingTokens{60};
    int32_t mMaxDraftTokensPerStep{60};
};

struct LoraWeights
{
    std::vector<std::pair<std::string, std::string>> weights;

    void add(std::string const& name, std::string const& path)
    {
        weights.emplace_back(name, path);
    }

    // Get the first LoRA weight (for backward compatibility)
    std::pair<std::string, std::string> getFirst() const
    {
        return weights.empty() ? std::make_pair("", "") : weights[0];
    }

    bool hasWeights() const
    {
        return !weights.empty();
    }

    static bool validateFormat(std::string const& loraArg)
    {
        size_t colonPos = loraArg.find(':');
        if (colonPos == std::string::npos)
        {
            std::cerr << "ERROR: LoRA weights must be in format name:path" << std::endl;
            return false;
        }
        return true;
    }

    static std::pair<std::string, std::string> parse(std::string const& loraArg)
    {
        size_t colonPos = loraArg.find(':');
        if (colonPos == std::string::npos)
        {
            return std::make_pair("", "");
        }
        std::string name = loraArg.substr(0, colonPos);
        std::string path = loraArg.substr(colonPos + 1);
        return std::make_pair(name, path);
    }
};

namespace CommonOptions
{

extern const struct option baseOptions[];
extern const struct option eagleOptions[];
extern const struct option eagleBuildOptions[];
bool parseBaseOptions(BaseParams& baseParams, int opt, char const* optarg, bool requireTokenizer = false);
bool parseEagleOptions(EagleParams& eagleParams, int opt, char const* optarg);
bool parseEagleBuildOptions(EagleBuildParams& eagleBuildParams, int opt, char const* optarg);
} // namespace CommonOptions

namespace CommonUsage
{
void printBaseOptions();
void printEagleOptions();
void printEagleBuildOptions();
void printLoraOptions();
void printBenchmarkOptions();
} // namespace CommonUsage