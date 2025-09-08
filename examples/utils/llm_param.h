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

#include <getopt.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct BaseParams
{
    std::string engineDir;
    bool help{false};
    bool debug{false};
    bool noCudaGraph{false};
};

struct EagleParams
{
    std::string baseModelDir{""};
    std::string draftModelDir{""};
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

struct VLMBuildParams
{
    int64_t imageTokens{512};
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
    bool usePromptTuning{false};
};

struct VLMRunParams
{
    std::string visualEngineDir{""};
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
extern const struct option vlmBuildOptions[];
extern const struct option vlmRunOptions[];

bool parseBaseOptions(BaseParams& baseParams, int opt, char const* optarg);
bool parseEagleOptions(EagleParams& eagleParams, int opt, char const* optarg);
bool parseEagleBuildOptions(EagleBuildParams& eagleBuildParams, int opt, char const* optarg);
bool parseVLMBuildOptions(VLMBuildParams& vlmBuildParams, int opt, char const* optarg);
bool parseVLMRunOptions(VLMRunParams& vlmRunParams, int opt, char const* optarg);

} // namespace CommonOptions

namespace CommonUsage
{
void printBaseOptions();
void printEagleOptions();
void printEagleBuildOptions();
void printVLMBuildOptions();
void printVLMRunOptions();
void printLoraOptions();
void printBenchmarkOptions();
} // namespace CommonUsage