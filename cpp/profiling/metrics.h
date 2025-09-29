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

#include <cstdint>
#include <string>

namespace drivellm
{
namespace metrics
{

//! Stage name constants to avoid hardcoding strings
namespace StageNames
{
inline std::string const kLLM_PREFILL = "llm_prefill";
inline std::string const kLLM_GENERATION = "llm_generation";
inline std::string const kMULTIMODAL_PROCESSING = "multimodal_processing";
} // namespace StageNames

//! Base class for all performance metrics
class BaseMetrics
{
public:
    virtual ~BaseMetrics() = default;

    //! Get total runs count
    int64_t getTotalRuns() const
    {
        return totalRuns;
    }

    //! Reset all metrics data
    virtual void reset() = 0;

protected:
    int64_t totalRuns{0};
};

//! LLM Prefill stage metrics
class LLMPrefillMetrics : public BaseMetrics
{
public:
    int64_t reusedTokens{0};
    int64_t computedTokens{0};

    void recordRun(int64_t reused, int64_t computed)
    {
        totalRuns++;
        reusedTokens += reused;
        computedTokens += computed;
    }

    void reset() override
    {
        totalRuns = 0;
        reusedTokens = 0;
        computedTokens = 0;
    }
};

//! LLM Generation stage metrics
class LLMGenerationMetrics : public BaseMetrics
{
public:
    int64_t generatedTokens{0};

    void recordRun(int64_t generated)
    {
        totalRuns++;
        generatedTokens += generated;
    }

    void reset() override
    {
        totalRuns = 0;
        generatedTokens = 0;
    }
};

//! Multimodal processing stage metrics
class MultimodalMetrics : public BaseMetrics
{
public:
    int64_t totalImages{0};
    int64_t totalImageTokens{0};

    void recordRun(int64_t imageCount, int64_t imageTokens)
    {
        totalRuns++;
        totalImages += imageCount;
        totalImageTokens += imageTokens;
    }

    void reset() override
    {
        totalRuns = 0;
        totalImages = 0;
        totalImageTokens = 0;
    }
};

} // namespace metrics
} // namespace drivellm
