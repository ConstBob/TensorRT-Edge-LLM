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

//! Global profiling control flag accessors (defined in timer.cpp)
//! When false, no profiling data (metrics or timing) will be recorded
bool getProfilingEnabled();
void setProfilingEnabled(bool enabled);

namespace metrics
{

//! Stage name constants to avoid hardcoding strings
namespace StageNames
{
inline std::string const kLLM_PREFILL = "llm_prefill";
inline std::string const kLLM_GENERATION = "llm_generation";
inline std::string const kMULTIMODAL_PROCESSING = "multimodal_processing";
inline std::string const kEAGLE_DRAFT_PREFILL = "eagle_draft_prefill";
inline std::string const kEAGLE_CONSTRUCT_DRAFT_TREE = "eagle_construct_draft_tree";
inline std::string const kEAGLE_BASE_VERIFICATION = "eagle_base_verification";
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
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        reusedTokens += reused;
        computedTokens += computed;
    }
};

//! LLM Generation stage metrics
class LLMGenerationMetrics : public BaseMetrics
{
public:
    int64_t generatedTokens{0};

    void recordRun(int64_t generated)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        generatedTokens += generated;
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
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalImages += imageCount;
        totalImageTokens += imageTokens;
    }
};

//! Eagle Generation stage metrics
class EagleGenerationMetrics : public BaseMetrics
{
public:
    int64_t totalIterations{0};
    int64_t totalGeneratedTokens{0};

    void recordRun(int64_t iterations, int64_t generatedTokens)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalIterations += iterations;
        totalGeneratedTokens += generatedTokens;
    }
};

} // namespace metrics
} // namespace drivellm
