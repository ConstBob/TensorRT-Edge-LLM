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

#include "profileFormatter.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "memoryMonitor.h"
#include "profiling/timer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <future>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <thread>

using namespace trt_edgellm;

namespace
{
//! Helper function to convert bytes to megabytes
double toMB(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

//! Utility function for calculating prefill tokens per second
float getPrefillTokensPerSecond(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    int64_t totalTokens = prefillMetrics.reusedTokens + prefillMetrics.computedTokens;
    if (totalTokens > 0)
    {
        return static_cast<float>(totalTokens) / (timingData->getTotalGpuTimeMs() / 1000.0f);
    }
    return 0.0f;
}

//! Utility function for calculating generation tokens per second
float getGenerationTokensPerSecond(metrics::LLMGenerationMetrics const& generationMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (generationMetrics.generatedTokens > 0)
    {
        return static_cast<float>(generationMetrics.generatedTokens) / (timingData->getTotalGpuTimeMs() / 1000.0f);
    }
    return 0.0f;
}

//! Utility function for calculating prefill average time per token
float getPrefillAverageTimePerToken(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    int64_t totalTokens = prefillMetrics.reusedTokens + prefillMetrics.computedTokens;
    if (totalTokens > 0)
    {
        return timingData->getTotalGpuTimeMs() / totalTokens;
    }
    return 0.0f;
}

//! Utility function for calculating prefill average tokens per run
float getPrefillAverageTokensPerRun(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    return static_cast<float>(prefillMetrics.reusedTokens + prefillMetrics.computedTokens)
        / prefillMetrics.getTotalRuns();
}

//! Utility function for calculating prefill average time per run
float getPrefillAverageTimePerRun(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getAverageTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    return timingData->getAverageTimeMs();
}

//! Utility function for calculating generation average time per token
float getGenerationAverageTimePerToken(metrics::LLMGenerationMetrics const& generationMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (generationMetrics.generatedTokens > 0)
    {
        return timingData->getTotalGpuTimeMs() / generationMetrics.generatedTokens;
    }
    return 0.0f;
}

//! Utility function for calculating multimodal average time per token
float getMultimodalAverageTimePerToken(metrics::MultimodalMetrics const& multimodalMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kMULTIMODAL_PROCESSING);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (multimodalMetrics.totalImageTokens > 0)
    {
        return timingData->getTotalGpuTimeMs() / multimodalMetrics.totalImageTokens;
    }
    return 0.0f;
}

//! Utility function for calculating Eagle overall tokens per second (excluding base model prefill)
float getEagleOverallTokensPerSecond(metrics::EagleGenerationMetrics const& eagleGenerationMetrics)
{
    if (eagleGenerationMetrics.totalGeneratedTokens <= 0)
    {
        return 0.0f;
    }

    // Calculate total time for all Eagle stages except base prefill
    float totalTimeMs = 0.0f;

    auto draftPrefillData = gTimer.getTimingData(metrics::StageNames::kEAGLE_DRAFT_PREFILL);
    if (draftPrefillData)
    {
        totalTimeMs += draftPrefillData->getTotalGpuTimeMs();
    }

    auto constructDraftTreeData = gTimer.getTimingData(metrics::StageNames::kEAGLE_CONSTRUCT_DRAFT_TREE);
    if (constructDraftTreeData)
    {
        totalTimeMs += constructDraftTreeData->getTotalGpuTimeMs();
    }

    auto baseVerificationData = gTimer.getTimingData(metrics::StageNames::kEAGLE_BASE_VERIFICATION);
    if (baseVerificationData)
    {
        totalTimeMs += baseVerificationData->getTotalGpuTimeMs();
    }

    if (totalTimeMs > 0.0f)
    {
        return static_cast<float>(eagleGenerationMetrics.totalGeneratedTokens) / (totalTimeMs / 1000.0f);
    }
    return 0.0f;
}

//! Utility function for calculating Eagle average acceptance rate
float getEagleAverageAcceptanceRate(metrics::EagleGenerationMetrics const& eagleGenerationMetrics)
{
    if (eagleGenerationMetrics.totalIterations <= 0)
    {
        return 0.0f;
    }

    return static_cast<float>(eagleGenerationMetrics.totalGeneratedTokens)
        / static_cast<float>(eagleGenerationMetrics.totalIterations);
}

//! Helper function to append timing data for a stage to an ostream
void appendStageTimingData(std::ostream& summary, std::string const& stageName, std::string const& displayName)
{
    auto timingData = gTimer.getTimingData(stageName);
    if (timingData && timingData->getTotalRuns() > 0)
    {
        summary << displayName << " - Total Runs: " << timingData->getTotalRuns() << ", Total GPU Time: " << std::fixed
                << std::setprecision(2) << timingData->getTotalGpuTimeMs()
                << " ms, Average: " << timingData->getAverageTimeMs() << " ms" << std::endl;
    }
}

} // anonymous namespace

StatisticalAnalysis StatisticalAnalysis::calculate(std::vector<float> const& data)
{
    StatisticalAnalysis stats;
    if (data.empty())
    {
        return stats;
    }

    stats.count = data.size();
    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    auto minmax = std::minmax_element(data.begin(), data.end());
    stats.min = static_cast<double>(*minmax.first);
    stats.max = static_cast<double>(*minmax.second);

    double variance = 0.0;
    for (float value : data)
    {
        double dValue = static_cast<double>(value);
        variance += (dValue - stats.mean) * (dValue - stats.mean);
    }
    stats.stddev = std::sqrt(variance / data.size());

    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t size = sorted_data.size();
    stats.median = static_cast<double>(sorted_data[size / 2]);

    size_t p95_index = std::min(static_cast<size_t>(size * 0.95), size - 1);
    size_t p99_index = std::min(static_cast<size_t>(size * 0.99), size - 1);

    stats.p95 = static_cast<double>(sorted_data[p95_index]);
    stats.p99 = static_cast<double>(sorted_data[p99_index]);

    return stats;
}

void outputPrefillProfile(std::ostream& output, metrics::LLMPrefillMetrics const& prefillMetrics)
{
    if (prefillMetrics.getTotalRuns() > 0)
    {
        output << "=== LLM Prefill ===" << std::endl;
        output << "Reused Tokens: " << prefillMetrics.reusedTokens << std::endl;
        output << "Computed Tokens: " << prefillMetrics.computedTokens << std::endl;
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << getPrefillAverageTokensPerRun(prefillMetrics) << std::endl;
        output << "Average Time per Run: " << std::fixed << std::setprecision(4)
               << getPrefillAverageTimePerRun(prefillMetrics) << " ms" << std::endl;
        output << "Tokens/Second: " << std::fixed << std::setprecision(1) << getPrefillTokensPerSecond(prefillMetrics)
               << std::endl;
        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getPrefillAverageTimePerToken(prefillMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kLLM_PREFILL, "LLM Prefill");
    }
}

void outputGenerationProfile(std::ostream& output, metrics::LLMGenerationMetrics const& generationMetrics)
{
    output << "=== LLM Generation (Excluding sampling after prefill) ===" << std::endl;

    if (generationMetrics.getTotalRuns() > 0)
    {
        output << "Generated Tokens: " << generationMetrics.generatedTokens << std::endl;
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << static_cast<float>(generationMetrics.generatedTokens) / generationMetrics.getTotalRuns() << std::endl;
        output << "Tokens/Second: " << std::fixed << std::setprecision(1)
               << getGenerationTokensPerSecond(generationMetrics) << std::endl;
        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getGenerationAverageTimePerToken(generationMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kLLM_GENERATION, "LLM Generation");
    }
    else
    {
        output << "max_generate_length = 1, the model only runs the prefill stage." << std::endl;
    }
}

void outputEagleGenerationProfile(std::ostream& output, metrics::EagleGenerationMetrics const& eagleGenerationMetrics)
{
    if (eagleGenerationMetrics.getTotalRuns() > 0)
    {
        output << "=== Eagle Generation ===" << std::endl;
        output << "Total Iterations: " << eagleGenerationMetrics.totalIterations << std::endl;
        output << "Total Generated Tokens: " << eagleGenerationMetrics.totalGeneratedTokens << std::endl;
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << static_cast<float>(eagleGenerationMetrics.totalGeneratedTokens)
                / eagleGenerationMetrics.getTotalRuns()
               << std::endl;
        output << "Average Acceptance Rate: " << std::fixed << std::setprecision(2)
               << getEagleAverageAcceptanceRate(eagleGenerationMetrics) << std::endl;
        output << "Overall Tokens/Second (excluding base prefill): " << std::fixed << std::setprecision(1)
               << getEagleOverallTokensPerSecond(eagleGenerationMetrics) << std::endl;

        // Individual Eagle stage timing
        appendStageTimingData(output, metrics::StageNames::kEAGLE_DRAFT_PREFILL, "Draft Model Prefill");
        appendStageTimingData(output, metrics::StageNames::kEAGLE_CONSTRUCT_DRAFT_TREE, "Construct Draft Tree");
        appendStageTimingData(output, metrics::StageNames::kEAGLE_BASE_VERIFICATION, "Base Model Verification");
    }
}

void outputMultimodalProfile(std::ostream& output, metrics::MultimodalMetrics const& multimodalMetrics)
{
    if (multimodalMetrics.getTotalRuns() > 0)
    {
        output << "=== Multimodal Processing ===" << std::endl;
        output << "Total Image Tokens: " << multimodalMetrics.totalImageTokens << std::endl;
        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getMultimodalAverageTimePerToken(multimodalMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kMULTIMODAL_PROCESSING, "Multimodal Processing");
    }
}

void outputMemoryProfile(std::ostream& output, size_t peakGpuMemoryBytes)
{
    if (peakGpuMemoryBytes > 0)
    {
        output << "=== Memory Usage ===" << std::endl;
        output << "Peak GPU Memory: " << std::fixed << std::setprecision(2) << toMB(peakGpuMemoryBytes) << " MB ("
               << peakGpuMemoryBytes << " bytes)" << std::endl;
    }
}

void addJsonPrefillSummary(nlohmann::json& summary, metrics::LLMPrefillMetrics const& prefillMetrics)
{
    if (prefillMetrics.getTotalRuns() > 0)
    {
        summary["prefill"] = {{"total_runs", prefillMetrics.getTotalRuns()},
            {"reused_tokens", prefillMetrics.reusedTokens}, {"computed_tokens", prefillMetrics.computedTokens},
            {"average_tokens_per_run", getPrefillAverageTokensPerRun(prefillMetrics)},
            {"average_time_per_run_ms", getPrefillAverageTimePerRun(prefillMetrics)},
            {"tokens_per_second", getPrefillTokensPerSecond(prefillMetrics)},
            {"average_time_per_token_ms", getPrefillAverageTimePerToken(prefillMetrics)}};
    }
}

void addJsonGenerationSummary(nlohmann::json& summary, metrics::LLMGenerationMetrics const& generationMetrics)
{
    if (generationMetrics.getTotalRuns() > 0)
    {
        summary["generation"] = {{"total_runs", generationMetrics.getTotalRuns()},
            {"generated_tokens", generationMetrics.generatedTokens},
            {"average_tokens_per_run",
                static_cast<float>(generationMetrics.generatedTokens) / generationMetrics.getTotalRuns()},
            {"tokens_per_second", getGenerationTokensPerSecond(generationMetrics)},
            {"average_time_per_token_ms", getGenerationAverageTimePerToken(generationMetrics)}};
    }
}

void addJsonEagleGenerationSummary(
    nlohmann::json& summary, metrics::EagleGenerationMetrics const& eagleGenerationMetrics)
{
    if (eagleGenerationMetrics.getTotalRuns() > 0)
    {
        summary["eagle_generation"] = {{"total_runs", eagleGenerationMetrics.getTotalRuns()},
            {"total_iterations", eagleGenerationMetrics.totalIterations},
            {"total_generated_tokens", eagleGenerationMetrics.totalGeneratedTokens},
            {"average_tokens_per_run",
                static_cast<float>(eagleGenerationMetrics.totalGeneratedTokens)
                    / eagleGenerationMetrics.getTotalRuns()},
            {"average_acceptance_rate", getEagleAverageAcceptanceRate(eagleGenerationMetrics)},
            {"overall_tokens_per_second_excluding_base_prefill",
                getEagleOverallTokensPerSecond(eagleGenerationMetrics)}};
    }
}

void addJsonMultimodalSummary(nlohmann::json& summary, metrics::MultimodalMetrics const& multimodalMetrics)
{
    if (multimodalMetrics.getTotalRuns() > 0)
    {
        summary["multimodal"] = {{"total_runs", multimodalMetrics.getTotalRuns()},
            {"total_images", multimodalMetrics.totalImages}, {"total_image_tokens", multimodalMetrics.totalImageTokens},
            {"average_time_per_token_ms", getMultimodalAverageTimePerToken(multimodalMetrics)}};
    }
}

void addJsonTimingStages(nlohmann::json& summary)
{
    summary["stages"] = nlohmann::json::array();
    for (auto const& [stageId, timingData] : gTimer.getAllTimingData())
    {
        if (timingData.gpuTimesMs.empty())
        {
            continue;
        }
        nlohmann::json stageJson;
        stageJson["stage_id"] = stageId;
        stageJson["total_runs"] = timingData.getTotalRuns();
        stageJson["total_gpu_time_ms"] = timingData.getTotalGpuTimeMs();
        stageJson["average_time_per_run_ms"] = timingData.getAverageTimeMs();

        // Add statistical analysis if available
        auto gpuStats = StatisticalAnalysis::calculate(timingData.gpuTimesMs);
        stageJson["gpu_time_stats"] = {{"count", gpuStats.count}, {"min_ms", gpuStats.min}, {"max_ms", gpuStats.max},
            {"mean_ms", gpuStats.mean}, {"median_ms", gpuStats.median}, {"p95_ms", gpuStats.p95},
            {"p99_ms", gpuStats.p99}, {"stddev_ms", gpuStats.stddev}};

        summary["stages"].push_back(stageJson);
    }
}

void addJsonMemorySummary(nlohmann::json& summary, size_t peakGpuMemoryBytes)
{
    if (peakGpuMemoryBytes > 0)
    {
        summary["peak_gpu_memory_bytes"] = peakGpuMemoryBytes;
        summary["peak_gpu_memory_mb"] = toMB(peakGpuMemoryBytes);
    }
}

std::string sanitizeUtf8ForJson(std::string const& input)
{
    // Use nlohmann::json's built-in UTF-8 validation by attempting to serialize to JSON
    // UTF-8 validation happens during dump(), not during assignment
    try
    {
        nlohmann::json testJson = input;
        // Actually call dump() to trigger UTF-8 validation
        testJson.dump();
        // If successful, the string is valid UTF-8
        return input;
    }
    catch (std::exception const& e)
    {
        // Catch any other exceptions
        LOG_WARNING("Error validating string for JSON: %s", e.what());
        LOG_WARNING("Original text (full): %s", input.c_str());

        // Return error message
        return "[ERROR: Invalid UTF-8 character detected in the output response. Check logs for original text.]";
    }
}
