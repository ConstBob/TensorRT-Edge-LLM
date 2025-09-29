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

using namespace drivellm;

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

//! Utility function for calculating multimodal average time per image
float getMultimodalAverageTimePerImage(metrics::MultimodalMetrics const& multimodalMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kMULTIMODAL_PROCESSING);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (multimodalMetrics.totalImages > 0)
    {
        return timingData->getTotalGpuTimeMs() / multimodalMetrics.totalImages;
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

void printSummary(metrics::LLMPrefillMetrics const& prefillMetrics,
    metrics::LLMGenerationMetrics const& generationMetrics, metrics::MultimodalMetrics const& multimodalMetrics,
    size_t peakGpuMemoryBytes)
{
    std::ostringstream summary;
    summary << "\n=== Performance Summary ===\n";

    // LLM Prefill metrics
    if (prefillMetrics.getTotalRuns() > 0)
    {
        auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
        summary << "=== LLM Prefill ===\n";
        summary << "Total Runs: " << prefillMetrics.getTotalRuns() << "\n";
        summary << "Reused Tokens: " << prefillMetrics.reusedTokens << "\n";
        summary << "Computed Tokens: " << prefillMetrics.computedTokens << "\n";
        summary << "Tokens/Second: " << std::fixed << std::setprecision(1) << getPrefillTokensPerSecond(prefillMetrics)
                << "\n";
        summary << "Average Time per Token: " << std::fixed << std::setprecision(4)
                << getPrefillAverageTimePerToken(prefillMetrics) << " ms\n";
        if (timingData)
        {
            summary << "Total GPU Time: " << std::fixed << std::setprecision(2) << timingData->getTotalGpuTimeMs()
                    << " ms\n";
            summary << "Average Time per Run: " << std::fixed << std::setprecision(2) << timingData->getAverageTimeMs()
                    << " ms\n";
        }
        summary << "\n";
    }

    // LLM Generation metrics
    if (generationMetrics.getTotalRuns() > 0)
    {
        auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
        summary << "=== LLM Generation ===\n";
        summary << "Total Runs: " << generationMetrics.getTotalRuns() << "\n";
        summary << "Generated Tokens: " << generationMetrics.generatedTokens << "\n";
        summary << "Tokens/Second: " << std::fixed << std::setprecision(1)
                << getGenerationTokensPerSecond(generationMetrics) << "\n";
        summary << "Average Time per Token: " << std::fixed << std::setprecision(4)
                << getGenerationAverageTimePerToken(generationMetrics) << " ms\n";
        if (timingData)
        {
            summary << "Total GPU Time: " << std::fixed << std::setprecision(2) << timingData->getTotalGpuTimeMs()
                    << " ms\n";
            summary << "Average Time per Run: " << std::fixed << std::setprecision(2) << timingData->getAverageTimeMs()
                    << " ms\n";
        }
        summary << "\n";
    }

    // Multimodal metrics
    if (multimodalMetrics.getTotalRuns() > 0)
    {
        auto timingData = gTimer.getTimingData(metrics::StageNames::kMULTIMODAL_PROCESSING);
        summary << "=== Multimodal Processing ===\n";
        summary << "Total Runs: " << multimodalMetrics.getTotalRuns() << "\n";
        summary << "Total Images: " << multimodalMetrics.totalImages << "\n";
        summary << "Total Image Tokens: " << multimodalMetrics.totalImageTokens << "\n";
        summary << "Average Time per Image: " << std::fixed << std::setprecision(2)
                << getMultimodalAverageTimePerImage(multimodalMetrics) << " ms\n";
        summary << "Average Time per Token: " << std::fixed << std::setprecision(4)
                << getMultimodalAverageTimePerToken(multimodalMetrics) << " ms\n";
        if (timingData)
        {
            summary << "Total GPU Time: " << std::fixed << std::setprecision(2) << timingData->getTotalGpuTimeMs()
                    << " ms\n";
            summary << "Average Time per Run: " << std::fixed << std::setprecision(2) << timingData->getAverageTimeMs()
                    << " ms\n";
        }
        summary << "\n";
    }

    // Add peak GPU memory information if provided
    if (peakGpuMemoryBytes > 0)
    {
        summary << "=== Memory Usage ===\n";
        summary << "Peak GPU Memory: " << std::fixed << std::setprecision(2) << toMB(peakGpuMemoryBytes) << " MB ("
                << peakGpuMemoryBytes << " bytes)\n";
        summary << "\n";
    }

    summary << "=====================================";
    LOG_INFO("%s", summary.str().c_str());
}

std::string getJsonSummary(metrics::LLMPrefillMetrics const& prefillMetrics,
    metrics::LLMGenerationMetrics const& generationMetrics, metrics::MultimodalMetrics const& multimodalMetrics,
    size_t peakGpuMemoryBytes)
{
    nlohmann::json summary;

    // Consolidated stages with all timing and metrics data
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

        // Add stage-specific metrics data
        if (stageId == metrics::StageNames::kLLM_PREFILL)
        {
            stageJson["reused_tokens"] = prefillMetrics.reusedTokens;
            stageJson["computed_tokens"] = prefillMetrics.computedTokens;
            stageJson["tokens_per_second"] = getPrefillTokensPerSecond(prefillMetrics);
            stageJson["average_time_per_token_ms"] = getPrefillAverageTimePerToken(prefillMetrics);
        }
        else if (stageId == metrics::StageNames::kLLM_GENERATION)
        {
            stageJson["generated_tokens"] = generationMetrics.generatedTokens;
            stageJson["tokens_per_second"] = getGenerationTokensPerSecond(generationMetrics);
            stageJson["average_time_per_token_ms"] = getGenerationAverageTimePerToken(generationMetrics);
        }
        else if (stageId == metrics::StageNames::kMULTIMODAL_PROCESSING)
        {
            stageJson["total_images"] = multimodalMetrics.totalImages;
            stageJson["total_image_tokens"] = multimodalMetrics.totalImageTokens;
            stageJson["average_time_per_image_ms"] = getMultimodalAverageTimePerImage(multimodalMetrics);
            stageJson["average_time_per_token_ms"] = getMultimodalAverageTimePerToken(multimodalMetrics);
        }

        summary["stages"].push_back(stageJson);
    }

    // Add peak GPU memory information if provided
    if (peakGpuMemoryBytes > 0)
    {
        summary["peak_gpu_memory_bytes"] = peakGpuMemoryBytes;
        summary["peak_gpu_memory_mb"] = toMB(peakGpuMemoryBytes);
    }

    return summary.dump(2);
}
