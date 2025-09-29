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

#include "profiling/metrics.h"
#include <string>
#include <vector>

//! Statistical analysis results for performance data.
struct StatisticalAnalysis
{
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double median{0.0};
    double p95{0.0};
    double p99{0.0};
    double stddev{0.0};
    size_t count{0};

    static StatisticalAnalysis calculate(std::vector<float> const& data);
};

//! Print comprehensive performance summary
void printSummary(drivellm::metrics::LLMPrefillMetrics const& prefillMetrics,
    drivellm::metrics::LLMGenerationMetrics const& generationMetrics,
    drivellm::metrics::MultimodalMetrics const& multimodalMetrics, size_t peakGpuMemoryBytes = 0);

//! Generate JSON summary
std::string getJsonSummary(drivellm::metrics::LLMPrefillMetrics const& prefillMetrics,
    drivellm::metrics::LLMGenerationMetrics const& generationMetrics,
    drivellm::metrics::MultimodalMetrics const& multimodalMetrics, size_t peakGpuMemoryBytes = 0);
