/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/state/contextCache/specReuseContract.h"

#include "common/checkMacros.h"

#include <cstdint>
#include <limits>

namespace trt_edgellm::rt
{

std::optional<SpecReuseContract> resolveSpecReuseContract(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.base.specDecodeType == SpecDecodeMode::kNONE
            || (deployment.draft.has_value() && deployment.specConfig.has_value()),
        "Speculative context reuse requires draft and speculative deployment configuration.");

    SpecReuseContract contract;
    int64_t workingTokens{};
    switch (deployment.base.specDecodeType)
    {
    case SpecDecodeMode::kNONE: return std::nullopt;
    case SpecDecodeMode::kMTP:
        contract = SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1,
            /*speculativeWorkingTokens=*/0};
        workingTokens = static_cast<int64_t>(deployment.specConfig->draftingStep)
            * static_cast<int64_t>(deployment.specConfig->draftingTopK);
        break;
    case SpecDecodeMode::kEAGLE:
        // EAGLE shifts draft token IDs by one relative to base hidden states and uses a padded step-by-fanout tree.
        contract = SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1,
            /*speculativeWorkingTokens=*/0};
        workingTokens = static_cast<int64_t>(deployment.specConfig->draftingStep)
            * static_cast<int64_t>(deployment.specConfig->draftingTopK);
        break;
    case SpecDecodeMode::kGemma4MTP:
        // The assistant mutates only target-owned KV lengths and owns no page pool.
        contract = SpecReuseContract{/*ownsPagedSpecState=*/false, /*futureDependencyTokens=*/0,
            /*speculativeWorkingTokens=*/0};
        break;
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kJetSpec:
        // One draft forward writes the complete configured block.
        contract = SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/0,
            /*speculativeWorkingTokens=*/0};
        workingTokens = deployment.specConfig->dflashBlockSize;
        break;
    case SpecDecodeMode::kDSpark:
        // DSpark's proposal is bounded by the exported block size.
        contract = SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/0,
            /*speculativeWorkingTokens=*/0};
        workingTokens = deployment.draft->specDraftBlockSize;
        break;
    default: ELLM_CHECK(false, "resolveSpecReuseContract: unhandled SpecDecodeMode");
    }

    ELLM_CHECK(workingTokens >= 0 && workingTokens <= std::numeric_limits<int32_t>::max(),
        "Speculative context reuse working-set geometry exceeds int32.");
    ELLM_CHECK(!contract.ownsPagedSpecState || workingTokens > 0,
        "Paged speculative context reuse requires a positive speculative working set.");
    contract.speculativeWorkingTokens = static_cast<int32_t>(workingTokens);
    return contract;
}

int32_t replayPages(SpecReuseContract const& contract, int32_t pageSize)
{
    ELLM_CHECK(pageSize > 0, "Context cache page size must be positive");
    ELLM_CHECK(
        contract.futureDependencyTokens >= 0, "Context cache speculative future dependency must be non-negative");
    return (contract.futureDependencyTokens + pageSize - 1) / pageSize;
}

} // namespace trt_edgellm::rt
