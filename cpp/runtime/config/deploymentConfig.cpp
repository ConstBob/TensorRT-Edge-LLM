/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/config/deploymentConfig.h"

#include "common/checkMacros.h"

#include "common/logger.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

int32_t DeploymentConfig::maxRuntimeBatchSize() const
{
    // When base and draft engines were built with different max batch sizes, fall
    // back to the smaller — the current runtime cannot drive either engine beyond
    // its engine-declared capacity, so the common ceiling is the safe choice. A
    // stricter "must match exactly" policy belongs to a follow-up that pairs with
    // an export-side guarantee; today we degrade gracefully and warn.
    int32_t const baseMax = base.maxSupportedBatchSize;
    if (!draft.has_value())
    {
        return baseMax;
    }
    int32_t const draftMax = draft->maxSupportedBatchSize;
    if (draftMax != baseMax)
    {
        LOG_WARNING(
            "base.maxSupportedBatchSize=%d vs draft.maxSupportedBatchSize=%d; "
            "using the smaller (%d). Re-export both engines against the same config to silence this warning.",
            baseMax, draftMax, std::min(baseMax, draftMax));
    }
    return std::min(baseMax, draftMax);
}

int32_t DeploymentConfig::effectiveMaxDraftTreeSize() const
{
    ELLM_CHECK(eagle.has_value(),
        "effectiveMaxDraftTreeSize: eagle configuration is not set. "
        "This method is EAGLE-only; guard the call with eagle.has_value().");
    return std::max(eagle->maxDraftTreeSize, eagle->verifyTreeSize);
}

DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath, std::optional<EagleDraftingConfig> const& drafting)
{
    DeploymentConfig cfg;

    // --- Structural precondition: drafting cannot be set without draft ---
    ELLM_CHECK(!drafting.has_value() || draftConfigPath.has_value(),
        "drafting configuration was provided but no draftConfigPath was set. "
        "An EAGLE drafting tree topology requires a draft engine config.");

    // --- Parse base ---
    cfg.base = parseEngineConfig(baseConfigPath);

    // --- Parse draft (if present) ---
    if (draftConfigPath.has_value())
    {
        cfg.draft = parseDraftEngineConfig(*draftConfigPath);
    }

    // No cross-engine consistency check needed: each engine's builder_config
    // carries only its own tree-size budget. The base emits
    // `max_verify_tree_size` (its verification budget); the draft emits
    // `max_draft_tree_size` (its proposal budget). There are no capacity
    // fields shared across the two configs, so there is nothing to
    // cross-check. Consumers read each field from the owning side.

    // --- Build consolidated EagleConfig and validate topology ---
    if (drafting.has_value())
    {
        // Positivity: each drafting field must be >= 1. Rejecting zero/negative
        // up front lets downstream arithmetic (the topK * step multiply below)
        // proceed under a clean invariant and produces a clearer error than a
        // far-away shape-mismatch at bind time.
        auto const requirePositiveField = [](int32_t value, char const* name) {
            ELLM_CHECK(
                value > 0, std::string("drafting.") + name + "=" + std::to_string(value) + " must be positive (>= 1).");
        };
        requirePositiveField(drafting->draftingTopK, "draftingTopK");
        requirePositiveField(drafting->draftingStep, "draftingStep");
        requirePositiveField(drafting->verifyTreeSize, "verifyTreeSize");

        EagleConfig eagle;
        // baseOutputHiddenDim comes from the draft config's `base_model_hidden_size`
        // (= base.hiddenSize * 3 for EAGLE-3, = base.hiddenSize for MTP). Don't
        // derive from base.hiddenSize directly — that's correct only for EAGLE-3.
        eagle.baseOutputHiddenDim = cfg.draft->baseModelHiddenSize;
        eagle.draftHiddenSize = cfg.draft->hiddenSize;
        eagle.maxVerifyTreeSize = cfg.base.maxVerifyTreeSize;
        eagle.maxDraftTreeSize = cfg.draft->maxDraftTreeSize;
        eagle.draftingTopK = drafting->draftingTopK;
        eagle.draftingStep = drafting->draftingStep;
        eagle.verifyTreeSize = drafting->verifyTreeSize;

        // In practice both `draftingStep` and `draftingTopK` are <= ~64 (bounded
        // downstream by `maxDraftTreeSize`, which is tens, not millions), so
        // int32 multiplication is overflow-safe; keeping it in int32 avoids
        // widening noise.
        int32_t const requiredDraftInputSize = eagle.draftingStep * eagle.draftingTopK;

        ELLM_CHECK(requiredDraftInputSize <= eagle.maxDraftTreeSize,
            "drafting.draftingStep=" + std::to_string(eagle.draftingStep) + " * drafting.draftingTopK="
                + std::to_string(eagle.draftingTopK) + " = " + std::to_string(requiredDraftInputSize)
                + " exceeds draft.maxDraftTreeSize=" + std::to_string(eagle.maxDraftTreeSize)
                + ". Drafting configuration exceeds engine draft tree size capability.");
        ELLM_CHECK(eagle.verifyTreeSize <= eagle.maxVerifyTreeSize,
            "drafting.verifyTreeSize=" + std::to_string(eagle.verifyTreeSize)
                + " exceeds base.maxVerifyTreeSize=" + std::to_string(eagle.maxVerifyTreeSize)
                + ". Verify tree size exceeds base engine maximum verify tree size.");

        cfg.eagle = eagle;
    }

    return cfg;
}

} // namespace rt
} // namespace trt_edgellm
