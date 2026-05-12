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

#pragma once

#include "runtime/config/llmEngineConfig.h"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief User-supplied drafting tree topology for Eagle speculative decoding.
 *
 * Caller-side input to `createDeploymentConfig`. The factory consumes this
 * together with the parsed engine configs to produce the consolidated
 * `EagleConfig` stored on `DeploymentConfig::eagle`.
 */
struct EagleDraftingConfig
{
    int32_t draftingTopK{0};   //!< Tokens to select from one predecessor for next draft tree level
    int32_t draftingStep{0};   //!< Number of drafting steps with draft model
    int32_t verifyTreeSize{0}; //!< Number of tokens for base model verification
};

/*!
 * @brief Consolidated EAGLE deployment configuration.
 *
 * Holds every EAGLE-specific value the runtime needs in one place, sourced
 * from three inputs:
 *   - the base engine's parsed config (`baseOutputHiddenDim`, `maxVerifyTreeSize`),
 *   - the draft engine's parsed config (`draftHiddenSize`, `maxDraftTreeSize`),
 *   - the caller-supplied `EagleDraftingConfig` tree topology
 *     (`draftingTopK`, `draftingStep`, `verifyTreeSize`).
 *
 * `createDeploymentConfig` populates this struct after both engine configs
 * are parsed and validates the topology against the engine capacities.
 */
struct EagleConfig
{
    // --- Engine-derived capacities ---
    //! Base engine output hidden dim as seen by the draft (= the third dim of
    //! the draft's `hidden_states_input` binding). Sourced from the draft
    //! config's `base_model_hidden_size`: `base.hiddenSize * 3` for EAGLE-3,
    //! `base.hiddenSize` for MTP. NOT a `base.hiddenSize * 3` derivation.
    int32_t baseOutputHiddenDim{};
    int32_t draftHiddenSize{};   //!< Draft engine hidden dim (= draft.hiddenSize)
    int32_t maxVerifyTreeSize{}; //!< Max seq_len the base engine accepts for tree verification
    int32_t maxDraftTreeSize{};  //!< Max seq_len the draft engine accepts for proposal / draft generation

    // --- User-supplied tree topology ---
    int32_t draftingTopK{};   //!< Tokens to select from one predecessor for next draft tree level
    int32_t draftingStep{};   //!< Number of drafting steps with draft model
    int32_t verifyTreeSize{}; //!< Number of tokens for base model verification
};

//! Complete EAGLE deployment configuration: the base engine's config, the
//! draft engine's config, and the consolidated EAGLE settings.
//!
//! For non-EAGLE deployments `draft` and `eagle` are both absent.
//! When `eagle` is present `draft` must also be present — the factory
//! enforces this invariant.
struct DeploymentConfig
{
    LLMEngineConfig base;                 //!< Parsed base engine configuration
    std::optional<LLMEngineConfig> draft; //!< Parsed draft engine configuration (EAGLE only)
    std::optional<EagleConfig> eagle;     //!< Consolidated EAGLE settings (EAGLE only)

    //! Maximum runtime batch size across the bundle. Returns the base engine's
    //! `maxSupportedBatchSize` when there is no draft; otherwise returns the
    //! `min` of base and draft. Logs a warning if base and draft disagree.
    int32_t maxRuntimeBatchSize() const;

    //! Effective maximum EAGLE tree size across drafting and verification.
    //! Returns `max(eagle->maxDraftTreeSize, eagle->verifyTreeSize)`.
    //! EAGLE-only — throws `std::runtime_error` if `eagle` is not set.
    int32_t effectiveMaxDraftTreeSize() const;
};

//! Create a `DeploymentConfig` from engine config paths and optional user-side drafting.
//!
//! - Parses `baseConfigPath` via `parseEngineConfig`.
//! - If `draftConfigPath` is set, parses it via `parseDraftEngineConfig`.
//! - If `drafting` is set, `draftConfigPath` must also be set (else throws).
//! - If `drafting` is set, builds `eagle` by combining the engines' EAGLE
//!   capacities with the user-supplied topology, and validates the topology
//!   against the engines' capacities:
//!     - `eagle->verifyTreeSize <= eagle->maxVerifyTreeSize`
//!     - `eagle->draftingStep * eagle->draftingTopK <= eagle->maxDraftTreeSize`
//!   Throws with named-fields message on violation.
//!
//! @throws std::runtime_error on any validation failure or parse failure.
DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath, std::optional<EagleDraftingConfig> const& drafting);

} // namespace rt
} // namespace trt_edgellm
