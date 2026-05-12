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

#include "testUtils.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Base config JSON with optional EAGLE fields. If `eagleMaxVerifyTreeSize`
//! is > 0, the config enables EAGLE and writes `max_verify_tree_size`.
//!
//! Note: `max_draft_tree_size` is draft-only and is not written on the base
//! side (the builder only emits it when `eagleDraft` is set). The
//! `eagleMaxDraftTreeSize` parameter is accepted for parallelism with
//! `makeDraftConfig` in call sites but intentionally ignored here.
Json makeBaseConfig(int32_t eagleMaxVerifyTreeSize = 0, int32_t /*eagleMaxDraftTreeSize*/ = 0, int32_t maxBatchSize = 2)
{
    Json config;
    config["num_hidden_layers"] = 12;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 64;
    config["hidden_size"] = 768;
    config["vocab_size"] = 32000;
    config["kv_cache_dtype"] = "fp16";

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_lora_rank"] = 0;
    if (eagleMaxVerifyTreeSize > 0)
    {
        bc["eagle_base"] = true;
        bc["max_verify_tree_size"] = eagleMaxVerifyTreeSize;
    }
    else
    {
        bc["eagle_base"] = false;
    }
    config["builder_config"] = bc;
    return config;
}

//! Draft config JSON. Mirrors `parseDraftEngineConfig`'s expected schema.
//!
//! Note: `max_verify_tree_size` is base-only and is not written on the draft
//! side (the builder only emits it when `eagleBase` is set). The
//! `maxVerifyTreeSize` parameter is accepted for parallelism with
//! `makeBaseConfig` but intentionally ignored here.
Json makeDraftConfig(int32_t /*maxVerifyTreeSize*/, int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config;
    config["num_hidden_layers"] = 1;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 64;
    config["hidden_size"] = 768;
    config["draft_vocab_size"] = 32000;
    config["base_model_hidden_size"] = 768 * 3;
    config["kv_cache_dtype"] = "fp16";

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_draft_tree_size"] = maxDraftTreeSize;
    config["builder_config"] = bc;
    return config;
}

//! Write a JSON object to a unique temp file and return its path.
std::filesystem::path writeJsonToTempFile(Json const& json, std::string const& suffix)
{
    auto tmpPath = std::filesystem::temp_directory_path() / ("deploymentConfigTest_" + suffix + ".json");
    std::ofstream ofs(tmpPath);
    ofs << json.dump(2);
    ofs.close();
    return tmpPath;
}

} // namespace

class DeploymentConfigTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        std::filesystem::remove(std::filesystem::temp_directory_path() / "deploymentConfigTest_base.json");
        std::filesystem::remove(std::filesystem::temp_directory_path() / "deploymentConfigTest_draft.json");
    }
};

TEST_F(DeploymentConfigTest, VanillaBundle)
{
    // Base only, no draft, no drafting → succeeds, draft/drafting absent.
    Json const baseJson = makeBaseConfig();
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);

    EXPECT_EQ(bundle.base.hiddenSize, baseJson["hidden_size"].get<int32_t>());
    EXPECT_FALSE(bundle.base.enableEagleSpecDecode);
    EXPECT_FALSE(bundle.draft.has_value());
    EXPECT_FALSE(bundle.eagle.has_value());
}

TEST_F(DeploymentConfigTest, EagleBundle)
{
    // Base + draft + drafting with valid values → succeeds, all fields populated.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;   // draftingStep * draftingTopK = 16 <= 16
    drafting.verifyTreeSize = 8; // 8 <= 16

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<EagleDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.enableEagleSpecDecode);
    ASSERT_TRUE(bundle.draft.has_value());
    // Draft engines set enableEagleSpecDecode=false (they ARE the draft, not the base).
    // Presence of a draft engine is indicated by maxDraftTreeSize > 0.
    EXPECT_FALSE(bundle.draft->enableEagleSpecDecode);
    EXPECT_GT(bundle.draft->maxDraftTreeSize, 0);
    ASSERT_TRUE(bundle.eagle.has_value());

    EXPECT_EQ(bundle.base.maxVerifyTreeSize, 16);
    // `maxDraftTreeSize` is only meaningful on the draft side (see makeBaseConfig).
    EXPECT_EQ(bundle.base.maxDraftTreeSize, 0);
    // `maxVerifyTreeSize` is only meaningful on the base side (see makeDraftConfig).
    EXPECT_EQ(bundle.draft->maxVerifyTreeSize, 0);
    EXPECT_EQ(bundle.draft->maxDraftTreeSize, 16);
    EXPECT_EQ(bundle.eagle->verifyTreeSize, 8);
    EXPECT_EQ(bundle.eagle->draftingStep, 4);
    EXPECT_EQ(bundle.eagle->draftingTopK, 4);
}

TEST_F(DeploymentConfigTest, DraftingWithoutDraftThrows)
{
    // Drafting set but draft not set → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTreeSize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::nullopt, std::optional<EagleDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsVerifyCapacityThrows)
{
    // User's verifyTreeSize > base.maxVerifyTreeSize → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 2;    // 2 * 2 = 4 <= 16 (OK)
    drafting.verifyTreeSize = 16; // 16 > 8 (violation)

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<EagleDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsDraftCapacityThrows)
{
    // User's draftingStep * draftingTopK > draft->maxDraftTreeSize → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4; // 4 * 4 = 16 > 8 (violation)
    drafting.verifyTreeSize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<EagleDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, ConsistentBundleValidatesOk)
{
    // All fields match → succeeds.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 3;
    drafting.draftingStep = 8;    // 3 * 8 = 24 <= 24
    drafting.verifyTreeSize = 32; // 32 <= 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<EagleDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.enableEagleSpecDecode);
    EXPECT_TRUE(bundle.draft.has_value());
    ASSERT_TRUE(bundle.eagle.has_value());
    EXPECT_EQ(bundle.base.maxVerifyTreeSize, 32);
    EXPECT_EQ(bundle.draft->maxDraftTreeSize, 24);
}

// ===========================================================================
// maxRuntimeBatchSize()
// ===========================================================================

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeVanilla)
{
    // Base only: returns base.maxSupportedBatchSize.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/0, /*maxDraft=*/0, /*maxBatch=*/4);
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 4);
}

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeBaseAndDraftAgree)
{
    // Base and draft both set the same batch → returns the common value.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/3);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/3);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    DeploymentConfig bundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 3);
}

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeMismatchReturnsMin)
{
    // Base and draft disagree on batch → fall back to the smaller of the two.
    // The runtime cannot drive either engine beyond its engine-declared capacity,
    // so the common ceiling (min) is the safe choice; a warning is logged.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/2);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/8);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    DeploymentConfig bundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 2);
}

// ===========================================================================
// effectiveMaxDraftTreeSize()
// ===========================================================================

TEST_F(DeploymentConfigTest, EffectiveMaxDraftTreeSizeEagle)
{
    // Both engine maxDraftTreeSize (24) and user verifyTreeSize (32) contribute.
    // Expected: max(24, 32) = 32.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 3;
    drafting.draftingStep = 8;    // 24
    drafting.verifyTreeSize = 32; // 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<EagleDraftingConfig>{drafting});
    EXPECT_EQ(bundle.effectiveMaxDraftTreeSize(), 32);
}

TEST_F(DeploymentConfigTest, EffectiveMaxDraftTreeSizeEngineCapacityWins)
{
    // Engine maxDraftTreeSize (16) is larger than verifyTreeSize (8).
    // Expected: max(16, 8) = 16.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EagleDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTreeSize = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<EagleDraftingConfig>{drafting});
    EXPECT_EQ(bundle.effectiveMaxDraftTreeSize(), 16);
}

TEST_F(DeploymentConfigTest, EffectiveMaxDraftTreeSizeNoDraftingThrows)
{
    // Vanilla bundle: drafting not set → throws.
    Json const baseJson = makeBaseConfig();
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);
    EXPECT_THROW(bundle.effectiveMaxDraftTreeSize(), std::runtime_error);
}
