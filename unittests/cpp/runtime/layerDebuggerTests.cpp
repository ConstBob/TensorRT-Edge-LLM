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

#include "runtime/debug/layerDebugger.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

//! Teacher forcing is addressed by *original* request row, not by active slot, because the
//! runtime compacts its per-slot vectors when a sequence finishes. These tests drive the
//! host-side entry points directly: they need neither a GPU nor an engine.
class LayerDebuggerForcedRows : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = std::filesystem::temp_directory_path() / "edgellm_layer_debugger_test";
        std::filesystem::create_directories(mDir);
        std::filesystem::path const tokensPath = mDir / "forced_tokens.txt";
        // Row r generates tokens 10*r + step, so a row mix-up is unmistakable.
        std::ofstream(tokensPath) << "0 1 2\n10 11 12\n20 21 22\n";

        setenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS", "1", 1);
        setenv("EDGELLM_DUMP_LOGITS_KVCACHE_DIR", mDir.c_str(), 1);
        setenv("EDGELLM_FORCE_TOKENS_FILE", tokensPath.c_str(), 1);
    }

    void TearDown() override
    {
        unsetenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS");
        unsetenv("EDGELLM_DUMP_LOGITS_KVCACHE_DIR");
        unsetenv("EDGELLM_FORCE_TOKENS_FILE");
        std::filesystem::remove_all(mDir);
    }

    std::filesystem::path mDir;
};

// One request's whole lifecycle, because a debugger claims its block of force-token rows on
// first use from a process-wide counter: a second instance would start at a different base.
//
// Slot 0 finishes mid-request, the runtime compacts the batch, and both the vanilla and the
// speculative entry points must keep addressing each survivor by the row it came from.
TEST_F(LayerDebuggerForcedRows, SurvivesBatchCompaction)
{
    auto debugger = rt::LayerDebugger::fromEnv();
    ASSERT_NE(debugger, nullptr);

    // Prefill: three active slots, still one-to-one with the request rows.
    std::vector<int32_t> const genLengths3{0, 0, 0};
    std::vector<int32_t> const identity{0, 1, 2};
    std::vector<int32_t> tokens{-1, -1, -1};
    debugger->applyForcedTokens(genLengths3, identity, tokens.data(), 3);
    EXPECT_EQ(tokens[0], 0);
    EXPECT_EQ(tokens[1], 10);
    EXPECT_EQ(tokens[2], 20);

    // Slot 0 finishes. Rows 1 and 2 move down to slots 0 and 1, and batchIndexMapping is what
    // records where they came from.
    std::vector<int32_t> const genLengths2{1, 1};
    std::vector<int32_t> const compacted{1, 2};
    std::vector<int32_t> survivors{-1, -1};
    debugger->applyForcedTokens(genLengths2, compacted, survivors.data(), 2);
    EXPECT_EQ(survivors[0], 11) << "survivor was fed the evicted sequence's golden tokens";
    EXPECT_EQ(survivors[1], 21);

    // The speculative path addresses rows the same way, and trims the acceptance at the first
    // token that disagrees rather than overwriting one in place. Slot 1 still carries row 2,
    // whose remaining golden tokens are 21 then 22; the second proposal disagrees.
    constexpr int32_t kMaxAcceptDepth = 3;
    std::vector<int32_t> const genLengthsSpec{1, 1};
    std::vector<int32_t> acceptedTokenIds{10, 11, 0, 21, 99, 0};
    std::vector<int32_t> acceptLengths{1, 3};
    std::vector<int32_t> ownTokens;

    bool const trimmed = debugger->applyForcedAcceptance(
        genLengthsSpec, compacted, acceptLengths.data(), acceptedTokenIds.data(), ownTokens, 2, kMaxAcceptDepth);

    EXPECT_TRUE(trimmed);
    EXPECT_EQ(acceptLengths[1], 2) << "acceptance must stop at the divergence so the replaced "
                                      "token's cache entry is left uncommitted";
    EXPECT_EQ(acceptedTokenIds[3], 21) << "the matching prefix is untouched";
    EXPECT_EQ(acceptedTokenIds[4], 22) << "the divergent token is replaced by the golden's";
    ASSERT_EQ(ownTokens.size(), 2u);
    EXPECT_EQ(ownTokens[1], 99) << "the engine's own choice is kept as the divergence signal";
}

} // namespace
