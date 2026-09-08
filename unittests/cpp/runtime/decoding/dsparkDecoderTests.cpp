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

// DSpark speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// DSpark speculative decoding.
//
// The one decoder here that genuinely needs a file: loadHeadSidecars() is
// unconditional and a missing tensor is a hard failure, so the test writes a
// real safetensors through the same production API the rest of the system uses.
// --------------------------------------------------------------------------

constexpr int32_t kDSparkBlockSize{4};

//! Rank of the low-rank Markov head. Only the shape matters here; the weights are zeroed.
constexpr int32_t kDSparkMarkovRank{8};

void addDSparkConfig(Json& config)
{
    Json dspark;
    dspark["target_layer_ids"] = Json::array({0, 1});
    dspark["block_size"] = kDSparkBlockSize;
    // The default is a real Qwen id, outside this deployment's 128-token vocabulary.
    dspark["mask_token_id"] = 3;
    dspark["markov_head_type"] = "vanilla";
    dspark["markov_rank"] = kDSparkMarkovRank;
    // Leaving the confidence head off keeps the sidecar to the two required matrices, and is also what lets the
    // scheduler stay off: the decoder refuses a scheduler without confidence tensors.
    dspark["enable_confidence_head"] = false;
    config["dspark_config"] = dspark;
}

Json makeDSparkBaseConfig()
{
    Json config = makeSpecBaseConfig("dspark", kDSparkBlockSize);
    addDSparkConfig(config);
    return config;
}

Json makeDSparkDraftConfig()
{
    Json config = makeSpecDraftConfig("dspark", kDSparkBlockSize);
    addDSparkConfig(config);
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

//! Write the heads sidecar DSpark loads unconditionally.
//!
//! Built with `saveSafetensors`, the same call the rest of the system writes weights with, so the test knows only
//! the tensor names and shapes the decoder validates -- which is the actual contract -- and not the file format.
void writeDSparkHeadsSidecar(std::filesystem::path const& dir, int32_t draftVocabSize, cudaStream_t stream)
{
    std::vector<rt::Tensor> heads;
    for (char const* name : {"markov_w1", "markov_w2"})
    {
        rt::Tensor matrix({draftVocabSize, kDSparkMarkovRank}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), stream));
        heads.push_back(std::move(matrix));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!rt::safetensors::saveSafetensors(dir / binding_names::kDSparkHeadsFileName, heads, stream))
    {
        throw std::runtime_error("failed to write the DSpark heads sidecar");
    }
}

class DSparkAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const dir = writeSpecModelDir("dsparkAssemblyTests", makeDSparkBaseConfig(), makeDSparkDraftConfig());
        writeDSparkHeadsSidecar(dir, static_cast<int32_t>(kVocabSize), mStream);
        return dir;
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig config{};
        config.draftingTopK = 1;
        config.draftingStep = 1;
        config.verifySize = kDSparkBlockSize;
        return config;
    }
};

TEST_F(DSparkAssemblyTest, AssemblesTwoEnginesAroundTheHeadsSidecar)
{
    expectAssembledFromArtifactsAlone("dspark");
    // The one file this deployment does need. DSpark loads it unconditionally, unlike EAGLE's d2t and DFlash's
    // draft vocab map, both of which are conditional and absent in their tests.
    EXPECT_TRUE(std::filesystem::exists(mModelDir / binding_names::kDSparkHeadsFileName));
}

// Given a heads sidecar whose Markov matrices disagree with the rank the config declares
// When the runtime is assembled
// Then assembly fails naming the tensor and the mismatched dimension
TEST_F(DSparkAssemblyTest, RejectsAHeadsSidecarWhoseMarkovMatricesAreTheWrongShape)
{
    // The load path validates rank, dtype and both dimensions of each matrix. Writing a rank that disagrees with
    // the config proves those checks run rather than being dead weight, and it is the failure a mismatched export
    // would actually produce.
    std::vector<rt::Tensor> heads;
    for (char const* name : {"markov_w1", "markov_w2"})
    {
        rt::Tensor matrix({static_cast<int64_t>(kVocabSize), kDSparkMarkovRank + 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), mStream));
        heads.push_back(std::move(matrix));
    }
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    ASSERT_TRUE(rt::safetensors::saveSafetensors(mModelDir / binding_names::kDSparkHeadsFileName, heads, mStream));

    auto artifacts = makeArtifacts();

    // Asserting the message, not merely that something threw: the assembly path has many ways to fail, and this
    // test is only meaningful if it fails on the column count.
    std::string message;
    try
    {
        auto runtime = makeRuntime(std::move(artifacts));
        FAIL() << "assembly accepted a heads sidecar whose markov matrices are the wrong shape";
    }
    catch (std::exception const& e)
    {
        message = e.what();
    }
    EXPECT_THAT(message, ::testing::HasSubstr("markov_w1"));
    EXPECT_THAT(message, ::testing::HasSubstr("column count mismatch"));
}

TEST_F(DSparkAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // The third decoder through the same eviction. DSpark carries per-slot Markov-head state on top of its draft
    // cache, so it has the most to keep in step.
    expectSurvivorKeepsDecodingAfterCompaction(kDSparkBlockSize);
}

} // namespace
