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

#include "common/bindingNames.h"
#include "common/pagedKvTypes.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

LLMEngineConfig makeBindingConfig(bool useSwa)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.numAttentionLayers = 3;
    cfg.numDecoderLayers = 3;
    cfg.numKVHeads = 1;
    cfg.headDim = 8;
    cfg.rotaryDim = 8;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8192;
    cfg.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity));
    cfg.numSwaPages = 64;
    cfg.kvCacheDtype = nvinfer1::DataType::kHALF;
    cfg.ropeConfig.type = RopeType::kNoRope;
    cfg.layerTypes.assign(3, HybridCacheManager::LayerType::kAttention);

    int32_t const window = useSwa ? 129 : 0;
    cfg.kvLayerConfigs = {
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8},
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/window},
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/window},
    };
    cfg.kvSharingDonors = {-1, -1, 1};
    return cfg;
}

std::unique_ptr<SharedResources> makeResources(LLMEngineConfig const& cfg)
{
    std::unordered_map<std::string, std::string> const noLoraWeights;
    return SharedResources::createForLLM(cfg, noLoraWeights, nullptr);
}

} // namespace

TEST(PipelineIOSwaBindingTest, SharedSwaConsumerUsesDonorPoolAndIndependentTable)
{
    LLMEngineConfig const cfg = makeBindingConfig(/*useSwa=*/true);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(1, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/false)), &kv.getCombinedKVCache(1));

    ASSERT_TRUE(map.contains(binding_names::kKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVCacheMode));
    ASSERT_NE(resources->getSwaKVPageTable(0), nullptr);
    EXPECT_EQ(map.get(binding_names::kKVPageTable), &resources->kvPageTables[0]->kernelView());
    EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &resources->getSwaKVPageTable(0)->kernelView());
    EXPECT_EQ(map.get(binding_names::kSwaKVCacheMode), &resources->swaKVCacheMode);
    EXPECT_EQ(cfg.prefillDims(/*batch=*/1, /*seqLen=*/1, /*kvCacheAllEmpty=*/true).swaKVCacheModeLen, 1);

    int32_t const logicalPages = computeMaxPagesPerSeq(cfg.maxKVCacheCapacity);
    EXPECT_EQ(resources->kvPageTables[0]->maxPagesPerSeq(), logicalPages);
    EXPECT_EQ(resources->getSwaKVPageTable(0)->maxPagesPerSeq(), logicalPages);
    EXPECT_NE(kv.numPages(), kv.numPages(1));

    // The two V namespaces are derived from their own physical page counts.
    int32_t const* fullRow = resources->kvPageTables[0]->hostRow(0);
    EXPECT_EQ(fullRow[logicalPages], kv.numPages());
    resources->getSwaKVPageTable(0)->setEntry(/*slot=*/0, /*logicalPage=*/0, /*kPageId=*/0);
    int32_t const* swaRow = resources->getSwaKVPageTable(0)->hostRow(0);
    EXPECT_EQ(swaRow[logicalPages], kv.numPages(1));
}

TEST(PipelineIOSwaBindingTest, DefaultOffKeepsFullTableAndExistingDonorRouting)
{
    LLMEngineConfig const cfg = makeBindingConfig(/*useSwa=*/false);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    ASSERT_EQ(resources->swaKVPageTables.size(), 1u);
    EXPECT_EQ(resources->getSwaKVPageTable(0), nullptr);
    EXPECT_FALSE(map.contains(binding_names::kSwaKVPageTable));
    EXPECT_FALSE(map.contains(binding_names::kSwaKVCacheMode));
    EXPECT_TRUE(map.contains(binding_names::kKVPageTable));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(kv.numPages(0), kv.numPages(1));
}

TEST(PipelineIOSwaBindingTest, FullModeAliasesOrdinaryTableAndUsesFullPhysicalPools)
{
    LLMEngineConfig cfg = makeBindingConfig(/*useSwa=*/true);
    int32_t const capabilityMarker = cfg.kvLayerConfigs[1].kvCacheCapacity;
    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    EXPECT_FALSE(kv.hasReducedKVCache());
    EXPECT_EQ(kv.numPages(0), cfg.kvPoolPages);
    EXPECT_EQ(kv.numPages(1), cfg.kvPoolPages);
    EXPECT_EQ(resources->getSwaKVPageTable(0), nullptr);
    ASSERT_TRUE(map.contains(binding_names::kSwaKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVCacheMode));
    EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &resources->kvPageTables[0]->kernelView());
    EXPECT_EQ(map.get(binding_names::kSwaKVCacheMode), &resources->swaKVCacheMode);
    EXPECT_EQ(cfg.prefillDims(/*batch=*/1, /*seqLen=*/1, /*kvCacheAllEmpty=*/true).swaKVCacheModeLen, 0);
    EXPECT_EQ(cfg.kvLayerConfigs[1].kvCacheCapacity, capabilityMarker);
}
