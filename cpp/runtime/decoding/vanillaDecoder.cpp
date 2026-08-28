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

#include "runtime/decoding/vanillaDecoder.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"

#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kDecodeProfile{1};

} // namespace

VanillaDecoder::VanillaDecoder(DecodingRuntimeContext& runtime)
    : mRuntime(runtime)
{
}

bool VanillaDecoder::decodeStep(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_GENERATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_vanilla_decoding,
        ("VANILLA_DECODING[R" + std::to_string(context.generationRound) + "," + std::to_string(context.activeBatchSize)
            + "]")
            .c_str(),
        nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(mRuntime.sampling.hostPackedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mRuntime.sampling.hostPackedTokenIds.dataPointer<int32_t>();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        hostPackedTokenIdsData[i] = context.tokenIds[i].back();
    }

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(
        cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mRuntime.sampling.hostPackedTokenIds.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, 1, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }

    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");

    mRuntime.preprocess.stepPreparer.prepare(
        InferencePhase::kDecode, activeBatchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, context.stream);
    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    auto const decodeDims = mRuntime.deployment.base.decodeDims(activeBatchSize);
    bool decodingStatus
        = mRuntime.base.executor.prepare(kDecodeProfile, decodeDims, mRuntime.base.tensorMap, context.stream);
    if (decodingStatus)
    {
        decodingStatus = mRuntime.base.executor.execute(context.stream);
    }
    if (decodingStatus)
    {
        mRuntime.base.cacheManager.commitSequenceLength(/*increment=*/1, context.stream);
    }
    if (!decodingStatus)
    {
        LOG_ERROR("Failed to execute vanilla decoding step for base model.");
        return false;
    }

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBias(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, context.stream);
    }
    // GCOVR_EXCL_STOP

    if (context.hasGuidedDecoding)
    {
        // One logits row per active slot. Sits after the forward was enqueued, so the host-side
        // mask fill overlaps it; the stream keeps the ordering.
        applyGuidedDecodingMask(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits, activeBatchSize,
            /*rowsPerSlot=*/1, context.stream);
    }

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        constexpr uint64_t kSAMPLING_SEED = 42U;
        constexpr uint64_t kSAMPLING_OFFSET = 0U;
        constexpr uint64_t kRANDOM_LANE = 0U;
        SamplingParams params(activeBatchSize, mRuntime.deployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        rt::Tensor const* rowUniforms{nullptr};
        if (context.useRequestStableSampling)
        {
            check::check(mRuntime.sampling.hostUniforms.reshape({activeBatchSize}), "Tensor reshape failed");
            check::check(mRuntime.sampling.uniforms.reshape({activeBatchSize}), "Tensor reshape failed");
            for (int32_t batch = 0; batch < activeBatchSize; ++batch)
            {
                uint64_t const position = requestStableNextAbsolutePosition(
                    context.rawBatchedInputIds[batch].size(), context.currentGenerateLengths[batch]);
                mRuntime.sampling.hostUniforms.dataPointer<float>()[batch] = requestStableUniform(
                    context.samplingSeeds[batch], position, SpecRandomPurpose::kTarget, kRANDOM_LANE);
            }
            CUDA_CHECK(
                cudaMemcpyAsync(mRuntime.sampling.uniforms.rawPointer(), mRuntime.sampling.hostUniforms.rawPointer(),
                    activeBatchSize * sizeof(float), cudaMemcpyHostToDevice, context.stream));
            rowUniforms = &mRuntime.sampling.uniforms;
        }
        topKtopPSamplingFromLogits(mRuntime.base.pipelineIO.outputLogits, mRuntime.sampling.indices, params,
            mRuntime.sampling.workspace, context.stream, kSAMPLING_SEED, kSAMPLING_OFFSET, rowUniforms);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::nullopt, mRuntime.sampling.indices, kSAMPLING_TOP_K,
            mRuntime.sampling.workspace, context.stream);
    }

    // Capture the indices while still in the output vocabulary, the space the matchers work
    // in. Rides the round's single sync below.
    if (context.hasGuidedDecoding)
    {
        check::check(mRuntime.sampling.hostOutputSpaceIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(),
            mRuntime.sampling.indices.rawPointer(), activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost,
            context.stream));
    }

    if (mRuntime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mRuntime.sampling.indices, mRuntime.sampling.baseVocabMappingTable, context.stream);
    }

    if (mRuntime.tokenBroadcast
        && !mRuntime.tokenBroadcast(mRuntime.sampling.indices.rawPointer(), activeBatchSize, context.stream))
    {
        LOG_ERROR("Failed to broadcast vanilla decode sampled tokens for parallel rank %d.", mRuntime.parallelRank);
        return false;
    }

    // Enqueue logprobs extraction + D2H before the round's single synchronization so the
    // copies ride the same sync as the sampled-token D2H below.
    if (context.numLogprobs > 0)
    {
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.base.pipelineIO.outputLogits, activeBatchSize, mRuntime, context.numLogprobs, context.stream);
    }

    check::check(mRuntime.sampling.hostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mRuntime.sampling.hostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mRuntime.sampling.indices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Few-layer-validation debug: dump this decode round. The KV cache is committed (line above) so
    // tokenIds[i].size() == the committed cache length for this round.
    if (context.layerDebugger != nullptr)
    {
        std::vector<int32_t> validLengths(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        context.layerDebugger->dumpRound(mRuntime.base.cacheManager, *mRuntime.base.sharedResources.kvPageTables[0],
            mRuntime.base.pipelineIO.outputLogits, validLengths, context.batchIndexMapping, hostSelectedTokenIdsData,
            activeBatchSize, context.stream);

        // Teacher-forcing — feed the golden's tokens instead of our own (no-op unless
        // EDGELLM_FORCE_TOKENS_FILE is set). After the dump, so the dump keeps our own sampled token.
        context.layerDebugger->applyForcedTokens(
            context.currentGenerateLengths, context.batchIndexMapping, hostSelectedTokenIdsData, activeBatchSize);
    }

    decoder_utils::appendSampledTokens(context, hostSelectedTokenIdsData, activeBatchSize);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecoding(mRuntime.guidedDecoder, context,
            mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(), activeBatchSize);
    }

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectLogprobsFromHost(mRuntime, context, activeBatchSize, context.numLogprobs);
    }

    return true;
}

bool VanillaDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool baseVanillaDecodingCaptureStatus{true};
    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, 1, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, mRuntime.deployment.base.outputVocabSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");
        check::check(mRuntime.preprocess.idsInput.reshape({batchSize, 1}), "Tensor reshape failed");
        if (mRuntime.preprocess.gemma4Ple)
        {
            mRuntime.preprocess.gemma4Ple->reshapeOutputs(batchSize, 1);
        }

        mRuntime.preprocess.stepPreparer.prepare(
            InferencePhase::kDecode, batchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, stream);
        if (mRuntime.preprocess.deepstack)
        {
            mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
        }

        auto const decodeDims = mRuntime.deployment.base.decodeDims(batchSize);
        baseVanillaDecodingCaptureStatus &= mRuntime.base.captureGraph(decodeDims, stream);
    }
    return baseVanillaDecodingCaptureStatus;
}

} // namespace rt
} // namespace trt_edgellm
