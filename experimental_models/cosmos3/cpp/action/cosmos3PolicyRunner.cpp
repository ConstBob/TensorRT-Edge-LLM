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

#include "action/cosmos3PolicyRunner.h"

#include "action/cosmos3Kernels.h"
#include "common/cosmos3Bindings.h"
#include "profiling/nvtx_wrapper.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

using namespace trt_edgellm;
using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace cosmos3
{

Cosmos3PolicyRunner::Cosmos3PolicyRunner(std::string const& engineDir, cudaStream_t stream)
{
    LOG_DEBUG("Loading Cosmos3 policy runner from %s", engineDir.c_str());

    std::string const enginePath = engineDir + "/gen.engine";
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "Failed to create execution context");
    ELLM_CHECK(mContext->setOptimizationProfileAsync(0, stream), "Failed to set optimization profile");

    parseModelConfig(engineDir + "/config.json"); // strict: throws on any missing contract key

    Cosmos3Scheduler::Config schedCfg;
    schedCfg.shift = mConfig.flowShift;
    mScheduler = std::make_unique<Cosmos3Scheduler>(schedCfg);

    allocateTensors(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Cosmos3PolicyRunner::~Cosmos3PolicyRunner() noexcept
{
    if (mGenGraphExec != nullptr)
    {
        static_cast<void>(cudaGraphExecDestroy(mGenGraphExec));
    }
    if (mGenGraph != nullptr)
    {
        static_cast<void>(cudaGraphDestroy(mGenGraph));
    }
}

void Cosmos3PolicyRunner::setNumInferenceSteps(int32_t steps)
{
    if (steps <= 0)
    {
        throw std::invalid_argument("Cosmos3PolicyRunner requires a positive denoise step count");
    }
    mConfig.numInferenceSteps = steps;
}

void Cosmos3PolicyRunner::parseModelConfig(std::string const& configPath)
{
    std::ifstream f(configPath);
    ELLM_CHECK(f.is_open(), "Failed to open Cosmos3 GEN config: " + configPath);
    Json const j = Json::parse(f);

    // Every field below is part of the exported component contract; parse strictly (no fallback
    // defaults) so a stale or truncated config fails load instead of silently mis-running.
    auto requireInt = [&](char const* key) {
        ELLM_CHECK(j.contains(key), "Cosmos3 GEN config missing required key: " + std::string(key));
        return j.at(key).get<int32_t>();
    };
    auto requireFloat = [&](char const* key) {
        ELLM_CHECK(j.contains(key), "Cosmos3 GEN config missing required key: " + std::string(key));
        return j.at(key).get<float>();
    };

    mConfig.numHiddenLayers = requireInt("num_hidden_layers");
    mConfig.numKVHeads = requireInt("num_key_value_heads");
    mConfig.headDim = requireInt("head_dim");
    mConfig.ropeTheta = requireFloat("rope_theta");
    mConfig.latentChannel = requireInt("latent_channel");
    mConfig.latentPatchSize = requireInt("latent_patch_size");
    mConfig.actionChunkSize = requireInt("action_chunk_size");
    mConfig.rawActionDim = requireInt("raw_action_dim");
    mConfig.maxActionDim = requireInt("max_action_dim");
    mConfig.numInferenceSteps = requireInt("num_inference_steps");
    mConfig.flowShift = requireFloat("flow_shift");
    mConfig.timestepScale = requireFloat("timestep_scale");
    mConfig.domainId = requireInt("domain_id");
    mConfig.videoLatentFrames = requireInt("video_latent_frames");
    mConfig.fps = requireFloat("fps");
    mConfig.baseFps = requireFloat("base_fps");
    mConfig.temporalCompressionFactor = requireInt("temporal_compression_factor");
    mConfig.temporalModalityMargin = requireInt("temporal_modality_margin");
    mConfig.actionStartFrameOffset = requireInt("action_start_frame_offset");

    ELLM_CHECK(j.contains("rope_scaling") && j.at("rope_scaling").contains("mrope_section"),
        "Cosmos3 GEN config missing required key: rope_scaling.mrope_section");
    auto const section = j.at("rope_scaling").at("mrope_section").get<std::vector<int32_t>>();
    ELLM_CHECK(section.size() == 3, "rope_scaling.mrope_section must have 3 entries");
    mConfig.mropeSectionH = section[1];
    mConfig.mropeSectionW = section[2];
}

void Cosmos3PolicyRunner::allocateTensors(cudaStream_t stream)
{
    // Allocate everything ONCE at the engine-profile MAXIMUM shapes: the latent grid is fixed by the
    // contract (min == opt == max), and the batch axis max is widened by the builder's --max-batch-size.
    // Per-request batches within the profile are metadata-only reshapes over these buffers.
    Dims const maxVideo = mEngine->getProfileShape(binding_names::kVideoLatent, 0, OptProfileSelector::kMAX);
    int32_t const batch = static_cast<int32_t>(maxVideo.d[0]);
    int32_t const channel = static_cast<int32_t>(maxVideo.d[1]);
    int32_t const tDim = static_cast<int32_t>(maxVideo.d[2]);
    int32_t const hDim = static_cast<int32_t>(maxVideo.d[3]);
    int32_t const wDim = static_cast<int32_t>(maxVideo.d[4]);
    mMaxBatch = batch;
    mActiveBatch = batch;
    mVideoShape = {batch, channel, tDim, hDim, wDim};
    mVideoElems = channel * tDim * hDim * wDim;
    mActionElems = mConfig.actionChunkSize * mConfig.maxActionDim;
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const genLen = numVideoTokens + mConfig.actionChunkSize;

    Dims const ropeDims
        = mEngine->getProfileShape(trt_edgellm::binding_names::kRopeCosSin, 0, OptProfileSelector::kOPT);
    mRopeHeadDim = static_cast<int32_t>(ropeDims.d[ropeDims.nbDims - 1]);

    // Packed [video ⧺ action] state and prediction buffers; the engine binds the video tensor at the
    // base pointer and the action tensor at base + videoBytes (see prepareStatic).
    int64_t const totalElems = static_cast<int64_t>(batch) * (mVideoElems + mActionElems);
    mStateDevice = rt::Tensor(rt::Coords{totalElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::state");
    mPredDevice = rt::Tensor(rt::Coords{totalElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::pred");
    mTimestepDevice
        = rt::Tensor(std::vector<int64_t>{batch}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::timestep");
    mTokenNoisyMaskDevice = rt::Tensor(
        rt::Coords{batch, numVideoTokens, 1}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::tokenNoisyMask");
    mActionNoisyMaskDevice = rt::Tensor(rt::Coords{batch, mConfig.actionChunkSize, 1}, rt::DeviceType::kGPU,
        DataType::kFLOAT, "cosmos3::actionNoisyMask");
    mRopeCosSinDevice = rt::Tensor(
        rt::Coords{batch, genLen, mRopeHeadDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::ropeCosSin");
    mPositionsDevice
        = rt::Tensor(rt::Coords{batch, 3, genLen}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::ropePositions");

    mTimestepHost
        = rt::Tensor(std::vector<int64_t>{batch}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::timestepHost");
    mTokenNoisyMaskHost = rt::Tensor(
        rt::Coords{batch, numVideoTokens, 1}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::tokenNoisyMaskHost");
    mActionNoisyMaskHost = rt::Tensor(rt::Coords{batch, mConfig.actionChunkSize, 1}, rt::DeviceType::kCPU,
        DataType::kFLOAT, "cosmos3::actionNoisyMaskHost");

    // attention_pos_id: identity gather indices 0..genLen-1; genLen is fixed for the policy grid, so the
    // buffer is filled once here.
    mPositionIdsDevice
        = rt::Tensor(rt::Coords{batch, genLen}, rt::DeviceType::kGPU, DataType::kINT32, "cosmos3::attnPosId");
    std::vector<int32_t> posIds(static_cast<size_t>(batch) * genLen);
    for (int32_t b = 0; b < batch; ++b)
    {
        std::iota(posIds.begin() + static_cast<size_t>(b) * genLen,
            posIds.begin() + (static_cast<size_t>(b) + 1) * genLen, 0);
    }
    CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), posIds.data(), posIds.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    // Device history buffers for the device-resident UniPC scheduler.
    mScheduler->prepare(totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int64_t Cosmos3PolicyRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool Cosmos3PolicyRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (sharedContextMemory.getMemoryCapacity() < getRequiredContextMemorySize())
    {
        return false;
    }
    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

void Cosmos3PolicyRunner::buildRopeAndPositions(int32_t actionLen, int32_t undLen, cudaStream_t stream)
{
    // Reproduce the reference unified_3d_mRoPE exactly (transformer_cosmos3.py:196-353,1224-1322).
    // The GEN sequence is [video tokens (t-major) ; action tokens]. Video and action positions both start
    // at media_temporal_offset = real_text_len + temporal_modality_margin and are fps-modulated, producing
    // *float* temporal positions. The per-token (T,H,W) positions and the [genLen, headDim] transcendental
    // cos/sin cache are both computed on device (the integer-position core kernel would truncate the float
    // positions).
    int32_t const tDim = static_cast<int32_t>(mVideoShape[2]);
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const hp = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize));
    int32_t const wp = static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize));
    int32_t const numVideoTokens = tDim * hp * wp;
    int32_t const genLen = numVideoTokens + actionLen;

    double const mediaOffset = static_cast<double>(undLen) + static_cast<double>(mConfig.temporalModalityMargin);
    bool const videoFpsMod = (mConfig.fps > 0.0F) && (tDim > 1);
    // Per the reference, the vision call passes base_temporal_compression_factor=None -> uses tcf, so the
    // video temporal stride reduces to base_fps/fps; the action call uses tcf as the *base* tcf with its own
    // tcf=1, giving stride (base_fps/base_tcf)/fps.
    double const videoBaseTps
        = static_cast<double>(mConfig.baseFps) / static_cast<double>(mConfig.temporalCompressionFactor);
    double const videoTps = static_cast<double>(mConfig.fps) / static_cast<double>(mConfig.temporalCompressionFactor);
    double const actionBaseTps
        = static_cast<double>(mConfig.baseFps) / static_cast<double>(mConfig.temporalCompressionFactor);
    double const actionTps = static_cast<double>(mConfig.fps); // action temporal_compression_factor = 1.

    // Fill the [B, 3, genLen] T/H/W position planes directly on device (one thread per token); the
    // planes depend only on the fixed latent grid and are identical across batch.
    int32_t const batch = mActiveBatch;
    kernel::launchBuildMRopePositions(mPositionsDevice, batch, genLen, numVideoTokens, hp, wp,
        mConfig.actionStartFrameOffset, videoFpsMod, mediaOffset, videoTps, videoBaseTps, actionTps, actionBaseTps,
        stream);
    // The core interleaved-mRoPE kernel (float-position overload: the fps-modulated temporal
    // positions are fractional) fills the packed [B, genLen, headDim] (cos | sin) cache on device.
    trt_edgellm::kernel::initializeMRopeCosSin(static_cast<float*>(mRopeCosSinDevice.rawPointer()),
        static_cast<float*>(mPositionsDevice.rawPointer()), mConfig.ropeTheta, mRopeHeadDim, genLen, batch,
        /*interleaved=*/true, mConfig.mropeSectionH, mConfig.mropeSectionW, stream);
}

void Cosmos3PolicyRunner::initializeLatents(rt::Tensor const& condLatent, cudaStream_t stream)
{
    // Standard-normal noise over the whole packed [video ; action] state (Philox, seeded, on
    // device), then the conditioning constraints: frame-0 latent injection and padded-action-dim
    // zeroing.
    kernel::launchNormalNoiseFill(mStateDevice, static_cast<uint64_t>(mNoiseSeed), /*offset=*/0, stream);
    reinjectConditioning(condLatent, stream);
}

void Cosmos3PolicyRunner::reinjectConditioning(rt::Tensor const& condLatent, cudaStream_t stream)
{
    int32_t const batch = mActiveBatch;
    int32_t const channel = static_cast<int32_t>(mVideoShape[1]);
    int32_t const tDim = static_cast<int32_t>(mVideoShape[2]);
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);

    rt::Coords const& condShape = condLatent.getShape();
    ELLM_CHECK(condShape.getNumDims() == 5 && condShape[0] == batch && condShape[1] == channel && condShape[2] >= 1
            && condShape[3] == hDim && condShape[4] == wDim,
        "Cosmos3 cond_latent must be [B,C,T,h,w] matching the policy latent grid");

    // video[b, c, 0, :, :] = cond[b, c, 0, :, :]: one strided device-to-device 2D copy
    // (rows = batch*channel frame-0 planes; dst row pitch spans t frames, src row pitch spans T frames).
    size_t const planeBytes = static_cast<size_t>(hDim) * wDim * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(mStateDevice.rawPointer(), static_cast<size_t>(tDim) * planeBytes,
        condLatent.rawPointer(), static_cast<size_t>(condShape[2]) * planeBytes, planeBytes,
        static_cast<size_t>(batch) * channel, cudaMemcpyDeviceToDevice, stream));

    // action[b, a, rawActionDim:maxActionDim] = 0: one strided 2D memset over the action rows.
    size_t const videoBytes = static_cast<size_t>(batch) * mVideoElems * sizeof(float);
    char* actionBase = static_cast<char*>(mStateDevice.rawPointer()) + videoBytes;
    size_t const rowPitch = static_cast<size_t>(mConfig.maxActionDim) * sizeof(float);
    size_t const tailBytes = static_cast<size_t>(mConfig.maxActionDim - mConfig.rawActionDim) * sizeof(float);
    CUDA_CHECK(cudaMemset2DAsync(actionBase + static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), rowPitch, 0,
        tailBytes, static_cast<size_t>(batch) * mConfig.actionChunkSize, stream));
}

void Cosmos3PolicyRunner::setDynamicInputShapes(int32_t batch, int32_t actionLen, int32_t undLen)
{
    int32_t const channel = static_cast<int32_t>(mVideoShape[1]);
    int32_t const tDim = static_cast<int32_t>(mVideoShape[2]);
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const genLen = numVideoTokens + actionLen;

    bool ok = true;
    ok &= mContext->setInputShape(binding_names::kVideoLatent, Dims{5, {batch, channel, tDim, hDim, wDim}});
    ok &= mContext->setInputShape(binding_names::kActionLatent, Dims{3, {batch, actionLen, mConfig.maxActionDim}});
    ok &= mContext->setInputShape(binding_names::kTimestep, Dims{1, {batch}});
    ok &= mContext->setInputShape(binding_names::kTokenNoisyMask, Dims{3, {batch, numVideoTokens, 1}});
    ok &= mContext->setInputShape(binding_names::kActionNoisyMask, Dims{3, {batch, actionLen, 1}});
    ok &= mContext->setInputShape(trt_edgellm::binding_names::kRopeCosSin, Dims{3, {batch, genLen, mRopeHeadDim}});
    ok &= mContext->setInputShape(trt_edgellm::binding_names::kAttentionPosId, Dims{2, {batch, genLen}});
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        Dims const kvShape{4, {batch, undLen, mConfig.numKVHeads, mConfig.headDim}};
        ok &= mContext->setInputShape(binding_names::formatUndKName(i).c_str(), kvShape);
        ok &= mContext->setInputShape(binding_names::formatUndVName(i).c_str(), kvShape);
    }
    if (!ok)
    {
        throw std::runtime_error("Cosmos3PolicyRunner::setDynamicInputShapes failed");
    }
}

void Cosmos3PolicyRunner::prepareStatic(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
    std::vector<rt::Tensor> const& undValues, cudaStream_t stream)
{
    // One-time per-request setup. Everything here is CONSTANT across the denoising steps: the input
    // shapes, the unified_3d_mRoPE cos/sin + position ids, the noisy masks, and all tensor-address
    // bindings (latent/pred buffers are reused in place; UND K/V are frozen). Hoisting this out of the
    // per-step loop avoids redundant TRT shape re-propagation, a 3551x64 transcendental rope rebuild, and
    // 56 UND-K/V setTensorAddress calls on every step.
    int32_t const batch = mActiveBatch;
    int32_t const tDim = static_cast<int32_t>(mVideoShape[2]);
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const framePatchTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize));

    setDynamicInputShapes(batch, actionLen, undLen);
    buildRopeAndPositions(actionLen, undLen, stream);

    // token_noisy_mask: frame-0 patch tokens are clean (0); all others noisy (1). Constant across steps.
    auto* tmask = static_cast<float*>(mTokenNoisyMaskHost.rawPointer());
    std::fill(tmask, tmask + static_cast<size_t>(batch) * numVideoTokens, 1.0F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t k = 0; k < framePatchTokens; ++k)
        {
            tmask[static_cast<size_t>(b) * numVideoTokens + k] = 0.0F;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mTokenNoisyMaskDevice.rawPointer(), tmask,
        static_cast<size_t>(batch) * numVideoTokens * sizeof(float), cudaMemcpyHostToDevice, stream));
    auto* amask = static_cast<float*>(mActionNoisyMaskHost.rawPointer()); // policy: all action tokens noisy
    std::fill(amask, amask + static_cast<size_t>(batch) * actionLen, 1.0F);
    CUDA_CHECK(cudaMemcpyAsync(mActionNoisyMaskDevice.rawPointer(), amask,
        static_cast<size_t>(batch) * actionLen * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Bind I/O addresses once (buffers are reused in place across steps). The video/action latents and
    // predictions are views into the packed [video ⧺ action] state/pred allocations at base/base+offset.
    size_t const videoBytes = static_cast<size_t>(batch) * mVideoElems * sizeof(float);
    char* stateBase = static_cast<char*>(mStateDevice.rawPointer());
    char* predBase = static_cast<char*>(mPredDevice.rawPointer());
    bool ok = true;
    ok &= mContext->setTensorAddress(binding_names::kVideoLatent, stateBase);
    ok &= mContext->setTensorAddress(binding_names::kActionLatent, stateBase + videoBytes);
    ok &= mContext->setTensorAddress(binding_names::kTimestep, mTimestepDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kTokenNoisyMask, mTokenNoisyMaskDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kActionNoisyMask, mActionNoisyMaskDevice.rawPointer());
    ok &= mContext->setTensorAddress(trt_edgellm::binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());
    ok &= mContext->setTensorAddress(trt_edgellm::binding_names::kAttentionPosId, mPositionIdsDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kVideoPred, predBase);
    ok &= mContext->setTensorAddress(binding_names::kActionPred, predBase + videoBytes);
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndKName(i).c_str(), undKeys[static_cast<size_t>(i)].rawPointer());
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndVName(i).c_str(), undValues[static_cast<size_t>(i)].rawPointer());
    }
    if (!ok)
    {
        throw std::runtime_error("Cosmos3PolicyRunner::prepareStatic failed to bind GEN engine tensors");
    }
}

bool Cosmos3PolicyRunner::runDenoiseStep(int32_t stepIdx, rt::Tensor const& condLatent, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(stepRange, "cosmos3::denoise_step");
    int32_t const batch = mActiveBatch;

    // timestep (raw; the graph applies timestep_scale internally). The only per-step engine input besides
    // the latents (a scalar per batch element).
    auto* tsHost = static_cast<float*>(mTimestepHost.rawPointer());
    std::fill(tsHost, tsHost + batch, mScheduler->timestepAt(stepIdx));
    CUDA_CHECK(cudaMemcpyAsync(mTimestepDevice.rawPointer(), tsHost, static_cast<size_t>(batch) * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    bool const engineOk
        = mGenGraphReady ? (cudaGraphLaunch(mGenGraphExec, stream) == cudaSuccess) : mContext->enqueueV3(stream);
    if (!engineOk)
    {
        LOG_ERROR("Cosmos3PolicyRunner: GEN forward failed at step %d (cudagraph=%d)", stepIdx,
            static_cast<int32_t>(mGenGraphReady));
        return false;
    }

    // Device-resident UniPC update over the packed [video ⧺ action] state, then re-impose the
    // conditioning constraints (frame-0 latent, zero padded action dims) — all on device; the
    // denoising state never visits the host.
    mScheduler->step(mPredDevice, mStateDevice, stepIdx, stream);
    reinjectConditioning(condLatent, stream);
    return true;
}

std::vector<float> Cosmos3PolicyRunner::generate(rt::Tensor const& condLatent, std::vector<rt::Tensor> const& undKeys,
    std::vector<rt::Tensor> const& undValues, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(genRange, "cosmos3::gen_denoise");

    std::vector<float> result;
    if (static_cast<int32_t>(undKeys.size()) != mConfig.numHiddenLayers
        || static_cast<int32_t>(undValues.size()) != mConfig.numHiddenLayers)
    {
        LOG_ERROR("Cosmos3PolicyRunner: expected %d UND K/V layers, got %zu/%zu", mConfig.numHiddenLayers,
            undKeys.size(), undValues.size());
        return result;
    }
    int32_t const undLen = static_cast<int32_t>(undKeys.front().getShape()[1]);

    // Per-request batch, derived from the conditioning latent (any value within the engine profile).
    ELLM_CHECK(condLatent.getShape().getNumDims() == 5, "cond_latent must be [B,C,T,h,w]");
    int32_t const batch = static_cast<int32_t>(condLatent.getShape()[0]);
    ELLM_CHECK(batch >= 1 && batch <= mMaxBatch,
        "Cosmos3 policy request batch " + std::to_string(batch) + " exceeds the engine profile maximum "
            + std::to_string(mMaxBatch) + " (build with a larger --max-batch-size)");
    ELLM_CHECK(static_cast<int32_t>(undKeys.front().getShape()[0]) == batch,
        "UND K/V batch does not match the conditioning latent batch");
    if (mGenGraphReady && batch != mGraphBatch)
    {
        // The captured graph bakes the request shapes; a batch change requires a re-capture.
        static_cast<void>(cudaGraphExecDestroy(mGenGraphExec));
        static_cast<void>(cudaGraphDestroy(mGenGraph));
        mGenGraphExec = nullptr;
        mGenGraph = nullptr;
        mGenGraphReady = false;
    }
    mActiveBatch = batch;

    // Metadata-only reshape of the packed state/pred buffers and the scheduler history to the
    // active batch (allocated at the profile maximum in the constructor).
    int64_t const activeElems = static_cast<int64_t>(batch) * (mVideoElems + mActionElems);
    ELLM_CHECK(mStateDevice.reshape(rt::Coords{activeElems}), "Cosmos3 state reshape failed");
    ELLM_CHECK(mPredDevice.reshape(rt::Coords{activeElems}), "Cosmos3 pred reshape failed");
    mScheduler->prepare(activeElems);

    mScheduler->initialize(mConfig.numInferenceSteps);

    initializeLatents(condLatent, stream);

    // One-time setup: shapes, mRoPE, masks, and tensor bindings are constant across the denoising steps.
    prepareStatic(mConfig.actionChunkSize, undLen, undKeys, undValues, stream);

    // Optionally capture the per-step GEN engine forward into a CUDA graph (once). Shapes and I/O
    // addresses are now fixed; only the timestep + latent buffer contents change per step (updated in
    // place), so a single capture replays for every step and across warm iterations.
    if (mUseCudaGraph && !mGenGraphReady)
    {
        if (!mContext->enqueueV3(stream)) // warmup so TRT internal state is initialized before capture
        {
            LOG_ERROR("Cosmos3PolicyRunner: warmup enqueueV3 failed before CUDA-graph capture");
            return result;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto captured = captureTRTCudaGraph(mContext.get(), stream);
        if (captured.has_value())
        {
            mGenGraph = captured->first;
            mGenGraphExec = captured->second;
            mGenGraphReady = true;
            mGraphBatch = mActiveBatch;
            LOG_INFO("Cosmos3PolicyRunner: captured GEN denoise step into a CUDA graph (batch %d)", mActiveBatch);
        }
        else
        {
            LOG_WARNING("Cosmos3PolicyRunner: CUDA-graph capture failed; using enqueueV3");
            mUseCudaGraph = false;
        }
    }

    int32_t const numSteps = mScheduler->numInferenceSteps();
    for (int32_t step = 0; step < numSteps; ++step)
    {
        if (!runDenoiseStep(step, condLatent, stream))
        {
            return result;
        }
    }

    // Slice the action chunk action_latent[:, :, :rawActionDim] straight out of the packed device state:
    // one strided 2D D2H copy (row = one action step; rawActionDim of maxActionDim columns).
    size_t const videoBytes = static_cast<size_t>(batch) * mVideoElems * sizeof(float);
    result.resize(static_cast<size_t>(batch) * mConfig.actionChunkSize * mConfig.rawActionDim);
    CUDA_CHECK(cudaMemcpy2DAsync(result.data(), static_cast<size_t>(mConfig.rawActionDim) * sizeof(float),
        static_cast<char const*>(mStateDevice.rawPointer()) + videoBytes,
        static_cast<size_t>(mConfig.maxActionDim) * sizeof(float),
        static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), static_cast<size_t>(batch) * mConfig.actionChunkSize,
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace cosmos3
} // namespace trt_edgellm
