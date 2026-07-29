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

#pragma once

#include "action/cosmos3Scheduler.h"
#include "common/tensor.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! \brief Configuration parsed from the Cosmos3 GEN engine's config.json.
//! Every field is required and parsed strictly from the component contract; there are no
//! fallback defaults (a missing key fails engine load instead of silently mis-running).
struct Cosmos3PolicyConfig
{
    int32_t numHiddenLayers{0};
    int32_t numKVHeads{0};
    int32_t headDim{0};
    float ropeTheta{0.0F};
    int32_t mropeSectionH{0};
    int32_t mropeSectionW{0};
    int32_t latentChannel{0};
    int32_t latentPatchSize{0};
    int32_t videoLatentFrames{0}; //!< latent temporal length t for the policy video latent.
    int32_t actionChunkSize{0};
    int32_t rawActionDim{0};
    int32_t maxActionDim{0};
    int32_t numInferenceSteps{0};
    float flowShift{0.0F};
    float timestepScale{0.0F};
    int32_t domainId{0};
    // unified_3d_mrope construction parameters (must match the reference transformer).
    float fps{0.0F};     //!< media/control fps used for temporal-position modulation.
    float baseFps{0.0F}; //!< model base fps.
    int32_t temporalCompressionFactor{0};
    int32_t temporalModalityMargin{0}; //!< gap separating text vs media position spaces.
    int32_t actionStartFrameOffset{0}; //!< action grid start-frame offset.
};

//! \brief Cosmos3 diffusion policy head. Runs the joint video+action flow-matching denoising loop in
//! process on the GPU (one GEN engine `enqueueV3` per step), cross-attending into the frozen UND KV.
//!
//! Mirrors the in-process / shared-context-memory / KV-reuse design of the core runtime,
//! but is a separate class with Cosmos3's joint-latent I/O and the host UniPC scheduler. The Wan VAE
//! decode is intentionally never run (video pixels are out of scope); only the action chunk is returned.
class Cosmos3PolicyRunner
{
public:
    //! \param engineDir Directory containing gen.engine and config.json.
    //! \param stream CUDA stream.
    Cosmos3PolicyRunner(std::string const& engineDir, cudaStream_t stream);
    ~Cosmos3PolicyRunner() noexcept;

    //! \brief Required TensorRT context-memory size (for the shared device-memory pool).
    int64_t getRequiredContextMemorySize() const;

    //! \brief Bind shared context memory (must be >= getRequiredContextMemorySize()).
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! \brief Set the random seed for the initial noise latents.
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseSeed = seed;
    }

    //! \brief Enable CUDA-graph capture/replay of the per-step GEN engine forward (captured lazily on
    //! the first generate() call once shapes and bindings are fixed).
    void setUseCudaGraph(bool enable) noexcept
    {
        mUseCudaGraph = enable;
    }

    //! \brief Override the number of diffusion denoise steps for this process.
    void setNumInferenceSteps(int32_t steps);

    Cosmos3PolicyConfig const& getConfig() const noexcept
    {
        return mConfig;
    }

    //! \brief Run the policy denoising loop and return the action chunk. The request batch B is
    //! derived from condLatent and may be any value in [1, engine-profile max batch]; buffers are
    //! preallocated at the profile maximum, so batch changes are metadata-only reshapes.
    //! \param condLatent Device FLOAT32 conditioning latent, [B, latentChannel, t, h, w] (frame 0 is used).
    //! \param undKeys    Per-layer frozen UND keys, FLOAT16 [B, undLen, numKVHeads, headDim] each.
    //! \param undValues  Per-layer frozen UND values, FLOAT16 [B, undLen, numKVHeads, headDim] each.
    //! \return Flattened action chunk of shape [B, actionChunkSize, rawActionDim] (row-major), empty on error.
    std::vector<float> generate(rt::Tensor const& condLatent, std::vector<rt::Tensor> const& undKeys,
        std::vector<rt::Tensor> const& undValues, cudaStream_t stream);

private:
    void parseModelConfig(std::string const& configPath);
    void allocateTensors(cudaStream_t stream);
    //! \brief Build the packed (cos|sin) unified_3d_mRoPE cache and attention position ids for the GEN
    //! stream. Per-token float (fps-modulated) positions are computed on host (tiny), and the
    //! [genLen, headDim] transcendental cache is computed on device by the interleaved-mRoPE kernel.
    void buildRopeAndPositions(int32_t actionLen, int32_t undLen, cudaStream_t stream);
    //! \brief Initialize the packed device state [video ⧺ action]: Philox device noise, frame-0
    //! conditioning injection (2D D2D copy), and padded-action-dim zeroing (2D memset).
    void initializeLatents(rt::Tensor const& condLatent, cudaStream_t stream);
    //! \brief Set dynamic input shapes for one denoising step.
    void setDynamicInputShapes(int32_t batch, int32_t actionLen, int32_t undLen);
    //! \brief One-time per-request setup of the constant-across-steps engine inputs: shapes, mRoPE cos/sin
    //! + position ids, noisy masks, and all tensor-address bindings. Hoisted out of the per-step loop.
    void prepareStatic(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
        std::vector<rt::Tensor> const& undValues, cudaStream_t stream);
    //! \brief Run one GEN engine forward + the device-resident UniPC update.
    bool runDenoiseStep(int32_t stepIdx, rt::Tensor const& condLatent, cudaStream_t stream);
    //! \brief Re-inject the clean frame-0 conditioning latent and re-zero the padded action dims
    //! (device-side: one 2D D2D copy + one 2D memset).
    void reinjectConditioning(rt::Tensor const& condLatent, cudaStream_t stream);

    int32_t mNoiseSeed{0};
    Cosmos3PolicyConfig mConfig{};
    std::unique_ptr<Cosmos3Scheduler> mScheduler;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};

    //! Optional CUDA-graph capture of the per-step GEN engine enqueue. The denoise steps enqueue the
    //! same engine with fixed shapes + fixed I/O addresses (after prepareStatic), so one capture replays
    //! for every step; only the timestep + latent buffer *contents* change per step (updated in place).
    bool mUseCudaGraph{false};
    bool mGenGraphReady{false};
    cudaGraph_t mGenGraph{nullptr};
    cudaGraphExec_t mGenGraphExec{nullptr};

    int32_t mVideoElems{0};  //!< latentChannel * t * h * w
    int32_t mActionElems{0}; //!< actionChunkSize * maxActionDim
    int32_t mRopeHeadDim{0};
    int32_t mMaxBatch{1};             //!< engine-profile maximum batch (allocation bound).
    int32_t mActiveBatch{1};          //!< per-request batch, derived from the conditioning latent.
    int32_t mGraphBatch{0};           //!< batch the CUDA graph was captured for (a change re-captures).
    std::vector<int64_t> mVideoShape; //!< {maxBatch, C, t, h, w}

    // Device buffers (allocated once at construction for the engine profile shape). The video/action
    // latents live as ONE packed [video ⧺ action] device allocation (the engine binds video at the base
    // pointer and action at base + videoBytes), matching the joint layout the device-resident UniPC
    // scheduler updates in place; the predictions are packed the same way. The denoising state never
    // visits the host; only the final action slice is copied back.
    rt::Tensor mStateDevice; //!< packed FLOAT32 [B*videoElems | B*actionElems] current latents.
    rt::Tensor mPredDevice;  //!< packed FLOAT32 model predictions.
    rt::Tensor mTimestepDevice;
    rt::Tensor mTokenNoisyMaskDevice;
    rt::Tensor mActionNoisyMaskDevice;
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsDevice; //!< attention_pos_id [B, genLen] (INT32)
    rt::Tensor mPositionsDevice;   //!< [maxB, 3, genLen] float T/H/W planes for the core mRoPE kernel.

    // Pinned host staging (kCPU tensors are cudaMallocHost-backed) for the small per-request /
    // per-step engine inputs, filled in place and uploaded async with no per-call allocation.
    rt::Tensor mTimestepHost;        //!< [maxB] float, refilled every denoise step.
    rt::Tensor mTokenNoisyMaskHost;  //!< [maxB, numVideoTokens, 1] float.
    rt::Tensor mActionNoisyMaskHost; //!< [maxB, actionChunkSize, 1] float.
};

} // namespace cosmos3
} // namespace trt_edgellm
