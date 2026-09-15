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

#include "cosmos3Scheduler.h"
#include "action/cosmos3Kernels.h"
#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace trt_edgellm
{
namespace cosmos3
{

namespace
{
//! Apply the resolution-dependent flow shift to a raw sigma in (0, 1].
float applyShift(double sigma, double shift)
{
    return static_cast<float>(shift * sigma / (1.0 + (shift - 1.0) * sigma));
}

//! Flow-matching half-log-SNR: lambda = log(alpha) - log(sigma), alpha = 1 - sigma.
//! Kept in float32 to match the reference, which evaluates torch.log on its float32
//! sigma buffer. At the first UniPC step sigma ~ 0.9998 leaves log(1-sigma) with only
//! a few significant bits, but that precision loss is part of the reference numerics:
//! computing it in double here would make this solver disagree with the reference.
double lambdaOf(double sigma)
{
    float const alpha = 1.0F - static_cast<float>(sigma);
    return static_cast<double>(std::log(alpha) - std::log(static_cast<float>(sigma)));
}
} // namespace

Cosmos3Scheduler::Cosmos3Scheduler(Config const& config)
    : mConfig(config)
{
}

void Cosmos3Scheduler::initialize(int32_t numSteps)
{
    mTimesteps.clear();
    mSigmas.clear();
    mTimesteps.reserve(static_cast<size_t>(numSteps));
    mSigmas.reserve(static_cast<size_t>(numSteps) + 1);

    // Matches the reference FlowUniPCMultistepScheduler.set_timesteps exactly:
    // sigmas = linspace(sigmaMax, sigmaMin, numSteps + 1)[:-1] followed by the flow shift,
    // with the terminal sigma 0 appended. The schedule starts at t = numTrainTimesteps - 1
    // (sigma == 1 has lambda = log(1-sigma) - log(sigma) = -inf, which would poison the
    // order-2 divided-difference term rk with inf - inf) and never reaches sigmaMin: the
    // final update jumps from sigmas[numSteps - 1] straight to the terminal sigma.
    double const sigmaMax
        = static_cast<double>(mConfig.numTrainTimesteps - 1) / static_cast<double>(mConfig.numTrainTimesteps);
    double const sigmaMin = 0.0;
    for (int32_t i = 0; i < numSteps; ++i)
    {
        double const frac = static_cast<double>(i) / static_cast<double>(numSteps);
        double const rawSigma = sigmaMax + frac * (sigmaMin - sigmaMax); // linspace(max, min, N+1)[:-1]
        float const sigma = applyShift(rawSigma, static_cast<double>(mConfig.shift));
        mSigmas.push_back(sigma);
        // The reference conditions the model on floor(sigma * numTrainTimesteps): integer
        // timesteps [999, 937, 833, 624] at 4 steps.
        mTimesteps.push_back(std::floor(static_cast<double>(sigma) * mConfig.numTrainTimesteps));
    }
    mSigmas.push_back(0.0F);
    reset();
}

void Cosmos3Scheduler::prepare(int64_t numElems)
{
    if (mNumElems == numElems)
    {
        return;
    }
    if (numElems > mCapacityElems)
    {
        // First prepare (at the runner's profile-maximum batch) allocates; smaller active batches
        // later are metadata-only reshapes within this capacity — no per-request allocation.
        mX0Ring.clear();
        mX0Ring.reserve(3);
        for (int32_t i = 0; i < 3; ++i)
        {
            mX0Ring.emplace_back(
                rt::Coords{numElems}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosmos3::schedX0");
        }
        mLastSample
            = rt::Tensor(rt::Coords{numElems}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosmos3::schedLast");
        mScratch = rt::Tensor(
            rt::Coords{numElems}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosmos3::schedScratch");
        mCapacityElems = numElems;
    }
    else
    {
        for (auto& t : mX0Ring)
        {
            ELLM_CHECK(t.reshape(rt::Coords{numElems}), "Cosmos3Scheduler x0 reshape failed");
        }
        ELLM_CHECK(mLastSample.reshape(rt::Coords{numElems}), "Cosmos3Scheduler lastSample reshape failed");
        ELLM_CHECK(mScratch.reshape(rt::Coords{numElems}), "Cosmos3Scheduler scratch reshape failed");
    }
    mNumElems = numElems;
    reset();
}

void Cosmos3Scheduler::reset()
{
    mHistCount = 0;
    mHaveLastSample = false;
    mLowerOrderNums = 0;
    mThisOrder = 0;
}

Cosmos3Scheduler::UniBHCoeffs Cosmos3Scheduler::computeUniBHCoeffs(
    int32_t order, int32_t idxT, int32_t idxS0, int32_t k, bool corrector) const
{
    double const sigmaT = static_cast<double>(mSigmas.at(static_cast<size_t>(idxT)));
    double const sigmaS0 = static_cast<double>(mSigmas.at(static_cast<size_t>(idxS0)));
    double const alphaT = 1.0 - sigmaT; // use_flow_sigmas: alpha = 1 - sigma
    double const lambdaS0 = lambdaOf(sigmaS0);

    // h, hh = -h (predict_x0). The terminal sigma (sigmaT == 0) gives lambda_t -> +inf, hh -> -inf,
    // for which expm1(-inf) = -1 (diffusers' inf-arithmetic limit).
    double const infD = std::numeric_limits<double>::infinity();
    double const lambdaT = (sigmaT <= 0.0) ? infD : lambdaOf(sigmaT);
    double const h = lambdaT - lambdaS0;
    double const hh = -h;
    bool const hhNegInf = (hh == -infD);
    double const hPhi1 = hhNegInf ? -1.0 : std::expm1(hh);
    double const bH = hhNegInf ? -1.0 : std::expm1(hh); // bh2 solver: B_h = expm1(hh)

    // Order-2 divided-difference term: rk and whether the previous x0 prediction participates.
    double rk = 1.0;
    bool useMPrev = false;
    if (order >= 2 && mHistCount >= 2)
    {
        // The prediction one slot before the most recent: si = k-1 (predictor) or k-2 (corrector).
        int32_t const si = corrector ? (k - 2) : (k - 1);
        if (si >= 0)
        {
            rk = (lambdaOf(static_cast<double>(mSigmas.at(static_cast<size_t>(si)))) - lambdaS0) / h;
            useMPrev = true;
        }
    }

    // Scalar B(h) coefficients. Predictor order 2 uses the simplified rho_p = 0.5; the corrector solves
    // the small R rho = b system (rho_c = 0.5 for order 1; 2x2 solve for order 2).
    double rhoFirst = 0.5; // coefficient on the divided difference D1s[0]
    double rhoLast = 0.5;  // coefficient on D1_t (corrector only)
    if (corrector && order >= 2)
    {
        // R = [[1, 1], [rk, 1]];  b = [hPhiK0/B_h, 2*hPhiK1/B_h]
        double const hPhiK0 = hhNegInf ? -1.0 : (hPhi1 / hh - 1.0);
        double const hPhiK1 = (hhNegInf ? 0.0 : (hPhiK0 / hh)) - 0.5;
        double const b0 = hPhiK0 / bH;
        double const b1 = 2.0 * hPhiK1 / bH;
        double const det = 1.0 - rk; // det([[1,1],[rk,1]]) = 1 - rk
        if (std::fabs(det) > std::numeric_limits<double>::epsilon())
        {
            rhoFirst = (b0 - b1) / det;
            rhoLast = (b1 - rk * b0) / det;
        }
    }

    // Fold the elementwise form
    //   out = ratio*x - alphaT*hPhi1*m0 - alphaT*bH*(rhoFirst*(mPrev - m0)/rk + rhoLast*(modelT - m0))
    // into one linear combination out = cX*x + cM0*m0 + cMPrev*mPrev + cMT*modelT.
    UniBHCoeffs c;
    c.cX = static_cast<float>(sigmaT / sigmaS0);
    c.useMPrev = useMPrev;
    c.useMT = corrector;
    double m0Coeff = -alphaT * hPhi1;
    if (useMPrev)
    {
        double const dd = -alphaT * bH * rhoFirst / rk;
        c.cMPrev = static_cast<float>(dd);
        m0Coeff -= dd;
    }
    if (corrector)
    {
        double const dt = -alphaT * bH * rhoLast;
        c.cMT = static_cast<float>(dt);
        m0Coeff -= dt;
    }
    c.cM0 = static_cast<float>(m0Coeff);
    return c;
}

void Cosmos3Scheduler::step(rt::Tensor const& modelOutput, rt::Tensor& sample, int32_t stepIdx, cudaStream_t stream)
{
    ELLM_CHECK(mNumElems > 0, "Cosmos3Scheduler::prepare must be called before step");
    int32_t const k = stepIdx;
    int32_t const numSteps = static_cast<int32_t>(mTimesteps.size());
    float const sigmaS0 = mSigmas.at(static_cast<size_t>(k));

    // Convert the raw model output to an x0 (data) prediction at the current sigma:
    // x0 = sample - sigma * velocity (one fused kernel into the current ring slot).
    rt::Tensor& x0Cur = mX0Ring[static_cast<size_t>(k % 3)];
    kernel::launchFusedCombine(x0Cur, 1.0F, sample, -sigmaS0, modelOutput, 0.0F, nullptr, 0.0F, nullptr, stream);

    // UniC corrector: refine the current sample using the previous step's prediction. Uses mThisOrder
    // from the previous step (set after the previous predictor), matching diffusers' self.this_order.
    bool const useCorrector = (k > 0) && mHaveLastSample;
    if (useCorrector)
    {
        UniBHCoeffs const c = computeUniBHCoeffs(mThisOrder, /*idxT=*/k, /*idxS0=*/k - 1, k, /*corrector=*/true);
        rt::Tensor const& m0 = mX0Ring[static_cast<size_t>((k - 1) % 3)];
        rt::Tensor const* mPrev = c.useMPrev ? &mX0Ring[static_cast<size_t>((k - 2) % 3)] : nullptr;
        kernel::launchFusedCombine(mScratch, c.cX, mLastSample, c.cM0, m0, c.cMPrev, mPrev, c.cMT, &x0Cur, stream);
        std::swap(mScratch, mLastSample); // corrected sample becomes the working sample
    }
    else
    {
        CUDA_CHECK(cudaMemcpyAsync(mLastSample.rawPointer(), sample.rawPointer(),
            static_cast<size_t>(mNumElems) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    mHistCount = std::min(mHistCount + 1, mConfig.solverOrder);

    // Effective order for the predictor (lower_order_final + multistep warmup).
    int32_t thisOrder = std::min(mConfig.solverOrder, numSteps - k);
    thisOrder = std::min(thisOrder, mLowerOrderNums + 1);
    mThisOrder = thisOrder;
    mHaveLastSample = true;

    // UniP predictor: advance the working sample to the next sigma, writing the state in place.
    UniBHCoeffs const c = computeUniBHCoeffs(thisOrder, /*idxT=*/k + 1, /*idxS0=*/k, k, /*corrector=*/false);
    rt::Tensor const* mPrev = c.useMPrev ? &mX0Ring[static_cast<size_t>((k - 1) % 3)] : nullptr;
    kernel::launchFusedCombine(sample, c.cX, mLastSample, c.cM0, x0Cur, c.cMPrev, mPrev, 0.0F, nullptr, stream);

    mLowerOrderNums = std::min(mLowerOrderNums + 1, mConfig.solverOrder);
}

} // namespace cosmos3
} // namespace trt_edgellm
