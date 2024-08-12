/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "samplingTopKKernels.h"
#include "topKSamplingLayer.h"

#include <algorithm>
#include <cstdint>
#include <float.h>
#include <numeric>

template <int32_t TOP_K_MAX>
__global__ void
setupTopKRuntimeArgs(std::int32_t batchSize, std::int32_t topK,
                     std::int32_t *topKs, std::int32_t topKsSize, float topP,
                     float *topPs, std::int32_t topPsSize, bool *skipDecode,
                     std::int32_t const *batchSlots) {
  auto const index =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  for (auto bi = index; bi < batchSize;
       bi += static_cast<std::int32_t>(gridDim.x * blockDim.x)) {
    auto const batchSlot = batchSlots != nullptr ? batchSlots[bi] : bi;
    auto k = topKsSize > 1 ? topKs[batchSlot] : topK;
    auto p = topPsSize > 1 ? topPs[batchSlot] : topP;
    if (k == 0 && p == 0.0f) {
      // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
      // equivalent to greedy search. So, we set the topk = 1 as an alternative
      // solution.
      k = 1;
    }
    if (k > 0 && p == 0.0f) {
      // This case corresponds to the old topk sampling, which is equivalent to
      // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
      // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
      // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
      // compatibility.
      p = 1.0f;
    }
    // Clip k value. A topk sampling kernel supports up to TOP_K_MAX.
    topKs[batchSlot] = k;
    // Clip p value if it is out of range. range = [0.0, 1.0].
    topPs[batchSlot] = p;
    skipDecode[batchSlot] = k == 0;
  }
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(DecoderDomain const &decoderDomain)
    : BaseLayer(decoderDomain) {
  allocateBuffer(mDecoderDomain.getBatchSize());
}

template <typename T> TopKSamplingLayer<T>::~TopKSamplingLayer() {
  deallocateBuffer(mDecoderDomain.getBatchSize());
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(std::int32_t const batchSize) {
  mWorkspaceSize = getTopKWorkspaceSize<T>(batchSize, 1, TOP_K_MAX,
                                           mDecoderDomain.getVocabSizePadded());

  std::int32_t *int32Buffer;
  bool *boolBuffer, *boolHostBuffer = new bool[batchSize];
  float *floatBuffer;
  check_cuda_error(
      cudaMalloc(&int32Buffer, sizeof(std::int32_t) * batchSize * 2));
  check_cuda_error(cudaMalloc(&boolBuffer, sizeof(bool) * batchSize));
  check_cuda_error(cudaMalloc(&floatBuffer, sizeof(float) * batchSize));

  mRuntimeTopKDevice =
      TensorWrapper(int32Buffer, {batchSize}, TRTDataType<std::int32_t>::value);
  mRuntimeTopPDevice =
      TensorWrapper(floatBuffer, {batchSize}, TRTDataType<float>::value);
  mSkipDecodeDevice =
      TensorWrapper(boolBuffer, {batchSize}, TRTDataType<bool>::value);
  mSetupWorkspaceDevice = TensorWrapper(int32Buffer + batchSize, {batchSize},
                                        TRTDataType<std::int32_t>::value);

  mSkipDecodeHost =
      TensorWrapper(boolHostBuffer, {batchSize}, TRTDataType<bool>::value);
}

template <typename T>
void TopKSamplingLayer<T>::deallocateBuffer(std::int32_t const batchSize) {

  cudaFree(mRuntimeTopKDevice.data());
  cudaFree(mRuntimeTopPDevice.data());
  cudaFree(mSkipDecodeDevice.data());
  delete static_cast<bool *>(mSkipDecodeHost.data());

}

template <typename T>
void TopKSamplingLayer<T>::setup(
    std::int32_t batchSize, std::int32_t beamWidth, BufferConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const &baseSetupParams) {
  auto setupParams =
      std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

  auto const defaultTopK = DefaultDecodingParams::getTopK();
  auto runtimeTopK = setupParams->runtimeTopK.value_or(
      std::vector<std::int32_t>(batchSize, defaultTopK));
  auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector<float>{});

  auto const runtimeTopKSize = runtimeTopK.size();
  auto const runtimeTopPSize = runtimeTopP.size();
  mNormalizeLogProbs = setupParams->normalizeLogProbs.has_value() &&
                       setupParams->normalizeLogProbs.value();

  for (auto &topP : runtimeTopP) {
    if (topP < 0.f || topP > 1.0f) {
      // TLLM_LOG_WARNING(
      //     "TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.",
      //     topP);
      topP = std::clamp(topP, 0.f, 1.f);
    }
  }
  for (auto &topK : runtimeTopK) {
    if (topK < 0 || topK > TOP_K_MAX) {
      // TLLM_LOG_WARNING("TopK (%d) is larger than max supported number (%d). "
      //                  "Clip to max supported number.",
      //                  topK, TOP_K_MAX);
      topK = std::clamp(topK, 0, static_cast<std::int32_t>(TOP_K_MAX));
    }
  }

  auto const topK =
      *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
  auto const topP = (runtimeTopPSize == 0) ? DefaultDecodingParams::getTopP()
                                           : runtimeTopP.front();

  auto batchSlotsPtr =
      batchSlots ? static_cast<std::int32_t const *>(batchSlots) : nullptr;
  auto setupWorkspaceDevicePtr =
      tensorCastOrNull<std::int32_t>(mSetupWorkspaceDevice);
  auto setupWorkspaceDeviceAsFloatPtr =
      reinterpret_cast<float const *>(setupWorkspaceDevicePtr);
  auto runtimeTopKDevicePtr =
      tensorCastOrNull<std::int32_t>(mRuntimeTopKDevice);
  auto runtimeTopPDevicePtr = tensorCastOrNull<float>(mRuntimeTopPDevice);
  if (runtimeTopKSize > 1) {
    // TLLM_CHECK_WITH_INFO(
    //     runtimeTopK.size() == batchSize,
    //     fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not
    //     satisfied!",
    //            runtimeTopK.size(), batchSize));
    // BufferPtr runtimeTopKSetupWorkspaceSlice =
    //     IBuffer::slice(mSetupWorkspaceDevice, 0, batchSize);
    // mBufferManager->copy(runtimeTopK.data(), *runtimeTopKSetupWorkspaceSlice,
    //                      runtime::MemoryType::kCPU);
    // invokeScatterDecodingParams(setupWorkspaceDevicePtr,
    // runtimeTopKDevicePtr,
    //                             batchSlotsPtr, batchSize, getStream());
    assert(false);
  }
  if (runtimeTopPSize > 1) {
    // TLLM_CHECK_WITH_INFO(
    //     runtimeTopP.size() == batchSize,
    //     fmtstr("runtimeTopP.size() (%lu) == batchSize (%d) is not
    //     satisfied!",
    //            runtimeTopP.size(), batchSize));
    // BufferPtr runtimeTopKSetupWorkspaceSlice =
    //     IBuffer::slice(mSetupWorkspaceDevice, 0, batchSize);
    // mBufferManager->copy(runtimeTopP.data(), *runtimeTopKSetupWorkspaceSlice,
    //                      runtime::MemoryType::kCPU);
    // invokeScatterDecodingParams(setupWorkspaceDeviceAsFloatPtr,
    //                             runtimeTopPDevicePtr, batchSlotsPtr,
    //                             batchSize, getStream());
    assert(false);
  }

  auto skipDecodeDevicePtr = tensorCastOrNull<bool>(mSkipDecodeDevice);
  {
    dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
    dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
    // support topK up to TOP_K_MAX.
    setupTopKRuntimeArgs<TOP_K_MAX><<<grid, block, 0, getStream()>>>(
        batchSize, topK, runtimeTopKDevicePtr, runtimeTopKSize, topP,
        runtimeTopPDevicePtr, runtimeTopPSize, skipDecodeDevicePtr,
        batchSlotsPtr);
  }

  cudaMemcpy(mSkipDecodeHost.data(), mSkipDecodeDevice.data(),
             mSkipDecodeDevice.getSizeInBytes(), cudaMemcpyDeviceToHost);
  std::vector<std::int32_t> runtimeTopKs(mDecoderDomain.getBatchSize());
  cudaMemcpy(runtimeTopKs.data(), mRuntimeTopKDevice.data(),
             mRuntimeTopKDevice.getSizeInBytes(), cudaMemcpyDeviceToHost);
  {
    std::int32_t maxTopK = 0;
    for (std::int32_t bi = 0; bi < batchSize; ++bi) {
      auto bid = bi;
      if (batchSlotsPtr) {
        bid = batchSlotsPtr[bi];
      }
      maxTopK = std::max(maxTopK, runtimeTopKs[bid]);
    }
    mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, maxTopK);
  }
}

template <typename T>
void TopKSamplingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const &outputs,
    std::shared_ptr<BaseDecodingInputs> const &baseInputs) {

  auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

  auto const batchSize = inputs->logits.value()->getDimension<0>();

  auto logits = tensorCastOrNull<T>(inputs->logits->get());
  auto endIds = tensorCastOrNull<std::int32_t>(inputs->endIds.get());
  auto batchSlots = tensorCastOrNull<std::int32_t>(inputs->batchSlots->get());
  auto curandStatesDevice = inputs->curandStates;
  auto samplingWorkspaceDevice = inputs->samplingWorkspace;
  auto const probsComputed = inputs->probsComputed;
  std::vector<int32_t> batchSlotsVec(batchSize);
  std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
  auto batchSlotsHost = inputs->batchSlots
                            ? tensorCastOrNull<int>(inputs->batchSlots->get())
                            : batchSlotsVec.data();
  auto skipDecodeHostPtr = tensorCastOrNull<bool>(mSkipDecodeHost);
  auto const skip =
      allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
  if (skip) {
    return;
  }

  FinishedState const *finishedInput =
      (inputs->finished) ? reinterpret_cast<FinishedState const *>(
                               tensorCastOrNull<FinishedState::UnderlyingType>(
                                   inputs->finished->get()))
                         : nullptr;
  FinishedState *finishedOutput =
      (outputs->finished) ? reinterpret_cast<FinishedState *>(
                                tensorCastOrNull<FinishedState::UnderlyingType>(
                                    outputs->finished->get()))
                          : nullptr;

  assert(curandStatesDevice && "No curand states provided");
  assert(samplingWorkspaceDevice && "No sampling workspace provided");

  TopKSamplingKernelParams<T> params;
  params.logProbs = logits;
  params.outputIdsPtrs =
      tensorCastOrNull<std::int32_t *>(outputs->outputIdsPtr);
  params.workspace = samplingWorkspaceDevice;
  params.maxTopP = 1.0f;
  params.topPs = tensorCastOrNull<float>(mRuntimeTopPDevice);
  params.maxTopK = mRuntimeMaxTopK;
  params.topKs = tensorCastOrNull<std::int32_t>(mRuntimeTopKDevice);
  params.sequenceLengths =
      tensorCastOrNull<std::int32_t>(outputs->sequenceLength);
  params.endIds = endIds;
  params.batchSlots = batchSlots;
  params.finishedInput = finishedInput;
  params.finishedOutput = finishedOutput;
  params.skipDecode = tensorCastOrNull<bool>(mSkipDecodeDevice);
  params.cumLogProbs = tensorCastOrNull<float>(outputs->cumLogProbs);
  params.outputLogProbs = tensorCastOrNull<float>(outputs->outputLogProbsTiled);
  params.curandState = curandStatesDevice;
  params.batchSize = batchSize;
  params.maxBatchSize = mDecoderDomain.getBatchSize();
  params.maxTokensPerStep = 1;
  params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
  params.normalizeLogProbs = mNormalizeLogProbs;
  params.logitsHasProbs = probsComputed;

  invokeBatchTopKSampling(params, getStream());
  sync_check_cuda_error();
}

template <typename T>
size_t TopKSamplingLayer<T>::getWorkspaceSize() const noexcept {
  return mWorkspaceSize;
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;
