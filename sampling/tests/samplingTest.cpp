#include "topKSamplingLayer.h"
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>

int main() {
  uint64_t seed = 42;
  int64_t batchSize = 1, vocabSize = 8, maxBatchSize = 64, maxSeqlen = 32,
          beam = 1;
  DecoderDomain domain(maxBatchSize, beam, vocabSize);
  TopKSamplingLayer<half> topkLayer(domain);

  auto setupParams = std::make_shared<SamplingSetupParams>();

  setupParams->randomSeed = std::make_optional<std::vector<uint64_t>>({seed});
  setupParams->runtimeTopK = std::make_optional<std::vector<std::int32_t>>({1});
  setupParams->runtimeTopP = std::nullopt;
  setupParams->topPDecay = std::nullopt;
  setupParams->topPMin = std::nullopt;
  setupParams->topPResetIds = std::nullopt;

  std::vector<float> logit = {
      -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163,  -1.2040,  -1.6094,
      -2.3026, // step 0
      -0.9163,  -1.2040,  -1.6094,  -2.3026,  -FLT_MAX, -FLT_MAX, -FLT_MAX,
      -FLT_MAX, // step 1
      -FLT_MAX, -FLT_MAX, -0.9163,  -1.2040,  -1.6094,  -2.3026,  -FLT_MAX,
      -FLT_MAX, // step 2
      -0.9163,  -1.2040,  -1.6094,  -2.3026,  -FLT_MAX, -FLT_MAX, -FLT_MAX,
      -FLT_MAX, // step 3
      -0.9163,  -1.2040,  -1.6094,  -2.3026,  -0.1,     -0.009,   -FLT_MAX,
      -FLT_MAX // step 4
  };

  std::vector<half> halfLogit(logit.size());
  for (int i = 0; i < halfLogit.size(); i++) {
    halfLogit[i] = logit[i];
  }

  printf("After setup\n");

  topkLayer.setup(batchSize, 1, nullptr, setupParams);

  printf("Before setup\n");

  half *logitsDevice;
  curandState *devStates;
  void *workspace;
  int64_t **outputIdsPtr, outputIdsHost[maxBatchSize * maxSeqlen], *outputIds;
  int32_t *seqlen, *endIds;
  FinishedState::UnderlyingType *finished;

  auto workspaceSize = topkLayer.getWorkspaceSize();
  cudaMalloc(&logitsDevice, sizeof(half) * batchSize * vocabSize);
  cudaMalloc(&devStates, sizeof(curandState) * maxBatchSize);
  cudaMalloc(&workspace, sizeof(int8_t) * workspaceSize);
  cudaMalloc(&seqlen, sizeof(int32_t) * maxBatchSize);
  cudaMalloc(&outputIds, sizeof(int64_t) * maxBatchSize * maxSeqlen);
  cudaMalloc(&endIds, sizeof(int32_t) * maxBatchSize);
  cudaMalloc(&finished, sizeof(FinishedState::UnderlyingType) * maxBatchSize);
  cudaMallocHost(&outputIdsPtr, sizeof(int64_t *) * maxBatchSize);
  for (int i = 0; i < batchSize; i++) {
    outputIdsPtr[i] = outputIds + i * maxSeqlen;
  }

  cudaMemset(endIds, -1, sizeof(int32_t) * maxBatchSize);
  cudaMemset(finished, 0, sizeof(FinishedState::UnderlyingType) * maxBatchSize);
  cudaMemcpy(logitsDevice, halfLogit.data(), sizeof(half) * logit.size(),
             cudaMemcpyHostToDevice);

  auto inputs = std::make_shared<SamplingInputs>(batchSize);
  inputs->endIds = std::make_shared<TensorWrapper>(
      endIds, std::vector<int64_t>{maxBatchSize}, TRTDataType<int32_t>::value);
  inputs->logits = std::make_shared<const TensorWrapper>(
      logitsDevice, std::vector<int64_t>{batchSize, vocabSize},
      TRTDataType<half>::value);
  inputs->probsComputed = false;
  inputs->curandStates = devStates;
  inputs->samplingWorkspace = workspace;
  inputs->finished = std::make_shared<TensorWrapper>(
      finished, std::vector<int64_t>{maxBatchSize},
      TRTDataType<FinishedState::UnderlyingType>::value);

  auto outputs =
      std::make_shared<BaseDecodingOutputs>(std::make_shared<TensorWrapper>(
          outputIds, std::vector<int64_t>{maxBatchSize, maxSeqlen},
          TRTDataType<int64_t>::value));
  outputs->outputIdsPtr = std::make_shared<TensorWrapper>(
      outputIdsPtr, std::vector<int64_t>{maxBatchSize},
      TRTDataType<int32_t *>::value);
  outputs->sequenceLength = std::make_shared<TensorWrapper>(
      seqlen, std::vector<int64_t>{maxBatchSize}, TRTDataType<int32_t>::value);
  outputs->finished = std::make_shared<TensorWrapper>(
      finished, std::vector<int64_t>{maxBatchSize},
      TRTDataType<FinishedState::UnderlyingType>::value);

  for (int step = 0; step < 5; step++) {
    cudaMemcpy(logitsDevice, halfLogit.data() + step * vocabSize,
               sizeof(half) * vocabSize, cudaMemcpyHostToDevice);

    topkLayer.forwardAsync(outputs, inputs);
    sync_check_cuda_error();

    cudaMemcpy(outputIdsHost, outputIds, sizeof(outputIdsHost),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++) {
      for (int j = 0; j < maxSeqlen; j++) {
        printf("%d ", int(outputIdsHost[i * maxSeqlen + j]));
      }
      printf("\n");
    }
  }

  cudaFree(logitsDevice);
  cudaFree(devStates);
  cudaFree(workspace);
  cudaFree(seqlen);
  cudaFree(outputIds);
  cudaFree(endIds);
  cudaFree(finished);
  cudaFreeHost(outputIdsPtr);
}
