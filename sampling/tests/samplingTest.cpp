#include "topKSamplingLayer.h"
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>

int main() {
  uint64_t seed = 42;
  DecoderDomain domain(64, 1, 8);
  TopKSamplingLayer<half> topkLayer(domain);
  int64_t batchSize = 1, vocabSize = 8;

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
  int32_t **outputIdsPtr;
  int32_t *seqlen, *outputIds, *endIds, outputIdsHost[64 * 64];
  FinishedState::UnderlyingType *finished;

  auto workspaceSize = topkLayer.getWorkspaceSize();
  cudaMalloc(&logitsDevice, sizeof(half) * batchSize * vocabSize);
  cudaMalloc(&devStates, sizeof(curandState) * 64);
  cudaMalloc(&workspace, sizeof(int8_t) * workspaceSize);
  cudaMalloc(&seqlen, sizeof(int32_t) * 64);
  cudaMalloc(&outputIds, sizeof(int32_t) * 64 * 64);
  cudaMalloc(&endIds, sizeof(int32_t) * 64);
  cudaMalloc(&finished, sizeof(FinishedState::UnderlyingType) * 64 * 64);
  cudaMallocHost(&outputIdsPtr, sizeof(int32_t *) * 64);
  for (int i = 0; i < batchSize; i++) {
    outputIdsPtr[i] = outputIds + i * 64;
  }

  cudaMemset(endIds, -1, sizeof(int32_t) * 64);
  cudaMemset(finished, 0, sizeof(FinishedState::UnderlyingType) * 64 * 64);
  cudaMemcpy(logitsDevice, halfLogit.data(), sizeof(half) * logit.size(),
             cudaMemcpyHostToDevice);

  auto inputs = std::make_shared<SamplingInputs>(
      std::make_shared<TensorWrapper>(endIds, std::vector<int64_t>{64},
                                      TRTDataType<int32_t>::value),
      64, 0, batchSize);
  inputs->logits = std::make_shared<TensorWrapper>(
      logitsDevice, std::vector<int64_t>{batchSize, vocabSize},
      TRTDataType<half>::value);
  inputs->probsComputed = false;
  inputs->curandStates = devStates;
  inputs->samplingWorkspace = workspace;
  inputs->finished = std::make_shared<TensorWrapper>(
      finished, std::vector<int64_t>{64, 64},
      TRTDataType<FinishedState::UnderlyingType>::value);

  auto outputs = std::make_shared<BaseDecodingOutputs>(
      std::make_shared<TensorWrapper>(outputIds, std::vector<int64_t>{64, 64},
                                      TRTDataType<int32_t>::value));
  outputs->outputIdsPtr = std::make_shared<TensorWrapper>(
      outputIdsPtr, std::vector<int64_t>{64}, TRTDataType<int32_t *>::value);
  outputs->sequenceLength = std::make_shared<TensorWrapper>(
      seqlen, std::vector<int64_t>{64}, TRTDataType<int32_t>::value);
  outputs->finished = std::make_shared<TensorWrapper>(
      finished, std::vector<int64_t>{64, 64},
      TRTDataType<FinishedState::UnderlyingType>::value);

  for (int step = 0; step < 5; step++) {
    cudaMemcpy(logitsDevice, halfLogit.data() + step * vocabSize,
               sizeof(half) * vocabSize, cudaMemcpyHostToDevice);

    sync_check_cuda_error();

    topkLayer.forwardAsync(outputs, inputs);

    sync_check_cuda_error();

    cudaMemcpy(outputIdsHost, outputIds, sizeof(outputIdsHost),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++) {
      for (int j = 0; j < 64; j++) {
        printf("%d ", outputIdsHost[i * 64 + j]);
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
