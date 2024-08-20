#include "sampler.h"
#include <cfloat>
#include <cuda_runtime_api.h>
#include <iostream>

int main() {
  int batchSize = 1, vocabSize = 8;
  Sampler<half> sampler(batchSize, vocabSize);

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

  half *logitsDevice;
  CUDA_CHECK(cudaMalloc(&logitsDevice, sizeof(half) * batchSize * vocabSize));

  for (int64_t i = 0; i < 5; i++) {
    CUDA_CHECK(cudaMemcpy(logitsDevice, halfLogit.data() + vocabSize * i,
                         sizeof(half) * vocabSize * batchSize,
                         cudaMemcpyHostToDevice));
    auto &a = sampler.greedySample(logitsDevice);
    std::cout << a[0] << std::endl;
  }
}
