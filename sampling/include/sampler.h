#pragma once

#include "baseLayer.h"
#include <cstdint>
#include <memory>
#include <vector>

template <typename T> class Sampler {
public:
  Sampler(int64_t batchSize, int64_t vocabSize);
  ~Sampler();

  const std::vector<int64_t> &greedySample(T *logits);

  Sampler(Sampler const &) = delete;
  Sampler &operator=(Sampler const &) = delete;

private:
  std::unique_ptr<BaseLayer> mLayer;
  void *mWorkspace;
  curandState *mDevStates;
  DecoderDomain mDecoderDomain;
  std::vector<int64_t> mOutputIds;
  int64_t *mOutputIdsDevice;
};
