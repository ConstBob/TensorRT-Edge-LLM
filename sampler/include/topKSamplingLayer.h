#pragma once

#include "baseLayer.h"
#include "cudaUtils.h"
#include <cstdint>

//! \brief Layer to randomly sample tokens from TopK logits.
//! When both TopK and TopP are specified, layer jointly samples using TopK and
//! TopP. When no TopK param is specified, sampling is skipped for particular
//! request.
template <typename T> class TopKSamplingLayer : public BaseLayer {
  using Base = BaseLayer;

public:
  TopKSamplingLayer(DecoderDomain const &decoderDomain);
  virtual ~TopKSamplingLayer();

  void setup(std::int32_t batchSize, std::int32_t beamWidth,
             BufferConstPtr batchSlots,
             std::shared_ptr<BaseSetupParams> const &setupParams) override;
  void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const &outputs,
                    std::shared_ptr<BaseDecodingInputs> const &inputs) override;

  //! @returns workspace needed for this layer in bytes
  [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

protected:
  bool mNormalizeLogProbs{true};
  std::int32_t mWorkspaceSize{0};
  std::int32_t mRuntimeMaxTopK{0};
  TensorWrapper mRuntimeTopKDevice;
  TensorWrapper mRuntimeTopPDevice;
  TensorWrapper mSetupWorkspaceDevice;
  TensorWrapper mSkipDecodeDevice;
  TensorWrapper mSkipDecodeHost;

  using Base::mDecoderDomain;

private:
  void allocateBuffer(std::int32_t batchSize);
  void deallocateBuffer(std::int32_t batchSize);
};

template <typename T>
[[nodiscard]] std::vector<size_t>
getTopKWorkspaceSizes(std::int32_t batchSize, std::int32_t maxTokensPerStep,
                      std::int32_t maxTopK, std::int32_t vocabSizePadded) {
  std::int32_t constexpr maxBlockPerBeam = 8;
  auto const tempLogProbsBufSize =
      sizeof(T) * batchSize * maxTokensPerStep * vocabSizePadded; // type T
  auto const topKTmpIdsBufSize = sizeof(std::int32_t) * batchSize *
                                 maxTokensPerStep * maxTopK *
                                 maxBlockPerBeam; // type int
  auto const topKTmpValBufSize = sizeof(T) * batchSize * maxTokensPerStep *
                                 maxTopK * maxBlockPerBeam; // type T

  return {tempLogProbsBufSize, topKTmpIdsBufSize, topKTmpValBufSize};
}

template <typename T>
[[nodiscard]] size_t
getTopKWorkspaceSize(std::int32_t batchSize, std::int32_t maxTokensPerStep,
                     std::int32_t maxTopK, std::int32_t vocabSizePadded) {
  auto const workspaceSizes = getTopKWorkspaceSizes<T>(
      batchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
  return calcAlignedSize(workspaceSizes, 256);
}
