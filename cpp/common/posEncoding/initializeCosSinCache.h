#pragma once

namespace drivellm
{
namespace kernel
{

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm