#pragma once

namespace drivellm
{
namespace kernel
{

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

void initializeLongRopeCosSin(float* shortCosSinCache, float* longCosSinCache, float* shortFactor, float* longFactor,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, 
    int32_t maxPositionEmbeddings, int32_t originalMaxPositionEmbeddings, cudaStream_t stream);

} // namespace kernel
} // namespace drivellm