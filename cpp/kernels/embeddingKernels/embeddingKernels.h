#pragma once

#include "common/safetensors_loader/safetensorsLoader.h"
#include "common/tensor.h"
#include <cuda_runtime.h>

namespace drivellm
{
namespace kernels
{

/**
 * @brief Standard embedding lookup kernel
 *
 * @param inputIds Input token IDs with shape [batchSize, seqLen]
 * @param embeddingTable Embedding table with shape [vocabSize, hiddenSize]
 * @param output Hidden states with shape [batchSize, seqLen, hiddenSize]
 * @param stream CUDA stream for execution
 */
void embeddingLookup(
    rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::Tensor& output, cudaStream_t stream = 0);

/**
 * @brief Embedding lookup with image embedding insertion following PromptTuningEmbedding logic
 *
 * @param inputIds Input token IDs with shape [batchSize, seqLen]
 * @param embeddingTable Embedding table with shape [vocabSize, hiddenSize]
 * @param imageEmbeds Image embeddings with shape [imageTokenLen, hiddenSize]
 * @param vocabSize Vocabulary size for normal tokens
 * @param output Hidden states with shape [batchSize, seqLen, hiddenSize]
 * @param stream CUDA stream for execution
 */
void embeddingLookupWithImageInsertion(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::Tensor const& imageEmbeds, rt::Tensor& output, cudaStream_t stream = 0);

} // namespace kernels
} // namespace drivellm