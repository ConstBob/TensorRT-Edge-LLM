# TensorRT Edge-LLM Sampling Unit

High-performance CUDA sampling API for language model inference supporting Top-K, Top-P, and combined sampling methods with manual workspace management.

## Supported Data Types
- **Logits**: `float` (32-bit floating point) only
- **Indices**: `int32_t` only

## API Reference

### Core Functions

#### Main Sampling Function
```cpp
void topKtopPSamplingFromLogits(float const* logits, int32_t* selectedIndices,
    SamplingParams const& params, void* workspace, size_t workspaceSize, 
    cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);
```

**Parameters:**
- `logits`: Input logits tensor of shape `[batchSize, vocabSize]` (float)
- `selectedIndices`: Output tensor for selected token indices of shape `[batchSize]` (int32_t)
- `params`: Sampling parameters (see SamplingParams structure)
- `workspace`: Allocated memory for temporary storage
- `workspaceSize`: Size of allocated workspace
- `stream`: CUDA stream for asynchronous execution
- `philoxSeed`: Random number generator seed (default: 42)
- `philoxOffset`: Random number generator offset (default: 0)

#### Select All Top-K Function
```cpp
void selectAllTopKFromLogits(float const* input, float* topKValues, int32_t* topKIndices,
    int32_t batchSize, int32_t vocabSize, int32_t topK, 
    void* workspace, size_t workspaceSize, cudaStream_t stream,
    bool returnLogProbs = false, bool normalizeLogProbs = true, bool inputHasProbs = false);
```

**Parameters:**
- `input`: Input tensor of shape `[batchSize, vocabSize]` (logits or probabilities)
- `topKValues`: Output tensor for top-K values of shape `[batchSize, topK]` (float)
- `topKIndices`: Output tensor for top-K indices of shape `[batchSize, topK]` (int32_t)
- `batchSize`: Number of sequences in the batch
- `vocabSize`: Size of the vocabulary
- `topK`: Number of top elements to select
- `workspace`: Allocated memory for temporary storage
- `workspaceSize`: Size of allocated workspace
- `stream`: CUDA stream for asynchronous execution
- `returnLogProbs`: If true, returns log probabilities instead of raw values
- `normalizeLogProbs`: If true and `returnLogProbs=true`, normalizes log probabilities
- `inputHasProbs`: If true, input is treated as probabilities; if false, input is treated as logits

### Workspace Size Calculation
```cpp
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params);
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK);
```

### SamplingParams Structure
```cpp
struct SamplingParams {
    int32_t batchSize;
    int32_t vocabSize;
    float temperature;
    int32_t topK;
    float topP;
    bool useTopK;
    bool useTopP;

    SamplingParams(int32_t batchSize, int32_t vocabSize, float temperature = 1.0f, 
                   int32_t topK = 0, float topP = 1.0f);
};
```

**Note**: Constructor automatically sets `useTopK = (topK > 0)` and `useTopP = (topP < 1.0f)`. At least one must be active.

## Usage Example

```cpp
#include "sampling.h"
using namespace drivellm;

// Setup
SamplingParams params(batchSize, vocabSize, 1.0f, 40, 0.9f);
size_t workspaceSize = getTopKtopPSamplingWorkspaceSize(batchSize, vocabSize, params);

// Allocate memory
void* workspace;
float* logits;
int32_t* selectedIndices;
cudaStream_t stream;

cudaMalloc(&workspace, workspaceSize);
cudaMalloc(&logits, batchSize * vocabSize * sizeof(float));
cudaMalloc(&selectedIndices, batchSize * sizeof(int32_t));
cudaStreamCreate(&stream);

// Perform sampling
topKtopPSamplingFromLogits(logits, selectedIndices, params, 
                          workspace, workspaceSize, stream);

// Cleanup
cudaFree(workspace);
cudaFree(logits);
cudaFree(selectedIndices);
cudaStreamDestroy(stream);
```

## Important Notes

1. **Workspace Required**: Manual workspace allocation is mandatory
2. **Stream Required**: CUDA stream parameter is required for all functions
3. **Data Types**: Only `float` for logits/values and `int32_t` for indices
4. **Combined Sampling**: Both topK and topP can be used together
5. **Parameter Validation**: At least one of topK or topP must be active