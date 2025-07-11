# DriveOS LLM SDK Sampling Unit

## Overview

This document describes a high-performance CUDA sampling API for language model inference that supports Top-K, Top-P, and combined sampling methods. The API is designed for optimal GPU performance through manual workspace management, enabling efficient token generation in large language models.

## Key Features

### Manual Workspace Management
```cpp
void* workspace;
size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<float>(batchSize, vocabSize, params);
cudaMalloc(&workspace, workspaceSize);
topKtopPSamplingFromLogits<float>(logits, selectedIndices, params, workspace, workspaceSize, stream, 42, 0);
```

**Benefits of manual workspace management:**
- **Optimal performance**: No allocation overhead during sampling
- **Memory efficiency**: Precise control over memory usage
- **Workspace reuse**: Allocate once, use for multiple sampling calls
- **Single API**: Consistent interface for all sampling methods

## API Reference

### Core Functions

#### Main Sampling Function
```cpp
template <typename T>
void topKtopPSamplingFromLogits(T const* logits, int64_t* selectedIndices,
    SamplingParams const& params, void* workspace, size_t workspaceSize, 
    cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);
```

**Required Parameters:**
- `logits`: Input logits tensor of shape `[batchSize, vocabSize]`
- `selectedIndices`: Output tensor for selected token indices of shape `[batchSize]`
- `params`: Sampling parameters (batch size, vocab size, temperature, topK, topP)
- `workspace`: Must be a valid pointer to allocated memory
- `workspaceSize`: Must be the correct size for the given parameters
- `stream`: CUDA stream for asynchronous execution

**Optional Parameters:**
- `philoxSeed`: Random number generator seed (default: 42)
- `philoxOffset`: Random number generator offset (default: 0)

**Features:**
- Supports all sampling methods: Top-K, Top-P, and combined
- **Note**: Output is `int64_t*` for selected indices

#### Select All Top-K Function
```cpp
template <typename T>
void selectAllTopKFromLogits(T const* input, float* topKValues, int64_t* topKIndices,
    int32_t batchSize, int32_t vocabSize, int32_t topK, 
    void* workspace, size_t workspaceSize, cudaStream_t stream,
    bool returnLogProbs = false, bool normalizeLogProbs = true, bool inputHasProbs = false);
```

**Required Parameters:**
- `input`: Input tensor of shape `[batchSize, vocabSize]` (logits or probabilities)
- `topKValues`: Output tensor for top-K values of shape `[batchSize, topK]` (must be non-null if `returnLogProbs=true`)
- `topKIndices`: Output tensor for top-K indices of shape `[batchSize, topK]`
- `batchSize`: Number of sequences in the batch
- `vocabSize`: Size of the vocabulary
- `topK`: Number of top elements to select
- `workspace`: Must be a valid pointer to allocated memory
- `workspaceSize`: Must be the correct size for the given parameters
- `stream`: CUDA stream for asynchronous execution

**Optional Parameters:**
- `returnLogProbs`: If true, returns log probabilities instead of raw values
- `normalizeLogProbs`: If true and `returnLogProbs=true`, normalizes log probabilities
- `inputHasProbs`: If true, input is treated as probabilities; if false, input is treated as logits

**Features:**
- Returns all top-K elements without sampling
- Same workspace behavior as main sampling function
- **Note**: Values are always output as `float*`, indices as `int64_t*`
- **Error**: If `returnLogProbs=true` and `topKValues` is nullptr, the function will return an error

**Input Types:**
- When `inputHasProbs=false`: Input is treated as logits (raw model outputs)
- When `inputHasProbs=true`: Input is treated as probabilities (already softmaxed)
- **Note**: The definition of logits will be formalized in the next release

### Workspace Size Calculation

```cpp
// Get workspace size for sampling
template <typename T>
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

// Get workspace size for selectAllTopK
template <typename T>
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

    // Constructor with default values
    SamplingParams(int32_t batchSize, int32_t vocabSize, float temperature = 1.0f, 
                   int32_t topK = 0, float topP = 1.0f);
};
```

**Important**: The constructor automatically sets `useTopK` and `useTopP` flags based on parameter values:
- `useTopK = (topK > 0)`
- `useTopP = (topP < 1.0f)`
- **At least one of topK or topP must be active** (throws exception otherwise)

**Combined Usage**: Both topK and topP can be used together:
- When both are set, topK is applied first, then topP filtering is applied to the topK results
- This provides more controlled sampling by limiting both the number of candidates and their cumulative probability

## Usage Examples

### 1. Top-K Sampling Only

```cpp
#include "sampling.h"
using namespace drivellm;

// Calculate workspace size
SamplingParams params(batchSize, vocabSize, 1.0f, 40, 1.0f);  // topK=40, topP=1.0 (disabled)
size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<float>(batchSize, vocabSize, params);

// Allocate workspace and buffers
void* dWorkspace;
float* dLogits;
int64_t* dSelectedIndices;
cudaStream_t stream;

cudaMalloc(&dWorkspace, workspaceSize);
cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(float));
cudaMalloc(&dSelectedIndices, batchSize * sizeof(int64_t));
cudaStreamCreate(&stream);

// Perform sampling with manual workspace
topKtopPSamplingFromLogits<float>(
    dLogits, dSelectedIndices, params, 
    dWorkspace, workspaceSize, stream, 42, 0);

// Cleanup
cudaFree(dWorkspace);
cudaFree(dLogits);
cudaFree(dSelectedIndices);
cudaStreamDestroy(stream);
```

### 2. Top-P Sampling Only

```cpp
// Calculate workspace size
SamplingParams params(batchSize, vocabSize, 1.0f, 0, 0.9f);  // topK=0 (disabled), topP=0.9
size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<float>(batchSize, vocabSize, params);

// Allocate workspace and buffers
void* dWorkspace;
float* dLogits;
int64_t* dSelectedIndices;
cudaStream_t stream;

cudaMalloc(&dWorkspace, workspaceSize);
cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(float));
cudaMalloc(&dSelectedIndices, batchSize * sizeof(int64_t));
cudaStreamCreate(&stream);

// Perform sampling with manual workspace
topKtopPSamplingFromLogits<float>(
    dLogits, dSelectedIndices, params, 
    dWorkspace, workspaceSize, stream, 42, 0);

// Cleanup
cudaFree(dWorkspace);
cudaFree(dLogits);
cudaFree(dSelectedIndices);
cudaStreamDestroy(stream);
```

### 3. Combined Top-K and Top-P Sampling

```cpp
// Calculate workspace size
SamplingParams params(batchSize, vocabSize, 1.0f, 40, 0.9f);  // topK=40, topP=0.9
size_t workspaceSize = getTopKtopPSamplingWorkspaceSize<float>(batchSize, vocabSize, params);

// Allocate workspace and buffers
void* dWorkspace;
float* dLogits;
int64_t* dSelectedIndices;
cudaStream_t stream;

cudaMalloc(&dWorkspace, workspaceSize);
cudaMalloc(&dLogits, batchSize * vocabSize * sizeof(float));
cudaMalloc(&dSelectedIndices, batchSize * sizeof(int64_t));
cudaStreamCreate(&stream);

// Perform sampling with manual workspace
topKtopPSamplingFromLogits<float>(
    dLogits, dSelectedIndices, params, 
    dWorkspace, workspaceSize, stream, 42, 0);

// Cleanup
cudaFree(dWorkspace);
cudaFree(dLogits);
cudaFree(dSelectedIndices);
cudaStreamDestroy(stream);
```

### 4. Using selectAllTopK with Logits Input

```cpp
// Get top-K values and indices without sampling
float* dTopKValues;
int64_t* dTopKIndices;
const int32_t topK = 10;
cudaStream_t stream;

cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float));
cudaMalloc(&dTopKIndices, batchSize * topK * sizeof(int64_t));
cudaStreamCreate(&stream);

// Calculate workspace size
size_t workspaceSize = getSelectAllTopKWorkspaceSize<float>(batchSize, vocabSize, topK);
void* workspace;
cudaMalloc(&workspace, workspaceSize);

// Get raw top-K values (input is logits)
selectAllTopKFromLogits<float>(dLogits, dTopKValues, dTopKIndices, 
                     batchSize, vocabSize, topK, workspace, workspaceSize, stream,
                     false,              // returnLogProbs
                     true,               // normalizeLogProbs
                     false);             // inputHasProbs (false for logits)

// Or get log probabilities (input is logits)
selectAllTopKFromLogits<float>(dLogits, dTopKValues, dTopKIndices, 
                     batchSize, vocabSize, topK, workspace, workspaceSize, stream,
                     true,               // returnLogProbs
                     true,               // normalizeLogProbs
                     false);             // inputHasProbs (false for logits)

cudaFree(workspace);
cudaStreamDestroy(stream);
```

### 5. Using selectAllTopK with Probabilities Input

```cpp
// Get top-K values and indices from probability input
float* dProbs;  // Pre-computed probabilities
float* dTopKValues;
int64_t* dTopKIndices;
const int32_t topK = 10;
cudaStream_t stream;

cudaMalloc(&dProbs, batchSize * vocabSize * sizeof(float));
cudaMalloc(&dTopKValues, batchSize * topK * sizeof(float));
cudaMalloc(&dTopKIndices, batchSize * topK * sizeof(int64_t));
cudaStreamCreate(&stream);

// Calculate workspace size
size_t workspaceSize = getSelectAllTopKWorkspaceSize<float>(batchSize, vocabSize, topK);
void* workspace;
cudaMalloc(&workspace, workspaceSize);

// Get raw top-K values (input is probabilities)
selectAllTopKFromLogits<float>(dProbs, dTopKValues, dTopKIndices, 
                     batchSize, vocabSize, topK, workspace, workspaceSize, stream,
                     false,              // returnLogProbs
                     true,               // normalizeLogProbs
                     true);              // inputHasProbs (true for probabilities)

// Or get log probabilities (input is probabilities)
selectAllTopKFromLogits<float>(dProbs, dTopKValues, dTopKIndices, 
                     batchSize, vocabSize, topK, workspace, workspaceSize, stream,
                     true,               // returnLogProbs
                     true,               // normalizeLogProbs
                     true);              // inputHasProbs (true for probabilities)

cudaFree(workspace);
cudaStreamDestroy(stream);
```

## Supported Data Types

The API supports the following data types:
- `float` (32-bit floating point)
- `half` (16-bit floating point)
- `__nv_bfloat16` (16-bit bfloat)

## Performance Guidelines

### Workspace Management Best Practices
- **Allocate once, reuse often**: Allocate workspace for your maximum expected batch size
- **Batch size planning**: Design your inference loop to reuse workspace across multiple calls
- **Memory efficiency**: Use the exact workspace size calculated for your parameters
- **Stream management**: Use separate streams for concurrent sampling operations

### Memory Layout and Alignment

The workspace is automatically partitioned with optimal memory alignment (256-byte alignment) for maximum GPU performance. The layout varies based on the sampling method:

- **Top-K**: Temporary logits, top-K indices, top-K values
- **Top-P**: CUB temp storage, probabilities, sorted data, offset buffers
- **Combined**: Uses top-K layout with top-P filtering

### Workspace Size Guidelines
- Top-K sampling: `O(batchSize * vocabSize + batchSize * topK)`
- Top-P sampling: `O(batchSize * vocabSize)` (larger due to sorting)
- Combined: Uses top-K layout

## Error Handling

The API includes comprehensive error handling:
- Workspace validation (size and pointer checks)
- Batch size and vocabulary size validation
- Memory allocation error handling
- CUDA error checking with detailed messages
- **Return value validation**: If `returnLogProbs=true`, `topKValues` must be non-null

## Important Notes

1. **Workspace Required**: Workspace allocation is mandatory - no automatic allocation
2. **Stream Required**: CUDA stream parameter is required for all functions
3. **Index Type**: Selected indices are returned as `int64_t*`, not `int32_t*`
4. **Value Type**: `selectAllTopKFromLogits` always returns values as `float*`, regardless of input type
5. **Parameter Setting**: Use constructor parameters directly, not setter methods
6. **Template Instantiation**: Make sure to specify the correct template parameter (`<float>`, `<half>`, `<__nv_bfloat16>`)
7. **RNG Parameters**: `philoxSeed` and `philoxOffset` control random number generation (defaults: 42, 0)
8. **Combined Sampling**: Both topK and topP can be used together for more controlled sampling
9. **Parameter Validation**: At least one of topK or topP must be active (throws exception otherwise)
10. **Input Type Flexibility**: `selectAllTopKFromLogits` supports both logits and probabilities input via `inputHasProbs` parameter
11. **Return Value Validation**: When `returnLogProbs=true`, the `topKValues` parameter must be non-null
12. **Logits Definition**: The formal definition of logits will be standardized in the next release