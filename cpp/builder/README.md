# TensorRT Edge LLM Builder

C++ API for building TensorRT engines from ONNX models for Large Language Models and visual encoders.

## Overview

The builder provides two main classes in the `drivellm::builder` namespace:
- `LLMBuilder`: For LLM engines (standard, Eagle, VLM, LoRA)
- `VisualBuilder`: For visual encoder engines

## API Reference

### LLMBuilderConfig

```cpp
struct LLMBuilderConfig {
    int64_t maxInputLen{128};           //!< Maximum input sequence length
    bool isVlm{false};                  //!< Whether this is a VLM
    int64_t minImageTokens{4};          //!< Min image tokens (VLM only)
    int64_t maxImageTokens{1024};       //!< Max image tokens (VLM only)
    bool eagleDraft{false};             //!< Whether this is an Eagle draft model
    bool eagleBase{false};              //!< Whether this is an Eagle base model
    int64_t maxBatchSize{4};            //!< Maximum batch size
    int64_t maxLoraRank{0};             //!< Maximum LoRA rank (0 = no LoRA)
    int64_t maxSeqLen{4096};            //!< Maximum sequence length
    int64_t maxDecodingTokens{60};      //!< Max decoding tokens (Eagle)
    int64_t maxDraftTokensPerStep{60};  //!< Max draft tokens per step (Eagle)
};
```

### VisualBuilderConfig

```cpp
struct VisualBuilderConfig {
    int64_t minImageTokens{4};      //!< Minimum number of image tokens
    int64_t maxImageTokens{1024};   //!< Maximum number of image tokens
    int64_t maxImageTokensPerImage{512}; //!< Maximum number of image tokens per image, used for preprocessing
};
```

### LLMBuilder

```cpp
class LLMBuilder {
public:
    LLMBuilder(std::filesystem::path const& onnxDir, 
               std::filesystem::path const& engineDir, 
               LLMBuilderConfig const& config);
    bool build();
};
```

### VisualBuilder

```cpp
class VisualBuilder {
public:
    VisualBuilder(std::filesystem::path const& onnxDir, 
                  std::filesystem::path const& engineDir,
                  VisualBuilderConfig const& config);
    bool build();
};
```

## Usage Examples

### Standard LLM

```cpp
#include "builder.h"
using namespace drivellm::builder;

LLMBuilderConfig config;
config.maxInputLen = 512;
config.maxBatchSize = 8;
config.maxSeqLen = 2048;
config.maxLoraRank = 16;

LLMBuilder builder("/path/to/onnx", "/path/to/engine", config);
if (!builder.build()) {
    std::cerr << "Build failed!" << std::endl;
    return -1;
}
```

### Eagle Draft Model

```cpp
LLMBuilderConfig config;
config.eagleDraft = true;
config.maxInputLen = 512;
config.maxBatchSize = 4;
config.maxSeqLen = 2048;

LLMBuilder builder("/path/to/onnx", "/path/to/engine", config);
builder.build();
```

### VLM Model

```cpp
LLMBuilderConfig config;
config.isVlm = true;
config.maxInputLen = 512;
config.maxBatchSize = 4;
config.minImageTokens = 256;
config.maxImageTokens = 1024;

LLMBuilder builder("/path/to/onnx", "/path/to/engine", config);
builder.build();
```

### Visual Encoder

```cpp
VisualBuilderConfig config;
config.minImageTokens = 256;
config.maxImageTokens = 1024;

VisualBuilder builder("/path/to/onnx", "/path/to/engine", config);
builder.build();
```

## Supported Models

### LLM Models
- Standard Transformer LLMs
- Eagle3 (draft/base)
- Vision-Language Models (VLM)
- LoRA-enabled models

### Visual Models
- Qwen2-VL, Qwen2.5-VL
- InternVL Vision

## Input Requirements

### ONNX Directory
- `model.onnx`: ONNX model file
- `config.json`: Model configuration
- `tokenizer_config.json`, `tokenizer.json`: For non-draft models
- `d2t.safetensors`: For Eagle3 draft models

### Config.json Example

```json
{
  "hidden_size": 4096,
  "num_key_value_heads": 32,
  "num_attention_heads": 32,
  "max_position_embeddings": 32768,
  "num_hidden_layers": 32,
  "vision_config": {
    "model_type": "qwen2_vl"
  }
}
```

## Output Files

### LLM Engines
- `llm.engine` / `eagle_base.engine` / `eagle_draft.engine`
- `config.json` (with builder config)
- `tokenizer_config.json`, `tokenizer.json`
- `d2t.safetensors` (Eagle3 draft)

### Visual Engines
- `visual.engine`
- `config.json` (with builder config)

## Build Process

1. Load TensorRT Edge LLM plugin library
2. Parse model configuration from config.json
3. Create TensorRT network from ONNX model
4. Set up optimization profiles (context/generation)
5. Build TensorRT engine
6. Copy necessary files to engine directory
7. Save builder configuration

## Dependencies

- TensorRT
- ONNX Runtime
- nlohmann/json
- C++17
- CUDA
