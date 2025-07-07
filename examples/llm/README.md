# DriveOS LLM SDK Example: Decoder-only Language Models

## Prerequisites

An ONNX that complies with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.


## Build engine

The `llm_build` binary is used to build the TensorRT engine. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command:
```
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096 --maxLoraRank=<max_rank> --loraWeights=<path_to_lora_weights_safetensors>
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
1. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node. This field can be adjusted using `--max_seq_len` during ONNX export.
1. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

### Example: Build Engine

**Without LoRA:**
```
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096
```
**With dynamic LoRA:**
```
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096 --maxLoraRank=16
```

## Infer engine

The `llm_benchmark`, `llm_accuracy` and `llm_chat` binaries are examples to show E2E C++ LLM inference using greedy decoding. Example usages:

### Chat
1. Interactive mode
    
    Chat will prompt for each batch until it has input prompt for all batches.
    ```
    ./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR --enginePath=llama3_fp16.engine --maxLength=64 --interactivte
    ```
2. Chat with input string
    ```
    ./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR --enginePath=llama3_fp16.engine --maxLength=64 --inputString="What is deep learning?"
    ```

#### Example: Runtime Inference

**Without LoRA:**
```
./build/examples/llm/llm_chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64
```
**With dynamic LoRA:**
```
./build/examples/llm/llm_chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64 --loraWeights=my_lora:processed_lora_weights.safetensors
```

### Benchmark Performance

```
./build/examples/llm/llm_benchmark --enginePath=llama3_fp16.engine --maxLength=256 --inputLength=24 [--warmUp 2 --numRuns 10]
```

### Evaluate with MMLU

To run MMLU accuracy evaluation, MMLU dataset is required.

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
./build/examples/llm/llm_accuracy --tokenizerPath=llama-v3-8b-instruct-hf/  --enginePath=llama3_fp16.engine --datasetPath data
```

Python reference

```
python scripts/mmlu.py
```

## LoRA Support

DriveOS LLM SDK supports dynamic LoRA (Low-Rank Adaptation) for efficient model adaptation.

- **Build-time:** Use `--maxLoraRank=<max_rank>` in `llm_build` to enable dynamic LoRA support.
- **Runtime:** Use `--loraWeights=name:path_to_lora_weights.safetensors` in `llm_chat`, `llm_benchmark`, or `llm_accuracy` to load LoRA weights.

> **Warning:**
> - You must process LoRA weights using `export/process_lora_weights.py` before use.
> - You are responsible for ensuring the LoRA weights are valid and compatible.
> - For static/merged LoRA, do **not** use these flags.

## Eagle2&3 Support
This part shows how to build and run a model using EAGLE decoding ([Github](https://github.com/SafeAILab/EAGLE/tree/main), [BLOG](https://sites.google.com/view/eagle-llm)). Different from other models, EAGLE decoding needs a base model and an EAGLE draft model. And both EAGLE2 and EAGLE3 are supported in DriveLLM with bath size equals 1.

## Prerequisite

ONNXS both for base model and draft model that compile with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.

## Eagle3 Usage
    In this example, we use the model from HuggingFace yuhuili/EAGLE3-LLaMA3.1-Instruct-8B (https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), which is a LLAMA-based model.

### Build base and draft model engines

The `llm_build` binary is used to build the TensorRT engine under EAGLE decoding. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command for building base model engine:
```
export EAGLE3_ONNX_BASE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Base"
export EAGLE3_BASE_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Base-Engine"
./build/examples/llm/llm_build --onnxPath=$EAGLE3_ONNX_BASE_DIR/model.onnx --enginePath=$EAGLE3_BASE_ENGINE_DIR/model_fp16_base_eagle3.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --isEagleBase --isEagle3
```

Example command for building draft model engine:
```
export EAGLE3_ONNX_DRAFT_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Draft"
export EAGLE3_DRAFT_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Draft-Engine"
./build/examples/llm/llm_build --onnxPath=$EAGLE3_ONNX_DRAFT_DIR/model.onnx --enginePath=$EAGLE3_DRAFT_ENGINE_DIR/model_fp16_draft_eagle3.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --isEagleDraft --isEagle3
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
1. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node. This field can be adjusted using `--max_seq_len` during ONNX export.
1. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

### Infer engine

The `llm_benchmark`, `llm_accuracy` and `llm_chat` binaries are examples to show E2E C++ LLM inference using eagle decoding. Example usages:
```
export TORCH_DIR="Meta-Llama-3.1-8B-Instruct"
export EAGLE3_BASE_ENGINE_PATH=$EAGLE3_BASE_ENGINE_DIR/model_fp16_base_eagle3.engine
export EAGLE3_DRAFT_ENGINE_PATH=$EAGLE3_DRAFT_ENGINE_DIR/model_fp16_draft_eagle3.engine
```
#### Interactive Chat
```
./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--maxLength=1024 --isEagle3 --interactivte
```
**Note**:
1. Chat will prompt for each batch until it has input prompt for all batches.

#### Chat with a spectific input
```
./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--maxLength=1024 --isEagle3 --inputString "<|begin_of_text|>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello ASSISTANT:"
```
#### Benchmark Performance

```
./build/examples/llm/llm_benchmark \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--tokenizerPath=$TORCH_DIR \
--maxLength=2048 --isEagle3 [--warmUp 2 --numRuns 5] 
```

#### Evaluate with MMLU

To run MMLU accuracy evaluation, MMLU dataset is required.

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
./build/examples/llm/llm_accuracy --tokenizerPath=$TORCH_DIR  --enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH --datasetPath data --isEagle3
```
## Eagle2 Usage
    In this example, we use the model from HuggingFace yuhuili/EAGLE-LLaMA3.1-Instruct-8B (https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B/tree/main), which is a LLAMA-based model.

### Build base and draft model engines
Example command for building base model engine:
```
export EAGLE2_ONNX_BASE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Base"
export EAGLE2_BASE_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Base-Engine"
./build/examples/llm/llm_build --onnxPath=$EAGLE2_ONNX_BASE_DIR/model.onnx --enginePath=$EAGLE2_BASE_ENGINE_DIR/model_fp16_base_eagle.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --isEagleBase
```

Example command for building draft model engine:
```
export EAGLE2_ONNX_DRAFT_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Draft"
export EAGLE2_DRAFT_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Draft-Engine"
./build/examples/llm/llm_build --onnxPath=$EAGLE2_ONNX_DRAFT_DIR/model.onnx --enginePath=$EAGLE2_DRAFT_ENGINE_DIR/model_fp16_draft_eagle.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --isEagleDraft
```
### Infer engine
```
export TORCH_DIR="Meta-Llama-3.1-8B-Instruct"
export EAGLE2_BASE_ENGINE_PATH=$EAGLE2_BASE_ENGINE_DIR/model_fp16_base_eagle.engine
export EAGLE2_DRAFT_ENGINE_PATH=$EAGLE2_DRAFT_ENGINE_DIR/model_fp16_draft_eagle.engine
```
#### Interactive Chat
```
./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE2_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_PATH \
--maxLength=1024 --interactivte
```

#### Chat with a specific input string
```
./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE2_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_PATH \
--maxLength=1024 --inputString "<|begin_of_text|>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello ASSISTANT:"
```
#### Benchmark Performance

```
./build/examples/llm/llm_benchmark \
--enginePath=$EAGLE2_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_PATH \
--tokenizerPath=$TORCH_DIR \
--maxLength=256 [--warmUp 2 --numRuns 5] 
```


