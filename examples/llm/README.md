# TensorRT Edge-LLM Example: Decoder-only Language Models

## Prerequisites

An ONNX model that complies with the TensorRT Edge-LLM runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.

### ONNX Folder Structure

The ONNX export process creates the following folder structure:

```
onnx_models/${MODEL_NAME}/
├── model.onnx                    # Main model ONNX file
├── config.json                   # Model configuration
├── tokenizer_config.json         # Tokenizer configuration
└── tokenizer.json               # Tokenizer vocabulary
```

## Engine Build

The `llm_build` binary is used to build TensorRT engines. All ONNX models have the same IO names and data types, making the building process agnostic to model and precision.

### Standard LLM (Naive Decoding)

```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME} \
--engineDir=engines/${MODEL_NAME} \
--maxBatchSize=1 \
--maxInputLen=128 \
--maxSeqLen=4096
```

### EAGLE Decoding

For EAGLE speculative decoding, build separate engines for base and draft models:

```bash
# Build base model engine
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}_eagle3_base \
--engineDir=engines/${MODEL_NAME}_eagle3_base \
--maxBatchSize=1 \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--isEagleBase \
--isEagle3

# Build draft model engine
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}_eagle3_draft \
--engineDir=engines/${MODEL_NAME}_eagle3_draft \
--maxBatchSize=1 \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--isEagleDraft \
--isEagle3
```

**Notes:**
- `--maxSeqLen` must be greater than `--maxInputLen`
- `maxSeqLen` must match the `kv_cache_capacity` field in the ONNX `AttentionPlugin` node

## Required Folder Structure

### Standard LLM Engine

```
engines/${MODEL_NAME}/
├── llm.engine                    # TensorRT engine file
├── config.json                   # Model configuration
├── tokenizer_config.json         # Tokenizer configuration
└── tokenizer.json               # Tokenizer vocabulary
```

### EAGLE Engine Structure

```
engines/${MODEL_NAME}_base/
├── eagle_base.engine            # Base model TensorRT engine
├── base_config.json             # Base model configuration
├── tokenizer_config.json        # Tokenizer configuration (shared)
└── tokenizer.json              # Tokenizer vocabulary (shared)

engines/${MODEL_NAME}_draft/
├── eagle_draft.engine           # Draft model TensorRT engine
├── draft_config.json            # Draft model configuration
└── d2t.bin                     # Draft-to-target mapping (Eagle3 only, optional)
```

## Running Inference

### Chat Interface

**Interactive Mode:**
```bash
./build/examples/llm/llm_chat \
--engineDir=engines/${MODEL_NAME} \
--maxLength=64 \
--interactive
```

**With Input String:**
```bash
./build/examples/llm/llm_chat \
--engineDir=engines/${MODEL_NAME} \
--maxLength=64 \
--inputString="What is deep learning?"
```

**EAGLE Chat:**
```bash
./build/examples/llm/llm_chat \
--baseModelDir=engines/${MODEL_NAME}_base \
--draftModelDir=engines/${MODEL_NAME}_draft \
--maxLength=1024 \
--isEagle3 \
--interactive
```

### Benchmark Performance

**Standard LLM:**
```bash
./build/examples/llm/llm_benchmark \
--engineDir=engines/${MODEL_NAME} \
--maxLength=256 \
--inputLength=24 \
--warmUp 2 \
--numRuns 10
```

**EAGLE:**
```bash
./build/examples/llm/llm_benchmark \
--baseModelDir=engines/${MODEL_NAME}_base \
--draftModelDir=engines/${MODEL_NAME}_draft \
--maxLength=2048 \
--isEagle3 \
--warmUp 2 \
--numRuns 5
```

### Accuracy Evaluation (MMLU)

**Download Dataset:**
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
```

**Standard LLM:**
```bash
./build/examples/llm/llm_accuracy \
--engineDir=engines/${MODEL_NAME} \
--datasetPath data
```

**EAGLE:**
```bash
./build/examples/llm/llm_accuracy \
--baseModelDir=engines/${MODEL_NAME}_base \
--draftModelDir=engines/${MODEL_NAME}_draft \
--datasetPath data \
--isEagle3
```

## Runtime LoRA Switching

TensorRT Edge-LLM supports dynamic LoRA (Low-Rank Adaptation) for efficient model adaptation.

### LoRA Weights Processing

Before using LoRA weights, process them using the provided script:

```bash
python export/process_lora_weights.py --input_dir /path/to/lora/adapter --output_dir /path/to/processed/lora --dtype fp16
```

### Build with LoRA Support

Add `--maxLoraRank=<max_rank>` to enable dynamic LoRA support:

```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME} \
--engineDir=engines/${MODEL_NAME} \
--batchSize=1 \
--maxInputLen=128 \
--maxSeqLen=4096 \
--maxLoraRank=16
```

### Runtime LoRA Usage

Use `--loraWeights=name:path_to_lora_weights.safetensors` in any inference binary:

```bash
./build/examples/llm/llm_chat \
--engineDir=engines/${MODEL_NAME} \
--maxLength=64 \
--loraWeights=my_lora:processed_lora_weights.safetensors
```


