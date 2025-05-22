# DriveOS LLM SDK Eagle Example: this document shows how to build and run a model using EAGLE decoding ([Github](https://github.com/SafeAILab/EAGLE/tree/main), [BLOG](https://sites.google.com/view/eagle-llm)). Different from other models, EAGLE decoding needs a base model and an EAGLE draft model. And both EAGLE2 and EAGLE3 are supported in DriveLLM with bath size equals 1.

## Prerequisite

ONNXS both for base model and draft model that compile with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.

## Eagle3 Usage
    In this example, we use the model from HuggingFace yuhuili/EAGLE3-LLaMA3.1-Instruct-8B (https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), which is a LLAMA-based model.

### Build base and draft model engines

The `llm_eagle_build` binary is used to build the TensorRT engine under EAGLE decoding. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command for building base model engine:
```
export EAGLE3_ONNX_BASE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Base"
export EAGLE3_BASE_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Base-Engine"
./build/examples/eagle/llm_eagle_build --onnxPath=$EAGLE3_ONNX_BASE_DIR/model.onnx --enginePath=$EAGLE3_BASE_ENGINE_DIR/model_fp16_base_eagle3.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --eagleBase --eagle3
```

Example command for building draft model engine:
```
export EAGLE3_ONNX_DRAFT_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Draft"
export EAGLE3_DRAFT_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle3-Draft-Engine"
./build/examples/eagle/llm_eagle_build --onnxPath=$EAGLE3_ONNX_DRAFT_DIR/model.onnx --enginePath=$EAGLE3_DRAFT_ENGINE_DIR/model_fp16_draft_eagle3.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --eagleDraft --eagle3
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
1. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node. This field can be adjusted using `--max_seq_len` during ONNX export.
1. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

### Infer engine

The `llm_eagle_benchmark`, `llm_eagle_accuracy` and `llm_eagle_chat` binaries are examples to show E2E C++ LLM inference using greedy decoding. Example usages:
```
export TORCH_DIR="Meta-Llama-3.1-8B-Instruct"
export EAGLE3_BASE_ENGINE_PATH=$EAGLE3_BASE_ENGINE_DIR/model_fp16_base_eagle3.engine
export EAGLE3_DRAFT_ENGINE_PATH=$EAGLE3_DRAFT_ENGINE_DIR/model_fp16_draft_eagle3.engine
```
#### Interactive Chat
```
./build/examples/eagle/llm_eagle_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--maxLength=1024 --isEagle3
```
**Note**:
1. Chat will prompt for each batch until it has input prompt for all batches.

#### Chat with a spectific input
```
./build/examples/eagle/llm_eagle_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--maxLength=1024 --isEagle3 --inputString "<|begin_of_text|>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello ASSISTANT:"
```
#### Benchmark Performance

```
./build/examples/eagle/llm_eagle_benchmark \
--enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH \
--tokenizerPath=$TORCH_DIR \
--maxLength=2048 --isEagle3 [--warmUp 2 --numRuns 10] 
```

#### Evaluate with MMLU

To run MMLU accuracy evaluation, MMLU dataset is required.

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
./build/examples/eagle/llm_eagle_accuracy --tokenizerPath=$TORCH_DIR  --enginePath=$EAGLE3_BASE_ENGINE_PATH --eagleEnginePath=$EAGLE3_DRAFT_ENGINE_PATH --datasetPath data --isEagle3
```
## Eagle2 Usage
    In this example, we use the model from HuggingFace yuhuili/EAGLE-LLaMA3.1-Instruct-8B (https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B/tree/main), which is a LLAMA-based model.

### Build base and draft model engines
Example command for building base model engine:
```
export EAGLE2_ONNX_BASE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Base"
export EAGLE2_BASE_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Base-Engine"
./build/examples/eagle/llm_eagle_build --onnxPath=$EAGLE2_ONNX_BASE_DIR/model.onnx --enginePath=$EAGLE2_BASE_ENGINE_DIR/model_fp16_base_eagle.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --eagleBase
```

Example command for building draft model engine:
```
export EAGLE2_ONNX_DRAFT_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Draft"
export EAGLE2_DRAFT_ENGINE_DIR="Meta-Llama-3.1-8B-Instruct-Eagle-Draft-Engine"
./build/examples/eagle/llm_eagle_build --onnxPath=$EAGLE2_ONNX_DRAFT_DIR/model.onnx --enginePath=$EAGLE2_DRAFT_ENGINE_DIR/model_fp16_draft_eagle.engine --batchSize=1 --maxInputLen=1024 --maxSeqLen=4096 --eagleDraft
```
### Infer engine
```
export TORCH_DIR="Meta-Llama-3.1-8B-Instruct"
export EAGLE2_BASE_ENGINE_DIR=$EAGLE_BASE_ENGINE_DIR/model_fp16_base_eagle.engine
export EAGLE2_DRAFT_ENGINE_DIR=$EAGLE_DRAFT_ENGINE_DIR/model_fp16_draft_eagle.engine
```
#### Interactive Chat
```
./build/examples/eagle/llm_eagle_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE2_BASE_ENGINE_DIR --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_DIR \
--maxLength=1024
```

#### Chat with a specific input string
```
./build/examples/eagle/llm_eagle_chat --tokenizerPath=$TORCH_DIR \
--enginePath=$EAGLE2_BASE_ENGINE_DIR --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_DIR \
--maxLength=1024 --inputString "<|begin_of_text|>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello ASSISTANT:"
```
#### Benchmark Performance

```
./build/examples/eagle/llm_eagle_benchmark \
--enginePath=$EAGLE2_BASE_ENGINE_DIR --eagleEnginePath=$EAGLE2_DRAFT_ENGINE_DIR \
--tokenizerPath=$TORCH_DIR \
--maxLength=256 [--warmUp 2 --numRuns 10] 
```

