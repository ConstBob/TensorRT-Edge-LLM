# DriveOS LLM SDK Example: Decoder-only Language Models

## Prerequisite

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
