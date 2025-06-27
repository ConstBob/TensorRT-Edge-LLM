# DriveOS LLM SDK Example: Decoder-only Language Models

## Prerequisites

An ONNX that complies with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.

## Build Engine

The `llm_build` binary is used to build the TensorRT engine. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command:
```bash
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096 [--maxLoraRank=<max_rank> --loraWeights=<path_to_lora_weights_safetensors>]
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
2. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node. This field can be adjusted using `--max_seq_len` during ONNX export.
3. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

### Example: Build Engine

**Without dynamic LoRA:**
```bash
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096
```

**With dynamic LoRA:**
```bash
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096 --maxLoraRank=16
```

## Infer Engine

The `llm_benchmark`, `llm_accuracy` and `llm_chat` binaries are examples to show E2E C++ LLM inference using greedy decoding. Example usages:

### Chat
1. Interactive mode
    
    Chat will prompt for each batch until it has input prompt for all batches.
    ```bash
    ./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR --enginePath=llama3_fp16.engine --maxLength=64 --interactivte
    ```

2. Chat with input string
    ```bash
    ./build/examples/llm/llm_chat --tokenizerPath=$TORCH_DIR --enginePath=llama3_fp16.engine --maxLength=64 --inputString="What is deep learning?"
    ```

#### Example: Runtime Inference

**Without dynamic LoRA:**
```bash
./build/examples/llm/llm_chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64
```

**With dynamic LoRA:**
When running `--interactive` mode with LoRA, it will prompt first whether you want to switch LoRA. For non interactive mode, it will only accept 1 LoRA weights and output the result for that particular LoRA weights.
```bash
./build/examples/llm/llm_chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64 --loraWeights=my_lora:processed_lora_weights.safetensors --interactive
```

### Benchmark Performance

```bash
./build/examples/llm/llm_benchmark --enginePath=llama3_fp16.engine --maxLength=256 --inputLength=24 [--warmUp 2 --numRuns 10]
```

### Evaluate with MMLU

To run MMLU accuracy evaluation, MMLU dataset is required.

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
./build/examples/llm/llm_accuracy --tokenizerPath=llama-v3-8b-instruct-hf/  --enginePath=llama3_fp16.engine --datasetPath data
```

Python reference:
```bash
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
