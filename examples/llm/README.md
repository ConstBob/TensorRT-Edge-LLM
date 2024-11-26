# DriveOS LLM SDK Example: Decoder-only Language Models

## Prerequisite

An ONNX that complies with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.


## Build engine

The `llm_build` binary is used to build the TensorRT engine. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command:
```
./build/examples/llm/llm_build --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
1. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node.
1. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

## Infer engine

The `llm_benchmark`, `llm_accuracy` and `llm_chat` binaries are examples to show E2E C++ LLM inference using greedy decoding. Example usages:

### Interactive Chat
```
./build/examples/llm/llm_chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64
```
**Note**:
1. Chat will prompt for each batch until it has input prompt for all batches.


### Benchmark Performance

```
./build/examples/llm/llm_benchmark --enginePath=llama3_fp16.engine --maxLength=256 --inputLength=24 [--warmUp 2 --numRuns 10]
```

#### Evaluate with MMLU

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

## Performance
Attached is the performance of all supported models collected on Thor with clock fixed at 756MHz with FP16 precision.

Model | Precision | Batch Size | Input/Output | First Token Latency(ms) | Generation Tokens/sec | Total Latency
--- | --- | --- | --- | --- | --- | ---
Llama3-8b-instruct | FP16 | 1 | 512+128 | 263.26 | 15.55 | 8492.5
Llama3.1-8B | FP16 | 1 | 512+128 | 263.21 | 15.91 | 8308.82
Llama3.2-3B | FP16 | 1 | 512+128 | 119.32 | 33.87 | 3899.13
Qwen2.5-7B-instruct | FP16 | 1 | 512+128 | 256.90 | 16.29 | 7858.04
Qwen2-7B-instruct | FP16 | 1 | 512+128 | 256.75 | 15.61 | 8455.55