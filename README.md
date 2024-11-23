# DriveOS LLM SDK: TensorRT's Large Language Model Inference Sample for Auto Platform

## Introduction

DriveOS LLM SDK is a light-weighted C++ software toolkit to showcase TensorRT's capability and performance to deploy Large Language Model(LLM) targeted Auto Platform. With DriveOS LLM SDK, users can:
1. Quantize and export PyTorch model to ONNX on Linux x86 system.
1. Build TensorRT Engine and run e2e LLM inference, including tokenization and sampling on Auto Platforms.


## Prerequisite

A Linux X86 host with GPU is required to export the model into ONNX format. Once the ONNX model is exported, the only dependency is TensorRT C++ library and CUDA runtime. DriveOS LLM SDK does not have any external C++ dependency.

## Supported platforms, models and precisions

### DriveOS 7.0.1 Release for Thor
DriveOS 7.0.1 is shipped with TensorRT 10.4 and CUDA 12.8 to support **Thor** platform. The following models and precisions are currently supported with good precision to run e2e inference by DriveOS LLM SDK.

Model | FP16 | INT4 | FP8
--- | --- | --- | ---
[LLaMa3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | No
[LLaMa3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | No
[LLaMa3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | No
[QWen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | No
[QWen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | No

#### Precisions explained and notes:
1. **FP16**: All the weights and compute are in FP16.
1. **FP8(W8A8)**: All the weights and GEMM are in FP8, but KV Cache, LayerNorm, Attention and lm_head are in FP16 precision. FP8 can both reduce memory footprint and kernel performance. FP8 GEMM is not supported by TensorRT 10.4 but it will be available in a later release.
1. **INT4(W4A16)**: All the weights are quantized in INT4 using awq recipe, but all the compute are in FP16 precision. INT4 can reduce memory footprint significantly, but in TensorRT 10.4 the latency is worse than FP16 due to unfused INT4 GEMM kernels.

#### Customized Models
1. Decoder-only LLaMa series and QWen series are likely to be supported if it fits in Thor memory, but they are not fully tested.
1. Other model series will likely not be supported due to the Tokenizer implementation and model architecture difference.

### Other Platforms
1. DriveOS LLM SDK does not support TensorRT 8.6 and therefore cannot be compatible with Orin and any DriveOS 6.
1. As a preview feature, DriveOS LLM SDK can also run on x86 Linux Data Center or Gaming GPUS with SM80, SM86 or SM89 in FP16 and INT4. SM89 also supports FP8 E2E inference. However, no model support is guaranteed and the resulting performance will not be comparable to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). It is recommended to deploy TensorRT-LLM for all Data Center use cases.


## Getting started

### 1. Build the C++ project

The C++ project can be built in Linux x86 host with cross build.

```
cd drive-llm
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_cross_toolchain.cmake
make
```

To build and run DriveOS LLM SDK in x86 machine, the `-DCMAKE_TOOCHAIN_FILE` is not needed. The binaries `builder`, `runtime` and `chat` generated to be used later. The AttentionPlugin library will also be there in `libAttentionPlugin.so`

### 2. Export ONNX from PyTorch checkpoint

First, it is needed to export the PyTorch model to ONNX on a x86 Linux host with GPU. If quantization is needed, it is recommended (or even required) to use a Data Center GPU like H100. Please see [README.MD](./export/README.md) for the detailed model export process. Once the ONNX model is available, no Python will be needed.

### 3. Build engine

The `builder` binary is used to build the TensorRT engine. All the ONNX have the same IO name and data type, so the building process is agnostic for all ONNX independent of model and precision.

Example command:
```
./build/builder --onnxPath=llama3_fp16/model.onnx --enginePath=llama3_fp16.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=4096
```

**Notes:**
1. `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.
1. Please notice that `maxSeqLen` must be identical to `kv_cache_capacity` field of the ONNX `AttentionPlugin` node.
1. We can support static multi-batch `batchSize < max_batch_size` field of ONNX `AttentionPlugin` node.

### 4. Infer engine

The `runtime` or `chat` binaries are examples to show E2E C++ LLM inference using greedy decoding. Example usages:

#### Interactive Chat
```
./build/chat --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama3_fp16.engine --maxLength=64
```
**Note**:
1. Chat will prompt for each batch until it has input prompt for all batches.

#### Inference with prompt

```
./build/runtime --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama.engine --maxLength=256 --inputString="What is the result of 1+1?" [--inputString="Where is Iceland?"]
```
**Note**:
1. To run inference with prompt for multi-batch engine, there should be exact n times `--inputString` inputs.

#### Benchmark Performance

```
./build/runtime --enginePath=llama.engine --maxLength=256 --inputLength=24 --mode=benchmark
```

#### Evaluate with MMLU

To run MMLU accuracy evaluation, it is first required to download the dataset.

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
./runtime --tokenizerPath=/home/scratch.trt_llm_data/llm-models/llama-models-v3/llama-v3-8b-instruct-hf/  --enginePath=llama.engine --mode evaluate --datasetPath ../data --debug
```

Python reference

```
python scripts/mmlu.py
```
