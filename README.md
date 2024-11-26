# DriveOS LLM SDK: TensorRT's Large Language Model Inference Sample for Auto Platform

## Introduction

DriveOS LLM SDK is a light-weighted C++ software toolkit to showcase TensorRT's capability and performance to deploy Large Language Model(LLM) targeted Auto Platform. With DriveOS LLM SDK, users can:
1. Quantize and export PyTorch model to ONNX on Linux x86 system.
1. Build TensorRT Engine and run e2e LLM inference, including tokenization and sampling on Auto Platform.


## Prerequisite

A Linux X86 host with GPU is required to export the model into ONNX format. Once the ONNX model is exported, the only dependency is TensorRT C++ library and CUDA runtime. DriveOS LLM SDK does not have any external C++ dependency.

## Supported platforms, models and precisions

### DriveOS 7.0.1 Release for Thor
DriveOS 7.0.1 is shipped with TensorRT 10.4 and CUDA 12.8 to support **Thor** platform.

The following LLM models under [./examples/llm](./examples/llm/) with corresponding precisions are supported by DriveOS LLM SDK with good accuracy:

Model | FP16 | INT4 | FP8
--- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | No
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | No
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | No
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | No
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | No

The following VLM models under [./examples/vlm](./examples/vlm/) with corresponding precisions are supported by DriveOS LLM SDK with good accuracy:

**TBA**

#### Precisions explained and notes:
1. **FP16**: All the weights and compute are in FP16.
1. **FP8(W8A8)**: All the weights and GEMM are in FP8, but KV Cache, LayerNorm, Attention and lm_head are in FP16 precision. FP8 can both reduce memory footprint and kernel performance. FP8 GEMM is not supported by TensorRT 10.4 but it will be available in a later release.
1. **INT4(W4A16)**: All the weights are quantized in INT4 using awq recipe, but all the compute are in FP16 precision. INT4 can reduce memory footprint significantly, but in TensorRT 10.4 the latency is worse than FP16 due to unfused INT4 GEMM kernels.

#### Customized Models
1. Decoder-only Llama series and Qwen series are likely to be supported if it fits in Thor memory, but they are not fully tested.
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

To build and run DriveOS LLM SDK in x86 machine, the `-DCMAKE_TOOCHAIN_FILE` is not needed. The binaries are generated in `examples` folder to be used later. The AttentionPlugin library will also be there in `libAttentionPlugin.so`.

### 2. Export ONNX from PyTorch checkpoint

First, it is needed to export the PyTorch model to ONNX on a x86 Linux host with GPU. If quantization is needed, it is recommended (or even required) to use a Data Center GPU like H100. Please see [export/README.md](./export/README.md) for the detailed model export process. Once the ONNX model is available, no Python will be needed.

### 3. Build engine and run E2E LLM inference on C++

Once the model is exported, you can follow the examples to build and run E2E LLM inference with C++. Please follow [examples/llm/README.md](./examples/llm/README.md) for decoder-only LLMs. The cpp files under [examples](./examples/) folder show the usage of the DriveOS LLM SDK runtime.
