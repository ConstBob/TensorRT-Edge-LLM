# DriveOS LLM SDK: TensorRT's Large Language Model Inference Sample for Auto Platform

## Introduction

DriveOS LLM SDK is a light-weighted C++ software toolkit to showcase TensorRT's capability and performance to deploy Large Language Models(LLMs) and Vision Language Models(VLMs) targeted Auto Platform. With DriveOS LLM SDK, users can:
1. Quantize and export PyTorch model to [ONNX Format](https://onnx.ai/) on Linux x86 system.
1. Build TensorRT Engine and run e2e LLM inference, including tokenization and sampling, on Auto Platform.


## Prerequisite

A Linux X86 host with GPU is required to export the model into ONNX format. Once the ONNX model is exported, the only dependency is TensorRT C++ library and CUDA runtime. DriveOS LLM SDK does not have any external C++ dependencies.

## Supported platforms, models and precisions

The following LLM models under [./examples/llm](./examples/llm/) with corresponding precisions are supported by DriveOS LLM SDK:

Model | FP16 | INT4 | FP8 | NVFP4
--- | --- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes | Yes
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes | Yes
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes | Yes
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes | Yes

The following VLM models under [./examples/vlm](./examples/vlm/) with corresponding precisions are supported by DriveOS LLM SDK with good accuracy. Note that the ViT will always be in FP16 precision.

Model | FP16 | INT4 | FP8 | NVFP4
--- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes


#### Precisions explained and notes:
1. **FP16**: All the weights and compute are in FP16.
1. **FP8(W8A8)**: All the weights and GEMMs are in FP8, but KV Cache, LayerNorm, Attention and lm_head are in FP16 precision. FP8 can both reduce memory footprint and improve inference latency.
1. **INT4(W4A16)**: All the weights are quantized in INT4 using awq recipe, but all the compute are in FP16 precision. INT4 can reduce memory footprint and improve significantly by reducing weights loading by 4x compared to FP16. Note that because TensorRT native(Or out-of-the-box or ootb) INT4 kernels have some performance issues, a [Int4GroupwiseGemmPlugin](./cpp/int4GroupwiseGemmPlugin/) is provided as the default option for INT4.
1. **NVFP4(W4A4)**: Similar to FP8, all the weights and GEMMs are in NVFP4 while the other parts are in FP16 precision. NVFP4 can significantly reduce memory footprint and improve inference latency, especially context phase. Current generation phase performance of NVFP4 GEMM is good but has room for improvements. The improvements will be shipped in the next few releases.

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
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_cross_toolchain.cmake -DAUTO_TARGET=thor
make
```

Built in Linux aarch64 with native build
```
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_native_toolchain.cmake -DAUTO_TARGET=thor
make
```

To build and run DriveOS LLM SDK in x86 machine, the `-DCMAKE_TOOCHAIN_FILE` is not needed. The binaries are generated in `examples` folder to be used later. The AttentionPlugin library will also be there in `libAttentionPlugin.so`.

### 2. Export ONNX from PyTorch checkpoint

First, it is needed to export the PyTorch model to ONNX on a x86 Linux host with GPU. If quantization is needed, it is recommended (or even required) to use a Data Center GPU like H100. Please see [export/README.md](./export/README.md) for the detailed model export process. Once the ONNX model is available, no Python will be needed.

### 3. Build engine and run E2E LLM inference on C++

Once the model is exported, you can follow the examples to build and run E2E LLM inference with C++. Please follow [examples/llm/README.md](./examples/llm/README.md) for decoder-only LLMs and [examples/multimodal/README.md](./examples/multimodal/README.md) for VLMs. The cpp files under [examples](./examples/) folder showcase the usage of the DriveOS LLM SDK runtime.
