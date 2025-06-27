# DriveOS LLM SDK: TensorRT's Large Language Model Inference Framework for Auto Platforms

## Introduction

DriveOS LLM SDK is a light-weighted C++ software toolkit to showcase TensorRT's capability and performance to deploy Large Language Models(LLMs) and Vision Language Models(VLMs) targeted Auto Platform. With DriveOS LLM SDK, users can:
1. Quantize and export PyTorch model to [ONNX Format](https://onnx.ai/) on Linux x86 system.
2. Build TensorRT Engine and run e2e LLM inference, including tokenization and sampling, on Auto Platform.

## Prerequisites

- A Linux X86 host with GPU is required to export the model into ONNX format
- Once the ONNX model is exported, the only dependency is TensorRT C++ library and CUDA runtime

## Supported Platforms, Models and Precisions

The following LLM models under [./examples/llm](./examples/llm/) with corresponding precisions are supported by DriveOS LLM SDK:

Model | FP16 | INT4 | FP8 | NVFP4
--- | --- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes | Yes
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes | Yes
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes | Yes
[Qwen2-0.5B-instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-1.5B-instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes | Yes

The following VLM models under [./examples/vlm](./examples/vlm/) with corresponding precisions are supported by DriveOS LLM SDK. Note that the ViT will always be in FP16 precision.

Model | FP16 | INT4 | FP8 | NVFP4
--- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Yes | Yes | Yes | Yes

### Precisions Explained

1. **FP16**: All the weights and compute are in FP16.
2. **FP8(W8A8)**: All the weights and GEMMs are in FP8, but KV Cache, LayerNorm, Attention and lm_head are in FP16 precision. FP8 can both reduce memory footprint and improve inference latency.
3. **INT4(W4A16)**: All the weights are quantized in INT4 using awq recipe, but all the compute are in FP16 precision. INT4 can reduce memory footprint and improve significantly by reducing weights loading by 4x compared to FP16. Note that because TensorRT native(Or out-of-the-box or ootb) INT4 kernels have some performance issues, a [Int4GroupwiseGemmPlugin](./cpp/int4GroupwiseGemmPlugin/) is provided as the default option for INT4.
4. **NVFP4(W4A4)**: Recommended precision on Thor. Similar to FP8, all the weights and GEMMs are in NVFP4 while the other parts are in FP16 precision. NVFP4 can significantly reduce memory footprint and improve inference latency, especially context phase. NVFP4 has huge advantage in context phase performance compared with INT4 and FP8.

### Customized Models

1. Decoder-only models with structure similar to Llama or Qwen are likely to be supported if it fits in Thor memory, but they are not fully tested. For VLM, visual encoder part is likely to be supported by native TensorRT and the LLM part applies to the same rule as pure LLM models.
2. Other model series might not be supported due to Attention plugin, tokenizer implementation and model architecture difference.

### Other Platforms

1. DriveOS LLM SDK for Orin is supported in a different package version. This package support TensorRT10 only and therefore cannot be compatible with Orin and any DriveOS 6.
2. As a preview feature, DriveOS LLM SDK can also run on x86 Linux Data Center or Gaming GPUS with SM80, SM86 or SM89 in FP16 and INT4. SM89 also supports FP8 E2E inference. However, no model support is guaranteed and the resulting performance will not be comparable to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). It is recommended to deploy TensorRT-LLM for all Data Center use cases.

## Getting Started

### 1. Build the C++ Project

The C++ project can be built in Linux x86 host with cross build:

```bash
cd drive-llm
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_cross_toolchain.cmake -DAUTO_TARGET=thor
make
```

Built in Linux aarch64 with native build:

```bash
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_native_toolchain.cmake -DAUTO_TARGET=thor
make
```

To build and run DriveOS LLM SDK in x86 machine for rapid development, the `-DCMAKE_TOOCHAIN_FILE` and `-DAUTO_TARGET` is not needed. The binaries are generated in `examples` folder to be used later. The AttentionPlugin library will also be there in `libAttentionPlugin.so`, and the Int4GemmPlugin in `libint4GemmPlugin.so`.

### 2. Export ONNX from PyTorch Checkpoint

First, it is needed to export the PyTorch model to ONNX on a x86 Linux host with GPU. If quantization is needed, it is recommended (or even required) to use a Data Center GPU like H100. Please see [export/README.md](./export/README.md) for the detailed model export process. Once the ONNX model is available, no Python will be needed.

### 3. Build Engine and Run E2E LLM Inference on C++

Once the model is exported, you can follow the examples to build and run E2E LLM inference with C++. Please follow [examples/llm/README.md](./examples/llm/README.md) for decoder-only LLMs and [examples/multimodal/README.md](./examples/multimodal/README.md) for VLMs. The cpp files under [examples](./examples/) folder showcase the usage of the DriveOS LLM SDK runtime.

## Limitations and Known Issues

### Python Export

1. `nvidia-modelopt>0.19.0` has accuracy issues for INT4 recipe, you need to downgrade `nvidia-modelopt` to 0.19.0 for INT4 export:
```bash
pip3 uninstall nvidia-modelopt
pip3 install -r requirements_int4.txt
```

2. Qwen2.5-VL 3B VIT with FP16 precision has occasional overflow issue from the last transformer block and we observed the same issue with HuggingFace using Pytorch backend. We applied a work-around to cast the last down_proj to FP32 in [multimodal_export.py](export/multimodal_export.py).

### Inference

3. There is a known issue up to DriveOS 7.0.3.0 that `cudaMallocAsync` will fail when allocated memory size is large (>~5G). If you build an engine that is larger than 5GB, it will fail to load the engine. Please use this as a WAR to prevent this issue. DriveOS team is aware of this issue and will fix it in the next release:
```bash
echo 24576 | sudo tee /proc/sys/vm/nr_hugepages
```

4. If you encounter issue with mmap while loading the engine, you can use `export DISABLE_MMAP_LOAD=1` to use the default IStreamReader to load engine.

5. When FP8 MHA is enabled for FP8 ViT and visual input is large, we discovered minor accuracy loss and device memory footprint increase. FP8 is not enabled by default in the export script.
