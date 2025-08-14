# TensorRT Edge LLM: Large Language Model Inference Framework for Edge Platforms

## Introduction

TensorRT Edge LLM is a lightweight C++ software toolkit that showcases TensorRT's capability and performance for deploying Large Language Models (LLMs) and Vision Language Models (VLMs) on Edge Platforms (Jetson and Auto). With TensorRT Edge LLM, users can:

1. Quantize and export PyTorch models to [ONNX Format](https://onnx.ai/) on Linux x86 systems
2. Build TensorRT engines and run end-to-end LLM inference, including tokenization and sampling, on Edge Platforms

## Prerequisites

- A Linux x86 host with GPU is required to export models to ONNX format
- Once the ONNX model is exported, the only dependency is TensorRT C++ library and CUDA runtime

## Supported Platforms, Models and Precisions

### LLM Models

The following LLM models under [./examples/llm](./examples/llm/) are supported:

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

### VLM Models

The following VLM models under [./examples/multimodal](./examples/multimodal/) are supported. Note that the ViT will always be in FP16 precision:

Model | FP16 | INT4 | FP8 | NVFP4
--- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Yes | Yes | Yes | Yes
[Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Yes | Yes | Yes | Yes
[InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | Yes | Yes | Yes | Yes
[InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | Yes | Yes | Yes | Yes

### Precision Types

1. **FP16**: All weights and compute are in FP16
2. **FP8(W8A8)**: Weights and GEMMs are in FP8, while KV Cache, LayerNorm, Attention and lm_head are in FP16 precision
3. **INT4(W4A16)**: Weights are quantized in INT4 using AWQ recipe, but all compute is in FP16 precision
4. **NVFP4(W4A4)**: Recommended precision on Thor. Similar to FP8, weights and GEMMs are in NVFP4 while other parts are in FP16 precision

### Custom Models

- Decoder-only models with structure similar to Llama or Qwen are likely to be supported if they fit in Thor memory
- For VLM, visual encoder part is likely to be supported by native TensorRT and the LLM part applies the same rules as pure LLM models
- Other model series might not be supported due to Attention plugin, tokenizer implementation and model architecture differences

### Platform Support

- **Thor**: Full support for all precisions
- **Orin**: Support FP16 and INT4
- **x86 Linux**: Preview feature for Data Center or Gaming GPUs with SM80, SM86 or SM89 in FP16 and INT4. SM89 also supports FP8 E2E inference

## Getting Started

### 1. Build the C++ Project

**Native build for x86 Linux:**
```bash
cd tensorrt-edgellm
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} 
make -j
```

**Cross and native build for aarch64 Linux systems:**
```bash
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET={hardware-type}
make -j
```

**Build options:**
- Add `-DBUILD_UNIT_TESTS=on` to build unit tests
- Add `-DCUDA_VERSION={version}` to override default CUDA toolkit version
- Supported hardware lists: [auto-thor, jetson-thor, orin, n1x]

### 2. Export ONNX from PyTorch Checkpoint

Export PyTorch models to ONNX on a x86 Linux host with GPU. For quantization, use a Data Center GPU like H100. See [export/README.md](./export/README.md) for detailed model export process.

### 3. Build Engine and Run Inference

Once the model is exported, follow the examples to build and run end-to-end LLM inference:

- **LLMs**: Follow [examples/llm/README.md](./examples/llm/README.md)
- **VLMs**: Follow [examples/multimodal/README.md](./examples/multimodal/README.md)

## Limitations and Known Issues

### Python Export

- `nvidia-modelopt>0.19.0` has accuracy issues for INT4 recipe. Downgrade to 0.19.0 for INT4 export
- Qwen2.5-VL 3B VIT with FP16 precision has occasional overflow issues from the last transformer block

### Inference

- Up to DriveOS 7.0.3.0, `cudaMallocAsync` fails when allocated memory size is large (>~5G). Use `echo 24576 | sudo tee /proc/sys/vm/nr_hugepages` as a workaround
- If you encounter mmap issues while loading the engine, use `export DISABLE_MMAP_LOAD=1`
- When FP8 MHA is enabled for FP8 ViT and visual input is large, minor accuracy loss and device memory footprint increase may occur
