# Key Components

> **Code Location:** [`tensorrt_edgellm/`](../../tensorrt_edgellm/) (Python), [`cpp/`](../../cpp/) (C++), [`examples/`](../../examples/) (Examples)

## Overview

TensorRT Edge-LLM uses a three-stage pipeline:

```
HuggingFace Model → Python Export → ONNX → Engine Builder → TensorRT Engine → C++ Runtime → Results
```

---

## Components

| Component | Language | Purpose | Details |
|-----------|----------|---------|---------|
| **[Python Export Pipeline](03.1_Python_Export_Pipeline.md)** | Python | Converts HuggingFace models to ONNX with quantization (FP8, INT4, NVFP4) | [Code →](../../tensorrt_edgellm/) |
| **[Engine Builder](03.2_Engine_Builder.md)** | C++ | Compiles ONNX models into optimized TensorRT engines | [Code →](../../cpp/builder/) |
| **[C++ Runtime](03.3_C++_Runtime.md)** | C++ | Executes TensorRT engines with CUDA graphs, LoRA, EAGLE support | [Code →](../../cpp/runtime/) |

---

## Quick Reference

### 1. Python Export Pipeline

**Command-line tools:**
- `tensorrt-edgellm-quantize-llm` - Quantize models
- `tensorrt-edgellm-export-llm` - Export LLMs to ONNX  
- `tensorrt-edgellm-export-visual` - Export vision encoders
- `tensorrt-edgellm-export-draft` - Export EAGLE draft models

**Supports:** FP16, FP8, INT4 (AWQ/GPTQ), NVFP4

### 2. Engine Builder

**Builders:**
- LLM Builder ([`llm_build`](../../examples/llm/llm_build.cpp)) - Language models
- Visual Builder ([`visual_build`](../../examples/multimodal/visual_build.cpp)) - Vision encoders

**Features:** Dual-phase optimization, EAGLE support, dynamic shapes

### 3. C++ Runtime

**Runtimes:**
- Standard ([`llmInferenceRuntime.h`](../../cpp/runtime/llmInferenceRuntime.h)) - LLM/VLM inference
- SpecDecode ([`llmInferenceSpecDecodeRuntime.h`](../../cpp/runtime/llmInferenceSpecDecodeRuntime.h)) - EAGLE decoding

**Features:** CUDA graphs, LoRA adapters, batch processing, KV-cache

---

## Example Usage

### 1. Export Model (x86 host)

```bash
pip3 install .

tensorrt-edgellm-quantize-llm \
  --model_dir Qwen/Qwen3-4B-Instruct-2507 \
  --quantization fp8 \
  --output_dir quantized/qwen3-4b

tensorrt-edgellm-export-llm \
  --model_dir quantized/qwen3-4b \
  --output_dir onnx_models/qwen3-4b
```

### 2. Build Engine (Thor device)

```bash
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-4b \
  --engineDir engines/qwen3-4b \
  --maxBatchSize 1
```

### 3. Run Inference (Thor device)

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json
```

---

**See also:** [Examples](04_Examples.md) | [Getting Started](01_Getting_Started.md) | [Supported Models](02_Supported_Models.md)
