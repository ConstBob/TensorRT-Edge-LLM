# Examples

> **Code Location:** [`examples/`](../../examples/) | **Build:** [`examples/llm/`](../../examples/llm/), [`examples/multimodal/`](../../examples/multimodal/)

## Table of Contents

- [Build Examples](#build-examples)
- [Inference Examples](#inference-examples)
- [Complete Workflows](#complete-workflows)
- [Common Parameters](#common-parameters)

---

## Overview

C++ examples for building engines and running inference. All examples include detailed README files and source code in [`examples/`](../../examples/).

---

## Example Flow

```
ONNX Models → [Build Examples] → TensorRT Engines → [Inference Examples] → Results
```

---

## Build Examples

### `llm_build` - [Source](../../examples/llm/llm_build.cpp)

Builds TensorRT engines for LLMs (standard, EAGLE, VLM, LoRA).

```bash
# Standard LLM
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-4b \
  --engineDir engines/qwen3-4b \
  --maxBatchSize 1

# EAGLE (speculative decoding)
./build/examples/llm/llm_build \
  --onnxDir onnx_models/model_eagle_base \
  --engineDir engines/model_eagle_base \
  --eagleBase
```

### `visual_build` - [Source](../../examples/multimodal/visual_build.cpp)

Builds TensorRT engines for vision encoders (Qwen-VL, InternVL).

```bash
./build/examples/multimodal/visual_build \
  --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx_qwen2 \
  --engineDir visual_engines/qwen2.5-vl-3b \
  --minImageTokens 128 \
  --maxImageTokens 512
```

---

## Inference Examples

### `llm_inference` - [Source](../../examples/llm/llm_inference.cpp)

Runs batch inference from JSON files. Supports standard, EAGLE, multimodal, and LoRA modes.

```bash
# Standard LLM
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json

# EAGLE (speculative decoding)
./build/examples/llm/llm_inference \
  --engineDir engines/model_eagle_base \
  --multimodalEngineDir engines/model_eagle_draft \
  --inputFile input.json \
  --outputFile output.json \
  --eagle

# Multimodal (VLM)
./build/examples/llm/llm_inference \
  --engineDir engines/qwen2.5-vl-3b \
  --multimodalEngineDir visual_engines/qwen2.5-vl-3b \
  --inputFile input_with_images.json \
  --outputFile output.json

# With profiling
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json \
  --dumpProfile \
  --profileOutputFile profile.json
```

---

## Complete Workflows

### Standard LLM (End-to-End)

```bash
# 1. Export ONNX (x86 host)
tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen3-4B-Instruct-2507 --quantization fp8 --output_dir quantized/qwen3-4b
tensorrt-edgellm-export-llm --model_dir quantized/qwen3-4b --output_dir onnx_models/qwen3-4b

# 2. Build Engine (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/qwen3-4b --engineDir engines/qwen3-4b --maxBatchSize 1

# 3. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/qwen3-4b --inputFile input.json --outputFile output.json
```

### Multimodal VLM (End-to-End)

```bash
# 1. Export (x86 host)
tensorrt-edgellm-export-llm --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b

# 2. Build Engines (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-3b --engineDir engines/qwen2.5-vl-3b --vlm
./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx_qwen2 --engineDir visual_engines/qwen2.5-vl-3b

# 3. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/qwen2.5-vl-3b --multimodalEngineDir visual_engines/qwen2.5-vl-3b --inputFile input.json --outputFile output.json
```

---

## Common Parameters

### Build Parameters (`llm_build`, `visual_build`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--onnxDir` | Input ONNX directory | Required |
| `--engineDir` | Output engine directory | Required |
| `--maxBatchSize` | Maximum batch size | 4 |
| `--maxInputLen` | Maximum input length | 128 |
| `--maxSeqLen` | Maximum sequence length | 4096 |
| `--vlm` | VLM mode | false |
| `--eagleBase/Draft` | EAGLE mode | false |
| `--maxLoraRank` | LoRA rank (0=disabled) | 0 |

### Inference Parameters (`llm_inference`)

| Parameter | Description |
|-----------|-------------|
| `--engineDir` | Engine directory (required) |
| `--multimodalEngineDir` | Visual/draft engine (for VLM/EAGLE) |
| `--inputFile` | Input JSON path (required) |
| `--outputFile` | Output JSON path (required) |
| `--eagle` | Enable EAGLE mode |
| `--dumpProfile` | Enable profiling |
| `--profileOutputFile` | Profile output path |

**Note:** Sampling parameters (temperature, top_p, top_k) go in the input JSON. See [`examples/llm/INPUT_FORMAT.md`](../../examples/llm/INPUT_FORMAT.md).

---

**See also:** [Getting Started](01_Getting_Started.md) | [Key Components](03_Key_Components.md) | [Supported Models](02_Supported_Models.md)
