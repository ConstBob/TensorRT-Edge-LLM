# Export ONNX with TensorRT-LLM AutoDeploy

> **Repository:** [github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

This guide shows how to export a model to ONNX using **TensorRT-LLM AutoDeploy** instead of the built-in EdgeLLM export tools. The resulting ONNX can be used with TensorRT Edge-LLM’s engine builder and runtime in the same way as ONNX produced by `tensorrt-edgellm-export-llm`.

**When to use this:** AutoDeploy runs graph rewrites on the PyTorch FX graph before exporting to ONNX, preserving high-level structure. Use this path if you are developing or debugging transformations in TensorRT-LLM, comparing export paths, or integrating with the AutoDeploy pipeline. For standard EdgeLLM workflows, the built-in [Quick Start Guide](quick-start-guide.md) (quantize → `tensorrt-edgellm-export-llm` → build → inference) is sufficient.

**Prerequisites:**

- [Installation](installation.md) of TensorRT Edge-LLM on the x86 host (for later build/inference steps).
- **TensorRT-LLM** is a separate repository from TensorRT Edge-LLM. You must clone and install it to run the AutoDeploy ONNX export:
  - **Repository:** [https://github.com/NVIDIA/TensorRT-LLM/](https://github.com/NVIDIA/TensorRT-LLM/)
  - Clone the repo, then install the Python package. The export script used in this guide lives under `examples/auto_deploy/` in that repository.
  - For full install options (Docker, build-from-source), see the [TensorRT-LLM installation documentation](https://github.com/NVIDIA/TensorRT-LLM#getting-started).

---

## Overview

Two ways to get ONNX with AutoDeploy:

1. **From a HuggingFace model directly**  
   Export directly from an HF model name or path. Output precision matches the model.

2. **From a model quantized by yourself**  
   If some model on HF is not quantized with the precision you want, you can quantize it by yourself.
   Quantize with TensorRT Edge-LLM’s `tensorrt-edgellm-quantize-llm` (e.g. with `--unified_checkpoint`).
   Then run the AutoDeploy export script with `--model` pointing at the quantized directory.
   Output precision matches the quantized checkpoint (e.g. FP8, NVFP4).

After export, use the same **build** and **inference** steps as in the [Quick Start Guide](quick-start-guide.md): transfer the ONNX directory to the device, run `llm_build`, then `llm_inference`.

---

## Step 1: Export to ONNX with AutoDeploy

Run the export script from inside a **TensorRT-LLM Docker container**. Most host environments do not have TensorRT pre-installed, so using the official container is strongly recommended.

> **Recommended: use Docker.** Follow the [TensorRT-LLM Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html) to pull the official container and start a shell inside it. The container ships with TensorRT, CUDA, and all Python dependencies already installed — no manual setup required.

If you prefer a bare-metal install, clone [https://github.com/NVIDIA/TensorRT-LLM/](https://github.com/NVIDIA/TensorRT-LLM/) and install with `pip install -e .` from the repo root, but be aware you will need to install TensorRT and its CUDA dependencies yourself.

### Option A: From a pre-quantized HuggingFace model directly

```bash
# Set up workspace and model (use HF name or path to local HF-style directory)
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace  # or wherever you like
export MODEL_NAME=nvidia/Qwen3-8B-NVFP4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# From TensorRT-LLM repo root:
cd /path/to/TensorRT-LLM

python examples/auto_deploy/onnx_export_llm.py \
    --model $MODEL_NAME \
    --output_dir $WORKSPACE_DIR/$MODEL_NAME/onnx
```

- `--model`: HuggingFace model name (e.g. `Qwen/Qwen3-0.6B`) or path to a local directory that contains a HuggingFace-style checkpoint (`config.json` and weights).
- `--output_dir`: Directory where the ONNX model and any sidecar files (e.g. `.safetensors` for embeddings) will be written.
- `--device`: `cuda` or `cpu`.

### Option B: From a quantized (unified) checkpoint

First quantize with TensorRT Edge-LLM and write a unified checkpoint:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace  # or wherever you like
export MODEL_NAME=Qwen/Qwen3-8B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Quantize and save as unified checkpoint (TensorRT Edge-LLM)
tensorrt-edgellm-quantize-llm \
    --model_dir $MODEL_NAME \
    --output_dir $WORKSPACE_DIR/$MODEL_NAME/quantized \
    --quantization fp8 \
    --unified_checkpoint
```

Then run the AutoDeploy export script with `--model` pointing at the quantized directory:

```bash
cd /path/to/TensorRT-LLM

python examples/auto_deploy/onnx_export_llm.py \
    --model $WORKSPACE_DIR/$MODEL_NAME/quantized \
    --output_dir $WORKSPACE_DIR/$MODEL_NAME/onnx
```

The exported ONNX will use the same precision as the quantized checkpoint (e.g. FP8). AutoDeploy does not change precision; it only applies graph transforms and exports to ONNX.

---

## Step 2: Build and run with TensorRT Edge-LLM

Use the exported ONNX with TensorRT Edge-LLM as usual: transfer the ONNX directory to the edge device, then build the engine and run inference. Commands match the [Quick Start Guide](quick-start-guide.md).

**Transfer to device:**

```bash
scp -r $WORKSPACE_DIR/$MODEL_NAME/onnx <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

**On the edge device: build engine and run inference:**

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096

./build/examples/llm/llm_inference \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json
```

For input file format and more options, see [Input Format](input-format.md) and [Examples](examples.md).

---

## Vision-language models (VLM)

This guide and the script `onnx_export_llm.py` cover **language-model** export only. For VLMs:

- **Language part:** You can use the same AutoDeploy export above for the LLM component (optionally from a quantized checkpoint).
- **Visual encoder:** Use TensorRT Edge-LLM’s `tensorrt-edgellm-export-visual` with the **original** HuggingFace model (see [Examples – VLM](examples.md#example-1-vlm-vision-language-model-inference)).

Then build both engines (`llm_build` and `visual_build`) and run inference with `--multimodalEngineDir` as in the VLM example.

---

## Summary

| Step | Tool | Input | Output |
|------|------|--------|--------|
| (Optional) Quantize | `tensorrt-edgellm-quantize-llm --unified_checkpoint` | HF model | Unified checkpoint dir |
| Export ONNX | `python examples/auto_deploy/onnx_export_llm.py` (TensorRT-LLM) | HF model or unified checkpoint dir | ONNX directory |
| Build | `llm_build` (Edge-LLM) | ONNX directory | TensorRT engine |
| Inference | `llm_inference` (Edge-LLM) | Engine directory | Output JSON |

The ONNX produced by AutoDeploy is compatible with TensorRT Edge-LLM’s `llm_build` and `llm_inference`; only the export step differs from the standard EdgeLLM pipeline.
