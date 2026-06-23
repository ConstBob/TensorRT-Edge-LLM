# MoE (Mixture of Experts)

Complete workflow for Mixture of Experts (MoE) models using pre-quantized INT4 or NVFP4 checkpoints.

**Currently supported models:**
- [Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4)
- [nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)
- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)

> **Note:** MoE export always runs on **CPU**. No GPU or device flag is required for the export step.

> **Note:** For very large NVFP4 MoE checkpoints such as Nemotron Super 120B,
> externalize NVFP4 MoE plugin weights during export and keep the generated
> safetensors file with the ONNX directory.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Export (x86 Host, CPU-only)

Export always runs on CPU; no GPU is required:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

tensorrt-edgellm-export \
  Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  $MODEL_NAME/exported

mkdir -p $MODEL_NAME/onnx
cp -a $MODEL_NAME/exported/llm/. $MODEL_NAME/onnx/
```

For Nemotron Super 120B NVFP4, externalize the MoE plugin weights to reduce engine build memory pressure:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

tensorrt-edgellm-export \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  $MODEL_NAME/exported \
  --externalize-weights nvfp4_moe \
  --max-kv-cache-capacity 4096

mkdir -p $MODEL_NAME/onnx
cp -a $MODEL_NAME/exported/llm/. $MODEL_NAME/onnx/
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engine (Edge Device)

`llm_build` is the same as for regular LLMs:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096
```

## Step 4: Run Inference (Edge Device)

`llm_inference` is the same as for regular LLMs:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```

MoE export uses CPU-only; build and inference use the same `llm_build` and `llm_inference` as standard LLMs.
