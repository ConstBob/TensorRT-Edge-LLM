# Speculative Decoding

## EAGLE3

EAGLE3 [https://arxiv.org/abs/2503.01840] uses a smaller draft model to accelerate generation. This guide covers EAGLE for both text-only LLMs and vision-language models.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

### Supported Draft Models

| Base Model | Draft Model |
|------------|-------------|
| Llama-3.1-8B-Instruct | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) |
| Qwen3-1.7B | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |
| Qwen3-4B | [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) |
| Qwen3-8B | [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) |
| Qwen2.5-VL-7B-Instruct | [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) |

Any EAGLE3-compatible draft model on HuggingFace can be tried however TensorRT Edge-LLM team does not test the accuracy or acceptance rate. Search for [eagle3 models](https://huggingface.co/models?search=eagle3) to find additional options.

Draft model quantization is supported via `tensorrt-edgellm-quantize-draft` with all precisions: `fp8`, `int4_awq`, `nvfp4`, and `int8_sq`. For example:

```bash
tensorrt-edgellm-quantize-draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-draft
```

Note that quantizing the draft model will cause a drop in acceptance rate compared to running it in FP16, which may reduce the overall speedup.

---

### LLM EAGLE

**Example model:** Llama-3.1-8B-Instruct with [EAGLE draft model](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
cd EAGLE3-LLaMA3.1-Instruct-8B && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize-llm \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base model with EAGLE flag
tensorrt-edgellm-export-llm \
  --model_dir $MODEL_NAME/quantized-base \
  --output_dir $MODEL_NAME/onnx/base \
  --is_eagle_base

# Quantize draft model
tensorrt-edgellm-quantize-draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export-draft \
  --draft_model_dir $MODEL_NAME/quantized-draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --output_dir $MODEL_NAME/onnx/draft
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd ~/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleDraft
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --eagle
```

**Note:** EAGLE speculative decoding provides 1.5-3x faster generation but is limited to batch size 1.

---

### VLM EAGLE

EAGLE for vision-language models combines accelerated text generation with image understanding.

**Example model:** Qwen2.5-VL-7B-Instruct with [EAGLE3 draft model](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl
cd qwen2.5-vl-7b-eagle3-sgl && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize-llm \
  --model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base model with EAGLE flag
tensorrt-edgellm-export-llm \
  --model_dir $MODEL_NAME/quantized-base \
  --output_dir $MODEL_NAME/onnx/base \
  --is_eagle_base

# Quantize draft model
tensorrt-edgellm-quantize-draft \
  --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --draft_model_dir qwen2.5-vl-7b-eagle3-sgl \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export-draft \
  --draft_model_dir $MODEL_NAME/quantized-draft \
  --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --output_dir $MODEL_NAME/onnx/draft

# Export visual encoder
tensorrt-edgellm-export-visual \
  --model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --output_dir $MODEL_NAME/onnx/visual
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd ~/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleDraft

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --eagle
```
