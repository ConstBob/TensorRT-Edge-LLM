# Phi-4-Multimodal

Phi-4-Multimodal requires merging its vision LoRA adapter into the base model before quantization and export.

**Example model:** Phi-4-multimodal-instruct

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Merge, Quantize, and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
export EDGE_LLM_PATH=$HOME/tensorrt-edge-llm   # path to TensorRT-Edge-LLM repo root
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Clone Phi-4-multimodal-instruct from HuggingFace
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
cd Phi-4-multimodal-instruct && git lfs pull && cd ..

# Merge vision LoRA adapter into base model
tensorrt-edgellm-merge-lora \
  --model_dir Phi-4-multimodal-instruct \
  --lora_dir Phi-4-multimodal-instruct/vision-lora \
  --output_dir $MODEL_NAME/merged

# Quantize merged model
tensorrt-edgellm-quantize-llm \
  --model_dir $MODEL_NAME/merged \
  --output_dir $MODEL_NAME/quantized \
  --quantization nvfp4

# Export language model
tensorrt-edgellm-export-llm \
  --model_dir $MODEL_NAME/quantized \
  --output_dir $MODEL_NAME/onnx/llm \
  --chat_template $EDGE_LLM_PATH/tensorrt_edgellm/chat_templates/templates/phi4mm.json

# Export visual encoder (use original weights, not merged)
tensorrt-edgellm-export-visual \
  --model_dir Phi-4-multimodal-instruct \
  --output_dir $MODEL_NAME/onnx/visual
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
cd ~/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

## Step 4: Run Inference (Thor Device)

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```
