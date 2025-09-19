# TensorRT Edge-LLM

A Python package for quantizing and exporting Large Language Models (LLMs) and visual models for edge deployment using NVIDIA TensorRT Edge-LLM runtime.

## Installation

```bash
git clone TBD tensorrt-edge-llm # TODO: Add repository URL
cd tensorrt-edge-llm
pip install -e .
```

## Quick Start

### Model Quantization

```bash
# Quantize LLM to FP8
tensorrt-edgellm-quantize-llm \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quantization fp8
```

### Language Model Export

```bash
# Standard model
tensorrt-edgellm-export-llm \
  --model_dir /path/to/model \
  --output_dir /path/to/output

# EAGLE model (base + draft)
tensorrt-edgellm-export-llm \
  --model_dir /path/to/base_model \
  --draft_model_dir /path/to/draft_model \
  --output_dir /path/to/output \
  --eagle2
```

### Visual Model Export

```bash
# Without quantization
tensorrt-edgellm-export-visual \
  --model_dir /path/to/model \
  --output_dir /path/to/output

# With FP8 quantization
tensorrt-edgellm-export-visual \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quantization fp8
```

## Command Line Interface

### Quantization using nvidia-modelopt

```bash
tensorrt-edgellm-quantize-llm [OPTIONS]
```

**Required Arguments:**
- `--model_dir`: Input model directory
- `--output_dir`: Output directory for quantized model

**Optional Arguments:**
- `--quantization`: Quantization method (`fp8`, `int4_awq`, `nvfp4`, `mxfp8`)
- `--torch_dtype`: Model loading dtype (`fp16`, default: `fp16`)
- `--dataset_dir`: Calibration dataset (default: `cnn_dailymail`)
- `--lm_head_quantization`: LM head quantization method

**Model Format:**
Quantized models are saved in uncompressed [HuggingFace format](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis) for PyTorch compatibility. Note that compressed checkpoint cannot be loaded by HuggingFace `from_pretrained` function so `tensorrt-edgellm-export-llm` cannot support it. It will be supported in the future.

```
output_dir/
├── config.json              # Model configuration
├── hf_quant_config.json     # Quantization configuration
├── modelopt_state.pth       # ModelOpt state
├── model.safetensors        # Model weights
├── tokenizer.json           # Tokenizer files
└── tokenizer_config.json    # Tokenizer configuration
```

### Language Model Export

```bash
tensorrt-edgellm-export-llm [OPTIONS]
```

**Required Arguments:**
- `--model_dir`: Input model directory
- `--output_dir`: Output directory for ONNX model

**Optional Arguments:**
- `--draft_model_dir`: Draft model directory (for EAGLE)
- `--eagle2`: Use EAGLE2 (default: EAGLE3)
- `--max_position_embeddings`: Max position embeddings (default: 4096)
- `--device`: Device for model loading (default: `cuda`)

### Visual Model Export

```bash
tensorrt-edgellm-export-visual [OPTIONS]
```

**Required Arguments:**
- `--model_dir`: Input model directory
- `--output_dir`: Output directory for ONNX model

**Optional Arguments:**
- `--dtype`: Export dtype (`fp16`, default: `fp16`)
- `--quantization`: Quantization method (`fp8`)
- `--device`: Device for model loading (default: `cuda`)

## LoRA Support

```bash
# Insert LoRA patterns into ONNX model
tensorrt-edgellm-insert-lora --onnx_dir /path/to/onnx_model

# Process LoRA weights for runtime use
tensorrt-edgellm-process-lora --input_dir /path/to/adapter --output_dir /path/to/output
```

The package supports adding LoRA weights into ONNX models. LoRA weights are treated as dynamic model inputs, allowing for efficient fine-tuning. The `lora_model.onnx` and `model.onnx` share the same base weights, with LoRA weights processed through `tensorrt-edgellm-process-lora` before runtime use.

## Quantization Methods

| Method | Description |
|--------|-------------|
| FP8 | 8-bit floating point quantization |
| INT4 AWQ | 4-bit integer weight quantization |
| INT4 GPTQ | 4-bit GPTQ weight quantization |
| NVFP4 | 4-bit floating point quantization |

**Note:** 

For INT4 GPTQ checkpoint, there is no need to run `tensorrt-edgellm-quantize-llm`. Please follow  additional requirement is needed. Please follow [GPTQModel](https://github.com/ModelCloud/GPTQModel) to run quantization, or directly acquire a checkpoint from HuggingFace Hub. Additional dependency is needed. `gptqmodel` only has wheel for certain CUDA version, but we only need the frontend to load and export the model.
```
BUILD_CUDA_EXT=0 pip install -v gptqmodel --no-build-isolation
```

## Limitations

- Only FP16 precision is currently supported, BF16 is not supported
- MXFP8 quantization is not supported
- When vocab_size is not a multiple of 4, the model cannot be quantized to int4_awq. Please use fp16 lm_head.


