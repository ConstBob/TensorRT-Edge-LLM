# TensorRT Edge-LLM Scripts

This directory contains command-line scripts for TensorRT Edge-LLM operations.

## Available Scripts

### quantize_llm.py
Quantizes language models using NVIDIA ModelOpt with support for various quantization schemes.

**Usage:**
```bash
python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
```

**Command-line interface:**
```bash
tensorrt-edgellm-quantize --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
```

**Parameters:**
- `--model_dir`: Path to the input model directory (required)
- `--output_dir`: Path to save the quantized model (required)
- `--quantization`: Quantization method (None, fp8, int4_awq, nvfp4, default: None)
- `--torch_dtype`: High precision dtype for model loading (fp16, bf16, default: fp16)
- `--dataset_dir`: Dataset for calibration (default: cnn_dailymail)
- `--lm_head_quantization`: Quantization method for language model head (None, fp8, int4_awq, nvfp4, default: None)

### export_llm.py
Exports language models to ONNX format with support for standard models, EAGLE base models, and EAGLE draft models.

**Usage:**
```bash
# Standard model export
python export_llm.py --model_dir /path/to/model --output_dir /path/to/output

# EAGLE complete export (both base and draft)
python export_llm.py --model_dir /path/to/base_model --draft_model_dir /path/to/draft_model --output_dir /path/to/output --eagle2
```

**Command-line interface:**
```bash
tensorrt-edgellm-export-llm --model_dir /path/to/model --output_dir /path/to/output
```

**Supported Models:**
- Standard language models (Llama, Qwen2, etc.)
- EAGLE2 base and draft models
- EAGLE3 base and draft models

**Parameters:**
- `--model_dir`: Path to the input model directory (required)
- `--draft_model_dir`: Path to the draft model directory (triggers complete EAGLE export)
- `--output_dir`: Path to save the exported ONNX model (required)
- `--eagle2`: Whether this is EAGLE2 (True) or EAGLE3 (False) for EAGLE models
- `--max_position_embeddings`: Maximum positional embedding length (default: 4096)
- `--device`: Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)

**Notes:**
- When `--draft_model_dir` is provided, both base and draft models are automatically exported
- Tokenizer files are saved to the base model output directory
- Standard models are exported as single models without EAGLE-specific processing

### export_visual.py
Exports visual models to ONNX format with optional quantization support.

**Usage:**
```bash
python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --dtype fp16 --quantization fp8
```

**Command-line interface:**
```bash
tensorrt-edgellm-export-visual --model_dir /path/to/model --output_dir /path/to/output --dtype fp16 --quantization fp8
```

**Supported Models:**
- Qwen2-VL
- Qwen2.5-VL  
- InternVL3

**Parameters:**
- `--model_dir`: Path to the input model directory (required)
- `--output_dir`: Path to save the exported ONNX model (required)
- `--dtype`: Data type for export (only fp16 supported, default: fp16)
- `--quantization`: Quantization method (None or fp8, default: None)

### safetensor_visualizer.py
Visualizes safetensor files for debugging and analysis.

## Installation

After installing the package, the scripts are available as command-line tools:

```bash
pip install tensorrt-edgellm
```

Then you can use:
- `tensorrt-edgellm-quantize` for model quantization
- `tensorrt-edgellm-export-llm` for language model export
- `tensorrt-edgellm-export-visual` for visual model export

## Examples

### Quantize a model with FP8 quantization:
```bash
tensorrt-edgellm-quantize \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quantization fp8 \
  --torch_dtype fp16
```

### Export a standard language model:
```bash
tensorrt-edgellm-export-llm \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --max_position_embeddings 8192
```

### Export EAGLE2 model with draft model:
```bash
tensorrt-edgellm-export-llm \
  --model_dir /path/to/base_model \
  --draft_model_dir /path/to/draft_model \
  --output_dir /path/to/output \
  --eagle2
```

### Export a model on CPU:
```bash
tensorrt-edgellm-export-llm \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --device cpu
```

### Export Qwen2-VL visual model without quantization:
```bash
tensorrt-edgellm-export-visual \
  --model_dir /path/to/qwen2-vl-model \
  --output_dir /path/to/output \
  --dtype fp16
```


### Export Qwen2.5-VL visual model with FP8 quantization:
```bash
tensorrt-edgellm-export-visual \
  --model_dir /path/to/qwen2.5-vl-model \
  --output_dir /path/to/output \
  --dtype fp16 \
  --quantization fp8
```

## Notes

- Only FP16 precision is currently supported for visual model export
- FP8 quantization is supported for Qwen2-VL and Qwen2.5-VL models
- InternVL3 models do not support FP8 quantization and will skip quantization if requested
- The exported ONNX model will be saved as `visual_model.onnx` in the output directory
- A `config.json` file will also be exported with model configuration
- Quantization parameters support explicit "None" values for clarity (e.g., `--quantization None`)
- All optional parameters have sensible defaults and can be omitted for common use cases
