# TensorRT Edge-LLM

A Python package for quantizing and exporting Large Language Models (LLMs) for edge deployment using NVIDIA ModelOpt.

## Features

- **Model Quantization**: Support for FP8, INT4 AWQ, and NVFP4 quantization
- **PyTorch Compatibility**: Quantized models can be loaded by PyTorch for ONNX export
- **HuggingFace Integration**: Seamless integration with HuggingFace models and datasets
- **Command-line Interface**: Easy-to-use CLI for model quantization

## Installation

```bash
git clone TBD tensorrt-edge-llm # TODO: Add repository URL
cd tensorrt-edge-llm
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Example with all options
tensorrt-edgellm-quantize \
    --torch_dir $TORCH_DIR \
    --output_dir $QUANTIZED_TORCH_DIR$ \
    --quantization [fp8 | int4_awq | nvfp4 | None] \
    --torch_dtype [fp16 | bf16] \
    --dataset_name_or_dir cnn_dailymail \
    --lm_head_quantization [fp8 | int4_awq | nvfp4 | None]
```

### Python API

```python
from tensorrt_edgellm import quantize_and_save_model

quantize_and_save_model(
    torch_dir="/path/to/your/model",
    output_dir="/path/to/output",
    quantization="fp8",
    torch_dtype="fp16"
)
```

## Quantization Methods

- **FP8 GEMM**: 8-bit floating point quantization for both activation and weights
- **INT4 AWQ**: 4-bit integer weight quantization with activation in 16-bit with smoothing
- **NVFP4**: 4-bit floating point quantization with dynamic scaling for both activation and weights

## Data Types

- **FP16**: 16-bit floating point (IEEE 754 half precision) - faster on most hardware, wider dynamic range
- **BF16**: 16-bit Brain Floating Point - better numerical stability, preserves more precision in extreme values

## Output Format

The quantized model is saved using [ModelOpt's HuggingFace checkpointing APIs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/6_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis) to ensure PyTorch compatibility:

```
output_dir/
├── config.json              # Model configuration
├── hf_quant_config.json     # Quantization configuration
├── modelopt_state.pth       # ModelOpt state (architecture modifications)
├── model.safetensors        # Model weights (safetensors format). May have multiple files.
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
└── ...                      # Other tokenizer files
```

**Note**: This format preserves original weights and amax values in safetensors format, ensuring the model can be loaded by PyTorch for ONNX export. Compressed checkpoints are not used to maintain compatibility.

## Loading Quantized Models

```python
import modelopt.torch.opt as mto
from transformers import AutoModelForCausalLM

mto.enable_huggingface_checkpointing()
model = AutoModelForCausalLM.from_pretrained("/path/to/quantized/model")
```
