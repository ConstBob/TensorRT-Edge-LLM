# TensorRT Edge LLM ONNX Exporter

This folder contains scripts to export ONNX models from PyTorch models. The exported ONNX model follows the format required by TensorRT Edge LLM runtime, so it can later be converted into a TensorRT engine for E2E LLM inference applications on the Auto platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Download Models from HuggingFace](#download-models-from-huggingface)
3. [Export LLM Models](#export-llm-models)
4. [Export Visual Models](#export-visual-models)
5. [Support Matrix](#support-matrix)
6. [LoRA Support](#lora-support)

## Prerequisites

### System Requirements

1. **Platform**: Linux x86 platform with Ampere or above GPUs
   - FP8 deployment requires Ada and above GPUs
   - NVFP4 deployment requires Blackwell and above GPUs
   - Simulated quantization scripts can run on any GPU

2. **Memory**: Recommended 80GB GPU memory to avoid OOM during quantization and ONNX export

3. **Environment Setup**:
   ```bash
   cd export
   pip3 install -r requirements.txt
   ```

   **For INT4 models**: Downgrade `nvidia-modelopt` to 0.19.0
   ```bash
   pip3 uninstall -y nvidia-modelopt
   pip3 install -r requirements_int4.txt
   ```

   **Note**: Refer to the main [README.md](../README.md#limitations-and-known-issues) for proper environment configuration.

## Download Models from HuggingFace

### Standard LLM and Multimodal Models

Download your desired model from HuggingFace and save it locally:

```bash
# Example: Download Qwen2-7B-Instruct
export MODEL_NAME="Qwen/Qwen2-7B-Instruct"
export TORCH_DIR="hf_models/Qwen2-7B-Instruct"
git lfs install
git clone https://huggingface.co/${MODEL_NAME} ${TORCH_DIR}
cd ${TORCH_DIR}
git lfs pull  # May be needed for larger models
```

### EAGLE Models

For EAGLE decoding, you need both base and draft models:

```bash
# Base model (standard LLM)
export TORCH_DIR="hf_models/Meta-Llama-3.1-8B-Instruct"
export EAGLE_TORCH_DIR="hf_models/EAGLE3-LLaMA3.1-Instruct-8B"

git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ${TORCH_DIR}
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B ${EAGLE_TORCH_DIR}
# May need to cd into the model directory and do git lfs pull 
```

## Export LLM Models

### Standard LLM Export

Export a standard LLM model to ONNX:

```bash
python3 llm_export.py \
    --torch_dir ${TORCH_DIR} \
    --dtype [fp16|fp8|int4|nvfp4|int4_ootb] \
    --output_dir onnx_models/${MODEL_NAME##*/}
```

**Parameters:**
- `--torch_dir`: Path to the PyTorch model directory
- `--dtype`: Precision for ONNX export
  - `fp16`: FP16 precision (default)
  - `fp8`: FP8 precision (requires Ada+ GPUs)
  - `int4`: INT4 precision with custom plugin
  - `nvfp4`: NVFP4 precision (requires Blackwell+ GPUs)
  - `int4_ootb`: INT4 precision with TensorRT OOTB (may have performance issues)
- `--output_dir`: Directory to store the generated ONNX model

**Additional Options:**
- `--keep_original`: Save original ONNX in `${ONNX_DIR}_raw` folder
- `--dataset_dir`: Skip downloading quantization calibration dataset
- `--tokenizer_dir`: Specify different directory for tokenizer files
- `--max_seq_length`: Maximum sequence length (default: 4096)

### EAGLE Model Export

```bash
# Export base model
export ONNX_BASE_DIR="onnx_models/Meta-Llama-3.1-8B-Instruct-Eagle-Base"
python3 llm_export.py \
    --torch_dir ${TORCH_DIR} \
    --dtype fp16 \
    --output_dir ${ONNX_BASE_DIR} \
    --eagle_base True

# Export draft model
export ONNX_DRAFT_DIR="onnx_models/Meta-Llama-3.1-8B-Instruct-Eagle-Draft"
python3 llm_export.py \
    --torch_dir ${TORCH_DIR} \
    --dtype fp16 \
    --output_dir ${ONNX_DRAFT_DIR} \
    --eagle_torch_dir ${EAGLE_TORCH_DIR} \
    --eagle_draft True
```

**Note**: EAGLE decoding is currently only verified with LLAMA-based models using FP16 precision. For EAGLE3, add `--eagle3 True` flag to both commands. EAGLE2 and EAGLE3 has different draft model format.

## Export Visual Models

Multimodal models require exporting both LLM and visual components to separate ONNX files.

### Supported Multimodal Models

**Supported Models:**
- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)
- [InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf)

**Export Commands:**
```bash
export MODEL_NAME="Qwen2-VL-7B-Instruct"
export TORCH_DIR="hf_models/${MODEL_NAME}"
export ONNX_DIR="onnx_models/${MODEL_NAME}"

# LLM part of the VLM export is the same as LLM.
python3 llm_export.py \
    --torch_dir ${TORCH_DIR} \
    --output_dir ${ONNX_DIR} \
    --dtype [fp16|fp8|int4|nvfp4|int4_ootb] \
    --use_prompt_tuning True

# Visual part export
export visualType=[fp16|fp8]
python3 multimodal_export.py \
    --torch_dir ${TORCH_DIR} \
    --output_dir ${ONNX_DIR} \
    --visualType ${visualType}
```

**Notes:**
- **VLM Requirement**: Use `--use_prompt_tuning True` flag for VLM models to enable prompt tuning support
- FP8 VIT quantization is supported and can preserve VLM accuracy while increasing VIT performance
- In TensorRT 10.10, disable `attn.proj` layers for best FP8 VIT performance
- Qwen2.5-VL 3B VIT with FP16 may have overflow issues in the last transformer block (workaround applied)
- InternVL3 uses 0.5 downsampling ratio, resulting in 4x fewer output tokens
- InternVL3 visual encoder currently only supports FP16

## Support Matrix

### LLM Models

The `llm_export.py` script supports the following LLM models:

| Model | FP16 | INT4 | FP8 | NVFP4 | Pre-built ONNX |
|-------|------|------|-----|-------|----------------|
| [Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | ✅ | ✅ | ✅ | ✅ | - |
| [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ | ✅ | ✅ | ✅ | - |
| [Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | ✅ | ✅ | ✅ | ✅ | - |
| [Qwen2-0.5B-instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2_0.5b.tgz](https://nvidia.box.com/shared/static/9buz5igx2unkerl1o23k4cpbvo2hvigf) |
| [Qwen2-1.5B-instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2_1.5b.tgz](https://nvidia.box.com/shared/static/0t4ucre0kc5nuuqqjlgj2ed6tzkkvjw1) |
| [Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2_7b.tgz](https://nvidia.box.com/shared/static/bfowygk8lj0vt55jxfl1cizenur6pjo4) |
| [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2.5_0.5b.tgz](https://nvidia.box.com/shared/static/5rvd43zi8b3ha4x3wm1vjfryb9az1xht) |
| [Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2.5_1.5b.tgz](https://nvidia.box.com/shared/static/0kg77vm50jw3nheci628mrse5sj1j4yn) |
| [Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | ✅ | ✅ | ✅ | ✅ | - |
| [Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2.5_7b.tgz](https://nvidia.box.com/shared/static/tjqxajzqz2ko25b3tuft7vsl3ffp56jm) |
| [Llama3.1-8B-Eagle2-Base](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ | ❌ | ❌ | ❌ | - |
| [Llama3.1-8B-Eagle2-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | ✅ | ❌ | ❌ | ❌ | [llama3.1_8b_eagle2_draft.tgz](https://nvidia.box.com/shared/static/apv5wtpd3twl8y4glz171t6eterht78q) |
| [Llama3.1-8B-Eagle3-Base](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ | ❌ | ❌ | ❌ | - |
| [Llama3.1-8B-Eagle3-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | ✅ | ❌ | ❌ | ❌ | [llama3.1_8b_eagle3_draft.tgz](https://nvidia.box.com/shared/static/wvqyb348j800l6yfks7icmlrog4oiil1) |

### Multimodal Models

The `multimodal_export.py` script supports the following multimodal models:

| Model | FP16 | INT4 | FP8 | NVFP4 | Pre-built ONNX |
|-------|------|------|-----|-------|----------------|
| [Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2_vl_2b.tgz](https://nvidia.box.com/shared/static/wt0c4ydbkwlqy33hc5u65c7hbqzydas4) |
| [Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2_vl_7b.tgz](https://nvidia.box.com/shared/static/818242meioy5ms3g0hgpb74uqw9pxhxl) |
| [Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | ✅ | ✅ | ✅ | ✅ | - |
| [Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | ✅ | ✅ | ✅ | ✅ | [qwen2.5_vl_7b.tgz](https://nvidia.box.com/shared/static/kwcqornn39km3ujzaor1erjflb9vlajs) |
| [InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | ✅ | ✅ | ✅ | ✅ | - |
| [InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | ✅ | ✅ | ✅ | ✅ | - |

**Notes:**
- InternVL3 visual encoder currently only supports FP16
- Other models may be supported with potential compatibility

## LoRA Support

For models with LoRA weights, use the following command:

```bash
python3 llm_export.py \
    --torch_dir ${TORCH_DIR} \
    --lora_dir ${LORA_DIR} \
    --lora_mode [merged|static|dynamic] \
    --dtype [fp16|fp8|int4|nvfp4|int4_ootb] \
    --output_dir ${ONNX_DIR}
```

### LoRA Modes

1. **Merged Mode** (`--lora_mode merged`):
   - LoRA weights are merged into the base model before quantization and ONNX export
   - Provides best performance without any loss compared to no LoRA
   - No runtime LoRA switching capability

2. **Static Mode** (`--lora_mode static`):
   - LoRA weights are kept in separate GEMMs as weights
   - Requires passing LoRA weights during model export
   - LoRA GEMMs are in FP16, mainstream GEMMs are quantized
   - Up to 20% performance loss
   - No runtime LoRA switching

3. **Dynamic Mode** (`--lora_mode dynamic`):
   - LoRA weights are passed as model inputs
   - No actual LoRA weights needed during export
   - Requires specifying `--maxLoraRank` during engine build
   - Multiple LoRA weights can be loaded and switched at runtime
   - Performance close to no LoRA when no weights loaded
   - Performance similar to static LoRA when weights loaded

### Adding Dynamic LoRA to Existing Models

To avoid re-quantizing models, use the convenience script:

```bash
python3 add_dynamic_lora.py \
    --onnx_path ${ONNX_PATH} \
    --dtype [fp16|fp8|int4|nvfp4|int4_ootb] \
    --output_dir ${OUTPUT_DIR}
```

### Processing LoRA Weights for Runtime

Before using dynamic LoRA at runtime, process your LoRA weights:

```bash
python3 process_lora_weights.py \
    --input_dir ${LORA_WEIGHTS_DIR} \
    --output_dir ${PROCESSED_LORA_DIR}
```

## Important Notes

1. **INT4 Performance**: TensorRT OOTB has known performance issues with INT4 GEMV. The custom `Int4GroupwiseGemmPlugin` is used by default for better performance. Use `int4_ootb` dtype to avoid the plugin.

2. **FP8/NVFP4 Limitations**: While FP8 and NVFP4 are supported for ONNX export, Orin does not support these precisions.

3. **Tokenizer Files**: Tokenizer files (`tokenizer_config.json`, `tokenizer.json`, etc.) are automatically copied from the PyTorch model directory to the ONNX output directory.

4. **Sequence Length**: Default `max_seq_length=4096` corresponds to `kv_cache_capacity` in AttentionPlugin. Use [prepare_mmmu_onnx.py](../../scripts/prepare_mmmu_onnx.py) to modify existing ONNX files without re-exporting.

5. **Model Compatibility**: There is potential for other LLMs to be supported beyond the listed models.