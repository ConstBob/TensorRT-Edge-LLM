# Supported Models

**Previous**: [Installation](01.3_Installation.md) | **Next**: [Python Export Pipeline](03.1_Python_Export_Pipeline.md)

---

> **Code Location:** [`tensorrt_edgellm/`](../../tensorrt_edgellm/) (export), [`cpp/`](../../cpp/) (runtime)

## Table of Contents

- [LLM Models](#llm-models)
- [VLM Models](#vlm-models)
- [Precision Support](#precision-support)
- [Platform Compatibility](#platform-compatibility)

---

## LLM Models

### Llama Family

| Model | Parameters | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8B | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 8B | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 3B | ✅ | ✅ | ✅ | ✅ |

### Qwen2/2.5 Family

| Model | Parameters | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 0.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |

### Qwen3 Family

| Model | Parameters | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | ✅ | ✅ | ✅ | ✅ |

### DeepSeek-R1 Distilled Family

| Model | Parameters | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7B | ✅ | ✅ | ✅ | ✅ |

---

## VLM Models

| Model | Parameters | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | 2B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | 1B | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 2B | ✅ | ✅ | ✅ | ✅ |

---

## Precision Support

| Precision | Memory | Compute | Platform Requirements | Best For |
|-----------|--------|---------|----------------------|----------|
| **FP16** | 1x (baseline) | FP16 | All platforms | Accuracy baseline, universal compatibility |
| **FP8** | 2x reduction | FP8 GEMMs + FP16 | **SM89+** (Ada Lovelace and newer) | Balanced performance on modern GPUs |
| **INT4 AWQ** | 4x reduction | FP16 (AWQ quantization) | All platforms | Memory-constrained devices |
| **INT4 GPTQ** | 4x reduction | FP16 (GPTQ quantization) | All platforms | Memory-constrained devices |
| **NVFP4** | 4x reduction | NVFP4 GEMMs + FP16 | **SM100+** (Blackwell and newer) | **Thor platforms (recommended)** |

### Additional Features
- **FP8 Vision Encoder**: Supported for visual models (Qwen2-VL, InternVL3) on SM89+
- **FP8/NVFP4 LM Head**: Supported for language model heads with platform-specific requirements

---

## Platform Compatibility

| GPU Architecture | Compute Capability | Supported Precisions |
|-----------------|-------------------|---------------------|
| **All Platforms** | Any | FP16, INT4 AWQ, INT4 GPTQ |
| **Ada Lovelace+** | SM89+ | FP16, FP8, INT4 AWQ, INT4 GPTQ |
| **Blackwell+** | SM100+ | FP16, FP8, INT4 AWQ, INT4 GPTQ, NVFP4 |

**Notes:**
- FP16 and INT4 (AWQ/GPTQ) quantization methods work on all CUDA-capable platforms
- FP8 quantization requires SM89+ (Ada Lovelace architecture or newer, e.g., RTX 40-series)
- NVFP4 quantization requires SM100+ (Blackwell architecture or newer, e.g., Thor platforms)
- Platform requirements apply to both model weights and operations (including ViT encoders and LM heads)

---

**See also:** [Overview](01.1_Overview.md) | [Quick Start Guide](01.2_Quick_Start_Guide.md) | [Examples](05_Examples.md)
