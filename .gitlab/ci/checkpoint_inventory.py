# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CI-internal inventory of public and private checkpoint mirrors.

This data-only file is deliberately outside product runtime packages. CI test
utilities may consume it, but release wheels and source distributions must not
contain this inventory.
"""

# Each key is also the fallback Hugging Face download ID. An empty value means
# that Edge-LLM supports the checkpoint but CI does not currently mirror it.
# Torch paths are relative to TORCH_DIR. Quantized checkpoint names are
# relative to QUANT_CHECKPOINT_DIR/public and expand as
# <name>-<quantization>. Optional per-quantization paths are relative to the
# managed Edge-LLM data root.
PUBLIC_CHECKPOINTS_BY_HF_ID = {
    "AngelSlim/Qwen3-1.7B_eagle3": {
        "torch": (
            "Qwen3-1.7B_eagle3",
            "Qwen3/Qwen3-1.7B_eagle3",
        ),
        "quantized": {
            "name":
            "Qwen3-1.7B_eagle3",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "INT8-SQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "deepseek-ai/dflash_gemma4_12b_block7": {
        "torch": ("dflash_gemma4_12b_block7", ),
    },
    "deepseek-ai/dspark_qwen3_4b_block7": {
        "torch": (
            "deepseek-ai/dspark_qwen3_4b_block7",
            "dspark/dspark_qwen3_4b_block7",
        ),
    },
    "deepseek-ai/eagle3_gemma4_12b_ttt7": {
        "torch": ("eagle3_gemma4_12b_ttt7", ),
    },
    "google/gemma-4-12B-it": {
        "torch": ("gemma/gemma-4-12B-it", ),
        "quantized": {
            "name": "gemma-4-12B-it",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "google/gemma-4-26B-A4B-it": {
        "torch": ("gemma/gemma-4-26B-A4B-it", ),
    },
    "google/gemma-4-26B-A4B-it-assistant": {
        "torch": (
            "gemma-4-26B-A4B-it-assistant",
            "gemma/gemma-4-26B-A4B-it-assistant",
        ),
    },
    "google/gemma-4-31B-it": {
        "torch": ("gemma/gemma-4-31B-it", ),
        "quantized": {
            "name": "gemma-4-31B-it",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
            ),
        },
    },
    "google/gemma-4-E2B-it": {
        "torch": (
            "gemma-4-E2B-it",
            "gemma/gemma-4-E2B-it",
            "source_models/gemma-4-E2B-it",
        ),
        "quantized": {
            "name": "gemma-4-E2B-it",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "google/gemma-4-E2B-it-assistant": {
        "torch": (
            "gemma-4-E2B-it-assistant",
            "gemma/gemma-4-E2B-it-assistant",
            "source_models/gemma-4-E2B-it-assistant",
        ),
    },
    "google/gemma-4-E4B-it": {
        "torch": ("gemma/gemma-4-E4B-it", ),
        "quantized": {
            "name": "gemma-4-E4B-it",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "JetSpec/jetspec-qwen3-8b": {
        "torch": ("JetSpec/jetspec-qwen3-8b", ),
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "torch": ("llama-3.1-model/Llama-3.1-8B-Instruct", ),
        "quantized": {
            "name":
            "Llama-3.1-8B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "meta-llama/Llama-3.2-1B": {
        "torch": ("llama-3.2-models/Llama-3.2-1B", ),
    },
    "meta-llama/Llama-3.2-3B": {
        "torch": ("llama-3.2-models/Llama-3.2-3B", ),
    },
    "microsoft/Phi-4-multimodal-instruct": {
        "torch": (
            "multimodals/Phi-4-multimodal-instruct",
            "Phi-4-multimodal-instruct",
        ),
        "quantized": {
            "name":
            "Phi-4-multimodal-instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "nvidia/Alpamayo-R1-10B": {
        "torch": ("Alpamayo-R1-10B", ),
    },
    "nvidia/Cosmos-Reason2-8B": {
        "torch": ("Cosmos-Reason2-8B", ),
        "quantized": {
            "name": "Cosmos-Reason2-8B",
            "quantizations": (
                "FP8",
                "NVFP4",
            ),
        },
    },
    "nvidia/Gemma-4-26B-A4B-NVFP4": {
        "torch": (
            "gemma/nvidia-Gemma-4-26B-A4B-NVFP4",
            "nvidia-Gemma-4-26B-A4B-NVFP4",
        ),
    },
    "nvidia/Gemma-4-31B-IT-NVFP4": {
        "torch": (
            "gemma/nvidia-Gemma-4-31B-IT-NVFP4",
            "nvidia-Gemma-4-31B-IT-NVFP4",
        ),
    },
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4": {
        "torch": (
            "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
        ),
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {
        "torch": ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", ),
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4": {
        "torch": ("NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", ),
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": {
        "torch": ("NVIDIA-Nemotron-3-Nano-4B-BF16", ),
        "quantized": {
            "name": "NVIDIA-Nemotron-3-Nano-4B",
            "quantizations": ("NVFP4", ),
            "paths": {
                "NVFP4": ("models/NVIDIA-Nemotron-3-Nano-4B-NVFP4", ),
            },
        },
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8": {
        "torch": ("NVIDIA-Nemotron-3-Nano-4B-FP8", ),
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": {
        "torch": ("NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", ),
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": {
        "torch": ("NVIDIA-Nemotron-Nano-9B-v2", ),
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8": {
        "torch": ("NVIDIA-Nemotron-Nano-9B-v2-FP8", ),
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4": {
        "torch": ("NVIDIA-Nemotron-Nano-9B-v2-NVFP4", ),
    },
    "nvidia/Phi-4-multimodal-instruct-FP8": {
        "torch": ("multimodals/Phi-4-multimodal-instruct-FP8", ),
    },
    "nvidia/Qwen3-30B-A3B-NVFP4": {
        "torch": ("Qwen3/nvidia-Qwen3-30B-A3B-NVFP4", ),
    },
    "nvidia/Qwen3.6-35B-A3B-NVFP4": {
        "torch": ("Qwen3.6-35B-A3B-NVFP4", ),
    },
    "OpenGVLab/InternVL3-1B-hf": {
        "torch": (
            "multimodals/InternVL3-1B-hf",
            "InternVL3-1B-hf",
        ),
        "quantized": {
            "name":
            "InternVL3-1B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "INT4-AWQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "OpenGVLab/InternVL3-2B-hf": {
        "torch": (
            "multimodals/InternVL3-2B-hf",
            "InternVL3-2B-hf",
        ),
        "quantized": {
            "name":
            "InternVL3-2B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-VITFP8",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen2-VL-2B-Instruct": {
        "torch": ("Qwen2-VL-2B-Instruct", ),
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "torch": ("Qwen2.5-0.5B-Instruct", ),
        "quantized": {
            "name":
            "Qwen2.5-0.5B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "torch": ("Qwen2.5-1.5B-Instruct", ),
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "torch": ("Qwen2.5-3B-Instruct", ),
        "quantized": {
            "name":
            "Qwen2.5-3B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "torch": ("Qwen2.5-7B-Instruct", ),
        "quantized": {
            "name":
            "Qwen2.5-7B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": {
        "torch": ("Qwen2.5-7B-Instruct-GPTQ-Int4", ),
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "torch": ("Qwen2.5-VL-3B-Instruct", ),
        "quantized": {
            "name":
            "Qwen2.5-VL-3B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "torch": ("Qwen2.5-VL-7B-Instruct", ),
        "quantized": {
            "name":
            "Qwen2.5-VL-7B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3-0.6B": {
        "torch": ("Qwen3/Qwen3-0.6B", ),
        "quantized": {
            "name":
            "Qwen3-0.6B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
            "paths": {
                "NVFP4": ("quantized_models/Qwen3-0.6B-NVFP4", ),
            },
        },
    },
    "Qwen/Qwen3-1.7B": {
        "torch": ("Qwen3/Qwen3-1.7B", ),
        "quantized": {
            "name":
            "Qwen3-1.7B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4": {
        "torch": (
            "Qwen3-30B-A3B-GPTQ-Int4",
            "Qwen3/Qwen3-30B-A3B-GPTQ-Int4",
        ),
    },
    "Qwen/Qwen3-4B": {
        "torch": (
            "Qwen/Qwen3-4B",
            "Qwen3/Qwen3-4B",
        ),
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "torch": ("Qwen3/Qwen3-4B-Instruct-2507", ),
        "quantized": {
            "name":
            "Qwen3-4B-Instruct-2507",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3-8B": {
        "torch": ("Qwen3/Qwen3-8B", ),
        "quantized": {
            "name":
            "Qwen3-8B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3-ASR-0.6B": {
        "torch": ("Qwen3/Qwen3-ASR-0.6B", ),
        "quantized": {
            "name": "Qwen3-ASR-0.6B",
            "quantizations": (
                "AUDFP8",
                "FP8",
                "NVFP4",
            ),
        },
    },
    "Qwen/Qwen3-ASR-1.7B": {
        "torch": ("Qwen3/Qwen3-ASR-1.7B", ),
        "quantized": {
            "name": "Qwen3-ASR-1.7B",
            "quantizations": (
                "AUDFP8",
                "FP8",
                "NVFP4",
            ),
        },
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "torch": ("Qwen3/Qwen3-Omni-30B-A3B-Instruct", ),
        "quantized": {
            "name": "Qwen3-Omni-30B-A3B-Instruct",
            "quantizations": ("NVFP4", ),
        },
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "torch": ("Qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice", ),
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "torch": ("Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice", ),
    },
    "Qwen/Qwen3-VL-2B-Instruct": {
        "torch": ("Qwen3/Qwen3-VL-2B-Instruct", ),
        "quantized": {
            "name":
            "Qwen3-VL-2B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3-VL-4B-Instruct": {
        "torch": ("Qwen3/Qwen3-VL-4B-Instruct", ),
        "quantized": {
            "name":
            "Qwen3-VL-4B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "torch": ("Qwen3/Qwen3-VL-8B-Instruct", ),
        "quantized": {
            "name":
            "Qwen3-VL-8B-Instruct",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "MXFP8",
                "MXFP8-FP8-KV",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3.5-0.8B": {
        "torch": ("Qwen3.5-0.8B", ),
        "quantized": {
            "name":
            "Qwen3.5-0.8B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3.5-27B": {
        "torch": ("Qwen3.5-27B", ),
        "quantized": {
            "name":
            "Qwen3.5-27B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3.5-2B": {
        "torch": ("Qwen3.5-2B", ),
        "quantized": {
            "name":
            "Qwen3.5-2B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "FP8-LMFP8-VITFP8",
                "FP8-LMFP8-VITFP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMFP8-VITFP8",
                "NVFP4-LMFP8-VITFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
                "VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "torch": (
            "Qwen3.5/Qwen3.5-35B-A3B",
            "Qwen3.5-35B-A3B",
        ),
    },
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": {
        "torch": ("Qwen3.5-35B-A3B-GPTQ-Int4", ),
    },
    "Qwen/Qwen3.5-4B": {
        "torch": ("Qwen3.5-4B", ),
        "quantized": {
            "name":
            "Qwen3.5-4B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "VITFP8",
            ),
        },
    },
    "Qwen/Qwen3.5-9B": {
        "torch": ("Qwen3.5-9B", ),
        "quantized": {
            "name":
            "Qwen3.5-9B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-KV",
                "FP8-LMFP8",
                "FP8-LMFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMFP8-FP8-KV",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "VITFP8",
            ),
        },
    },
    "Qwen/Qwen3.6-27B": {
        "torch": ("Qwen3.6-27B", ),
        "quantized": {
            "name":
            "Qwen3.6-27B",
            "quantizations": (
                "FP8",
                "FP8-FP8-KV",
                "FP8-VITFP8",
                "FP8-VITFP8-FP8-KV",
                "INT4-AWQ",
                "INT8-SQ",
                "MXFP8",
                "NVFP4",
                "NVFP4-FP8-KV",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
                "NVFP4-LMNVFP4-FP8-KV",
                "NVFP4-LMNVFP4-VITFP8",
                "NVFP4-LMNVFP4-VITFP8-FP8-KV",
                "NVFP4-VITFP8",
                "NVFP4-VITFP8-FP8-KV",
            ),
        },
    },
    "Qwen/Qwen3.8-27B": {
        "torch": ("Qwen3.8-27B", ),
    },
    "Qwen/Qwen3.6-35B-A3B": {
        "torch": ("Qwen3.6-35B-A3B", ),
    },
    "tencent/HY-MT1.5-7B": {
        "torch": ("HY-MT1.5-7B", ),
    },
    "tencent/Hy-MT2-1.8B": {
        "torch": ("Hy-MT2-1.8B", ),
    },
    "tencent/Hy-MT2-7B": {
        "torch": ("Hy-MT2-7B", ),
    },
    "Tengyunw/qwen3_8b_eagle3": {
        "torch": (
            "Qwen3/qwen3_8b_eagle3",
            "qwen3_8b_eagle3",
        ),
        "quantized": {
            "name":
            "qwen3_8b_eagle3",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B": {
        "torch": ("EAGLE3-LLaMA3.1-Instruct-8B", ),
        "quantized": {
            "name":
            "EAGLE3-LLaMA3.1-Instruct-8B",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "z-lab/gemma-4-26B-A4B-it-DFlash": {
        "torch": ("gemma-4-26B-A4B-it-DFlash", ),
    },
    "z-lab/gemma-4-31B-it-DFlash": {
        "torch": ("gemma-4-31B-it-DFlash", ),
    },
    "z-lab/gemma4-12B-it-DFlash": {
        "torch": ("gemma4-12B-it-DFlash", ),
    },
    "z-lab/Qwen3-4B-DFlash-b16": {
        "torch": ("Qwen3-4B-DFlash-b16", ),
        "quantized": {
            "name": "Qwen3-4B-DFlash-b16",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "z-lab/Qwen3-8B-DFlash-b16": {
        "torch": ("Qwen3-8B-DFlash-b16", ),
        "quantized": {
            "name": "Qwen3-8B-DFlash-b16",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "z-lab/Qwen3.5-27B-DFlash": {
        "torch": ("Qwen3.5-27B-DFlash", ),
        "quantized": {
            "name": "Qwen3.5-27B-DFlash",
            "quantizations": (
                "FP8",
                "NVFP4",
            ),
        },
    },
    "z-lab/Qwen3.5-35B-A3B-DFlash": {
        "torch": ("Qwen3.5-35B-A3B-DFlash", ),
    },
    "z-lab/Qwen3.5-4B-DFlash": {
        "torch": ("Qwen3.5-4B-DFlash", ),
        "quantized": {
            "name": "Qwen3.5-4B-DFlash",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "z-lab/Qwen3.5-9B-DFlash": {
        "torch": ("Qwen3.5-9B-DFlash", ),
        "quantized": {
            "name": "Qwen3.5-9B-DFlash",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "z-lab/Qwen3.6-35B-A3B-DFlash": {
        "torch": ("Qwen3.6-35B-A3B-DFlash", ),
    },
    # Publicly documented checkpoints that are downloaded from Hugging Face
    # when a CI mirror is unavailable.
    "AngelSlim/Qwen3-4B_eagle3": {
        "torch": ("Qwen3/Qwen3-4B_eagle3", ),
    },
    "AngelSlim/Qwen3-8B_eagle3": {},
    "OpenGVLab/InternVL3-14B-AWQ": {},
    "OpenGVLab/InternVL3-14B-hf": {},
    "OpenGVLab/InternVL3-1B-AWQ": {},
    "OpenGVLab/InternVL3-2B-AWQ": {},
    "OpenGVLab/InternVL3-8B-AWQ": {},
    "OpenGVLab/InternVL3-8B-hf": {},
    "OpenGVLab/InternVL3-9B": {},
    "OpenGVLab/InternVL3-9B-Instruct": {},
    "OpenGVLab/InternVL3_5-14B-HF": {},
    "OpenGVLab/InternVL3_5-1B-HF": {},
    "OpenGVLab/InternVL3_5-2B-HF": {},
    "OpenGVLab/InternVL3_5-4B-HF": {},
    "OpenGVLab/InternVL3_5-8B-HF": {},
    "Qwen/Qwen2-0.5B": {
        "torch": ("Qwen2-0.5B", ),
    },
    "Qwen/Qwen2-0.5B-Instruct": {
        "torch": ("Qwen2-0.5B-Instruct", ),
    },
    "Qwen/Qwen2-0.5B-Instruct-AWQ": {},
    "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2-1.5B": {
        "torch": ("Qwen2-1.5B", ),
    },
    "Qwen/Qwen2-1.5B-Instruct": {},
    "Qwen/Qwen2-1.5B-Instruct-AWQ": {},
    "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2-7B": {},
    "Qwen/Qwen2-7B-Instruct": {
        "torch": ("Qwen2-7B-Instruct", ),
    },
    "Qwen/Qwen2-7B-Instruct-AWQ": {
        "torch": ("Qwen2-7B-Instruct-AWQ", ),
    },
    "Qwen/Qwen2-7B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2-Math-1.5B": {},
    "Qwen/Qwen2-Math-1.5B-Instruct": {},
    "Qwen/Qwen2-Math-7B": {},
    "Qwen/Qwen2-Math-7B-Instruct": {},
    "Qwen/Qwen2.5-0.5B": {},
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-1.5B": {},
    "Qwen/Qwen2.5-1.5B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-14B": {},
    "Qwen/Qwen2.5-14B-Instruct": {},
    "Qwen/Qwen2.5-14B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4": {
        "torch": ("Qwen2.5-14B-Instruct-GPTQ-Int4", ),
    },
    "Qwen/Qwen2.5-3B": {},
    "Qwen/Qwen2.5-3B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-7B": {},
    "Qwen/Qwen2.5-7B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-0.5B": {},
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {},
    "Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-Coder-1.5B": {},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-Coder-14B": {},
    "Qwen/Qwen2.5-Coder-14B-Instruct": {},
    "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-Coder-3B": {},
    "Qwen/Qwen2.5-Coder-3B-Instruct": {},
    "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-Coder-7B": {},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {},
    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4": {},
    "Qwen/Qwen2.5-Math-1.5B": {},
    "Qwen/Qwen2.5-Math-1.5B-Instruct": {},
    "Qwen/Qwen2.5-Math-7B": {},
    "Qwen/Qwen2.5-Math-7B-Instruct": {},
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": {},
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": {},
    "Qwen/Qwen3-0.6B-Base": {
        "torch": ("Qwen3/Qwen3-0.6B-Base", ),
    },
    "Qwen/Qwen3-1.7B-Base": {},
    "Qwen/Qwen3-14B": {
        "torch": ("Qwen3/Qwen3-14B", ),
    },
    "Qwen/Qwen3-14B-AWQ": {},
    "Qwen/Qwen3-14B-Base": {},
    "Qwen/Qwen3-4B-AWQ": {},
    "Qwen/Qwen3-4B-Base": {},
    "Qwen/Qwen3-4B-Thinking-2507": {},
    "Qwen/Qwen3-8B-AWQ": {},
    "Qwen/Qwen3-8B-Base": {},
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {},
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {},
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {},
    "Qwen/Qwen3-VL-2B-Thinking": {},
    "Qwen/Qwen3-VL-4B-Thinking": {},
    "Qwen/Qwen3-VL-8B-Thinking": {},
    "Qwen/Qwen3.5-0.8B-Base": {},
    "Qwen/Qwen3.5-2B-Base": {},
    "Qwen/Qwen3.5-4B-Base": {},
    "Qwen/Qwen3.5-9B-Base": {},
    "Rayzl/qwen2.5-vl-7b-eagle3-sgl": {
        "quantized": {
            "name":
            "qwen2.5-vl-7b-eagle3-sgl",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "torch": ("DeepSeek-R1-Distill-Qwen-1.5B", ),
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {},
    "deepseek-ai/dspark_gemma4_12b_block7": {
        "torch": ("dspark/dspark_gemma4_12b_block7", ),
    },
    "deepseek-ai/dspark_qwen3_8b_block7": {
        "torch": ("dspark/dspark_qwen3_8b_block7", ),
    },
    "google/gemma-4-12B-it-assistant": {
        "torch": ("gemma/gemma-4-12B-it-assistant", ),
    },
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized": {},
    "google/gemma-4-31B-it-assistant": {
        "torch": ("gemma/gemma-4-31B-it-assistant", ),
    },
    "google/gemma-4-E4B-it-assistant": {
        "torch": ("gemma/gemma-4-E4B-it-assistant", ),
    },
    "meta-llama/Llama-3.2-1B-Instruct": {},
    "meta-llama/Llama-3.2-3B-Instruct": {},
    "meta-llama/Meta-Llama-3-8B-Instruct": {},
    "nvidia/Cosmos-Reason2-2B": {
        "quantized": {
            "name": "Cosmos-Reason2-2B",
            "quantizations": (
                "FP8",
                "NVFP4",
            ),
        },
    },
    "nvidia/Cosmos-Reason2-2B-FP8": {},
    "nvidia/Cosmos-Reason2-2B-NVFP4": {},
    "nvidia/Cosmos-Reason2-8B-FP8": {},
    "nvidia/Cosmos-Reason2-8B-NVFP4": {},
    "nvidia/Cosmos3-Edge": {
        "torch": ("Cosmos3-Edge", ),
    },
    "nvidia/Cosmos3-Edge-Policy-DROID": {},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": {},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash": {},
    "nvidia/Llama-3.1-8B-Instruct-FP8": {
        "torch": ("Llama-3.1-8B-Instruct-FP8", ),
    },
    "nvidia/Llama-3.1-8B-Instruct-NVFP4": {
        "torch": ("Llama-3.1-8B-Instruct-NVFP4", ),
    },
    "nvidia/Qwen2.5-VL-7B-Instruct-FP8": {
        "torch": ("multimodals/Qwen2.5-VL-7B-Instruct-FP8", ),
    },
    "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4": {},
    "nvidia/Qwen3-14B-FP8": {
        "torch": ("Qwen3/nvidia-Qwen3-14B-FP8", ),
    },
    "nvidia/Qwen3-14B-NVFP4": {
        "torch": ("Qwen3/nvidia-Qwen3-14B-NVFP4", ),
    },
    "nvidia/Qwen3-8B-FP8": {
        "torch": ("Qwen3/nvidia-Qwen3-8B-FP8", ),
    },
    "nvidia/Qwen3-8B-NVFP4": {
        "torch": ("Qwen3/nvidia-Qwen3-8B-NVFP4", ),
    },
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": {},
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {},
}

# These identifiers and mirror layouts are CI-only. They deliberately have no
# Hugging Face fallback and are restricted to QUANT_CHECKPOINT_DIR/private.
PRIVATE_CHECKPOINTS_BY_CI_ID = {
    "private:EAGLE3-Qwen3-4B-v2": {
        "quantized": {
            "name":
            "EAGLE3-Qwen3-4B-v2",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "private:EAGLE3-Qwen3-4B-v2.1": {
        "quantized": {
            "name":
            "EAGLE3-Qwen3-4B-v2.1",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "private:EAGLE3-Qwen3-VL-4B-v1.1": {
        "quantized": {
            "name":
            "EAGLE3-Qwen3-VL-4B-v1.1",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "private:Qwen3-Omni-4B-Instruct-multilingual": {
        "quantized": {
            "name": "Qwen3-Omni-4B-Instruct-multilingual",
            "quantizations": (
                "FP8",
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
    "private:qwen2.5-vl-7b-eagle3-v2": {
        "quantized": {
            "name":
            "qwen2.5-vl-7b-eagle3-v2",
            "quantizations": (
                "FP8",
                "FP8-LMFP8",
                "INT4-AWQ",
                "NVFP4",
                "NVFP4-LMFP8",
                "NVFP4-LMNVFP4",
            ),
        },
    },
    "private:qwen3_5_omni_23a2.6b_final_multilingual_0315": {
        "quantized": {
            "name": "qwen3_5_omni_23a2.6b_final_multilingual_0315",
            "quantizations": ("NVFP4", ),
        },
    },
    "private:qwen3_5_omni_3b_final_multilingual_0324": {
        "quantized": {
            "name": "qwen3_5_omni_3b_final_multilingual_0324",
            "quantizations": (
                "INT4-AWQ",
                "NVFP4",
            ),
        },
    },
}
