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
"""
LLM Quantization Module for TensorRT Edge-LLM.

This module provides quantization utilities for large language models using NVIDIA ModelOpt.
It supports various quantization schemes including FP8, INT4 AWQ, and NVFP4.
"""

import json
import os
import time
from typing import Any, Dict, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.utils import is_quantized
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer)

from ..llm_models.model_utils import (_is_qwen3_asr_model,
                                      load_eagle3_draft_model, load_hf_model)
from ..llm_models.models.eagle3_draft import Eagle3DraftModel
from .calib_dataloaders import (get_audio_llm_calib_dataloader,
                                get_text_calib_dataloader)
from .quantization_utils import (enable_huggingface_checkpointing_patch,
                                 quantize_draft_model, quantize_model)

enable_huggingface_checkpointing_patch()

# Quantization configuration constants
# FP8 quantization configuration for language model head.
FP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "default": {
            "enable": False
        }
    }
}

# INT4 AWQ quantization configuration for language model head.
INT4_AWQ_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {
                -1: 128,
                "type": "static"
            },
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# NVFP4 quantization configuration for language model head.
NVFP4_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# MXFP8 quantization configuration for language model head.
MXFP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "default": {
            "enable": False
        }
    }
}

# FP8 attention configuration: enables FP8 Q/K/V BMM quantizers + attention output quantizer.
# ModelOpt's QuantAttention creates q/k/v_bmm_quantizer on attention modules.
# FP8_KV_CFG sets "default": {"enable": false}, so we must explicitly enable each.
FP8_ATTN_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    }
}

# Configuration to disable visual model quantization.
DISABLE_VISUAL_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*visual.*": {
            "enable": False
        },
    }
}

# Configuration to disable audio tower quantization (e.g. Qwen3-ASR).
DISABLE_AUDIO_TOWER_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*audio_tower.*": {
            "enable": False
        },
    }
}

# Already merged the vision LoRA and audio is currently not supported, so disable them.
DISABLE_Phi4MM_VISUAL_AUDIO_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*audio_embed.*": {
            "enable": False
        },
        "*image_embed.*": {
            "enable": False
        },
    }
}


def get_llm_quant_config(
        quantization: Optional[str], lm_head_quantization: Optional[str],
        kv_cache_quantization: Optional[str]) -> Dict[str, Any]:
    """
    Get quantization configuration for LLM models.
    
    Args:
        quantization: Optional quantization method
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method
            (enables FP8 KV cache + FP8 FMHA compute)
        
    Returns:
        Dict containing quantization configuration
        
    Raises:
        ValueError: If quantization method is not supported
    """
    # Get base config
    if quantization is None:
        quant_cfg = {"quant_cfg": {}, "algorithm": "max"}
    elif quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    elif quantization == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    elif quantization == "mxfp8":
        quant_cfg = mtq.MXFP8_DEFAULT_CFG.copy()
    elif quantization == "int8_sq":
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    # Add LM head quantization if specified
    if lm_head_quantization is not None:
        # Remove any existing lm_head configuration
        quant_cfg["quant_cfg"] = {
            k: v
            for k, v in quant_cfg["quant_cfg"].items() if "*lm_head" not in k
        }

        if lm_head_quantization == "fp8":
            quant_cfg["quant_cfg"].update(FP8_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "nvfp4":
            quant_cfg["quant_cfg"].update(NVFP4_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "mxfp8":
            quant_cfg["quant_cfg"].update(MXFP8_LM_HEAD_CONFIG["quant_cfg"])

    # Add attention module quantization if specified (FP8 KV cache + FP8 FMHA compute)
    if kv_cache_quantization is not None:
        if kv_cache_quantization == "fp8":
            quant_cfg["quant_cfg"].update(FP8_ATTN_CONFIG["quant_cfg"])

    # Disable visual model
    quant_cfg["quant_cfg"].update(DISABLE_VISUAL_CONFIG["quant_cfg"])

    # Disable audio tower (e.g. Qwen3-ASR)
    quant_cfg["quant_cfg"].update(DISABLE_AUDIO_TOWER_CONFIG["quant_cfg"])

    # Disable vision and audio models in Phi-4MM
    quant_cfg["quant_cfg"].update(
        DISABLE_Phi4MM_VISUAL_AUDIO_CONFIG["quant_cfg"])
    return quant_cfg


def quantize_llm(
    model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    quantization: Optional[str],
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
    model_dir: Optional[str] = None,
) -> Union[AutoModelForCausalLM, AutoModelForImageTextToText]:
    """
    Quantize a language model using the specified quantization method.

    For ASR models (e.g. Qwen3-ASR), audio calibration data from LibriSpeech
    is used instead of text so that the LLM backbone sees realistic activations
    produced by the audio encoder, following the same pattern as Flux
    transformer quantization in TensorRT OSS.

    Args:
        model: The model to quantize (causal LM or image-text model)
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method
            (enables FP8 KV cache + FP8 FMHA compute)
        model_dir: Original model directory (used to detect ASR models and
            load the audio processor).

    Returns:
        Quantized model

    Raises:
        AssertionError: If quantization method is not supported
    """
    assert (quantization is not None) or (lm_head_quantization is not None) or (kv_cache_quantization is not None), \
        "At least one of 'quantization', 'lm_head_quantization', or 'kv_cache_quantization' must be set (not all None)."
    assert quantization in [
        None, "fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"
    ]
    assert lm_head_quantization in [None, "fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # Get calibration dataloader — use audio data for ASR models
    use_audio_calib = model_dir is not None and _is_qwen3_asr_model(model_dir)

    if use_audio_calib:
        if dataset_dir == "cnn_dailymail":
            dataset_dir = "openslr/librispeech_asr"
            print("ASR model detected; switching calibration dataset to "
                  f"'{dataset_dir}' (override with --dataset_dir).")
        data_loader = get_audio_llm_calib_dataloader(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            num_samples=512,
        )
    else:
        if quantization is None or "int4" in quantization:
            batch_size = 16
        else:
            batch_size = 1
        data_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                                dataset_dir=dataset_dir,
                                                batch_size=batch_size,
                                                num_samples=512,
                                                max_length=512)

    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization)
    model = quantize_model(model, quant_config, data_loader)

    return model


def quantize_draft(
    base_model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    draft_model: Union[Eagle3DraftModel],
    tokenizer: AutoTokenizer,
    quantization: str,
    dataset_dir: str,
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
) -> Union[Eagle3DraftModel]:
    """
    Quantize a language model using the specified quantization method.
    
    Args:
        base_model: Based model which is used to generate inputs for the draft model.
        draft_model: The draft model to quantize
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4", "int8_sq")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method
            (enables FP8 KV cache + FP8 FMHA compute)

    Returns:
        Quantized draft model
        
    Raises:
        AssertionError: If quantization method is not supported
    """
    assert quantization in ["fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8"]
    assert lm_head_quantization in [None, "fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # Get calibration dataloader
    if "int4" in quantization:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                            dataset_dir=dataset_dir,
                                            batch_size=batch_size,
                                            num_samples=512,
                                            max_length=512)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization)
    model = quantize_draft_model(base_model, draft_model, quant_config,
                                 data_loader)

    return model


def quantize_and_save_llm(model_dir: str,
                          output_dir: str,
                          quantization: Optional[str] = None,
                          dtype: str = "fp16",
                          dataset_dir: str = "cnn_dailymail",
                          lm_head_quantization: Optional[str] = None,
                          kv_cache_quantization: Optional[str] = None,
                          device: str = "cuda",
                          unified_checkpoint: bool = False) -> None:
    """
    Load a model, quantize it if specified, and save the result.
    
    This is the main entry point for quantizing language models. It supports various
    quantization schemes including FP8, INT4 AWQ, and NVFP4.
    
    Args:
        model_dir: Directory containing the input HuggingFace model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply (None, "fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8")
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for calibration data
        lm_head_quantization: Optional separate quantization for language model head (only "fp8", "nvfp4", and "mxfp8" are currently supported)
        kv_cache_quantization: Optional attention quantization (enables FP8 KV cache + FP8 FMHA compute)
        device: Device to use for model loading and quantization ("cuda", "cpu")
        unified_checkpoint: Whether to export unified checkpoint

    Raises:
        ValueError: If model loading fails or quantization parameters are invalid
    """
    start_time = time.time()
    # Load model and tokenizer
    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)

    # Qwen3ASRForConditionalGeneration has no forward(); add one that
    # delegates to the thinker so the calibration loop can call
    # model(input_ids).
    if _is_qwen3_asr_model(model_dir):
        type(model).forward = lambda self, *args, **kwargs: self.thinker(
            *args, **kwargs)

    if is_quantized(model):
        print(f"Model is already quantized, skipping quantization.")
    else:
        model = quantize_llm(model,
                             tokenizer,
                             dataset_dir,
                             quantization,
                             lm_head_quantization,
                             kv_cache_quantization,
                             model_dir=model_dir)

    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize generation_config so save_pretrained doesn't reject it.
    # Some upstream models ship with conflicting flags (e.g. temperature set
    # while do_sample=False).
    if hasattr(model, "generation_config"):
        gc = model.generation_config
        if getattr(gc, "temperature",
                   None) is not None and not getattr(gc, "do_sample", True):
            gc.temperature = None

    if unified_checkpoint:  # Original checkpoint read by ModelOpt
        from modelopt.torch.export import export_hf_checkpoint
        with torch.inference_mode():
            export_hf_checkpoint(
                model,  # The quantized model.
                export_dir=
                output_dir,  # The directory where the exported files will be stored.
            )
    else:  # Unified checkpoint read by AutoDeploy
        with torch.inference_mode():
            model.save_pretrained(output_dir)
        # Save the quant config
        quant_config = get_quant_config(model)
        with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
            json.dump(quant_config, f)

    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")


def quantize_and_save_draft(
    base_model_dir: str,
    draft_model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "fp16",
    dataset_dir: str = "cnn_dailymail",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
) -> None:
    """
    Load an EAGLE draft model, quantize it if specified, and save the result.
    
    This is the main entry point for quantizing EAGLE draft models. It requires
    both a base model and draft model directory.
    
    Args:
        base_model_dir: Directory containing the base HuggingFace model
        draft_model_dir: Directory containing the EAGLE draft model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply (None, "fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8")
        device: Device to use for model loading and quantization ("cuda", "cpu")
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for calibration data
        lm_head_quantization: Optional separate quantization for language model head (only "fp8", "nvfp4", and "mxfp8" are currently supported)
        kv_cache_quantization: Optional attention quantization (enables FP8 KV cache + FP8 FMHA compute)

    Raises:
        ValueError: If model loading fails or quantization parameters are invalid
    """
    start_time = time.time()

    draft_model = load_eagle3_draft_model(draft_model_dir, base_model_dir,
                                          dtype, device)

    if is_quantized(draft_model):
        print(f"Draft Model is already quantized, skipping quantization.")
    else:
        base_model, tokenizer, _ = load_hf_model(base_model_dir, dtype, device)
        draft_model = quantize_draft(base_model, draft_model, tokenizer,
                                     quantization, dataset_dir,
                                     lm_head_quantization,
                                     kv_cache_quantization)
    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        draft_model.save_pretrained(output_dir)

    # Save the quant config
    quant_config = get_quant_config(draft_model)
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(quant_config, f)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")
