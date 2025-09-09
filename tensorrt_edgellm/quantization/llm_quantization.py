# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Optional, Tuple, Union

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer)

from .quantization_utils import quantize_model

mto.enable_huggingface_checkpointing()

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

# Configuration to disable visual model quantization.
DISABLE_VISUAL_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*visual.*": {
            "enable": False
        },
    }
}


def get_llm_calib_dataloader(
    tokenizer: AutoTokenizer,
    dataset_dir: str = "cnn_dailymail",
    batch_size: int = 1,
    num_samples: int = 512,
    max_length: int = 512,
) -> DataLoader:
    """
    Create a calibration dataloader for LLM quantization.
    
    Args:
        tokenizer: HuggingFace tokenizer for text processing
        dataset_dir: Dataset name or local directory path
        batch_size: Batch size for the dataloader
        num_samples: Number of samples to use for calibration
        max_length: Maximum sequence length for tokenization
        
    Returns:
        DataLoader: Calibration dataloader with tokenized inputs
        
    Raises:
        NotImplementedError: If dataset format is not supported
    """
    print(f"Loading calibration dataset from {dataset_dir}")
    if "cnn_dailymail" in dataset_dir:
        dataset = load_dataset(dataset_dir, name="3.0.0", split="train")
        dataset = dataset["article"][:num_samples]
    elif os.path.isdir(dataset_dir):
        print(
            f"Recognized local dataset repo {dataset_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_dir, split="train")
        dataset = dataset["text"][:num_samples]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=max_length)

    calib_dataloader = DataLoader(batch_encoded["input_ids"],
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_llm_quant_config(
        quantization: str,
        lm_head_quantization: Optional[str] = None) -> Dict[str, Any]:
    """
    Get quantization configuration for LLM models.
    
    Args:
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4")
        lm_head_quantization: Optional LM head quantization method
        
    Returns:
        Dict containing quantization configuration
        
    Raises:
        ValueError: If quantization method is not supported
    """
    # Get base config
    if quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    elif quantization == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    elif quantization == "mxfp8":
        quant_cfg = mtq.MXFP8_DEFAULT_CFG.copy()
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
        elif lm_head_quantization == "int4_awq":
            quant_cfg["quant_cfg"].update(INT4_AWQ_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "nvfp4":
            quant_cfg["quant_cfg"].update(NVFP4_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "mxfp8":
            quant_cfg["quant_cfg"].update(MXFP8_LM_HEAD_CONFIG["quant_cfg"])

    # Disable visual model
    quant_cfg["quant_cfg"].update(DISABLE_VISUAL_CONFIG["quant_cfg"])

    return quant_cfg


def quantize_llm(
    model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    tokenizer: AutoTokenizer,
    quantization: str,
    dataset_dir: str = "cnn_dailymail",
    lm_head_quantization: Optional[str] = None,
) -> Union[AutoModelForCausalLM, AutoModelForImageTextToText]:
    """
    Quantize a language model using the specified quantization method.
    
    Args:
        model: The model to quantize (causal LM or image-text model)
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        
    Returns:
        Quantized model
        
    Raises:
        AssertionError: If quantization method is not supported
    """
    assert quantization in ["fp8", "int4_awq", "nvfp4", "mxfp8"]
    assert lm_head_quantization in [None, "fp8", "int4_awq", "nvfp4", "mxfp8"]

    # Get calibration dataloader
    if "int4" in quantization:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_llm_calib_dataloader(tokenizer=tokenizer,
                                           dataset_dir=dataset_dir,
                                           batch_size=batch_size)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization)
    quantized_model = quantize_model(model, quant_config, data_loader)

    return quantized_model


def load_hf_model(
    model_dir: str,
    torch_dtype: str = "fp16"
) -> Tuple[Union[AutoModelForCausalLM, AutoModelForImageTextToText],
           AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer with automatic model type detection.
    
    Args:
        model_dir: Directory containing the model files
        torch_dtype: Torch data type ("fp16")
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If torch_dtype is not supported or model loading fails
    """
    # Convert torch_dtype string to torch dtype
    if torch_dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    # Try loading as AutoModelForCausalLM first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype,
            trust_remote_code=True).to(dtype).cuda()
    except Exception:
        # If that fails, try AutoModelForImageTextToText
        try:
            # TODO: Need a WAR to quantize only the language model.
            # In VLMs, the model has both model.language_model and model.vision_model.
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=dtype,
                trust_remote_code=True).to(dtype).cuda()
        except Exception as e:
            raise ValueError(
                f"Could not load model from {model_dir}. Error: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def quantize_and_save_llm(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    torch_dtype: str = "fp16",
    dataset_dir: str = "cnn_dailymail",
    lm_head_quantization: Optional[str] = None,
) -> None:
    """
    Load a model, quantize it if specified, and save the result.
    
    Args:
        model_dir: Directory containing the input model
        output_dir: Directory to save the quantized model
        quantization: Optional quantization method to apply (None, fp8, int4_awq, nvfp4)
        torch_dtype: Torch data type for model loading (fp16)
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method (None, fp8, int4_awq, nvfp4)
        
    Raises:
        ValueError: If model loading fails
    """
    start_time = time.time()
    # Load model and tokenizer
    model, tokenizer = load_hf_model(model_dir, torch_dtype)

    if quantization is not None:
        if is_quantized(model):
            print(f"Model is already quantized, skipping quantization.")
        else:
            model = quantize_llm(model, tokenizer, quantization, dataset_dir,
                                 lm_head_quantization)
    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the quant config
    quant_config = get_quant_config({
        name: module
        for name, module in model.named_modules()
    })
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(quant_config, f)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")
