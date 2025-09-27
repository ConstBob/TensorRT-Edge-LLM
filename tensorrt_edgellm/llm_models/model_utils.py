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
Model utility functions for loading and setting up LLM models for models quantization and ONNX export.

This module contains functions for loading Hugging Face models,
checking model types, and setting up quantization.
"""

import gc
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from modelopt.torch.quantization.utils import is_quantized_linear
from safetensors.torch import safe_open
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoTokenizer,
                          PreTrainedModel)

from .models.eagle3_draft import Eagle3DraftModel
from .models.llm_model import EdgeLLMModelForCausalLM


def is_nvfp4_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with NVFP4 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (4, 3)
    return False


def is_mxfp8_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with MXFP8 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (8, 0)
    return False


def set_dynamic_quant(model: nn.Module, dtype: str) -> None:
    """Set quantization for nvfp4 and mxfp8 quantization."""
    for module in model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == "fp16" else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
        elif is_mxfp8_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"


def is_vlm(model_dir: str) -> bool:
    """Check if the model is a VLM."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if "vision_config" in cfg:
        print("Set use_prompt_tuning to True")
        return True
    else:
        print("Set use_prompt_tuning to False")
        return False


def is_gptq_model(model: PreTrainedModel) -> bool:
    """Check if the model is a GPTQ model by config."""
    config = model.config.to_dict()
    quant_config = config.get("quantization_config", None)
    return quant_config and quant_config.get("quant_method") == "gptq"


def load_hf_model(
    model_dir: str, dtype: str, device: str
) -> Tuple[Union[AutoModelForCausalLM, AutoModelForImageTextToText],
           AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer with automatic model type detection.
    
    Args:
        model_dir: Directory containing the model files
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If dtype is not supported or model loading fails
    """
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    device = torch.device(device)

    # Try loading as AutoModelForCausalLM first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)
    except Exception:
        # If that fails, try AutoModelForImageTextToText
        try:
            # TODO: Need a WAR to quantize only the language model.
            # In VLMs, the model has both model.language_model and model.vision_model.
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=torch_dtype,
                trust_remote_code=True).to(device)
        except Exception as e:
            raise ValueError(
                f"Could not load model from {model_dir}. Error: {e}")
    if not is_gptq_model(model):
        model.to(torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_llm_model(model_dir: str, dtype: str, max_position_embeddings: int,
                   device: str, enable_reuse_kv_cache: bool,
                   is_eagle_base: bool) -> tuple[nn.Module, bool]:
    """
    Load a language model (standard or EAGLE base).
    
    Args:
        model_dir: Directory containing the torch model
        dtype: Model dtype
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        enable_reuse_kv_cache: Whether to enable persistent KV cache
        is_eagle_base: Whether this is an EAGLE3 base model
        
    Returns:
        tuple: (model, use_prompt_tuning)
    """
    # Determine model type and print message
    if is_eagle_base:
        print(f"Loading eagle3 base model from {model_dir}")
    else:
        print(f"Loading standard model from {model_dir}")

    model, _ = load_hf_model(model_dir, dtype, device)
    use_prompt_tuning = is_vlm(model_dir)
    set_dynamic_quant(model, dtype)

    # Create EdgeLLMModelForCausalLM wrapper.
    # max_position_embeddings is set in EdgeLLMModelForCausalLM
    edge_model = EdgeLLMModelForCausalLM(model, is_eagle_base,
                                         use_prompt_tuning,
                                         max_position_embeddings,
                                         enable_reuse_kv_cache)

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return edge_model, use_prompt_tuning


def load_eagle3_draft_model(draft_model_dir: str, base_model_dir: str,
                            use_prompt_tuning: bool,
                            max_position_embeddings: int, dtype: str,
                            device: str,
                            enable_reuse_kv_cache: bool) -> nn.Module:
    """
    Load an EAGLE draft model with base model for weight copying.
    
    Args:
        draft_model_dir: Directory containing the draft model
        base_model_dir: Directory containing the base model 
        use_prompt_tuning: Whether the model uses prompt tuning
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        enable_reuse_kv_cache: Whether to enable KV cache reuse
        
    Returns:
        nn.Module: Draft model
    """
    print(f"Loading eagle3 draft model from {draft_model_dir}")
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Load draft model using from_pretrained. Draft model only support fp16.
    draft_model = Eagle3DraftModel.from_pretrained(
        draft_model_dir=draft_model_dir,
        base_model_dir=base_model_dir,
        use_prompt_tuning=use_prompt_tuning,
        max_position_embeddings=max_position_embeddings,
        enable_reuse_kv_cache=enable_reuse_kv_cache,
        device=device).eval().to(device)
    if not is_gptq_model(draft_model):
        draft_model.to(torch_dtype)

    set_dynamic_quant(draft_model, dtype)

    return draft_model


def load_tensor_by_candidate_keys(model_dir: str, keys_candidate: List[str],
                                  device: str) -> Optional[torch.Tensor]:
    """
    Search all .safetensors shards in `model_dir` and lazily load
    the first matching tensor in `candidate_keys`.

    Returns
    -------
    tensor : Optional[torch.Tensor]
        The requested tensor moved to `device`.
    """
    model_dir = Path(model_dir)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for shard_path in safetensor_files:
        # Lazy/MMAP open – only metadata is read
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in keys_candidate:
                # try candidates in order
                if key in keys:
                    print(f"Using {key} from {model_dir}/{shard_path.name}")
                    tensor = f.get_tensor(key)  # actually loads data
                    return tensor.to(device)  # move to desired device

    return None
