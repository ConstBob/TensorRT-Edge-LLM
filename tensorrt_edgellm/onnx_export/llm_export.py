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
ONNX Export Module for LLM Models with Custom Attention Plugin

This module provides functionality to export different types of LLM models to ONNX format
with custom attention plugin integration. It supports standard models and EAGLE models.

ONNX Input Naming Conventions:
- input_ids: Input token IDs for all models (standard and prompt tuning)
- image_embeds: Image embeddings for prompt tuning models only
- hidden_states_input: Renamed from hidden_states_from_base for ONNX export
- attention_pos_id: Renamed from position_ids for ONNX export

Model Loading Strategy:
- Standard models: Use AutoModelForCausalLM/AutoModelForImageTextToText detection
- EAGLE models: Load both base and draft models with weight copying
"""

import gc
import json
import os
import shutil
import time
from typing import Any, Dict

import modelopt.torch.opt as mto
import numpy as np
import torch
import torch.nn as nn
from modelopt.torch.quantization.utils import is_quantized_linear
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

mto.enable_huggingface_checkpointing()

from ..llm_models.layers.attention_plugin import \
    register_attention_plugin_onnx_symbolic_functions
from ..llm_models.layers.gather_nd import \
    register_gather_nd_onnx_symbolic_functions
from ..llm_models.layers.int4_gemm_plugin import (
    register_int4_gemm_plugin_onnx_symbolic_functions,
    replace_torch_quant_linear_with_plugin)
from ..llm_models.models.eagle2_draft import Eagle2DraftModel
from ..llm_models.models.eagle3_draft import Eagle3DraftModel
from ..llm_models.models.llm_model import EdgeLLMModelForCausalLM
from .config_export import export_llm_config
from .onnx_utils import export_onnx


def is_nvfp4_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with NVFP4 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (4, 3)
    else:
        return False


def is_mxfp8_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with MXFP8 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (8, 0)
    else:
        return False


def save_tokenizer_to_output_dir(model_dir: str, output_dir: str) -> None:
    """Save tokenizer files from model_dir to output_dir."""
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]

    saved_files = []
    for filename in tokenizer_files:
        src_path = os.path.join(model_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                saved_files.append(filename)
                print(f"Saved tokenizer file: {filename}")
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")

    if saved_files:
        print(
            f"Tokenizer files saved to {output_dir}: {', '.join(saved_files)}")
    else:
        print(f"Warning: No tokenizer files found in {model_dir}")


def save_d2t_for_eagle3_draft(draft_model: nn.Module, output_dir: str) -> None:
    """Save d2t.bin for Eagle3 draft model."""
    d2t_tensor = draft_model.d2t
    d2t_path = os.path.join(output_dir, "d2t.bin")
    with open(d2t_path, 'wb') as f:
        f.write(d2t_tensor.cpu().numpy().astype(np.int32).tobytes())
    print(f"Saved d2t.bin to {output_dir}")


def load_model(model_dir: str,
               dtype: torch.dtype = torch.float16,
               max_position_embeddings: int = 4096,
               device: str = "cuda",
               is_eagle2_base: bool = False,
               is_eagle3_base: bool = False) -> tuple[nn.Module, bool]:
    """
    Load a language model (standard or EAGLE base).
    
    Args:
        model_dir: Directory containing the torch model
        dtype: Model dtype
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        is_eagle2_base: Whether this is an EAGLE2 base model
        is_eagle3_base: Whether this is an EAGLE3 base model
        
    Returns:
        tuple: (model, use_prompt_tuning)
    """
    # Determine model type and print message
    if is_eagle2_base:
        print(f"Loading eagle2 base model from {model_dir}")
    elif is_eagle3_base:
        print(f"Loading eagle3 base model from {model_dir}")
    else:
        print(f"Loading standard model from {model_dir}")

    # Try loading as AutoModelForCausalLM first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=dtype).eval().to(device)
        use_prompt_tuning = False
    except Exception:
        # If that fails, try AutoModelForImageTextToText
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir, trust_remote_code=True,
                torch_dtype=dtype).eval().to(device)
            use_prompt_tuning = True
            print(
                "Detected AutoModelForImageTextToText, enabling prompt tuning")
        except Exception as e:
            raise ValueError(
                f"Could not load model from {model_dir}. Error: {e}")

    # Create EdgeLLMModelForCausalLM wrapper.
    # max_position_embeddings is set in EdgeLLMModelForCausalLM
    edge_model = EdgeLLMModelForCausalLM(model, is_eagle2_base, is_eagle3_base,
                                         use_prompt_tuning,
                                         max_position_embeddings)

    # WAR for nvfp4 and mxfp8 quantization
    for module in edge_model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == torch.float16 else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
        elif is_mxfp8_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return edge_model, use_prompt_tuning


def load_eagle_draft_model(model_dir: str,
                           eagle2: bool,
                           base_model: nn.Module,
                           max_position_embeddings: int = 4096,
                           use_prompt_tuning: bool = False,
                           device: str = "cuda") -> nn.Module:
    """
    Load an EAGLE draft model with base model for weight copying.
    
    Args:
        model_dir: Directory containing the draft model
        eagle2: Whether this is EAGLE2 (True) or EAGLE3 (False)
        base_model: Base model to copy weights from if needed
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        use_prompt_tuning: Whether the model uses prompt tuning
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        
    Returns:
        nn.Module: Draft model
    """
    eagle_type = "eagle2" if eagle2 else "eagle3"
    print(f"Loading {eagle_type} draft model from {model_dir}")

    # Load draft model using from_pretrained. Draft model only support fp16.
    if not eagle2:  # EAGLE3
        draft_model = Eagle3DraftModel.from_pretrained(
            draft_model_dir=model_dir,
            base_model=base_model,
            use_prompt_tuning=use_prompt_tuning,
            max_position_embeddings=max_position_embeddings).to(
                torch.float16).eval().to(device)
    else:  # EAGLE2
        draft_model = Eagle2DraftModel.from_pretrained(
            draft_model_dir=model_dir,
            base_model=base_model,
            use_prompt_tuning=use_prompt_tuning,
            max_position_embeddings=max_position_embeddings).to(
                torch.float16).eval().to(device)

    # Overwrite the max_position_embeddings in the model config
    if hasattr(draft_model, 'config'):
        draft_model.config.max_position_embeddings = max_position_embeddings

    return draft_model


def create_dummy_inputs(model: nn.Module,
                        is_eagle_base: bool = False,
                        is_eagle_draft: bool = False,
                        eagle2: bool = False,
                        use_prompt_tuning: bool = False) -> Dict[str, Any]:
    """
    Create dummy inputs for ONNX export.
    
    Args:
        model: The model to create inputs for
        is_eagle_base: Whether this is an EAGLE base model
        is_eagle_draft: Whether this is an EAGLE draft model
        eagle2: Whether this is EAGLE2 (True) or EAGLE3 (False)
        use_prompt_tuning: Whether the model uses prompt tuning
        
    Returns:
        dict: Dictionary containing dummy inputs
    """
    # Use hardcoded values
    batch_size = 1
    seq_len = 2
    past_len = 2
    image_token_len = 2

    print(
        f"Creating dummy inputs with batch_size={batch_size}, seq_len={seq_len}, past_len={past_len}"
    )

    # Get model configuration
    model_config = model.config
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    head_dim = hidden_size // num_heads
    max_position_embeddings = model_config.max_position_embeddings

    device = next(model.parameters()).device

    # Create dummy past key values
    past_key_values = []
    for _ in range(num_layers):
        # Only FP16 KV Cache is supported for now. More precision will be supported in the future.
        past_key_value = torch.randn(batch_size,
                                     2,
                                     num_kv_heads,
                                     seq_len,
                                     head_dim,
                                     dtype=torch.float16,
                                     device=device)
        past_key_values.append(past_key_value)

    # Create last_token_ids
    if not is_eagle_base and not is_eagle_draft:
        last_token_ids = torch.full([batch_size, 1],
                                    seq_len - 1,
                                    dtype=torch.int64,
                                    device=device)
    else:
        num_selected_tokens = 2
        last_token_ids = torch.full([batch_size * num_selected_tokens],
                                    seq_len - 1,
                                    dtype=torch.int64,
                                    device=device)

    # Create rope_rotary_cos_sin
    rope_rotary_cos_sin = torch.randn(batch_size,
                                      max_position_embeddings,
                                      head_dim,
                                      dtype=torch.float32,
                                      device=device)

    # Create context_lengths
    context_lengths = torch.full([batch_size],
                                 past_len + seq_len,
                                 dtype=torch.int32,
                                 device=device)

    # Base inputs that all models need
    base_inputs = {
        'past_key_values': tuple(past_key_values),
        'last_token_ids': last_token_ids,
        'rope_rotary_cos_sin': rope_rotary_cos_sin,
        'context_lengths': context_lengths
    }

    # Create input_ids for all models
    input_ids = torch.randint(0,
                              model_config.vocab_size, (batch_size, seq_len),
                              dtype=torch.int32,
                              device=device)
    base_inputs['input_ids'] = input_ids

    # Create image_embeds for all models (needed for ONNX export alignment)
    if use_prompt_tuning:
        image_embeds = torch.randn(image_token_len,
                                   hidden_size,
                                   dtype=torch.float16,
                                   device=device)
        base_inputs['image_embeds'] = image_embeds

    # Create position_ids and attention_mask for all models
    position_ids = torch.arange(seq_len, dtype=torch.int32,
                                device=device).unsqueeze(0).expand(
                                    batch_size, -1)
    attention_mask = torch.ones(batch_size,
                                seq_len,
                                seq_len + past_len,
                                dtype=torch.int32,
                                device=device)
    base_inputs['position_ids'] = position_ids
    base_inputs['attention_mask'] = attention_mask

    # Add EAGLE-specific inputs
    if is_eagle_draft:
        target_hidden_size = getattr(model_config, 'target_hidden_size',
                                     hidden_size)
        if not eagle2:  # EAGLE3
            target_hidden_size = target_hidden_size * 3
        base_inputs['hidden_states_from_base'] = torch.randn(
            batch_size,
            seq_len,
            target_hidden_size,
            dtype=torch.float16,
            device=device)
        base_inputs['hidden_states_from_draft'] = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=torch.float16,
            device=device)

    return base_inputs


def export_model_to_onnx(model: nn.Module,
                         dummy_inputs: Dict[str, Any],
                         output_dir: str,
                         is_eagle_base: bool = False,
                         is_eagle_draft: bool = False,
                         use_prompt_tuning: bool = False) -> None:
    """
    Export the model to ONNX format.
    
    Args:
        model: The model to export
        dummy_inputs: Dummy inputs for tracing
        output_dir: Directory to save the ONNX model
        is_eagle_base: Whether this is an EAGLE base model
        is_eagle_draft: Whether this is an EAGLE draft model
        use_prompt_tuning: Whether the model uses prompt tuning
    """
    print(f"Exporting model to ONNX format: {output_dir}")

    try:
        # Set model to evaluation mode
        model.eval()

        # Get model configuration for dynamic shapes
        model_config = model.config
        num_layers = model_config.num_hidden_layers

        # Prepare inputs
        base_inputs = [
            dummy_inputs['past_key_values'],
            dummy_inputs['rope_rotary_cos_sin'],
            dummy_inputs['context_lengths'], dummy_inputs['last_token_ids']
        ]

        if is_eagle_draft:
            base_inputs.append(dummy_inputs['hidden_states_from_base'])
            base_inputs.append(dummy_inputs['hidden_states_from_draft'])

        if is_eagle_base or is_eagle_draft:
            # EAGLE models use position_ids and attention_mask
            base_inputs.extend(
                [dummy_inputs['position_ids'], dummy_inputs['attention_mask']])
        else:
            # Standard models pass None for position_ids and attention_mask
            base_inputs.extend([None, None])

        # Add input_ids (always present)
        base_inputs.append(dummy_inputs['input_ids'])

        # Add image_embeds (only for prompt tuning models)
        if use_prompt_tuning:
            base_inputs.append(dummy_inputs['image_embeds'])

        inputs = tuple(base_inputs)

        # Create input names
        input_names = [f'past_key_values.{i}' for i in range(num_layers)] + [
            'rope_rotary_cos_sin', 'context_lengths', 'last_token_ids'
        ]

        # TODO: Change this name to hidden_states_from_base
        if is_eagle_draft:
            input_names.append('hidden_states_input')
            input_names.append('hidden_states_from_draft')

        if is_eagle_base or is_eagle_draft:
            input_names.extend(['attention_pos_id', 'attention_mask'])

        # Add input_ids (always present)
        input_names.append('input_ids')

        if use_prompt_tuning:
            input_names.append('image_embeds')

        # Create output names
        if is_eagle_base or is_eagle_draft:
            output_names = ['logits', 'hidden_states'] + [
                f'present_key_values.{i}' for i in range(num_layers)
            ]
        else:
            output_names = ['logits'] + [
                f'present_key_values.{i}' for i in range(num_layers)
            ]

        # Create dynamic shapes
        past_key_values_shapes = {
            f"past_key_values.{i}": {
                0: "batch_size",
                3: "past_len"
            }
            for i in range(num_layers)
        }

        present_key_values_shapes = {
            f"present_key_values.{i}": {
                0: "batch_size"
            }
            for i in range(num_layers)
        }

        dynamic_axes = {
            "input_ids": {
                0: "batch_size",
                1: "seq_len"
            },
            **past_key_values_shapes, "rope_rotary_cos_sin": {
                0: "rope_batch_size",
                1: "max_position_embeddings"
            },
            "context_lengths": {
                0: "batch_size"
            },
            "last_token_ids": {
                0: "indices_len"
            },
            **present_key_values_shapes
        }

        if is_eagle_draft:
            dynamic_axes.update({
                "hidden_states_input": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "hidden_states_from_draft": {
                    0: "batch_size",
                    1: "seq_len"
                }
            })

        if is_eagle_base or is_eagle_draft:
            dynamic_axes.update({
                "attention_pos_id": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "seq_len",
                    2: "seq_len_padded"
                }
            })

        if use_prompt_tuning:
            dynamic_axes["image_embeds"] = {0: "image_token_len"}

        # Register ONNX symbolic functions
        register_attention_plugin_onnx_symbolic_functions()
        register_gather_nd_onnx_symbolic_functions()

        # Export to ONNX
        export_onnx(model, inputs, output_dir, input_names, output_names,
                    dynamic_axes)

    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def export_standard_model(model_dir: str,
                          output_dir: str,
                          max_position_embeddings: int = 4096,
                          device: str = "cuda") -> str:
    """
    Export a standard model to ONNX format.
    
    Args:
        model_dir: Directory containing the model
        output_dir: Directory to save the exported ONNX model
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
    
    Returns:
        str: Path to the output directory
    """
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, use_prompt_tuning = load_model(
        model_dir,
        dtype=torch.float16,
        max_position_embeddings=max_position_embeddings,
        device=device)

    if hasattr(model.config, "quantization_config"):
        print("Detected quantization config in the loaded model")
        if model.config.quantization_config.quant_method == "gptq":
            print(
                "Detected GPTQ quantization, replacing TorchQuantLinear with Int4GemmPluginModule"
            )
            register_int4_gemm_plugin_onnx_symbolic_functions()
            model = replace_torch_quant_linear_with_plugin(model)

    # Create dummy inputs
    dummy_inputs = create_dummy_inputs(model,
                                       is_eagle_base=False,
                                       is_eagle_draft=False,
                                       eagle2=False,
                                       use_prompt_tuning=use_prompt_tuning)

    # Export to ONNX
    export_model_to_onnx(model,
                         dummy_inputs,
                         output_dir,
                         is_eagle_base=False,
                         is_eagle_draft=False,
                         use_prompt_tuning=use_prompt_tuning)

    # Save model configuration
    model_config = export_llm_config(model.config, 'llm')
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Model configuration saved to {config_path}")

    # Save tokenizer files
    save_tokenizer_to_output_dir(model_dir, output_dir)

    end_time = time.time()
    print(
        f"Export completed successfully in {end_time - start_time}s. Files saved to: {output_dir}"
    )
    return output_dir


def export_eagle_models(base_model_dir: str,
                        draft_model_dir: str,
                        output_dir: str,
                        eagle2: bool,
                        max_position_embeddings: int = 4096,
                        device: str = "cuda") -> str:
    """
    Export complete EAGLE model (both base and draft) to ONNX format.
    
    Args:
        base_model_dir: Directory containing the base model
        draft_model_dir: Directory containing the draft model
        output_dir: Directory to save the exported ONNX models
        eagle2: Whether this is EAGLE2 (True) or EAGLE3 (False)
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
    
    Returns:
        str: Path to the output directory
    """
    start_time = time.time()

    eagle_type = "eagle2" if eagle2 else "eagle3"
    print(f"Exporting {eagle_type} model (base + draft)")

    # Create subdirectories
    base_output_dir = os.path.join(output_dir, "base")
    draft_output_dir = os.path.join(output_dir, "draft")
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(draft_output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model from {base_model_dir}")
    base_model, use_prompt_tuning = load_model(base_model_dir,
                                               torch.float16,
                                               max_position_embeddings,
                                               device,
                                               is_eagle2_base=eagle2,
                                               is_eagle3_base=not eagle2)

    # Load draft model with base model for weight copying
    print(f"Loading draft model from {draft_model_dir}")
    draft_model = load_eagle_draft_model(draft_model_dir, eagle2, base_model,
                                         max_position_embeddings,
                                         use_prompt_tuning, device)

    # Export draft model
    print(f"Exporting draft model to {draft_output_dir}")
    draft_dummy_inputs = create_dummy_inputs(
        draft_model,
        is_eagle_base=False,
        is_eagle_draft=True,
        eagle2=eagle2,
        use_prompt_tuning=use_prompt_tuning)
    export_model_to_onnx(draft_model,
                         draft_dummy_inputs,
                         draft_output_dir,
                         is_eagle_base=False,
                         is_eagle_draft=True,
                         use_prompt_tuning=use_prompt_tuning)

    # Save draft model configuration
    draft_config = export_llm_config(draft_model.config, 'eagle_draft', eagle2)
    config_path = os.path.join(draft_output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(draft_config, f, indent=2)
    print(f"Draft model configuration saved to {config_path}")

    if not eagle2:
        save_d2t_for_eagle3_draft(draft_model, draft_output_dir)

    # Export base model
    print(f"Exporting base model to {base_output_dir}")
    base_dummy_inputs = create_dummy_inputs(
        base_model,
        is_eagle_base=True,
        is_eagle_draft=False,
        eagle2=eagle2,
        use_prompt_tuning=use_prompt_tuning)
    base_onnx_path = os.path.join(base_output_dir, "model.onnx")
    export_model_to_onnx(base_model,
                         base_dummy_inputs,
                         base_onnx_path,
                         is_eagle_base=True,
                         is_eagle_draft=False,
                         use_prompt_tuning=use_prompt_tuning)

    # Save base model configuration
    base_config = export_llm_config(base_model.config, 'eagle_base', eagle2)
    config_path = os.path.join(base_output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(base_config, f, indent=2)
    print(f"Base model configuration saved to {config_path}")

    # Save tokenizer files to base output directory
    save_tokenizer_to_output_dir(base_model_dir, base_output_dir)

    end_time = time.time()
    print(
        f"Complete {eagle_type} export completed successfully in {end_time - start_time}s!"
    )
    print(f"Base model saved to: {base_output_dir}")
    print(f"Draft model saved to: {draft_output_dir}")

    return output_dir


def llm_export(model_dir: str,
               output_dir: str,
               eagle2: bool = False,
               draft_model_dir: str = None,
               max_position_embeddings: int = 4096,
               device: str = "cuda") -> str:
    """
    Export LLM model to ONNX format.
    
    This function automatically detects the export type based on whether draft_model_dir is provided.
    
    Args:
        model_dir: Directory containing the torch model
        output_dir: Directory to save the exported ONNX model
        eagle2: Whether this is EAGLE2 (True) or EAGLE3 (False) for EAGLE models
        draft_model_dir: Directory containing the draft model (triggers complete EAGLE export)
        max_position_embeddings: Maximum positional embedding length to use for model initialization
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
    
    Returns:
        str: Path to the output directory where the exported model is saved
        
    Raises:
        ValueError: If model loading fails, unsupported model type, or invalid configuration
        RuntimeError: If export fails
    """
    if draft_model_dir is not None:
        return export_eagle_models(model_dir, draft_model_dir, output_dir,
                                   eagle2, max_position_embeddings, device)
    else:
        return export_standard_model(model_dir, output_dir,
                                     max_position_embeddings, device)
