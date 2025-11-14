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

import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..quantization.quantization_utils import \
    enable_huggingface_checkpointing_patch

enable_huggingface_checkpointing_patch()

from ..llm_models.layers.attention_plugin import \
    register_attention_plugin_onnx_symbolic_functions
from ..llm_models.layers.gather_nd import \
    register_gather_nd_onnx_symbolic_functions
from ..llm_models.layers.int4_gemm_plugin import (
    register_int4_gemm_plugin_onnx_symbolic_functions,
    replace_torch_quant_linear_with_plugin)
from ..llm_models.model_utils import (is_gptq_model, load_eagle3_draft_model,
                                      load_llm_model)
from .chat_template import process_chat_template
from .config_export import export_llm_config
from .onnx_utils import export_onnx


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
    """Save d2t.safetensors for Eagle3 draft model."""
    from safetensors.torch import save_file

    d2t_tensor = draft_model.d2t
    # Convert to int32 and move to CPU if needed
    d2t_tensor_int32 = d2t_tensor.cpu().to(torch.int32)

    # Save as safetensors with key 'd2t'
    d2t_path = os.path.join(output_dir, "d2t.safetensors")
    save_file({"d2t": d2t_tensor_int32}, d2t_path)
    print(f"Saved d2t.safetensors to {output_dir}")


def create_dummy_inputs(model: nn.Module, enable_reuse_kv_cache: bool,
                        is_eagle_base: bool, is_eagle_draft: bool,
                        use_prompt_tuning: bool) -> Dict[str, Any]:
    """
    Create dummy inputs for ONNX export.
    
    Args:
        model: The model to create inputs for
        is_eagle_base: Whether this is an EAGLE base model
        is_eagle_draft: Whether this is an EAGLE draft model
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
    # Use head_dim from config if available, otherwise calculate from hidden_size
    if hasattr(model_config, 'head_dim'):
        head_dim = model_config.head_dim
    else:
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
        # For EAGLE models, maintain batch dimension for proper GatherND support
        num_selected_tokens = 2
        last_token_ids = torch.full([batch_size, num_selected_tokens],
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

    if model_config.model_type == "qwen3_vl_text":
        deepstack_visual_embeds = [
            torch.randn(image_token_len,
                        hidden_size,
                        dtype=torch.float16,
                        device=device) for _ in range(3)
        ]
        base_inputs['deepstack_visual_embeds'] = deepstack_visual_embeds

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

    # Add input for persistent KV cache
    if enable_reuse_kv_cache:
        base_inputs['kvcache_start_index'] = torch.zeros(batch_size,
                                                         dtype=torch.int32,
                                                         device=device)

    # Add EAGLE-specific inputs
    if is_eagle_draft:
        target_hidden_size = getattr(model_config, 'target_hidden_size',
                                     hidden_size)
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


def replace_torch_quant_linear_with_int4_plugin(model: nn.Module) -> nn.Module:
    """
    Replace all TorchQuantLinear modules in a model with Int4GemmPluginModule.
        
    Args:
        model: PyTorch model containing TorchQuantLinear modules
        
    Returns:
        nn.Module: Model with TorchQuantLinear modules replaced by Int4GemmPluginModule
    """
    if is_gptq_model(model):
        print(
            "Detected GPTQ quantization, replacing TorchQuantLinear with Int4GemmPluginModule"
        )
        register_int4_gemm_plugin_onnx_symbolic_functions()
        model = replace_torch_quant_linear_with_plugin(model)
    return model


def export_model_to_onnx(model: nn.Module, dummy_inputs: Dict[str, Any],
                         output_dir: str, enable_reuse_kv_cache: bool,
                         is_eagle_base: bool, is_eagle_draft: bool,
                         use_prompt_tuning: bool) -> None:
    """
    Export the model to ONNX format.
    
    Args:
        model: The model to export
        dummy_inputs: Dummy inputs for tracing
        output_dir: Directory to save the ONNX model
        enable_reuse_kv_cache: Whether to enable persistent KV cache
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

        if enable_reuse_kv_cache:
            base_inputs.append(dummy_inputs['kvcache_start_index'])
        else:
            base_inputs.append(None)

        # Add input_ids (always present)
        base_inputs.append(dummy_inputs['input_ids'])

        # Add image_embeds (only for prompt tuning models)
        if use_prompt_tuning:
            base_inputs.append(dummy_inputs['image_embeds'])

        if model_config.model_type == "qwen3_vl_text":
            base_inputs.append(dummy_inputs['deepstack_visual_embeds'])

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

        if enable_reuse_kv_cache:
            input_names.append('kvcache_start_index')

        # Add input_ids (always present)
        input_names.append('input_ids')

        if use_prompt_tuning:
            input_names.append('image_embeds')

        if model_config.model_type == "qwen3_vl_text":
            input_names += [f'deepstack_features.{i}' for i in range(3)]

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

        # Define dynamic axes for last_token_ids based on model type
        if is_eagle_base or is_eagle_draft:
            # EAGLE models: (batch_size, num_selected_tokens)
            last_token_ids_axes = {0: "batch_size", 1: "num_selected_tokens"}
        else:
            # Standard models: (batch_size, 1)
            last_token_ids_axes = {0: "batch_size"}

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
            "last_token_ids": last_token_ids_axes,
            **present_key_values_shapes
        }

        if enable_reuse_kv_cache:
            dynamic_axes.update({"kvcache_start_index": {0: "batch_size"}})

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
                    1: "q_len"
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "q_len",
                    2: "q_len_padded"
                }
            })

        if use_prompt_tuning:
            dynamic_axes["image_embeds"] = {0: "image_token_len"}

        if model_config.model_type == "qwen3_vl_text":
            dynamic_axes.update({
                f"deepstack_features.{i}": {
                    0: "image_token_len"
                }
                for i in range(3)
            })

        # Add dynamic axes for outputs
        if is_eagle_base or is_eagle_draft:
            # EAGLE models: logits shape (batch_size, num_selected_tokens, vocab_size)
            dynamic_axes["logits"] = {
                0: "batch_size",
                1: "num_selected_tokens"
            }
            # EAGLE models hidden_states shape
            # Eagle base: (batch_size, seq_len, 3*hidden_dim)
            # Eagle draft: (batch_size, seq_len, hidden_dim)
            dynamic_axes["hidden_states"] = {0: "batch_size", 1: "seq_len"}
        else:
            # Standard models: logits shape (batch_size, num_tokens, vocab_size)
            dynamic_axes["logits"] = {0: "batch_size", 1: "num_tokens"}

        # Register ONNX symbolic functions
        register_attention_plugin_onnx_symbolic_functions()
        register_gather_nd_onnx_symbolic_functions()

        # Export to ONNX
        export_onnx(model, inputs, output_dir, input_names, output_names,
                    dynamic_axes)

    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def export_llm_model(model_dir: str,
                     output_dir: str,
                     max_position_embeddings: int = 4096,
                     device: str = "cuda",
                     enable_reuse_kv_cache: bool = True,
                     is_eagle_base: bool = False) -> None:
    """
    Export a language model to ONNX format with custom attention plugin.
    
    This is the main entry point for exporting standard LLM models and EAGLE base models
    to ONNX format with TensorRT Edge-LLM optimizations.
    
    Args:
        model_dir: Directory containing the HuggingFace model
        output_dir: Directory to save the exported ONNX model
        max_position_embeddings: Maximum sequence length for positional embeddings
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        enable_reuse_kv_cache: Whether to enable persistent KV cache for system prompts
        is_eagle_base: Whether the model is an EAGLE3 base model (vs standard LLM)
    """
    start_time = time.time()

    if is_eagle_base:
        print(f"Exporting EAGLE3 base model to ONNX format")
    else:
        print(f"Exporting standard model to ONNX format")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, use_prompt_tuning = load_llm_model(
        model_dir,
        dtype='fp16',
        max_position_embeddings=max_position_embeddings,
        device=device,
        enable_reuse_kv_cache=enable_reuse_kv_cache,
        is_eagle_base=is_eagle_base)

    model = replace_torch_quant_linear_with_int4_plugin(model)

    # Create dummy inputs
    dummy_inputs = create_dummy_inputs(
        model,
        enable_reuse_kv_cache=enable_reuse_kv_cache,
        is_eagle_base=is_eagle_base,
        is_eagle_draft=False,
        use_prompt_tuning=use_prompt_tuning)

    # Export to ONNX
    export_model_to_onnx(model,
                         dummy_inputs,
                         output_dir,
                         enable_reuse_kv_cache=enable_reuse_kv_cache,
                         is_eagle_base=is_eagle_base,
                         is_eagle_draft=False,
                         use_prompt_tuning=use_prompt_tuning)

    # Save model configuration
    model_type = 'eagle3_base' if is_eagle_base else 'llm'
    model_config = export_llm_config(model.config, model_type,
                                     enable_reuse_kv_cache)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Model configuration saved to {config_path}")

    # Save tokenizer files
    save_tokenizer_to_output_dir(model_dir, output_dir)

    # Process and save chat template
    process_chat_template(model_dir, output_dir)

    end_time = time.time()
    print(
        f"Export completed successfully in {end_time - start_time}s. Files saved to: {output_dir}"
    )


def export_draft_model(draft_model_dir: str,
                       output_dir: str,
                       use_prompt_tuning: bool = False,
                       base_model_dir: Optional[str] = None,
                       max_position_embeddings: int = 4096,
                       device: str = "cuda",
                       enable_reuse_kv_cache: bool = False) -> None:
    """
    Export an EAGLE draft model to ONNX format with custom attention plugin.
    
    This is the main entry point for exporting EAGLE draft models to ONNX format.
    The draft model requires a base model for weight copying.
    
    Args:
        draft_model_dir: Directory containing the EAGLE draft model
        output_dir: Directory to save the exported ONNX model
        use_prompt_tuning: Whether the model uses prompt tuning (for VLM models)
        base_model_dir: Directory containing the base model (for weight copying)
        max_position_embeddings: Maximum sequence length for positional embeddings
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        enable_reuse_kv_cache: Whether to enable persistent KV cache for system prompts
    """
    start_time = time.time()

    print(f"Exporting EAGLE3 draft model")

    # Create subdirectories
    os.makedirs(output_dir, exist_ok=True)

    # Load draft model with base model for weight copying
    print(f"Loading draft model from {draft_model_dir}")

    draft_model = load_eagle3_draft_model(draft_model_dir, base_model_dir,
                                          use_prompt_tuning,
                                          max_position_embeddings, 'fp16',
                                          device, enable_reuse_kv_cache)

    draft_model = replace_torch_quant_linear_with_int4_plugin(draft_model)

    # Export draft model
    print(f"Exporting draft model to {output_dir}")
    draft_dummy_inputs = create_dummy_inputs(
        draft_model,
        enable_reuse_kv_cache=enable_reuse_kv_cache,
        is_eagle_base=False,
        is_eagle_draft=True,
        use_prompt_tuning=use_prompt_tuning)
    export_model_to_onnx(draft_model,
                         draft_dummy_inputs,
                         output_dir,
                         enable_reuse_kv_cache=enable_reuse_kv_cache,
                         is_eagle_base=False,
                         is_eagle_draft=True,
                         use_prompt_tuning=use_prompt_tuning)

    # Save draft model configuration
    draft_config = export_llm_config(draft_model.config, 'eagle_draft',
                                     enable_reuse_kv_cache)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(draft_config, f, indent=2)
    print(f"Draft model configuration saved to {config_path}")

    save_d2t_for_eagle3_draft(draft_model, output_dir)

    draft_end_time = time.time()
    print(
        f"Complete draft model export completed successfully in {draft_end_time - start_time}s!"
    )
    print(f"Draft model saved to: {output_dir}")
