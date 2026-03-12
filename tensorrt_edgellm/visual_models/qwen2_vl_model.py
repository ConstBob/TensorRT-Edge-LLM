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
Qwen2-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen2-VL visual models,
enabling ONNX export with proper attention mechanism handling.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

from typing import Optional

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel, VisionAttention,
    apply_rotary_pos_emb_vision)

from ..llm_models.layers.attention_plugin import (
    register_attention_plugin_onnx_symbolic_functions, vit_attention_plugin)
from ..onnx_export.onnx_utils import export_onnx


class Qwen2VisionAttentionPatch(VisionAttention):
    """
    Patched version of Qwen2-VL vision attention for ONNX export.
    Uses vit attention plugin to support ragged attention via cu_seqlens.
    """

    def __init__(self, attention_module: VisionAttention) -> None:
        """
        Initialize the patched attention module.
        
        Args:
            attention_module: Original attention module to extract components from
        """
        super().__init__(attention_module.config)
        self.qkv = attention_module.qkv
        self.proj = attention_module.proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            cu_seqlens: Prefix sum of sequence lengths
            max_seqlen_carrier: Shape-only input carrying max sequence length for FMHA launch
            position_embeddings: Position embeddings for rotary attention
            
        Returns:
            Attention output
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Convert to FP16 for plugin compatibility
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        # Use ViT attention plugin with separate Q, K, V
        # q, k, v are already in shape [total_S, H, D]
        attn_output = vit_attention_plugin(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen_carrier,
            num_heads=self.num_heads,
            head_size=self.head_dim,
        )

        # Plugin output layout is [total_S, H, D], reshape to [total_S, H * D]
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VisionTransformerPretrainedModelPatch(
        Qwen2VisionTransformerPretrainedModel):
    """
    Patched version of Qwen2VisionTransformerPretrainedModel for ONNX export.
    
    This class provides a wrapper around the original Qwen2-VL vision transformer
    with custom blocks that are compatible with ONNX export.
    """

    def __init__(
            self,
            original_model: Qwen2VisionTransformerPretrainedModel) -> None:
        """
        Initialize the patched vision transformer from original model.
        
        Args:
            original_model: Original Qwen2VisionTransformerPretrainedModel instance
        """
        super().__init__(original_model.config)

        # Reuse all original components
        self.patch_embed = original_model.patch_embed
        self.blocks = original_model.blocks
        self.merger = original_model.merger

        # Replace attention modules, reusing existing components to preserve quantization
        for block in self.blocks:
            block.attn = Qwen2VisionAttentionPatch(block.attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the vision transformer.
        
        Args:
            hidden_states: Input hidden states
            rotary_pos_emb: Rotary position embeddings
            cu_seqlens: Prefix sum of sequence lengths
            max_seqlen_carrier: Shape-only input carrying max sequence length for FMHA launch
        
        Returns:
            torch.Tensor: Output embeddings after processing through all blocks
        """
        # Apply patch embedding
        hidden_states = self.patch_embed(hidden_states)

        # Prepare position embeddings for rotary attention
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Process through all vision blocks
        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                max_seqlen_carrier=max_seqlen_carrier,
                                position_embeddings=position_embeddings)

        # Apply final merger to get output embeddings
        res = self.merger(hidden_states)
        return res


def export_qwen2_vl_visual(
    model: Qwen2VisionTransformerPretrainedModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen2-VL visual model to ONNX format.
    
    This function takes a patched Qwen2-VL visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen2-VL vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy inputs for ONNX export
    hw = 16  # Height * width for the input
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.embed_dim // model.config.num_heads // 2

    # Create input tensors with appropriate shapes and dtypes
    pixel_values = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch_dtype,
        device=model.device)
    rotary_pos_emb = torch.randn(
        (hw, rotary_pos_emb_dim),
        dtype=torch.float32,  # Keep as float32 for rotary embeddings
        device=model.device)
    cu_seqlens = torch.tensor([0, hw], dtype=torch.int32, device=model.device)
    max_seqlen_carrier = torch.zeros(hw,
                                     dtype=torch.int32,
                                     device=model.device)

    inputs = (pixel_values, rotary_pos_emb, cu_seqlens, max_seqlen_carrier)

    input_names = [
        "input", "rotary_pos_emb", "cu_seqlens", "max_seqlen_carrier"
    ]
    output_names = ["output"]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'cu_seqlens': {
            0: 'batch_size + 1'
        },
        'max_seqlen_carrier': {
            0: 'max_seqlen'
        },
        'output': {
            0: 'image_token_len'
        },
    }

    register_attention_plugin_onnx_symbolic_functions()
    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
