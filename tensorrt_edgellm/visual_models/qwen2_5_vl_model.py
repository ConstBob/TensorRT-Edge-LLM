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
Qwen2.5-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen2.5-VL visual models,
enabling ONNX export with proper attention mechanism handling and FP16 overflow fixes.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

from typing import Any, Optional

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLMLP,
    Qwen2_5_VLPatchMerger, Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionBlock,
    apply_rotary_pos_emb_vision)

from ..llm_models.layers.attention_plugin import (
    register_attention_plugin_onnx_symbolic_functions, vit_attention_plugin)
from ..onnx_export.onnx_utils import export_onnx


class Qwen2_5_VLVisionAttentionPatch(Qwen2_5_VLVisionAttention):
    """
    Patched version of Qwen2.5-VL vision attention for ONNX export.
    Uses vit attention plugin to support ragged attention via cu_seqlens.
    """

    def __init__(self, attention_module: Qwen2_5_VLVisionAttention) -> None:
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
        max_seqlen_carrier: Optional[torch.Tensor],
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


class Qwen2_5_VLMLPPatch(Qwen2_5_VLMLP):
    """
    Patched version of Qwen2_5_VLMLP to cast Down Proj to FP32 to avoid FP16 overflow.
    
    This class addresses numerical stability issues in FP16 by casting the down projection
    layer to FP32 during computation.
    """

    def __init__(self, config: Any, mlp_module: Qwen2_5_VLMLP) -> None:
        """
        Initialize the patched MLP from original MLP module.
        
        Args:
            config: Model configuration object
            mlp_module: Original Qwen2_5_VLMLP module to reuse components from
        """
        super().__init__(config, bias=mlp_module.down_proj.bias is not None)
        # Reuse all MLP components to preserve quantization information
        self.gate_proj = mlp_module.gate_proj
        self.up_proj = mlp_module.up_proj
        self.down_proj = mlp_module.down_proj
        self.act_fn = mlp_module.act_fn

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 casting for numerical stability.
        
        Args:
            hidden_state: Input hidden states
        
        Returns:
            torch.Tensor: Output after MLP processing
        """
        # Apply gate and up projections
        hidden_state = self.act_fn(
            self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        # Cast to FP32 for numerical stability
        hidden_state = hidden_state.to(torch.float32)
        # Cast down projection weights and bias to FP32
        self.down_proj.weight.data = self.down_proj.weight.data.to(
            torch.float32)
        self.down_proj.bias.data = self.down_proj.bias.data.to(torch.float32)
        return self.down_proj(hidden_state)


class Qwen2_5_VLVisionBlockPatchWAR(Qwen2_5_VLVisionBlock):
    """
    Workaround patch for Qwen2.5-VL 3B FP16 overflow issues.
    
    This class provides a workaround for FP16 overflow issues that occur specifically
    in the Qwen2.5-VL 3B model by using FP32 casting in critical operations.
    """

    def __init__(self, config: Any,
                 block_module: Qwen2_5_VLVisionBlock) -> None:
        """
        Initialize the workaround vision block from original block module.
        
        Args:
            config: Model configuration object
            block_module: Original Qwen2_5_VLVisionBlock module to reuse components from
        """
        super().__init__(config)
        # Reuse normalization layers
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        # Create patched attention and MLP, reusing original components
        self.attn = Qwen2_5_VLVisionAttentionPatch(block_module.attn)
        self.mlp = Qwen2_5_VLMLPPatch(config, block_module.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with FP32 casting for overflow prevention.
        
        Args:
            hidden_states: Input hidden states
            cu_seqlens: Prefix sum of sequence lengths
            position_embeddings: Position embeddings
        
        Returns:
            torch.Tensor: Output hidden states
        """
        # Apply attention with residual connection
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Apply MLP with FP32 casting and residual connection
        hidden_states = hidden_states.to(torch.float32) + self.mlp(
            self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPatchMergerWAR(Qwen2_5_VLPatchMerger):
    "WAR for Qwen2.5-VL 3B FP16 overflow"

    def __init__(self, config: Any,
                 merger_module: Qwen2_5_VLPatchMerger) -> None:
        """
        Initialize the workaround merger from original merger module.
        
        Args:
            config: Model configuration object
            merger_module: Original Qwen2_5_VLPatchMerger module to reuse components from
        """
        super().__init__(config.out_hidden_size, config.hidden_size,
                         config.spatial_merge_size)
        # Reuse all merger components to preserve quantization information
        self.ln_q = merger_module.ln_q
        self.mlp = merger_module.mlp
        self.hidden_size = merger_module.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).to(torch.float16).view(-1, self.hidden_size))
        return x


class Qwen2_5_VisionTransformerPretrainedModelPatch(
        Qwen2_5_VisionTransformerPretrainedModel):
    """
    Patched version of Qwen2.5_VisionTransformerPretrainedModel for ONNX export.
    
    This class provides a wrapper around the original Qwen2.5-VL vision transformer
    with custom blocks that are compatible with ONNX export and includes workarounds
    for FP16 overflow issues in the 3B model.
    """

    def __init__(
            self,
            original_model: Qwen2_5_VisionTransformerPretrainedModel) -> None:
        """
        Initialize the patched vision transformer from original model.
        
        Args:
            original_model: Original Qwen2_5_VisionTransformerPretrainedModel instance
        """
        config = original_model.config
        super().__init__(config)

        # Reuse all original components
        self.patch_embed = original_model.patch_embed
        self.blocks = original_model.blocks
        self.merger = original_model.merger

        # Replace attention modules
        for block in self.blocks:
            block.attn = Qwen2_5_VLVisionAttentionPatch(block.attn)

        # Qwen2.5-VL 3B VIT has overflow issue with FP16 and only happens in /blocks.31/mlp/down_proj
        # Apply Patch to cast /blocks.31/mlp/down_proj to FP32 to avoid this issue.
        if config.out_hidden_size == 2048:
            self.blocks[-1] = Qwen2_5_VLVisionBlockPatchWAR(
                config, self.blocks[-1])
            self.merger = Qwen2_5_VLPatchMergerWAR(config, self.merger)

        # Calculate window attention max_seqlen: (window_size // patch_size) ** 2
        self.window_max_seqlen = (config.window_size // config.patch_size)**2

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the vision transformer with ragged attention support.
        
        Args:
            hidden_states: Input hidden states
            rotary_pos_emb: Rotary position embeddings
            cu_seqlens: Prefix sum of sequence lengths for full attention blocks
            max_seqlen_carrier: Shape-only input carrying max sequence length hint for full attention
            cu_window_seqlens: Prefix sum of sequence lengths for window attention blocks
            window_index: Window index for attention
            reverse_window_index: Reverse window index for attention
        
        Returns:
            torch.Tensor: Output embeddings after processing through all blocks
        """
        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_carrier_now = max_seqlen_carrier
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_carrier_now = torch.zeros(
                    self.window_max_seqlen,
                    dtype=torch.int32,
                    device=hidden_states.device)
            # max_seqlen is passed via kwargs, which will be forwarded to self.attn by the base class
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                max_seqlen_carrier=max_seqlen_carrier_now,
                                position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_window_index, :]

        return hidden_states


def export_qwen2_5_vl_visual(
    model: Qwen2_5_VisionTransformerPretrainedModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen2.5-VL visual model to ONNX format.
    
    This function takes a patched Qwen2.5-VL visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen2.5-VL vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy input sizes (will be replaced by dynamic axes)
    grid_t = 1
    grid_h = 8
    grid_w = 16
    hw = grid_t * grid_h * grid_w
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.hidden_size // model.config.num_heads // 2

    # Create input tensors with appropriate shapes and dtypes
    input_tensor = torch.randn(
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
    cu_window_seqlens = torch.tensor([0, hw],
                                     dtype=torch.int32,
                                     device=model.device)

    # Create window index tensors
    window_index = torch.arange(hw // 4,
                                dtype=torch.int64,
                                device=model.device)
    window_index = window_index.reshape(grid_t, grid_h // 8, 4, grid_w // 8, 4)
    window_index = window_index.permute(0, 1, 3, 2, 4).reshape(-1)
    # TensorRT TopK max K = 3840. Compute reverse index outside to support longer image tokens.
    reverse_window_index = torch.argsort(window_index)

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
        'cu_window_seqlens': {
            0: 'num_windows + 1'
        },
        'window_index': {
            0: 'hw//4'
        },
        'reverse_window_index': {
            0: 'hw//4'
        },
        "output": {
            0: 'image_token_len'
        },
    }

    inputs = (input_tensor, rotary_pos_emb, cu_seqlens, max_seqlen_carrier,
              cu_window_seqlens, window_index, reverse_window_index)
    input_names = [
        "input", "rotary_pos_emb", "cu_seqlens", "max_seqlen_carrier",
        "cu_window_seqlens", "window_index", "reverse_window_index"
    ]
    output_names = ["output"]

    register_attention_plugin_onnx_symbolic_functions()
    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
