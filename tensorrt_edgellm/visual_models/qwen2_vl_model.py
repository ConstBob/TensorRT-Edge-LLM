"""
Qwen2-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen2-VL visual models,
enabling ONNX export with proper attention mechanism handling.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

import io
import math
import os
import time
from typing import Any, Optional

import modelopt.torch.quantization as mtq
import onnx
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock, VisionAttention,
    apply_rotary_pos_emb_vision)

from ..onnx_config import (all_tensors_to_one_file, convert_attribute,
                           do_constant_folding, location, opset_version,
                           save_as_external_data)


class Qwen2VisionAttentionPatch(VisionAttention):
    """
    Patched version of Qwen2-VL vision attention for ONNX export.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
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

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class QuantQwen2VisionAttentionPatch(Qwen2VisionAttentionPatch):
    """
    Quantized MHA version of Qwen2VisionAttentionPatch.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._setup()

    def _setup(self) -> None:
        """Initialize quantization components."""
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()
        self.softmax_quantizer = TensorQuantizer()

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantized forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
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

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        q = self.q_bmm_quantizer(q)
        k = self.k_bmm_quantizer(k)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       v.dtype)
        attn_weights = self.softmax_quantizer(attn_weights)
        v = self.v_bmm_quantizer(v)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


mtq.register(original_cls=Qwen2VisionAttentionPatch,
             quantized_cls=QuantQwen2VisionAttentionPatch)


class Qwen2VLVisionBlockPatch(Qwen2VLVisionBlock):
    """
    Patched version of Qwen2VLVisionBlock with custom attention mechanism.
    
    This class replaces the original attention mechanism with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        """
        Initialize the patched vision block.
        
        Args:
            config: Model configuration object
            attn_implementation: Attention implementation type
        """
        super().__init__(config)
        self.attn = Qwen2VisionAttentionPatch(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision block.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask for the attention mechanism
            position_embeddings: Position embeddings for rotary attention
        
        Returns:
            torch.Tensor: Output hidden states after processing
        """
        # Apply attention with residual connection
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        # Apply MLP with residual connection
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VisionTransformerPretrainedModelPatch(
        Qwen2VisionTransformerPretrainedModel):
    """
    Patched version of Qwen2VisionTransformerPretrainedModel for ONNX export.
    
    This class provides a wrapper around the original Qwen2-VL vision transformer
    with custom blocks that are compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the patched vision transformer.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        # Replace all blocks with patched versions
        self.blocks = nn.ModuleList([
            Qwen2VLVisionBlockPatch(config, config._attn_implementation)
            for _ in range(config.depth)
        ])

    def forward(self, hidden_states: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision transformer.
        
        Args:
            hidden_states: Input hidden states
            rotary_pos_emb: Rotary position embeddings
            attention_mask: Attention mask
        
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
                                attention_mask=attention_mask,
                                position_embeddings=position_embeddings)

        # Apply final merger to get output embeddings
        res = self.merger(hidden_states)
        return res


def export_qwen2_vl_visual(model: Qwen2VisionTransformerPretrainedModelPatch,
                           output_dir: str,
                           torch_dtype: torch.dtype,
                           quantization: Optional[str] = None) -> None:
    """
    Export Qwen2-VL visual model to ONNX format.
    
    This function takes a patched Qwen2-VL visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen2-VL vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
        quantization: Quantization type (currently not used for Qwen2-VL)
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
    attention_mask = torch.randn((1, hw, hw),
                                 dtype=torch_dtype,
                                 device=model.device)

    inputs = (pixel_values, rotary_pos_emb, attention_mask)

    input_names = ["input", "rotary_pos_emb", "attention_mask"]
    output_names = ["output"]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        },
        'output': {
            0: 'image_token_len'
        },
    }

    start_time = time.time()

    # Export to BytesIO first to avoid intermediate file I/O
    bytes_io = io.BytesIO()
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            bytes_io,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
        )

    # Load from bytes and apply post-processing
    onnx_bytes = bytes_io.getvalue()
    onnx_model = onnx.load_model_from_string(onnx_bytes)

    # Save the final ONNX model
    output_path = f'{output_dir}/model.onnx'
    onnx.save_model(onnx_model,
                    output_path,
                    save_as_external_data=save_as_external_data,
                    all_tensors_to_one_file=all_tensors_to_one_file,
                    location=location,
                    convert_attribute=convert_attribute)

    end_time = time.time()
    print(
        f"Qwen2-VL visual encoder ONNX Export from torch completed in {end_time - start_time}s. "
        f"ONNX file is saved to {output_dir}.")
