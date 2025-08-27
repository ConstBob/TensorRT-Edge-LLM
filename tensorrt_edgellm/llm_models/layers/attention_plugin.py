"""
Dummy Attention Plugin for TensorRT Integration

This module provides a custom TensorRT operation for attention computation that can be
exported to ONNX format. It includes RoPE (Rotary Position Embedding) application,
KV cache management, and attention computation in a single fused operation.

The module contains:
- attention_plugin: Dummy TensorRT operation for attention computation, this is not used in the actual inference.
- ONNX export utilities for the custom operation
"""

from typing import Optional, Tuple

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...onnx_config import opset_version

# Define ONNX OpSchema for AttentionPlugin
attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain="trt",
    since_version=opset_version,
    doc=
    "Custom TensorRT attention plugin with RoPE, KV cache, and attention computation.",
    inputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description="Concatenated QKV tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context length tensor",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="RoPE rotary embeddings (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Attention mask tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="attention_pos_id",
            description="Position IDs tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description="Updated KV cache tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="kv_cache_capacity",
            type=OpSchema.AttrType.INT,
            description="Maximum capacity of KV cache",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0 or 1)",
            required=True,
        ),
        OpSchema.Attribute(
            name="max_batch_size",
            type=OpSchema.AttrType.INT,
            description="Maximum batch size",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(attention_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "i", "b", "i", "v",
                            "v")
def symbolic_attention_plugin(
    g: torch.onnx._internal.jit_utils.GraphContext,
    qkv: torch._C.Value,
    past_key_value: torch._C.Value,
    context_lengths: torch._C.Value,
    rope_rotary_cos_sin: torch._C.Value,
    num_q_heads: torch._C.Value,
    num_kv_heads: torch._C.Value,
    kv_cache_capacity: torch._C.Value,
    enable_tree_attention: torch._C.Value,
    head_size: torch._C.Value,
    attention_mask: Optional[torch._C.Value] = None,
    position_ids: Optional[torch._C.Value] = None,
):
    """Custom attention plugin operation for ONNX export."""

    # Build inputs list - only include required inputs
    inputs = [qkv, past_key_value, context_lengths, rope_rotary_cos_sin]
    if attention_mask is not None and attention_mask.type().kind(
    ) != 'NoneType':
        inputs.append(attention_mask)
    if position_ids is not None and position_ids.type().kind() != 'NoneType':
        inputs.append(position_ids)

    qkv_type = qkv.type()
    past_key_value_type = past_key_value.type()
    attn_output, present_key_value = g.op(
        "trt::AttentionPlugin",
        *inputs,
        num_q_heads_i=num_q_heads,
        num_kv_heads_i=num_kv_heads,
        head_size_i=head_size,
        kv_cache_capacity_i=kv_cache_capacity,
        enable_tree_attention_i=1 if enable_tree_attention else 0,
        max_batch_size_i=16,
        outputs=2)

    qkv_sizes = _get_tensor_sizes(qkv)
    attn_output_sizes = qkv_sizes[:-1] + [num_q_heads, head_size]
    attn_output.setType(qkv_type.with_sizes(attn_output_sizes))
    present_key_value_sizes = _get_tensor_sizes(past_key_value)
    # KV Cache output should have static length dimension with kv_cache_capacity
    present_key_value_sizes[3] = kv_cache_capacity
    present_key_value.setType(
        past_key_value_type.with_sizes(present_key_value_sizes))

    return attn_output, present_key_value


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    qkv: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    kv_cache_capacity: int,
    enable_tree_attention: bool,
    head_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dummy TensorRT operation for attention computation, this is not used in the actual inference.
    
    This operation wraps the logic after v_proj and before o_proj into a single 
    AttentionPlugin operation during ONNX export. It handles RoPE application,
    KV cache management, and attention computation in a fused manner.
    
    Args:
        qkv: Concatenated QKV tensor of shape (batch_size, seq_len, num_q_heads * head_size + 2 * num_kv_heads * head_size)
        past_key_value: KV cache tensor of shape (batch_size, 2, num_kv_heads, kv_cache_capacity, head_size)
        rope_rotary_cos_sin: RoPE tensor of shape (batch_size, seq_len, 2 * head_size) containing cos and sin values
        context_lengths: Context length tensor of shape (batch_size,) indicating current position in cache
        num_q_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        kv_cache_capacity: Maximum capacity of KV cache
        enable_tree_attention: Whether to enable tree attention
        head_size: Size of each attention head
        attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
        position_ids: Position IDs tensor of shape (batch_size, seq_len), optional
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention output tensor and updated KV cache
            - Attention output: shape (batch_size, seq_len, num_q_heads * head_size)
            - Updated KV cache: shape (batch_size, 2, num_kv_heads, kv_cache_capacity, head_size)
        
    Raises:
        AssertionError: If enable_tree_attention is True but required tensors are missing
    """
    if enable_tree_attention:
        assert attention_mask is not None, "attention_mask should be provided for tree attention"
        assert position_ids is not None, "position_ids should be provided for tree attention"

    batch_size, seq_len, qkv_size = qkv.shape
    assert head_size * (
        num_q_heads + 2 * num_kv_heads
    ) == qkv_size, "qkv_size should be equal to head_size * (num_q_heads + 2 * num_kv_heads)"
    assert past_key_value.shape[
        0] == batch_size, "batch_size of kv_cache should be equal to batch_size of qkv"
    assert past_key_value.shape[1] == 2, "kv_cache should have 2 tensors"
    assert past_key_value.shape[
        2] == num_kv_heads, "num_kv_heads of kv_cache should be equal to num_kv_heads of qkv"
    assert past_key_value.shape[
        4] == head_size, "head_size of kv_cache should be equal to head_size of qkv"

    assert qkv.dtype == torch.float16, "qkv should be in float16"
    assert past_key_value.dtype == torch.float16, "past_key_value should be in float16"

    # Dummy implementation for ONNX export, this is not used in the actual inference
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=qkv.dtype,
                              device=qkv.device)

    return attn_output, past_key_value.clone()


def register_attention_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::attention_plugin",
                                symbolic_attention_plugin, opset_version)

    print("Registered ONNX symbolic functions for custom attention plugin")
