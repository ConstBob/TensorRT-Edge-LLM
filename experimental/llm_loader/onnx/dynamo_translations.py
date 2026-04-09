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
onnxscript translations for dynamo ONNX export.

Maps ``torch.ops.trt.*`` and ``torch.ops.trt_edgellm.*`` stubs to ONNX graphs.
Implementation files must be importable; ``onnxscript.script`` parses their AST.
"""

from typing import Sequence

import onnx
import onnxscript
import torch
from onnxscript import opset21 as _op21
from onnxscript import script

# Custom ONNX domains
_trt = onnxscript.values.Opset("trt", 1)
_trt_edgellm = onnxscript.values.Opset("trt_edgellm", 1)

# ---------------------------------------------------------------------------
# FP8 ops
# ---------------------------------------------------------------------------


@script()
def _attention_plugin_translation(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    kvcache_start_index: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    enable_fp8_kv_cache: int,
    sliding_window_size: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    attn_4d, present_kv = _trt_edgellm.AttentionPlugin(
        query_states,
        key_states,
        value_states,
        past_key_value,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        enable_tree_attention=0,
        enable_fp8_kv_cache=0,  # non-FP8-KV path always uses 0
        sliding_window_size=sliding_window_size,
        _outputs=2,
    )
    return attn_4d, present_kv


@script()
def _attention_plugin_fp8kv_translation(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    kvcache_start_index: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    qkv_scales: Sequence[float],
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    # qkv_scales is a FLOATS ONNX attribute [q_scale, k_scale, v_scale]
    # read by TRT's AttentionPlugin as a PluginField at engine-build time.
    # Defaults to [1.0, 1.0, 1.0] when the checkpoint has no KV scales.
    attn_4d, present_kv = _trt_edgellm.AttentionPlugin(
        query_states,
        key_states,
        value_states,
        past_key_value,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        enable_tree_attention=0,
        enable_fp8_kv_cache=1,
        sliding_window_size=sliding_window_size,
        qkv_scales=qkv_scales,
        _outputs=2,
    )
    return attn_4d, present_kv


@script()
def _fp8_quantize_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT8E4M3FN:
    """Standard ONNX QuantizeLinear (see onnx.ai QuantizeLinear)."""
    return _op21.QuantizeLinear(
        hidden_states,
        scale,
        output_dtype=int(onnx.TensorProto.FLOAT8E4M3FN),
    )


@script()
def _fp8_dequantize_translation(
    x: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT16:
    """Standard ONNX DequantizeLinear; FP16 scale -> FP16 dequantized output."""
    return _op21.DequantizeLinear(x, scale)


# ---------------------------------------------------------------------------
# NVFP4 ops
# ---------------------------------------------------------------------------


@script()
def _nvfp4_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16,
    global_scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """DynQ + 2x trt::DQ for NVFP4 activation quantization.

    Emits the same graph as ModelOpt's ``export_fp4(dynamic)``::

        TRT_FP4DynamicQuantize(x, scale, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, scale)
            -> dq_scale
        trt::DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
            -> x_dq
    """
    x_f4, sx_f8 = _trt.TRT_FP4DynamicQuantize(
        hidden_states,
        global_scale,
        axis=-1,
        block_size=16,
        scale_type=17,
        _outputs=2,
    )
    # Cast fp32 global_scale -> fp16 so DQ outputs are fp16 (not fp32)
    global_scale_f16 = _op21.Cast(global_scale, to=10)  # 10 = FLOAT16
    dq_scale = _trt.DequantizeLinear(sx_f8, global_scale_f16)
    x_dq = _trt.DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
    return x_dq


@script()
def _nvfp4_dequantize_translation(
    weight: onnxscript.INT8,
    weight_scale: onnxscript.FLOAT8E4M3FN,
    weight_scale_2: onnxscript.FLOAT,
    group_size: int,
) -> onnxscript.FLOAT16:
    """2xstandard-ONNX DequantizeLinear for NVFP4 weight dequantization.

    Emits the same graph as ModelOpt's ``fp4qdq_to_2dq()``::

        DequantizeLinear(weight_scale_fp8, weight_scale_2_fp32) -> ws
        DequantizeLinear(weight_fp4, ws, axis=-1, block_size=group_size) -> w_dq
    """
    # DQ1 (standard ONNX): fp8 per-block scales -> float32 scales
    ws = _op21.DequantizeLinear(weight_scale, weight_scale_2)
    # DQ2 (standard ONNX): FLOAT4E2M1 weight -> float32 (scale type propagates)
    # Note: weight initializer is rewritten from INT8 to FLOAT4E2M1 post-export.
    w_dq = _op21.DequantizeLinear(weight, ws, axis=-1, block_size=group_size)
    # Cast float32 -> float16 to match activation dtype for MatMul
    return _op21.Cast(w_dq, to=10)  # 10 = ONNX TensorProto.FLOAT16


# ---------------------------------------------------------------------------
# AWQ / INT4 op
# ---------------------------------------------------------------------------


@script()
def _int4_groupwise_gemm_translation(
    hidden_states: onnxscript.FLOAT16,
    qweight: onnxscript.INT8,
    scales: onnxscript.FLOAT16,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Int4GroupwiseGemmPlugin(
        hidden_states,
        qweight,
        scales,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# INT8 SmoothQuant ops
# ---------------------------------------------------------------------------


@script()
def _int8_sq_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """Per-tensor INT8 activation QDQ: QuantizeLinear + DequantizeLinear.

    Emits the standard ONNX QDQ pattern TRT recognises for INT8 GEMM fusion::

        QuantizeLinear(x, scale, output_dtype=INT8) -> q
        DequantizeLinear(q, scale)                  -> dq  [float32]
        Cast(dq, to=FLOAT16)                        -> output
    """
    # output_dtype=3 -> INT8 (symmetric, zero_point=0)
    quantized = _op21.QuantizeLinear(hidden_states, scale, output_dtype=3)
    dq = _op21.DequantizeLinear(quantized, scale)
    return _op21.Cast(dq, to=10)  # 10 = FLOAT16


@script()
def _int8_sq_weight_dq_translation(
    weight: onnxscript.INT8,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """Per-channel INT8 weight DequantizeLinear (axis=0).

    Emits::

        DequantizeLinear(weight, scale, axis=0) -> dq  [float32]
        Cast(dq, to=FLOAT16)                    -> output
    """
    dq = _op21.DequantizeLinear(weight, scale, axis=0)
    return _op21.Cast(dq, to=10)  # 10 = FLOAT16


# ---------------------------------------------------------------------------
# Hybrid (Mamba) ops
# ---------------------------------------------------------------------------


@script()
def _causal_conv1d_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, conv_state_out = _trt_edgellm.causal_conv1d(
        hidden_states,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        _outputs=2,
    )
    return output, conv_state_out


@script()
def _update_ssm_state_translation(
    hidden_states: onnxscript.FLOAT16,
    ssm_a: onnxscript.FLOAT,
    ssm_b: onnxscript.FLOAT16,
    ssm_c: onnxscript.FLOAT16,
    ssm_d: onnxscript.FLOAT16,
    dt: onnxscript.FLOAT16,
    dt_bias: onnxscript.FLOAT16,
    state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, state_out = _trt_edgellm.update_ssm_state(
        hidden_states,
        ssm_a,
        ssm_b,
        ssm_c,
        ssm_d,
        dt,
        dt_bias,
        state,
        context_lengths,
        dt_softplus=dt_softplus,
        ngroups=ngroups,
        chunk_size=chunk_size,
        _outputs=2,
    )
    return output, state_out


# ---------------------------------------------------------------------------
# GatherND op  (token selection)
# ---------------------------------------------------------------------------


@script()
def _gather_nd_translation(
    value: onnxscript.FLOAT16,
    indices: onnxscript.INT64,
) -> onnxscript.FLOAT16:
    """ONNX GatherND with batch_dims=1 for last-token selection.

    Converts [B, S, H] hidden_states + [B, T] indices -> [B, T, H].
    indices is unsqueezed to [B, T, 1] as required by the GatherND spec.

    GatherND (opset 16+), Unsqueeze (opset 13+), and Constant are all
    available at opset 21.
    """
    axes = _op21.Constant(value_ints=[-1])
    indices_3d = _op21.Unsqueeze(indices, axes)
    return _op21.GatherND(value, indices_3d, batch_dims=1)


# ---------------------------------------------------------------------------
# ViT attention op
# ---------------------------------------------------------------------------


@script()
def _vit_attention_plugin_translation(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    cu_seqlens: onnxscript.INT32,
    max_seqlen_carrier: onnxscript.INT32,
    num_heads: int,
    head_size: int,
) -> onnxscript.FLOAT16:
    """ViT ragged self-attention without KV cache."""
    return _trt_edgellm.ViTAttentionPlugin(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen_carrier,
        num_heads=num_heads,
        head_size=head_size,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_custom_translation_table() -> dict:
    """Return the ``custom_translation_table`` for ``torch.onnx.export(dynamo=True)``.

    Maps each ``torch.ops.trt.*`` / ``torch.ops.trt_edgellm.*`` op to its
    onnxscript translation function.
    """
    from .onnx_custom_schemas import register_llm_loader_onnx_custom_schemas

    register_llm_loader_onnx_custom_schemas()

    # Ensure custom ops are registered before accessing torch.ops.trt.*
    from ..models import \
        ops  # noqa: F401 - side-effect: registers all custom_ops

    return {
        torch.ops.trt.attention_plugin.default: _attention_plugin_translation,
        torch.ops.trt.attention_plugin_fp8kv.default:
        _attention_plugin_fp8kv_translation,
        torch.ops.trt.fp8_quantize.default: _fp8_quantize_translation,
        torch.ops.trt.fp8_dequantize.default: _fp8_dequantize_translation,
        torch.ops.trt.nvfp4_act_qdq.default: _nvfp4_act_qdq_translation,
        torch.ops.trt.nvfp4_dequantize.default: _nvfp4_dequantize_translation,
        torch.ops.trt.int4_groupwise_gemm.default:
        _int4_groupwise_gemm_translation,
        torch.ops.trt.int8_sq_act_qdq.default: _int8_sq_act_qdq_translation,
        torch.ops.trt.int8_sq_weight_dq.default:
        _int8_sq_weight_dq_translation,
        torch.ops.trt_edgellm.causal_conv1d.default:
        _causal_conv1d_translation,
        torch.ops.trt_edgellm.update_ssm_state.default:
        _update_ssm_state_translation,
        torch.ops.trt.vit_attention_plugin.default:
        _vit_attention_plugin_translation,
        torch.ops.trt.gather_nd.default: _gather_nd_translation,
    }
