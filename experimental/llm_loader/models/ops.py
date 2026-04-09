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
``torch.library.custom_op`` stubs for inference export.

Each op is a trace-time dummy (returns zero tensors of the correct shape/dtype)
paired with a ``register_fake`` for shape propagation in the dynamo exporter.
Domains ``trt::`` / ``trt_edgellm::`` map to ONNX nodes consumed by the
TensorRT plugin runtime.
"""

from typing import List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Custom op: trt::attention_plugin
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    enable_fp8_kv_cache: bool,
    sliding_window_size: int,
    k_v_scale_quant_orig: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub for AttentionPlugin - returns zero tensors of the right shape.

    The TRT AttentionPlugin kernel returns a 4-D tensor
    [batch, seq_len, num_q_heads, head_size].
    The caller (Attention.forward) is responsible for reshaping to
    [batch, seq_len, num_q_heads * head_size].
    """
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=query_states.dtype,
                              device=query_states.device)
    present_key_value = torch.zeros(batch_size,
                                    2,
                                    num_kv_heads,
                                    past_len + seq_len,
                                    head_size,
                                    dtype=past_key_value.dtype,
                                    device=past_key_value.device)
    return attn_output, present_key_value


@attention_plugin.register_fake
def _(query_states,
      key_states,
      value_states,
      past_key_value,
      context_lengths,
      rope_rotary_cos_sin,
      kvcache_start_index,
      num_q_heads,
      num_kv_heads,
      head_size,
      enable_fp8_kv_cache,
      sliding_window_size,
      k_v_scale_quant_orig=None):
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    return (torch.empty(batch_size,
                        seq_len,
                        num_q_heads,
                        head_size,
                        dtype=query_states.dtype,
                        device=query_states.device),
            torch.empty(batch_size,
                        2,
                        num_kv_heads,
                        past_len + seq_len,
                        head_size,
                        dtype=past_key_value.dtype,
                        device=past_key_value.device))


# ---------------------------------------------------------------------------
# Custom op: trt::attention_plugin_fp8kv  (FP8 KV cache variant, 8 inputs)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_plugin_fp8kv", mutates_args=())
def attention_plugin_fp8kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    qkv_scales: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub for AttentionPlugin with FP8 KV cache.

    qkv_scales is a list of three floats [q_scale, k_scale, v_scale]
    embedded as the ``qkv_scales`` ONNX node attribute (PluginField) so
    TRT's AttentionPlugin can read them at engine-build time.  Defaults
    to [1.0, 1.0, 1.0] when the checkpoint carries no explicit scale
    tensors (q_scale is never in the checkpoint; k/v_scale default to 1.0).
    """
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=query_states.dtype,
                              device=query_states.device)
    present_key_value = torch.zeros(batch_size,
                                    2,
                                    num_kv_heads,
                                    past_len + seq_len,
                                    head_size,
                                    dtype=past_key_value.dtype,
                                    device=past_key_value.device)
    return attn_output, present_key_value


@attention_plugin_fp8kv.register_fake
def _(query_states,
      key_states,
      value_states,
      past_key_value,
      context_lengths,
      rope_rotary_cos_sin,
      kvcache_start_index,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window_size,
      qkv_scales=None):
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    return (torch.empty(batch_size,
                        seq_len,
                        num_q_heads,
                        head_size,
                        dtype=query_states.dtype,
                        device=query_states.device),
            torch.empty(batch_size,
                        2,
                        num_kv_heads,
                        past_len + seq_len,
                        head_size,
                        dtype=past_key_value.dtype,
                        device=past_key_value.device))


# ---------------------------------------------------------------------------
# Custom op: trt::vit_attention_plugin  (ViT ragged self-attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    query_states: torch.Tensor,  # [T, num_heads, head_size]
    key_states: torch.Tensor,  # [T, num_heads, head_size]
    value_states: torch.Tensor,  # [T, num_heads, head_size]
    cu_seqlens: torch.Tensor,  # [batch+1] int32
    max_seqlen_carrier: torch.Tensor,  # [] or [1] int32 (scalar)
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """ViT ragged self-attention.

    In eager mode, implements varlen SDPA using cu_seqlens to process each
    sequence segment independently.  During dynamo/ONNX tracing the
    register_fake shape propagation is used and this body is not executed.

    Unlike AttentionPlugin, ViT attention has no KV cache and takes ragged
    input with cu_seqlens instead of context_lengths.  RoPE is applied before
    this call.
    """
    import torch.nn.functional as F
    out = torch.empty_like(query_states)
    seqlens = cu_seqlens.tolist()
    for i in range(len(seqlens) - 1):
        start, end = int(seqlens[i]), int(seqlens[i + 1])
        if start >= end:
            continue
        # q/k/v: [S, H, D] -> [1, H, S, D] for SDPA
        q = query_states[start:end].permute(1, 0, 2).unsqueeze(0)
        k = key_states[start:end].permute(1, 0, 2).unsqueeze(0)
        v = value_states[start:end].permute(1, 0, 2).unsqueeze(0)
        attn = F.scaled_dot_product_attention(q, k, v)  # [1, H, S, D]
        out[start:end] = attn.squeeze(0).permute(1, 0, 2)
    return out


@vit_attention_plugin.register_fake
def _(query_states, key_states, value_states, cu_seqlens, max_seqlen_carrier,
      num_heads, head_size):
    return torch.empty_like(query_states)


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_quantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::fp8_quantize", mutates_args=())
def fp8_quantize(
        hidden_states: torch.Tensor,  # float16 input
        scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Stub: quantize float16 -> FP8; ONNX export -> QuantizeLinear."""
    return torch.zeros_like(hidden_states)


@fp8_quantize.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::fp8_dequantize", mutates_args=())
def fp8_dequantize(
        weight: torch.Tensor,  # fp8_e4m3fn [out, in]
        weight_scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Stub: dequantize FP8 -> float16; ONNX export -> DequantizeLinear."""
    return torch.zeros_like(weight, dtype=torch.float16)


@fp8_dequantize.register_fake
def _(weight, weight_scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_act_qdq  (activation DynQ + 2DQ -> float16)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_act_qdq", mutates_args=())
def nvfp4_act_qdq(
        hidden_states: torch.Tensor,  # float16 activation
        global_scale: torch.Tensor,  # float32 scalar: amax / (6.0 * 448.0)
) -> torch.Tensor:
    """Stub: NVFP4 activation QDQ (DynQ + 2 trt::DQ). Returns float16.

    In the ONNX graph this emits three nodes matching ModelOpt's
    ``export_fp4(onnx_quantizer_type="dynamic")`` pattern::

        TRT_FP4DynamicQuantize(x, scale_f32, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, scale_f32)
            -> dq_scale
        trt::DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
            -> x_dq  [float16]
    """
    return torch.zeros_like(hidden_states)


@nvfp4_act_qdq.register_fake
def _(hidden_states, global_scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_dequantize", mutates_args=())
def nvfp4_dequantize(
    weight: torch.Tensor,  # uint8 [out, in//2] packed fp4
    weight_scale: torch.Tensor,  # fp8_e4m3fn [out, in//group_size]
    weight_scale_2: torch.Tensor,  # float32 scalar
    group_size: int,
) -> torch.Tensor:
    """Stub: dequantize NVFP4 packed weight to float16."""
    out_features, packed_in = weight.shape
    in_features = packed_in * 2
    return torch.zeros(out_features,
                       in_features,
                       dtype=torch.float16,
                       device=weight.device)


@nvfp4_dequantize.register_fake
def _(weight, weight_scale, weight_scale_2, group_size):
    out_features, packed_in = weight.shape
    return torch.empty(out_features,
                       packed_in * 2,
                       dtype=torch.float16,
                       device=weight.device)


# ---------------------------------------------------------------------------
# Custom op: trt::int4_groupwise_gemm
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int4_groupwise_gemm", mutates_args=())
def int4_groupwise_gemm(
    hidden_states: torch.Tensor,  # [*, in_features] float16
    qweight: torch.Tensor,  # [out_features//2, in_features] int8 (swizzled)
    scales: torch.Tensor,  # [in_features//group_size, out_features] float16
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Stub: INT4 groupwise GEMM - returns zero tensor of correct shape."""
    *leading, _ = hidden_states.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_groupwise_gemm.register_fake
def _(hidden_states, qweight, scales, gemm_n, gemm_k, group_size):
    *leading, _ = hidden_states.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_act_qdq  (INT8 SmoothQuant activation QDQ)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_act_qdq", mutates_args=())
def int8_sq_act_qdq(
        hidden_states: torch.Tensor,  # float16 smoothed activation [*, in]
        scale: torch.Tensor,  # float32 per-tensor input scale []
) -> torch.Tensor:
    """Stub: symmetric per-tensor INT8 QuantizeLinear + DequantizeLinear.

    In the ONNX graph emits::

        QuantizeLinear(x, scale, output_dtype=INT8) -> q
        DequantizeLinear(q, scale)                  -> dq  [float32]
        Cast(dq, to=FLOAT16)                        -> output
    """
    return torch.zeros_like(hidden_states)


@int8_sq_act_qdq.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_weight_dq  (INT8 per-channel weight dequantize)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_weight_dq", mutates_args=())
def int8_sq_weight_dq(
        weight: torch.Tensor,  # int8 [out, in]
        scale: torch.Tensor,  # float32 [out] per-channel scale
) -> torch.Tensor:
    """Stub: per-channel INT8 DequantizeLinear (axis=0), output float16.

    In the ONNX graph emits::

        DequantizeLinear(weight, scale, axis=0) -> dq  [float32]
        Cast(dq, to=FLOAT16)                    -> output
    """
    return torch.zeros_like(weight, dtype=torch.float16)


@int8_sq_weight_dq.register_fake
def _(weight, scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::causal_conv1d
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::causal_conv1d", mutates_args=())
def causal_conv1d(
    hidden_states: torch.Tensor,  # [batch, seq_len, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [batch, conv_dim, conv_kernel]
    context_lengths: torch.Tensor,  # [batch] int32
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d. Returns same-shape activations and cloned conv_state."""
    return torch.zeros_like(hidden_states), conv_state.clone()


@causal_conv1d.register_fake
def _(hidden_states, weight, bias, conv_state, context_lengths, stride,
      padding, dilation, groups):
    return torch.empty_like(hidden_states), conv_state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::update_ssm_state
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::update_ssm_state", mutates_args=())
def update_ssm_state(
    hidden_states: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
    ssm_a: torch.Tensor,  # [num_heads] float32
    ssm_b: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_c: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_d: torch.Tensor,  # [num_heads] float16
    dt: torch.Tensor,  # [batch, seq_len, num_heads]
    dt_bias: torch.Tensor,  # [num_heads] float16
    state: torch.Tensor,  # [batch, num_heads, head_dim, ssm_state_size]
    context_lengths: torch.Tensor,  # [batch] int32
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: Mamba SSM state update. Returns zeros for hidden_states and cloned state."""
    return torch.zeros_like(hidden_states), state.clone()


@update_ssm_state.register_fake
def _(hidden_states,
      ssm_a,
      ssm_b,
      ssm_c,
      ssm_d,
      dt,
      dt_bias,
      state,
      context_lengths,
      dt_softplus,
      ngroups,
      chunk_size=0):
    return torch.empty_like(hidden_states), state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt::gather_nd  (token selection: GatherND with batch_dims=1)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::gather_nd", mutates_args=())
def gather_nd(
        value: torch.Tensor,  # [batch, seq_len, hidden_size] float16
        indices: torch.Tensor,  # [batch, num_tokens] int64
) -> torch.Tensor:
    """Stub: gather tokens from seq_len dim. Exports as GatherND(batch_dims=1).

    Equivalent to ``value[b, indices[b, t], :]`` for each batch b and
    token position t.  Exports as GatherND(batch_dims=1).
    """
    batch_size, num_tokens = indices.shape
    return torch.zeros(batch_size,
                       num_tokens,
                       value.shape[-1],
                       dtype=value.dtype,
                       device=value.device)


@gather_nd.register_fake
def _(value, indices):
    batch_size, num_tokens = indices.shape
    hidden_size = value.shape[-1]
    return torch.empty(batch_size,
                       num_tokens,
                       hidden_size,
                       dtype=value.dtype,
                       device=value.device)
