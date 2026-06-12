# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Gemma4 text decoder implementation for checkpoint-based export."""

from __future__ import annotations

import itertools
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from ...config import ModelConfig
from ..default.modeling_default import (MLP, Attention, CausalLM, DecoderLayer,
                                        OnnxSpec, RMSNorm)
from ..linear import make_linear
from ..ops import attention_plugin

__all__ = [
    "Gemma4ForCausalLM",
    "Gemma4DecoderLayer",
    "Gemma4Transformer",
]

# These are dummy tensor extents used only to seed torch.export/ONNX export.
# Runtime limits are controlled by dynamic_shapes and the builder profiles.
_DUMMY_BATCH_SIZE = 1
_DUMMY_SEQ_LEN = 1
_DUMMY_PAST_LEN = 1
_DUMMY_ROPE_CACHE_LEN = 4096


def _attention_type_for_layer(config: ModelConfig, layer_idx: int) -> str:
    """Return Gemma4's per-layer attention type."""
    if layer_idx < len(config.attention_layer_types):
        return config.attention_layer_types[layer_idx]
    if config.sliding_window_size >= 0:
        return "sliding_attention"
    return "full_attention"


def _rotary_dim_from_rope_config(config: ModelConfig,
                                 rope_config: dict | None) -> int:
    """Return the RoPE table width for one Gemma4 runtime RoPE config."""
    if isinstance(rope_config, dict):
        rope_scaling = rope_config.get("rope_scaling")
        partial_rotary_factor = float(
            rope_config.get("partial_rotary_factor",
                            config.partial_rotary_factor))
    else:
        rope_scaling = config.rope_scaling
        partial_rotary_factor = config.partial_rotary_factor

    if isinstance(rope_scaling, dict):
        rope_type = str(
            rope_scaling.get("rope_type", rope_scaling.get("type", "default")))
        if rope_type == "proportional":
            return int(config.head_dim)
    return int(config.head_dim * partial_rotary_factor)


def _select_rope_for_layer(
    layer: nn.Module,
    rope_rotary_cos_sin: torch.Tensor | None,
    rope_rotary_cos_sin_sliding: torch.Tensor | None,
    rope_rotary_cos_sin_full: torch.Tensor | None,
) -> torch.Tensor:
    """Select the Gemma4 RoPE table matching ``layer`` attention type."""
    if (rope_rotary_cos_sin_sliding is None
            and rope_rotary_cos_sin_full is None):
        if rope_rotary_cos_sin is None:
            raise ValueError(
                "rope_rotary_cos_sin is required for single-RoPE Gemma4 export."
            )
        return rope_rotary_cos_sin

    attention_type = getattr(layer.self_attn, "attention_type",
                             "full_attention")
    if attention_type == "sliding_attention":
        if rope_rotary_cos_sin_sliding is None:
            raise ValueError(
                "rope_rotary_cos_sin_sliding is required for Gemma4 sliding attention layers."
            )
        return rope_rotary_cos_sin_sliding

    if rope_rotary_cos_sin_full is None:
        raise ValueError(
            "rope_rotary_cos_sin_full is required for Gemma4 full attention layers."
        )
    return rope_rotary_cos_sin_full


def _make_gemma4_flat_wrapper(model: nn.Module,
                              Na: int,
                              num_ple_inputs: int,
                              use_dual_rope: bool = False,
                              eagle_base: bool = False,
                              emit_hidden_states: bool = False) -> nn.Module:
    """Build a Gemma4 export wrapper with explicit PLE/RoPE tensor inputs."""
    has_hidden_output = eagle_base or emit_hidden_states

    param_names: List[str] = (
        ["inputs_embeds"] +
        [f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
        [f"past_key_values_{i}" for i in range(Na)])
    if use_dual_rope:
        param_names += [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full"
        ]
    else:
        param_names += ["rope_rotary_cos_sin"]
    param_names += ["context_lengths", "kvcache_start_index", "last_token_ids"]
    if eagle_base:
        param_names += ["attention_pos_id", "attention_mask"]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    ple_tuple = "({},)".format(", ".join(
        f"ple_token_embeds_{i}"
        for i in range(num_ple_inputs))) if num_ple_inputs else "()"
    ple_kwarg = f", ple_token_embeds={ple_tuple}" if num_ple_inputs > 0 else ""
    eagle_kwargs = (", attention_mask=attention_mask"
                    ", attention_pos_id=attention_pos_id"
                    if eagle_base else "")
    if use_dual_rope:
        rope_arg = "None"
        rope_kwargs = (
            ", rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding"
            ", rope_rotary_cos_sin_full=rope_rotary_cos_sin_full")
    else:
        rope_arg = "rope_rotary_cos_sin"
        rope_kwargs = ""

    if has_hidden_output:
        body = (
            f"    logits, hidden_states, present_key_values = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, {rope_arg}, "
            f"context_lengths, kvcache_start_index, last_token_ids"
            f"{eagle_kwargs}{ple_kwarg}{rope_kwargs})\n"
            f"    return (logits, hidden_states) + tuple(present_key_values)\n"
        )
    else:
        body = (f"    logits, present_key_values = self._model(\n"
                f"        inputs_embeds, {past_kv_tuple}, {rope_arg}, "
                f"context_lengths, kvcache_start_index, last_token_ids"
                f"{eagle_kwargs}{ple_kwarg}{rope_kwargs})\n"
                f"    return (logits,) + tuple(present_key_values)\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


def _resolve_hidden_activation(
        activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the Gemma4 PLE gate activation."""
    if activation_name in ACT2FN:
        return ACT2FN[activation_name]
    raise ValueError(
        f"Unsupported hidden_activation for Gemma4 PLE gate: {activation_name!r}"
    )


class Gemma4ValueRMSNorm(nn.Module):
    """Weightless per-head RMSNorm used by Gemma4 attention values."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return hidden_states.to(input_dtype)


class Gemma4Attention(Attention):
    """Gemma4 attention with HF-compatible value norm and QK scaling."""

    def __init__(self,
                 config: ModelConfig,
                 layer_idx: int,
                 in_features: int = 0) -> None:
        super().__init__(config, layer_idx, in_features)
        self.qk_scale = (float(config.attention_scaling)
                         if config.attention_scaling > 0.0 else self.head_dim**
                         -0.5)
        self.v_norm = (Gemma4ValueRMSNorm(self.head_dim, config.rms_norm_eps)
                       if config.has_value_norm else None)
        self.attention_type = _attention_type_for_layer(config, layer_idx)
        self.sliding_window_size = (config.sliding_window_size
                                    if self.attention_type
                                    == "sliding_attention" else -1)

    def _attention_plugin_query_scale(self) -> float:
        default_plugin_qk_scale = self.head_dim**-0.5
        return self.qk_scale / default_plugin_qk_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(batch_size, seq_len, self.num_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len,
                                         self.num_heads * self.head_dim)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                   self.head_dim)).reshape(
                                       batch_size, seq_len,
                                       self.num_kv_heads * self.head_dim)

        if self.v_norm is not None:
            value_states = self.v_norm(
                value_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len,
                                         self.num_kv_heads * self.head_dim)

        query_scale = self._attention_plugin_query_scale()
        if query_scale != 1.0:
            query_states = query_states * query_states.new_tensor(query_scale)

        enable_tree = attention_mask is not None and attention_pos_id is not None
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": enable_tree,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
        }
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])

        attn_output, present_key_value = attention_plugin(
            query_states,
            key_states,
            value_states,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


class Gemma4MLP(MLP):
    """Gemma4 MLP using the checkpoint-configured activation."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.act_fn = _resolve_hidden_activation(config.hidden_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


class Gemma4DecoderLayer(DecoderLayer):
    """Gemma4 decoder layer with per-layer input injection."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = Gemma4Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx=layer_idx)
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input)
        self.act_fn = _resolve_hidden_activation(config.hidden_activation)

        if self.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
            )
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size,
                                                     config.rms_norm_eps)
            self.register_buffer("layer_scalar", torch.ones(1))

    def _apply_per_layer_input(
            self, hidden_states: torch.Tensor,
            per_layer_input: torch.Tensor | None) -> torch.Tensor:
        """Apply Gemma4 PLE gate/projection/post-norm/residual injection."""
        if per_layer_input is None:
            return hidden_states
        if self.hidden_size_per_layer_input <= 0:
            raise ValueError(
                "per_layer_input was provided but Gemma4 PLE is disabled.")
        if per_layer_input.ndim != 3:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expects "
                "per_layer_input to be rank-3, got shape "
                f"{tuple(per_layer_input.shape)}.")
        if per_layer_input.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expects "
                "per_layer_input batch/sequence dimensions to match "
                f"hidden_states. Got {tuple(per_layer_input.shape)} and "
                f"{tuple(hidden_states.shape)}.")
        if per_layer_input.shape[-1] != self.hidden_size_per_layer_input:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expected final "
                f"dimension {self.hidden_size_per_layer_input}, got "
                f"{per_layer_input.shape[-1]}.")

        gated = self.per_layer_input_gate(hidden_states)
        gated = self.act_fn(gated)
        gated = gated * per_layer_input.to(dtype=gated.dtype)
        gated = self.per_layer_projection(gated)
        gated = self.post_per_layer_input_norm(gated)
        return hidden_states + gated

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_per_layer_input(hidden_states,
                                                    per_layer_input)
        if self.hidden_size_per_layer_input > 0:
            hidden_states = hidden_states * self.layer_scalar

        return hidden_states, present_key_value


class Gemma4Transformer(nn.Module):
    """Gemma4 attention decoder with optional per-layer input injection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.use_dual_rope and config.eagle_base:
            raise ValueError(
                "Gemma4 dual RoPE is incompatible with EAGLE base mode.")
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input)
        self.vocab_size_per_layer_input = int(
            config.vocab_size_per_layer_input)
        self.ple_enabled = self.hidden_size_per_layer_input > 0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.ple_enabled:
            if self.vocab_size_per_layer_input <= 0:
                raise ValueError(
                    "Gemma4 PLE requires vocab_size_per_layer_input > 0.")
            # Weight holder for the token-identity PLE table. This module is
            # not called by the ONNX forward path; checkpoint export writes its
            # weight to ple_embedding.safetensors for runtime-side gather.
            self.embed_tokens_per_layer = nn.Embedding(
                self.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0, dtype=torch.float32)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_model_projection_scale",
                torch.tensor(float(config.hidden_size)**-0.5,
                             dtype=torch.float32),
                persistent=False,
            )

        self.last_pre_norm_hidden_states: torch.Tensor | None = None

    def _project_per_layer_inputs(
            self, inputs_embeds: torch.Tensor) -> torch.Tensor | None:
        """Project runtime ``inputs_embeds`` into Gemma4 per-layer inputs."""
        if not self.ple_enabled:
            return None

        projected = self.per_layer_model_projection(inputs_embeds)
        scale = self.per_layer_model_projection_scale.to(
            dtype=projected.dtype, device=projected.device)
        projected = projected * scale
        projected = projected.reshape(
            *inputs_embeds.shape[:-1],
            len(self.layers),
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(projected)

    def _combine_per_layer_input(self, projected_per_layer_inputs: torch.Tensor
                                 | None, ple_token_embeds: Tuple[torch.Tensor,
                                                                 ...],
                                 layer_index: int) -> torch.Tensor | None:
        """Combine context projection and runtime token-identity PLE input."""
        if not self.ple_enabled:
            return None
        if projected_per_layer_inputs is None:
            raise ValueError("Gemma4 PLE projection unexpectedly missing.")
        if len(ple_token_embeds) != len(self.layers):
            raise ValueError(
                "Gemma4 PLE expects one ple_token_embeds input per layer; "
                f"got {len(ple_token_embeds)} for {len(self.layers)} layers.")

        combined = (projected_per_layer_inputs[:, :, layer_index, :] +
                    ple_token_embeds[layer_index].to(
                        dtype=projected_per_layer_inputs.dtype))
        scale = self.per_layer_input_scale.to(dtype=combined.dtype,
                                              device=combined.device)
        return combined * scale

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor | None,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        ple_token_embeds: Tuple[torch.Tensor, ...] = (),
        rope_rotary_cos_sin_sliding: torch.Tensor | None = None,
        rope_rotary_cos_sin_full: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Tuple, Tuple | None]:
        hidden_states = inputs_embeds
        projected_per_layer_inputs = self._project_per_layer_inputs(
            inputs_embeds)
        present_key_values_list: List[torch.Tensor] = []
        all_hidden_states: list = []

        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_rope_rotary_cos_sin = _select_rope_for_layer(
                layer,
                rope_rotary_cos_sin,
                rope_rotary_cos_sin_sliding,
                rope_rotary_cos_sin_full,
            )
            per_layer_input = self._combine_per_layer_input(
                projected_per_layer_inputs, ple_token_embeds, layer_index)
            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                layer_rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                attention_mask=attention_mask,
                attention_pos_id=attention_pos_id,
                per_layer_input=per_layer_input,
            )
            present_key_values_list.append(next_key_value)

        self.last_pre_norm_hidden_states = hidden_states
        normed = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(normed)

        return (normed, tuple(present_key_values_list),
                tuple(all_hidden_states) if output_hidden_states else None)


class Gemma4ForCausalLM(CausalLM):
    """Gemma4 CausalLM wrapper for the checkpoint exporter."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Gemma4Transformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    @property
    def ple_enabled(self) -> bool:
        """Whether this Gemma4 export uses per-layer embedding inputs."""
        return self.model.ple_enabled

    def onnx_export_spec(self) -> OnnxSpec:
        """Return Gemma4-specific ONNX export parameters."""
        if not self.ple_enabled and not self.config.use_dual_rope:
            return super().onnx_export_spec()

        config = self.config
        if config.use_dual_rope and config.eagle_base:
            raise NotImplementedError(
                "Gemma4 dual RoPE export is not supported for EAGLE base models."
            )

        Na = config.num_hidden_layers
        eagle_base = config.eagle_base
        num_ple_inputs = Na if self.ple_enabled else 0
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len, past_len, max_pos = (_DUMMY_BATCH_SIZE,
                                                  _DUMMY_SEQ_LEN,
                                                  _DUMMY_PAST_LEN,
                                                  _DUMMY_ROPE_CACHE_LEN)

        inputs_embeds = torch.zeros(batch_size,
                                    seq_len,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        ple_token_embeds_list: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        seq_len,
                        config.hidden_size_per_layer_input,
                        dtype=dtype16,
                        device=device) for _ in range(num_ple_inputs)
        ]
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        2,
                        config.num_key_value_heads,
                        past_len,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
        ]

        args = (inputs_embeds, *ple_token_embeds_list, *past_key_values_list)
        input_names = (
            ["inputs_embeds"] +
            [f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
            [f"past_key_values_{i}" for i in range(Na)])

        if config.use_dual_rope:
            sliding_rotary_dim = _rotary_dim_from_rope_config(
                config, config.sliding_rope_config)
            full_rotary_dim = _rotary_dim_from_rope_config(
                config, config.full_rope_config)
            rope_rotary_cos_sin_sliding = torch.zeros(batch_size,
                                                      max_pos,
                                                      sliding_rotary_dim,
                                                      dtype=torch.float32,
                                                      device=device)
            rope_rotary_cos_sin_full = torch.zeros(batch_size,
                                                   max_pos,
                                                   full_rotary_dim,
                                                   dtype=torch.float32,
                                                   device=device)
            args = args + (rope_rotary_cos_sin_sliding,
                           rope_rotary_cos_sin_full)
            input_names = input_names + [
                "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full"
            ]
        else:
            rotary_dim = _rotary_dim_from_rope_config(config, None)
            rope_rotary_cos_sin = torch.zeros(batch_size,
                                              max_pos,
                                              rotary_dim,
                                              dtype=torch.float32,
                                              device=device)
            args = args + (rope_rotary_cos_sin, )
            input_names = input_names + ["rope_rotary_cos_sin"]

        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)

        args = args + (context_lengths, kvcache_start_index, last_token_ids)
        input_names = input_names + [
            "context_lengths", "kvcache_start_index", "last_token_ids"
        ]
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)])
        if self.emit_hidden_states and not eagle_base:
            output_names = (["logits", "hidden_states"] +
                            [f"present_key_values_{i}" for i in range(Na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        past = torch.export.Dim("past_len", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)

        num_selected = torch.export.Dim("num_selected", min=1,
                                        max=256) if eagle_base else None
        all_shapes: list = [{0: batch, 1: seq}]
        for _ in range(num_ple_inputs):
            all_shapes.append({0: batch, 1: seq})
        for _ in range(Na):
            all_shapes.append({0: batch, 3: past})
        all_shapes.append({0: rope_batch, 1: pos})
        if config.use_dual_rope:
            all_shapes.append({0: rope_batch, 1: pos})
        all_shapes.append({0: batch})
        all_shapes.append({0: kv_batch})
        if eagle_base:
            all_shapes.append({0: batch, 1: num_selected})
        else:
            all_shapes.append({0: batch})
        if eagle_base:
            attention_pos_id = torch.zeros(batch_size,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=device)
            attention_mask = torch.zeros(batch_size,
                                         seq_len,
                                         seq_len + past_len,
                                         dtype=torch.int32,
                                         device=device)
            args = args + (attention_pos_id, attention_mask)
            input_names = input_names + ["attention_pos_id", "attention_mask"]
            output_names = ["logits", "hidden_states"
                            ] + [f"present_key_values_{i}" for i in range(Na)]

            eagle_seq = torch.export.Dim("eagle_seq_len", min=1, max=32768)
            mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
            all_shapes.append({0: batch, 1: eagle_seq})
            all_shapes.append({0: batch, 1: eagle_seq, 2: mask_kv_len})

        wrapped = _make_gemma4_flat_wrapper(
            self,
            Na,
            num_ple_inputs=num_ple_inputs,
            use_dual_rope=config.use_dual_rope,
            eagle_base=eagle_base,
            emit_hidden_states=self.emit_hidden_states)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor | None,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        ple_token_embeds: Tuple[torch.Tensor, ...] = (),
        rope_rotary_cos_sin_sliding: torch.Tensor | None = None,
        rope_rotary_cos_sin_full: torch.Tensor | None = None,
    ) -> Tuple:
        eagle_base = self.config.eagle_base
        hidden_states, present_key_values, all_hidden_states = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            output_hidden_states=eagle_base,
            ple_token_embeds=ple_token_embeds,
            rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding,
            rope_rotary_cos_sin_full=rope_rotary_cos_sin_full,
        )

        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)

        if eagle_base and all_hidden_states is not None:
            n_layers = len(all_hidden_states) - 1
            idx = [2, n_layers // 2, n_layers - 4]
            eagle_hidden = torch.cat([
                all_hidden_states[idx[0]],
                all_hidden_states[idx[1]],
                all_hidden_states[idx[2]],
            ],
                                     dim=-1).to(torch.float16)
            return logits, eagle_hidden, present_key_values

        if self.emit_hidden_states:
            return logits, self.model.last_pre_norm_hidden_states, \
                present_key_values

        return logits, present_key_values
