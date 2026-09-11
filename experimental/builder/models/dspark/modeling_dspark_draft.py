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
"""DSpark checkpoint-direct draft backbone."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import config as core_config
from ...core import quantization
from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .. import registry as model_registry


class DSparkProposalAttention(TreeAttention):
    """Proposal attention over persistent target-derived and proposal K/V."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.attention_sinks = None
        if ctx.cfg.attention_sink_bias:
            name = self.key("attention_sink_bias")
            if not self.weights.has(name):
                raise ValueError(
                    f"DSpark attention sink is enabled but {name!r} is missing"
                )
            sinks = self.weights.f32(name).reshape(-1)
            if sinks.shape != (ctx.cfg.num_attention_heads, ):
                raise ValueError(f"{name} has shape {sinks.shape}, expected "
                                 f"{(ctx.cfg.num_attention_heads,)}")
            self.attention_sinks = np.ascontiguousarray(sinks, np.float32)

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        cfg = self.cfg
        key_delta = self.k_proj(hidden_delta).reshape(
            (0, cfg.num_key_value_heads, cfg.head_dim))
        key_delta = self.k_norm(key_delta, 3)
        value_delta = self.v_proj(hidden_delta).reshape(
            (0, cfg.num_key_value_heads, cfg.head_dim))
        updated = F.update_dflash_target_cache(key_delta, value_delta, past,
                                               delta_rope, delta_positions,
                                               delta_token_to_sequence,
                                               ragged.kv_page_table)

        query = self.q_proj(hidden).reshape(
            (0, cfg.num_attention_heads, cfg.head_dim))
        query = self.q_norm(query, 3).reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        key = self.k_proj(hidden).reshape(
            (0, cfg.num_key_value_heads, cfg.head_dim))
        key = self.k_norm(key, 3).reshape(
            (0, cfg.num_key_value_heads * cfg.head_dim))
        value = self.v_proj(hidden)
        qkv = pack_qkv(query, key, value, self.v_proj)
        attention, present = F.attention(
            qkv,
            updated,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            sliding_window_size=cfg.sliding_window_size,
            attention_scale=cfg.attention_scaling,
            enable_fp8_kv_cache=False,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            attention_sinks=(F.constant(self.attention_sinks,
                                        "attention_sinks")
                             if self.attention_sinks is not None else None),
            enable_contiguous_query_swa=cfg.dspark_contiguous_query_swa,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(attention), present


class DSparkDecoderLayer(Module):
    """One DSpark cached proposal decoder layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.input_norm = RMSNorm(ctx, self.key("input_layernorm"),
                                  ctx.cfg.rms_norm_eps)
        self.attention = DSparkProposalAttention(ctx, self.key("self_attn"))
        self.post_norm = RMSNorm(ctx, self.key("post_attention_layernorm"),
                                 ctx.cfg.rms_norm_eps)
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        attention, present = self.attention(self.input_norm(hidden),
                                            hidden_delta, past, rope, ragged,
                                            delta_rope, delta_positions,
                                            delta_token_to_sequence,
                                            attention_mask, attention_pos_id)
        hidden = hidden + attention
        feed_forward = self.mlp(self.post_norm(hidden))
        return hidden + feed_forward, present


class DSparkTargetProjection(Module):
    """Project concatenated target states in FP32 before normalization."""

    def forward(self, hidden):
        descriptor = self.weights.linear_descriptor(self.prefix,
                                                    quantization.QUANT_FP16)
        return F.linear_f32_from_weights(hidden, descriptor, self.prefix,
                                         hidden.ndim)


class DSparkDraftModel(NetworkModule):
    """Parallel proposal backbone with hidden output for sequential heads."""

    @classmethod
    def from_config(cls, ctx):
        if (ctx.weights.has("lm_head.weight")
                or ctx.weights.has("lm_head.qweight")):
            return cls(ctx)

        args = ctx.args
        target_cfg = core_config.DeviceConfig.from_pretrained(
            args.target_model_dir, tp_size=args.tp_size, tp_rank=args.tp_rank)
        target_bundle = core_config.BundleConfig.from_pretrained(
            args.target_model_dir)
        conversion = model_registry.weight_conversion_for(
            target_bundle.root_model_type)
        target_weights = ctx.open_weights(
            args.target_model_dir,
            group_size=target_cfg.group_size,
            quant=target_cfg.quant,
            component="llm",
            vocab_map=ctx.weights.vocab_map,
            conversion=conversion,
            int4_gemm_plugin_version=args.int4_gemm_plugin_version,
            checkpoint_source="target",
            tie_word_embeddings=target_cfg.tie_word_embeddings)
        try:
            target_context = ctx.with_checkpoint(target_cfg, target_weights)
            model = cls(ctx,
                        lm_head=Linear(target_context,
                                       target_weights.causal_lm_head_prefix()))
        except Exception:
            target_weights.close()
            raise
        model._target_weights = target_weights
        return model

    def __init__(self, ctx, lm_head=None) -> None:
        super().__init__(ctx)
        self._target_weights = None
        self.fc = DSparkTargetProjection(ctx, "fc")
        self.hidden_norm = RMSNorm(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            DSparkDecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = lm_head or Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        target_layers = cfg.dspark_target_layer_ids
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "base_hidden":
            self.add_input("dflash_target_hidden_concat", trt.float16,
                           (-1, len(target_layers) * cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32, (-1, )),
            "attention_mask":
            self.add_input("packed_attention_mask", trt.int32, (-1, -1)),
            "delta_rope":
            self.add_input("dflash_delta_rope_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "delta_positions":
            self.add_input("dflash_delta_positions", trt.int32, (-1, )),
            "delta_token_to_sequence":
            self.add_input("dflash_delta_token_to_sequence", trt.int32,
                           (-1, )),
        }
        io.update(
            add_ragged_decoder_inputs(self.add_input,
                                      include_logits_indices=False).as_dict())
        return io

    def forward(self, **io):
        hidden = io["inputs_embeds"]
        delta = self.hidden_norm(self.fc(io["base_hidden"]))
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, delta,
                                  io["past_key_values"][index], io["rope"],
                                  RaggedDecoderInputs.from_dict(io),
                                  io["delta_rope"], io["delta_positions"],
                                  io["delta_token_to_sequence"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)
        hidden = self.norm(hidden)
        outputs = {
            "logits": F.cast(self.lm_head(hidden), trt.float32),
            "dspark_hidden_states": F.cast(hidden, trt.float16),
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs

    def close(self) -> None:
        if self._target_weights is not None:
            self._target_weights.close()
            self._target_weights = None
