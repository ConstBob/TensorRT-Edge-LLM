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
"""EAGLE3 checkpoint-direct draft graph."""

from typing import Dict

import tensorrt as trt

from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F


class Eagle3DecoderLayer(Module):
    """One EAGLE3 tree-attention decoder layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.hidden_norm = RMSNorm(ctx, self.key("hidden_norm"), eps)
        self.input_norm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.post_norm = RMSNorm(ctx, self.key("post_attention_layernorm"),
                                 eps)
        self.attention = TreeAttention(ctx, self.key("self_attn"))
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, embeds, past, rope, context_lengths, cache_start,
                kv_page_table, attention_mask, attention_pos_id):
        attention_input = F.concatenate(
            (self.input_norm(embeds), self.hidden_norm(hidden)), 2)
        attention, present = self.attention(attention_input, past, rope,
                                            context_lengths, cache_start,
                                            kv_page_table, attention_mask,
                                            attention_pos_id)
        hidden = hidden + attention
        feed_forward = self.mlp(self.post_norm(hidden))
        hidden = hidden + feed_forward
        return hidden, present


class Eagle3DraftModel(NetworkModule):
    """EAGLE3 draft model with runtime tree-attention metadata."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Eagle3DecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        target_hidden = int(
            cfg.raw_component.get("eagle3_target_hidden_size",
                                  cfg.hidden_size))
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", trt.float16,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, cfg.rotary_dim)),
            "context_lengths":
            self.add_input("context_lengths", trt.int32, (-1, )),
            "cache_start":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, -1)),
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, -1, target_hidden * 3)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
            "attention_mask":
            self.add_input("attention_mask", trt.int32, (-1, -1, -1)),
            "last_token_ids":
            self.add_input("last_token_ids", trt.int64, (-1, -1)),
        }

    def forward(self, **io):
        outputs = {}
        hidden = self.fc(io["base_hidden"])
        hidden = hidden + io["draft_hidden"]
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, io["inputs_embeds"],
                                  io["past_key_values"][index], io["rope"],
                                  io["context_lengths"], io["cache_start"],
                                  io["kv_page_table"], io["attention_mask"],
                                  io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_last_tokens(hidden, io["last_token_ids"])
        logits = self.lm_head(self.norm(selected)).cast(trt.float32)
        outputs["logits"] = logits.log_softmax(2)
        outputs["hidden_states"] = selected
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
