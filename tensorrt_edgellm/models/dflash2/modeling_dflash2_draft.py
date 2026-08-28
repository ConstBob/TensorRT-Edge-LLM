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
"""Official ONNX frontend for the Qwen3.8 DFlash2 draft graph."""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn

from ..default.modeling_default import OnnxSpec
from ..dflash.modeling_dflash_draft import (DFlashCachedDecoderLayer,
                                            DFlashDraftModel)
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE, dflash2_grouped_dynamic_conv

_BATCH_SIZE = 2
_CTX_LEN = 2
_KV_CAPACITY = 64
_MAX_BLOCK_SIZE = 16


class DFlash2GroupedConv(nn.Module):
    """Checkpoint-native dynamic grouped convolution around one sublayer."""

    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        self.block_size = _MAX_BLOCK_SIZE
        self.kernel_size = config.dflash2_conv_kernel_size
        self.group_size = config.dflash2_conv_group_size
        if config.hidden_size % self.group_size:
            raise ValueError("DFlash2 conv group_size must divide hidden_size")
        self.num_groups = config.hidden_size // self.group_size
        self.kernel_projection = make_linear(
            config,
            config.hidden_size,
            2 * self.kernel_size * self.num_groups,
            bias=False,
            module_name=f"{prefix}.kernel_projection")
        self.base_kernel = nn.Parameter(torch.empty(2,
                                                    self.kernel_size,
                                                    config.hidden_size,
                                                    dtype=torch.float16),
                                        requires_grad=False)

    def prepare(self,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden)
        side_size = self.kernel_size * self.num_groups
        pre = coefficients[..., :side_size].reshape(*coefficients.shape[:-1],
                                                    self.kernel_size,
                                                    self.num_groups)
        post = coefficients[..., side_size:].reshape(*coefficients.shape[:-1],
                                                     self.kernel_size,
                                                     self.num_groups)
        convolved = dflash2_grouped_dynamic_conv(hidden,
                                                 pre,
                                                 self.base_kernel[0],
                                                 None,
                                                 block_size=self.block_size,
                                                 kernel_size=self.kernel_size,
                                                 group_size=self.group_size,
                                                 fuse_residual=0)
        return convolved, post

    def finish(self, hidden: torch.Tensor, coefficients: torch.Tensor,
               residual: torch.Tensor) -> torch.Tensor:
        return dflash2_grouped_dynamic_conv(hidden,
                                            coefficients,
                                            self.base_kernel[1],
                                            residual,
                                            block_size=self.block_size,
                                            kernel_size=self.kernel_size,
                                            group_size=self.group_size,
                                            fuse_residual=1)


class DFlash2DecoderLayer(DFlashCachedDecoderLayer):
    """DFlash attention/MLP layer with DFlash2 pre/post dynamic conv."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.attention_conv = DFlash2GroupedConv(
            config, f"layers.{layer_idx}.attention_conv")
        self.mlp_conv = DFlash2GroupedConv(config,
                                           f"layers.{layer_idx}.mlp_conv")

    def forward(self, hidden_states, h_delta, past_key_value, rope_cos_sin,
                kvcache_start_index, kv_page_table, delta_lengths,
                context_lengths, attention_mask, attention_pos_id):
        residual = hidden_states.to(torch.float32)
        attention_input = self.input_layernorm(residual).to(torch.float16)
        attention_input, attention_post = self.attention_conv.prepare(
            attention_input)
        attention, present_kv = self.self_attn(attention_input, h_delta,
                                               past_key_value, rope_cos_sin,
                                               kvcache_start_index,
                                               kv_page_table, delta_lengths,
                                               context_lengths, attention_mask,
                                               attention_pos_id)
        hidden_states = self.attention_conv.finish(attention, attention_post,
                                                   residual)

        residual = hidden_states
        mlp_input = self.post_attention_layernorm(residual).to(torch.float16)
        mlp_input, mlp_post = self.mlp_conv.prepare(mlp_input)
        feed_forward = self.mlp(mlp_input)
        hidden_states = self.mlp_conv.finish(feed_forward, mlp_post, residual)
        return hidden_states, present_kv


class DFlash2CandidateSelector(nn.Module):
    """Checkpoint-shaped selector parameters."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_projection = make_linear(
            config,
            config.hidden_size,
            config.dflash2_selector_rank,
            bias=False,
            module_name="candidate_selector.hidden_projection")
        codebook_shape = (config.vocab_size, config.dflash2_selector_rank)
        self.predecessor_codebook = nn.Parameter(torch.empty(
            codebook_shape, dtype=torch.float16),
                                                 requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(
            codebook_shape, dtype=torch.float16),
                                               requires_grad=False)


def _make_flat_wrapper(model: nn.Module, num_layers: int) -> nn.Module:
    parameters = [
        "inputs_embeds", "dflash_target_hidden_concat", "rope_rotary_cos_sin",
        "context_lengths", "kvcache_start_index", "kv_page_table",
        "dflash_delta_lengths", "attention_mask", "attention_pos_id"
    ] + [f"past_key_values_{index}" for index in range(num_layers)]
    past = "({},)".format(", ".join(f"past_key_values_{index}"
                                    for index in range(num_layers)))
    body = (
        "    outputs, present = self._model(\n"
        "        inputs_embeds, dflash_target_hidden_concat,\n"
        "        rope_rotary_cos_sin, context_lengths, kvcache_start_index,\n"
        "        kv_page_table, dflash_delta_lengths, attention_mask,\n"
        f"        attention_pos_id, list({past}))\n"
        "    return tuple(outputs) + tuple(present)\n")
    namespace = {}
    exec("def _forward(self, {}):\n{}".format(", ".join(parameters), body),
         namespace)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, wrapped):
            super().__init__()
            self._model = wrapped

    _Wrapper.forward = namespace["_forward"]
    return _Wrapper(model)


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 draft backbone, target head, TopK, and selector walk."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([
            DFlash2DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ])
        self.candidate_selector = DFlash2CandidateSelector(config)

    def forward(self, inputs_embeds, target_hidden_concat, rope_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                delta_lengths, attention_mask, attention_pos_id,
                past_key_values: List[torch.Tensor]):
        bias = (self.fc.bias.to(torch.float32)
                if self.fc.bias is not None else None)
        h_delta = torch.nn.functional.linear(
            target_hidden_concat.to(torch.float32),
            self.fc.weight.to(torch.float32), bias)
        h_delta = self.hidden_norm(h_delta).to(torch.float16)

        hidden_states = inputs_embeds.to(torch.float32)
        present_key_values = []
        for index, layer in enumerate(self.layers):
            hidden_states, present = layer(hidden_states, h_delta,
                                           past_key_values[index],
                                           rope_cos_sin, kvcache_start_index,
                                           kv_page_table, delta_lengths,
                                           context_lengths, attention_mask,
                                           attention_pos_id)
            present_key_values.append(present)

        prediction_hidden = self.norm(hidden_states).to(torch.float16)[:,
                                                                       1:, :]
        projected = self.candidate_selector.hidden_projection(
            prediction_hidden).to(torch.float16)
        unary_logits = self.lm_head(prediction_hidden).to(torch.float32)
        unary_values, candidate_ids = torch.topk(
            unary_logits, self.config.dflash2_selector_top_k, dim=2)
        outputs = (candidate_ids.to(torch.int32), unary_values, projected)
        return outputs, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        batch, block, delta_len = _BATCH_SIZE, config.dflash2_block_size, _CTX_LEN
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        args = [
            torch.zeros(batch,
                        block,
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device),
            torch.zeros(batch,
                        delta_len,
                        len(config.dflash2_target_layer_ids) *
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device),
            torch.zeros(1,
                        _KV_CAPACITY,
                        rotary_dim,
                        dtype=torch.float32,
                        device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, 2, 1, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch,
                        block, (block + 31) // 32,
                        dtype=torch.int32,
                        device=device),
            torch.zeros(batch, block, dtype=torch.int32, device=device),
        ]
        args.extend(
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=torch.float16,
                        device=device)
            for _ in range(config.num_hidden_layers))
        input_names = [
            "inputs_embeds", "dflash_target_hidden_concat",
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "dflash_delta_lengths", "attention_mask",
            "attention_pos_id"
        ] + [f"past_key_values_{i}" for i in range(config.num_hidden_layers)]
        output_names = [
            "spec_proposal_support_ids", "spec_proposal_unary_values",
            "spec_proposal_projected_hidden"
        ] + [
            f"present_key_values_{i}" for i in range(config.num_hidden_layers)
        ]
        batch_dim = torch.export.Dim("batch", min=1, max=256)
        block_dim = torch.export.Dim("dflash2_block",
                                     min=2,
                                     max=_MAX_BLOCK_SIZE)
        delta_dim = torch.export.Dim("delta_seq", min=1, max=32768)
        kv_dim = torch.export.Dim("kv_len", min=1, max=32768)
        pages_dim = torch.export.Dim("num_pages", min=1, max=1048576)
        max_pages_dim = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        dynamic_shapes = [
            {
                0: batch_dim,
                1: block_dim
            },
            {
                0: batch_dim,
                1: delta_dim
            },
            {
                1: kv_dim
            },
            {
                0: batch_dim
            },
            {
                0: batch_dim
            },
            {
                0: batch_dim,
                2: max_pages_dim
            },
            {
                0: batch_dim
            },
            {
                0: batch_dim,
                1: block_dim
            },
            {
                0: batch_dim,
                1: block_dim
            },
        ] + [{
            1: pages_dim
        } for _ in range(config.num_hidden_layers)]
        wrapped = _make_flat_wrapper(self, config.num_hidden_layers)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=tuple(args),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dynamic_shapes)
