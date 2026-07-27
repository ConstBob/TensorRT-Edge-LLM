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
"""Reusable MoE modules whose checkpoint and routing semantics are identical."""

from typing import Callable, Dict

import numpy as np
import tensorrt as trt

from ..weight_packing import int4 as int4_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import functional as F
from .module import BuildContext, Module
from .tensor import Tensor


def prepare_gated_nvfp4_weights(
        ctx: BuildContext, experts: "GatedExperts",
        repack_experts: Callable) -> Dict[str, np.ndarray]:
    """Prepare the common nine-buffer NVFP4 gated-expert op layout."""
    cfg = ctx.cfg
    layout = "concat" if ctx.options.sm12x else "interleave"
    packed = repack_experts(
        experts.load_expert_dense,
        cfg.num_experts,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.group_size,
        fc1_layout=layout,
    )
    (fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights, fc2_blocks_scale,
     fc2_alpha) = packed
    return {
        "fc1_qweights": fc1_qweights,
        "fc1_blocks_scale": fc1_blocks_scale,
        "fc1_alpha": fc1_alpha,
        "fc2_qweights": fc2_qweights,
        "fc2_blocks_scale": fc2_blocks_scale,
        "fc2_alpha": fc2_alpha,
        "input_global_scale": np.ones(cfg.num_experts, np.float32),
        "down_input_scale": np.ones(cfg.num_experts, np.float32),
        "e_score_correction_bias": np.zeros(cfg.num_experts, np.float32),
    }


def prepare_gated_int4_weights(ctx: BuildContext,
                               prefix: str) -> Dict[str, np.ndarray]:
    """Prepare stacked GPTQ experts for the common Marlin op layout."""
    cfg = ctx.cfg
    weights = ctx.weights
    gate_up_weights = []
    gate_up_scales = []
    down_weights = []
    down_scales = []
    for expert_index in range(cfg.num_experts):
        expert = f"{prefix}.experts.{expert_index}"

        def extract(projection: str):
            projection_prefix = f"{expert}.{projection}"
            qzeros = (weights.array(projection_prefix + ".qzeros") if
                      weights.has(projection_prefix + ".qzeros") else np.empty(
                          (1, 0), dtype=np.int32))
            return int4_pack.extract_gptq_for_moe(
                weights.array(projection_prefix + ".qweight"), qzeros,
                weights.f16(projection_prefix + ".scales"), cfg.group_size,
                cfg.quant.gptq_zero_point_offset)

        gate_weight, gate_scale = extract("gate_proj")
        up_weight, up_scale = extract("up_proj")
        down_weight, down_scale = extract("down_proj")
        gate_up_weights.append(np.concatenate((gate_weight, up_weight),
                                              axis=0))
        gate_up_scales.append(np.concatenate((gate_scale, up_scale), axis=0))
        down_weights.append(down_weight)
        down_scales.append(down_scale)
    gate_weight, gate_scale = int4_pack.pack_moe_marlin(
        np.stack(gate_up_weights), np.stack(gate_up_scales), cfg.group_size)
    down_weight, down_scale = int4_pack.pack_moe_marlin(
        np.stack(down_weights), np.stack(down_scales), cfg.group_size)
    return {
        "fc_gate_up_qweights": gate_weight,
        "fc_gate_up_scales": gate_scale,
        "fc_down_qweights": down_weight,
        "fc_down_scales": down_scale,
    }


class TopKRouter(Module):
    """FP16 router projection producing FP32 logits."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        weight = F.constant(self.weights.f16(self.key("weight")), "gate")
        hidden_states = hidden_states.reshape((-1, self.cfg.hidden_size))
        return F.matmul(hidden_states, weight,
                        transpose_rhs=True).cast(trt.float32)


class GatedExperts(Module):
    """Load stacked or individually stored gate/up/down expert weights."""

    def __len__(self) -> int:
        return self.cfg.num_experts

    def load_expert_dense(self, expert_index: int) -> Dict[str, np.ndarray]:
        gate_up_key = self.key("gate_up_proj")
        down_key = self.key("down_proj")
        if self.weights.has(gate_up_key) and self.weights.has(down_key):
            gate_up = self.weights.f32(gate_up_key)[expert_index]
            gate, up = np.split(gate_up, 2, axis=0)
            down = self.weights.f32(down_key)[expert_index]
            return {
                "gate": np.ascontiguousarray(gate),
                "up": np.ascontiguousarray(up),
                "down": np.ascontiguousarray(down),
            }

        prefix = self.key(str(expert_index))
        return {
            "gate": self.weights.expert_dense_f32(f"{prefix}.gate_proj"),
            "up": self.weights.expert_dense_f32(f"{prefix}.up_proj"),
            "down": self.weights.expert_dense_f32(f"{prefix}.down_proj"),
        }


class GroupedSigmoidRouter(Module):
    """Float32 router used by grouped sigmoid top-k models."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        correction_key = self.key("e_score_correction_bias")
        self.correction = (self.weights.f32(correction_key)
                           if self.weights.has(correction_key) else np.zeros(
                               self.cfg.num_experts, dtype=np.float32))

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.reshape(
            (-1, self.cfg.hidden_size)).cast(trt.float32)
        weight = F.constant(self.weights.f32(self.key("weight")),
                            "router_weight")
        return hidden_states.matmul(weight,
                                    rhs_op=trt.MatrixOperation.TRANSPOSE)


class NonGatedNvfp4Experts(Module):
    """NVFP4 non-gated expert bank shared by equivalent Nemotron models."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 repack_experts: Callable) -> None:
        super().__init__(ctx, prefix)
        self.repack_experts = repack_experts

    def _has_stacked_experts(self) -> bool:
        return (self.weights.has(self.key("up_proj"))
                and self.weights.has(self.key("down_proj")))

    def _load_expert(self, expert_index: int) -> dict:
        prefix = self.key(str(expert_index))
        up = self.weights.expert_raw_nvfp4(prefix + ".up_proj")
        down = self.weights.expert_raw_nvfp4(prefix + ".down_proj")
        return {
            "up_packed": up["packed"],
            "up_sf": up["sf"],
            "up_alpha": up["alpha"],
            "down_packed": down["packed"],
            "down_sf": down["sf"],
            "down_alpha": down["alpha"],
        }

    def _pack_stacked_experts(self, hidden_size: int, hidden_alignment: int):
        cfg = self.cfg
        padded_intermediate = ((cfg.moe_intermediate_size + 127) // 128) * 128
        padded_hidden = ((hidden_size + hidden_alignment - 1) //
                         hidden_alignment) * hidden_alignment
        if padded_hidden % cfg.group_size:
            raise ValueError("padded hidden size must divide the FP4 group")
        stacked_up = self.weights.f32(self.key("up_proj"))
        stacked_down = self.weights.f32(self.key("down_proj"))
        fc1_weight = []
        fc1_scale = []
        fc2_weight = []
        fc2_scale = []
        for expert_index in range(cfg.num_experts):
            up = np.zeros((padded_intermediate, padded_hidden), np.float32)
            down = np.zeros((padded_hidden, padded_intermediate), np.float32)
            up[:cfg.
               moe_intermediate_size, :hidden_size] = stacked_up[expert_index]
            down[:hidden_size, :cfg.
                 moe_intermediate_size] = stacked_down[expert_index]
            packed_up, scale_up = nvfp4_pack.pack_nvfp4_moe_weight(
                up, cfg.group_size)
            packed_down, scale_down = nvfp4_pack.pack_nvfp4_moe_weight(
                down, cfg.group_size)
            fc1_weight.append(packed_up)
            fc1_scale.append(scale_up)
            fc2_weight.append(packed_down)
            fc2_scale.append(scale_down)
        ones = np.ones(cfg.num_experts, np.float32)
        return (np.stack(fc1_weight), np.stack(fc1_scale), ones,
                np.stack(fc2_weight), np.stack(fc2_scale), ones.copy(),
                padded_intermediate, padded_hidden)

    def forward(self, hidden_states: Tensor, router_logits: Tensor,
                correction: np.ndarray) -> Tensor:
        cfg = self.cfg
        hidden_size = cfg.moe_latent_size or cfg.hidden_size
        hidden_alignment = 256 if self.ctx.options.sm12x else 1
        if self._has_stacked_experts():
            packed = self._pack_stacked_experts(hidden_size, hidden_alignment)
        else:
            packed = self.repack_experts(self._load_expert, cfg.num_experts,
                                         hidden_size,
                                         cfg.moe_intermediate_size,
                                         cfg.group_size, hidden_alignment)
        (fc1_weight, fc1_scale, fc1_alpha, fc2_weight, fc2_scale, fc2_alpha,
         padded_intermediate, padded_hidden) = packed
        if padded_hidden != hidden_size:
            hidden_states = F.pad_last_dim(hidden_states,
                                           padded_hidden - hidden_size, 3)
        weights = {
            "fc1_qweights": fc1_weight,
            "fc1_blocks_scale": fc1_scale,
            "fc1_alpha": fc1_alpha,
            "fc2_qweights": fc2_weight,
            "fc2_blocks_scale": fc2_scale,
            "fc2_alpha": fc2_alpha,
            "input_global_scale": np.ones(cfg.num_experts, np.float32),
            "down_input_scale": np.ones(cfg.num_experts, np.float32),
            "e_score_correction_bias": correction.astype(np.float32),
        }
        output = F.nvfp4_moe(router_logits,
                             hidden_states,
                             weights,
                             cfg.num_experts,
                             cfg.num_experts_per_tok,
                             padded_hidden,
                             padded_intermediate,
                             4,
                             cfg.n_group,
                             cfg.topk_group,
                             int(cfg.norm_topk_prob),
                             cfg.routed_scaling_factor,
                             1,
                             self.ctx.options.sm12x,
                             weight_prefix=self.prefix)
        if padded_hidden != hidden_size:
            output = F.slice_last_dim(output, 0, hidden_size, 3)
        return output
