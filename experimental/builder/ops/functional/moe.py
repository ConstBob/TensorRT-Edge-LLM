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
"""Mixture-of-experts operations.

The implementation selects the matching lowering for the target architecture
without exposing that choice to model definitions.
"""

from enum import IntEnum

import numpy as np

from ..tensor import Tensor
from ._operation import operation, parameter
from .core import constant

__all__ = ["MoeActivation", "MoeRouting", "int4_moe", "nvfp4_moe"]


class MoeActivation(IntEnum):
    """Supported fused expert activations."""

    SWIGLU = 2
    GEGLU = 5


class MoeRouting(IntEnum):
    """Supported fused expert routing modes."""

    SOFTMAX_TOPK = 0
    SOFTMAX_TOPK_POST_SCALE = 2


_BACKEND_AUTO = 0
_IO_DTYPE_FP16 = 1
_MAX_ROUTED_ROWS_AUTO = 0


def int4_moe(router_logits: Tensor, hidden_states: Tensor, weights: dict,
             num_experts: int, top_k: int, hidden_size: int,
             moe_inter_size: int, group_size: int, *,
             weight_prefix: str) -> Tensor:
    """Run GPTQ-Marlin mixture-of-experts."""

    inputs = [
        router_logits,
        hidden_states,
        parameter(weight_prefix + ".fc_gate_up_qweights",
                  weights["fc_gate_up_qweights"].astype(np.int8, copy=False),
                  "int4_moe"),
        constant(weights["fc_gate_up_scales"].astype(np.float16),
                 "fc_gate_up_scales"),
        parameter(weight_prefix + ".fc_down_qweights",
                  weights["fc_down_qweights"].astype(np.int8,
                                                     copy=False), "int4_moe"),
        constant(weights["fc_down_scales"].astype(np.float16),
                 "fc_down_scales"),
    ]
    return operation("int4_moe",
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=0,
                     quantization_group_size=group_size)


def nvfp4_moe(router_logits: Tensor, hidden_states: Tensor, moe_weights: dict,
              num_experts: int, top_k: int, hidden_size: int,
              moe_inter_size: int, activation_type: MoeActivation,
              n_group: int, topk_group: int, norm_topk_prob: int,
              routed_scaling_factor: float, routing_mode: MoeRouting,
              sm12x: bool, *, weight_prefix: str) -> Tensor:
    """Run NVFP4 mixture-of-experts and return ``[B,S,H]`` FP16.

    ``moe_weights`` provides the 9 constant inputs as numpy arrays:
    ``fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights,
    fc2_blocks_scale, fc2_alpha, input_global_scale, down_input_scale,
    e_score_correction_bias``.
    """

    order = [
        ("fc1_qweights", np.int8),
        ("fc1_blocks_scale", np.int8),
        ("fc1_alpha", np.float32),
        ("fc2_qweights", np.int8),
        ("fc2_blocks_scale", np.int8),
        ("fc2_alpha", np.float32),
        ("input_global_scale", np.float32),
        ("down_input_scale", np.float32),
        ("e_score_correction_bias", np.float32),
    ]
    weight_inputs = [
        parameter(weight_prefix + "." + name,
                  np.ascontiguousarray(moe_weights[name], dt), "nvfp4_moe")
        for name, dt in order
    ]
    inputs = [router_logits, hidden_states] + weight_inputs
    operation_name = "nvfp4_moe_sm12x" if sm12x else "nvfp4_moe"
    return operation(operation_name,
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=int(activation_type),
                     n_group=n_group,
                     topk_group=topk_group,
                     norm_topk_prob=norm_topk_prob,
                     routed_scaling_factor=routed_scaling_factor,
                     routing_mode=int(routing_mode),
                     backend=_BACKEND_AUTO,
                     max_routed_rows=_MAX_ROUTED_ROWS_AUTO,
                     io_dtype=_IO_DTYPE_FP16)
