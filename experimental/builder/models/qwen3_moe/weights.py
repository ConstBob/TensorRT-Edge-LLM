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
"""Qwen3 MoE checkpoint weight mapping."""

from dataclasses import replace
from typing import Sequence

import numpy as np

from ...core import quantization
from ...core.weights import LinearWeights
from ...weight_packing import nvfp4 as nvfp4_pack


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3 MoE checkpoint aliases."""
    del component, spec_type, spec_role
    if name == "lm_head.weight" and quant_type == "fp16":
        return ("model.embed_tokens.weight", )
    return ()


def fuse_gptq_qkv(projections: Sequence[LinearWeights]) -> LinearWeights:
    """Fuse Q/K/V GPTQ descriptors into one family-owned projection.

    Qwen3 checkpoints store three projections, matching Transformers. The
    compiled backend uses one GEMM when their activation permutations agree;
    this also gives attention a physically contiguous packed-QKV tensor.
    """
    query, key, value = projections
    for name, projection in zip(("query", "key", "value"), projections):
        if projection.quant_type != quantization.QUANT_INT4_GPTQ:
            raise ValueError(f"{name} projection is not GPTQ INT4")
        if projection.in_features != query.in_features:
            raise ValueError("Qwen3 Q/K/V projections must share input size")
        if projection.group_size != query.group_size:
            raise ValueError("Qwen3 Q/K/V projections must share group size")
        scales = projection.weight_scale
        if (scales is None or scales.ndim != 2
                or scales.shape[1] != projection.out_features):
            raise ValueError(f"unsupported Qwen3 GPTQ scale shape for {name}: "
                             f"{None if scales is None else scales.shape}")

    permutation = query.activation_permutation
    if any(not np.array_equal(permutation, projection.activation_permutation)
           for projection in (key, value)):
        raise ValueError("Qwen3 GPTQ Q/K/V activation permutations must match")

    biases = [projection.bias for projection in projections]
    if any(bias is None for bias in biases):
        if not all(bias is None for bias in biases):
            raise ValueError("Qwen3 Q/K/V projections must use uniform bias")
        bias = None
    else:
        bias = np.ascontiguousarray(np.concatenate(biases))

    return replace(
        query,
        weight=np.ascontiguousarray(
            np.concatenate([projection.weight for projection in projections],
                           axis=0)),
        bias=bias,
        weight_scale=np.ascontiguousarray(
            np.concatenate(
                [projection.weight_scale for projection in projections],
                axis=1)),
    )


def repack_nvfp4_experts(load_expert, num_experts: int, hidden_size: int,
                         intermediate_size: int, group_size: int,
                         fc1_layout: str):
    """Pack this family's SwiGLU experts for the Edge-LLM MoE operation."""
    if fc1_layout not in ("interleave", "concat"):
        raise ValueError(f"unsupported FC1 layout {fc1_layout!r}")

    def build_fc1(gate, up):
        if fc1_layout == "concat":
            return np.concatenate([up, gate],
                                  axis=0).reshape(2 * intermediate_size,
                                                  hidden_size)
        rows = 64
        if intermediate_size % rows:
            raise ValueError("moe_intermediate_size must be a multiple of 64")
        chunks = intermediate_size // rows
        up_chunks = up.reshape(chunks, rows, hidden_size)
        gate_chunks = gate.reshape(chunks, rows, hidden_size)
        return np.stack([up_chunks, gate_chunks],
                        axis=1).reshape(2 * intermediate_size, hidden_size)

    fc1_weights, fc1_scales = [], []
    fc2_weights, fc2_scales = [], []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        fc1 = build_fc1(expert["gate"], expert["up"])
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(fc1, group_size)
        fc1_weights.append(weight)
        fc1_scales.append(scale)
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(
            expert["down"], group_size)
        fc2_weights.append(weight)
        fc2_scales.append(scale)

    ones = np.ones(num_experts, dtype=np.float32)
    return (np.stack(fc1_weights), np.stack(fc1_scales), ones,
            np.stack(fc2_weights), np.stack(fc2_scales), ones.copy())
