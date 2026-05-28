#!/usr/bin/env python3
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

"""Probe the SM110 decomposed NVFP4 MoE contract with plugin tensors.

This is a contract probe, not the production plugin path. It consumes synthetic
tensors with the current ``Nvfp4MoePlugin`` shapes, builds grouped-MoE
metadata, runs a deterministic FP4-aware reference, and optionally reports
whether the local machine has enough CuTeDSL/CUDA Python support to compile or
run the future SM110 kernels.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


SF_VEC_SIZE = 16
M_TILE_SIZE = 128
FP4_E2M1_VALUES = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ProbeCase:
    router_logits: np.ndarray
    hidden_states: np.ndarray
    fc1_qweights: np.ndarray
    fc1_blocks_scale: np.ndarray
    fc1_alpha: np.ndarray
    fc2_qweights: np.ndarray
    fc2_blocks_scale: np.ndarray
    fc2_alpha: np.ndarray
    input_global_scale: np.ndarray
    down_input_scale: np.ndarray


@dataclass(frozen=True)
class MoELayout:
    tile_idx_to_group_idx: np.ndarray
    tile_idx_to_mn_limit: np.ndarray
    permuted_idx_to_expanded_idx: np.ndarray
    num_non_exiting_tiles: int
    m_padded: int


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def validate_problem(args: argparse.Namespace) -> None:
    if args.hidden_size % (2 * SF_VEC_SIZE) != 0:
        raise ValueError("hidden_size must be divisible by 32 for packed FP4 and 16-wide scale blocks")
    if args.intermediate_size % SF_VEC_SIZE != 0:
        raise ValueError("intermediate_size must be divisible by 16")
    if not 0 < args.top_k <= args.experts:
        raise ValueError("top_k must be in the range [1, experts]")
    if args.tokens <= 0:
        raise ValueError("tokens must be positive")


def scale_shape(rows: int, cols: int, experts: int) -> tuple[int, int, int, int, int, int]:
    scale_cols = cols // SF_VEC_SIZE
    return (experts, ceil_div(rows, M_TILE_SIZE), ceil_div(scale_cols, 4), 32, 4, 4)


def make_probe_case(args: argparse.Namespace) -> ProbeCase:
    rng = np.random.default_rng(args.seed)
    n1 = args.intermediate_size * 2 if args.activation == "swiglu" else args.intermediate_size

    hidden_states = rng.normal(0.0, 0.30, size=(args.tokens, args.hidden_size)).astype(np.float32)
    router_logits = rng.normal(0.0, 1.0, size=(args.tokens, args.experts)).astype(np.float32)

    fc1_qweights = rng.integers(
        0, 256, size=(args.experts, n1, args.hidden_size // 2), dtype=np.uint8
    )
    fc2_qweights = rng.integers(
        0, 256, size=(args.experts, args.hidden_size, args.intermediate_size // 2), dtype=np.uint8
    )

    fc1_blocks_scale = rng.integers(
        8, 40, size=scale_shape(n1, args.hidden_size, args.experts), dtype=np.uint8
    )
    fc2_blocks_scale = rng.integers(
        8, 40, size=scale_shape(args.hidden_size, args.intermediate_size, args.experts), dtype=np.uint8
    )

    fc1_alpha = rng.uniform(0.20, 0.90, size=(args.experts,)).astype(np.float32)
    fc2_alpha = rng.uniform(0.20, 0.90, size=(args.experts,)).astype(np.float32)
    input_global_scale = rng.uniform(0.80, 1.20, size=(args.experts,)).astype(np.float32)
    down_input_scale = rng.uniform(0.80, 1.20, size=(args.experts,)).astype(np.float32)

    return ProbeCase(
        router_logits=router_logits,
        hidden_states=hidden_states,
        fc1_qweights=fc1_qweights,
        fc1_blocks_scale=fc1_blocks_scale,
        fc1_alpha=fc1_alpha,
        fc2_qweights=fc2_qweights,
        fc2_blocks_scale=fc2_blocks_scale,
        fc2_alpha=fc2_alpha,
        input_global_scale=input_global_scale,
        down_input_scale=down_input_scale,
    )


def topk_softmax(router_logits: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    topk_unsorted = np.argpartition(-router_logits, top_k - 1, axis=1)[:, :top_k]
    topk_scores_unsorted = np.take_along_axis(router_logits, topk_unsorted, axis=1)
    order = np.argsort(-topk_scores_unsorted, axis=1)
    topk_ids = np.take_along_axis(topk_unsorted, order, axis=1).astype(np.int32)
    topk_scores = np.take_along_axis(topk_scores_unsorted, order, axis=1)
    topk_scores = topk_scores - np.max(topk_scores, axis=1, keepdims=True)
    exp_scores = np.exp(topk_scores, dtype=np.float32)
    topk_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return topk_ids, topk_weights.astype(np.float32)


def build_moe_layout(topk_ids: np.ndarray, num_experts: int, tile_size: int = M_TILE_SIZE) -> MoELayout:
    num_tokens, top_k = topk_ids.shape
    expanded = topk_ids.reshape(num_tokens * top_k)
    counts = np.bincount(expanded, minlength=num_experts).astype(np.int32)

    offsets = np.zeros(num_experts, dtype=np.int32)
    tile_idx_to_group_idx: list[int] = []
    tile_idx_to_mn_limit: list[int] = []
    running_offset = 0

    for expert, count in enumerate(counts.tolist()):
        padded = ceil_div(count, tile_size) * tile_size if count > 0 else 0
        offsets[expert] = running_offset
        for tile in range(padded // tile_size):
            tile_idx_to_group_idx.append(expert)
            tile_idx_to_mn_limit.append(running_offset + min((tile + 1) * tile_size, count))
        running_offset += padded

    m_padded = running_offset
    permuted = np.full((m_padded,), -1, dtype=np.int32)
    scatter_counters = np.zeros((num_experts,), dtype=np.int32)

    for expanded_idx, expert in enumerate(expanded.tolist()):
        position = offsets[expert] + scatter_counters[expert]
        permuted[position] = expanded_idx
        scatter_counters[expert] += 1

    return MoELayout(
        tile_idx_to_group_idx=np.asarray(tile_idx_to_group_idx, dtype=np.int32),
        tile_idx_to_mn_limit=np.asarray(tile_idx_to_mn_limit, dtype=np.int32),
        permuted_idx_to_expanded_idx=permuted,
        num_non_exiting_tiles=len(tile_idx_to_group_idx),
        m_padded=m_padded,
    )


def unpack_fp4(packed: np.ndarray) -> np.ndarray:
    values = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32)
    values[..., 0::2] = FP4_E2M1_VALUES[packed & 0x0F]
    values[..., 1::2] = FP4_E2M1_VALUES[packed >> 4]
    return values


def pack_fp4(codes: np.ndarray) -> np.ndarray:
    if codes.shape[-1] % 2 != 0:
        raise ValueError("FP4 code rows must have an even number of elements")
    low = codes[..., 0::2].astype(np.uint8)
    high = codes[..., 1::2].astype(np.uint8)
    return low | (high << 4)


def reference_scale_to_float(scale_bytes: np.ndarray) -> np.ndarray:
    # This probe only needs a deterministic positive scale interpretation. The
    # generated kernel will consume the same bytes through CuTeDSL's FP8 type.
    return np.maximum(scale_bytes.astype(np.float32), 1.0) / 32.0


def atom_scales_to_logical(scale6d: np.ndarray, rows: int, cols: int) -> np.ndarray:
    expected = scale_shape(rows, cols, scale6d.shape[0])
    if tuple(scale6d.shape) != expected:
        raise ValueError(f"scale shape mismatch: got {scale6d.shape}, expected {expected}")

    scale_cols = cols // SF_VEC_SIZE
    logical = np.empty((scale6d.shape[0], rows, scale_cols), dtype=np.float32)
    for row in range(rows):
        m_tile = row // M_TILE_SIZE
        row_in_tile = row % M_TILE_SIZE
        row_group = row_in_tile // 32
        row_in_group = row_in_tile % 32
        for scale_col in range(scale_cols):
            k_tile = scale_col // 4
            k_mod = scale_col % 4
            logical[:, row, scale_col] = reference_scale_to_float(
                scale6d[:, m_tile, k_tile, row_in_group, row_group, k_mod]
            )
    return logical


def dequant_sm110_weight(packed: np.ndarray, scale6d: np.ndarray, rows: int, cols: int) -> np.ndarray:
    fp4_values = unpack_fp4(packed)
    if fp4_values.shape != (packed.shape[0], rows, cols):
        raise ValueError(f"packed weight shape mismatch after unpack: {fp4_values.shape}")
    logical_scales = atom_scales_to_logical(scale6d, rows, cols)
    repeated_scales = np.repeat(logical_scales, SF_VEC_SIZE, axis=2)
    return fp4_values * repeated_scales


def nearest_fp4_codes(values: np.ndarray) -> np.ndarray:
    distances = np.abs(values[..., None] - FP4_E2M1_VALUES.reshape((1,) * values.ndim + (16,)))
    return np.argmin(distances, axis=-1).astype(np.uint8)


def quantize_fp4_blocks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = values.shape
    if cols % SF_VEC_SIZE != 0:
        raise ValueError("quantize_fp4_blocks expects cols divisible by 16")
    scale_cols = cols // SF_VEC_SIZE
    dequant = np.empty_like(values, dtype=np.float32)
    codes = np.empty(values.shape, dtype=np.uint8)
    scales = np.empty((rows, scale_cols), dtype=np.float32)

    for row in range(rows):
        for scale_col in range(scale_cols):
            begin = scale_col * SF_VEC_SIZE
            end = begin + SF_VEC_SIZE
            block = values[row, begin:end]
            scale = max(float(np.max(np.abs(block))) / float(np.max(np.abs(FP4_E2M1_VALUES))), 1.0e-6)
            block_codes = nearest_fp4_codes(block / scale)
            codes[row, begin:end] = block_codes
            dequant[row, begin:end] = FP4_E2M1_VALUES[block_codes] * scale
            scales[row, scale_col] = scale

    return pack_fp4(codes), scales, dequant


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values, dtype=np.float32))


def apply_activation(values: np.ndarray, activation: str, intermediate_size: int) -> np.ndarray:
    if activation == "swiglu":
        gate = values[:intermediate_size]
        up = values[intermediate_size:]
        return silu(gate) * up
    if activation == "relu2":
        return np.square(np.maximum(values, 0.0))
    raise ValueError(f"unsupported activation {activation!r}")


def run_reference(case: ProbeCase, args: argparse.Namespace) -> dict[str, Any]:
    topk_ids, topk_weights = topk_softmax(case.router_logits, args.top_k)
    layout = build_moe_layout(topk_ids, args.experts)

    n1 = args.intermediate_size * 2 if args.activation == "swiglu" else args.intermediate_size
    fc1_weight = dequant_sm110_weight(case.fc1_qweights, case.fc1_blocks_scale, n1, args.hidden_size)
    fc2_weight = dequant_sm110_weight(
        case.fc2_qweights, case.fc2_blocks_scale, args.hidden_size, args.intermediate_size
    )

    fc1_output = np.zeros((layout.m_padded, args.intermediate_size), dtype=np.float32)
    for row, expanded_idx in enumerate(layout.permuted_idx_to_expanded_idx.tolist()):
        if expanded_idx < 0:
            continue
        token_idx = expanded_idx // args.top_k
        topk_idx = expanded_idx % args.top_k
        expert = int(topk_ids[token_idx, topk_idx])
        projection = fc1_weight[expert] @ case.hidden_states[token_idx]
        projection *= case.fc1_alpha[expert] * case.input_global_scale[expert]
        fc1_output[row] = apply_activation(projection, args.activation, args.intermediate_size)

    fc1_packed, fc1_scales, fc1_dequant = quantize_fp4_blocks(fc1_output)

    output = np.zeros((args.tokens, args.hidden_size), dtype=np.float32)
    for row, expanded_idx in enumerate(layout.permuted_idx_to_expanded_idx.tolist()):
        if expanded_idx < 0:
            continue
        token_idx = expanded_idx // args.top_k
        topk_idx = expanded_idx % args.top_k
        expert = int(topk_ids[token_idx, topk_idx])
        projection = fc2_weight[expert] @ fc1_dequant[row]
        projection *= case.fc2_alpha[expert] * case.down_input_scale[expert]
        output[token_idx] += projection * topk_weights[token_idx, topk_idx]

    non_empty_experts = int(np.count_nonzero(np.bincount(topk_ids.reshape(-1), minlength=args.experts)))
    padded_rows = int(layout.m_padded - args.tokens * args.top_k)

    return {
        "activation": args.activation,
        "tokens": args.tokens,
        "experts": args.experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "top_k": args.top_k,
        "fc1_qweights_shape": list(case.fc1_qweights.shape),
        "fc1_blocks_scale_shape": list(case.fc1_blocks_scale.shape),
        "fc2_qweights_shape": list(case.fc2_qweights.shape),
        "fc2_blocks_scale_shape": list(case.fc2_blocks_scale.shape),
        "num_non_exiting_tiles": layout.num_non_exiting_tiles,
        "m_padded": layout.m_padded,
        "padded_rows": padded_rows,
        "non_empty_experts": non_empty_experts,
        "fc1_packed_shape": list(fc1_packed.shape),
        "fc1_scale_shape": list(fc1_scales.shape),
        "output_shape": list(output.shape),
        "output_sum": float(np.sum(output)),
        "output_l2": float(np.linalg.norm(output)),
        "output_max_abs": float(np.max(np.abs(output))),
        "permuted_map_valid": bool(
            np.array_equal(
                np.sort(layout.permuted_idx_to_expanded_idx[layout.permuted_idx_to_expanded_idx >= 0]),
                np.arange(args.tokens * args.top_k, dtype=np.int32),
            )
        ),
    }


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def collect_cutedsl_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "nvidia_smi": run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
    }

    try:
        import cupy as cp  # type: ignore[import-not-found]

        props = cp.cuda.runtime.getDeviceProperties(0)
        env["cupy"] = cp.__version__
        env["cuda_device_name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        env["cuda_compute_capability"] = f"{props['major']}.{props['minor']}"
    except Exception as exc:  # noqa: BLE001 - this is an environment probe.
        env["cupy_error"] = str(exc)

    try:
        import cutlass  # type: ignore[import-not-found]

        env["cutlass"] = getattr(cutlass, "__version__", "imported")
    except Exception as exc:  # noqa: BLE001 - this is an environment probe.
        env["cutlass_error"] = str(exc)

    env["cutedsl_ready"] = "cupy" in env and "cutlass" in env
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--activation", choices=["swiglu", "relu2"], default="swiglu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-cutedsl", action="store_true")
    parser.add_argument("--require-cutedsl", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_problem(args)
    case = make_probe_case(args)
    report = run_reference(case, args)

    if args.check_cutedsl or args.require_cutedsl:
        report["environment"] = collect_cutedsl_environment()
        if args.require_cutedsl and not report["environment"]["cutedsl_ready"]:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("SM110 NVFP4 MoE probe passed")
        print(f"  output_shape={report['output_shape']}")
        print(f"  m_padded={report['m_padded']} padded_rows={report['padded_rows']}")
        print(f"  num_non_exiting_tiles={report['num_non_exiting_tiles']}")
        print(f"  output_l2={report['output_l2']:.6f} output_max_abs={report['output_max_abs']:.6f}")
        print(f"  permuted_map_valid={report['permuted_map_valid']}")
        if "environment" in report:
            print(f"  cutedsl_ready={report['environment']['cutedsl_ready']}")
            if not report["environment"]["cutedsl_ready"]:
                print("  CuTeDSL runtime is not available on this machine.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
