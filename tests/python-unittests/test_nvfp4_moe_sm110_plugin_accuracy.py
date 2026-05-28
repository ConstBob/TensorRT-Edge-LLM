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
"""SM110 TensorRT accuracy tests for Nvfp4MoePlugin.

The tests auto-skip unless the host is a Thor SM110 (compute capability 11.0)
with TensorRT Python and a plugin built with ``-DENABLE_CUTE_DSL=nvfp4_moe``.
Hardware/runtime gating lives in :func:`check_requirements`. To point the test
at a specific plugin ``.so`` (rather than the default ``build*/`` discovery),
set ``EDGELLM_NVFP4_MOE_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so``.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ENV = "EDGELLM_NVFP4_MOE_PLUGIN_SO"
SF_VEC_SIZE = 16
ROW_TILE = 128
FP4_LEVELS = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
SM110_NUM_EXPERTS = 128
SM110_TOP_K = 8


@dataclass(frozen=True)
class Sm110CaseConfig:
    name: str
    hidden_size: int
    intermediate_size: int
    seed: int
    non_uniform_scales: bool
    min_cosine: float
    min_mag_ratio: float
    max_mag_ratio: float
    activation_type: int = 2
    scale_mode: str = "config"
    # Number of input tokens. ``num_tokens=1`` exercises the SM110 fast-decode
    # fused setup path (only fires for hidden_size=2048 per
    # cuteDslNvfp4MoeSm110Runner.cpp::useFastDecodeSetup); ``num_tokens>1``
    # exercises the general gather + grouped-GEMM path.
    num_tokens: int = 1


@dataclass
class Sm110Case:
    config: Sm110CaseConfig
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
    e_score_correction_bias: np.ndarray


def resolve_plugin_so() -> Path:
    env = os.environ.get(PLUGIN_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        REPO_ROOT / "build_thor_sm110" / "libNvInfer_edgellm_plugin.so",
        REPO_ROOT / "build" / "cpp" / "libNvInfer_edgellm_plugin.so",
        REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def check_requirements() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch", reason="torch required for CUDA execution and FP8 reference")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip(f"Thor SM110 required, got compute capability {torch.cuda.get_device_capability()}")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn required for FP8 scale reference")
    trt = pytest.importorskip("tensorrt", reason="TensorRT Python required")
    plugin_so = resolve_plugin_so()
    if not plugin_so.is_file():
        pytest.skip(f"missing plugin library: {plugin_so}")
    return torch, trt


def scale_shape(rows: int, cols: int, experts: int = SM110_NUM_EXPERTS) -> tuple[int, int, int, int, int, int]:
    sf_cols = cols // SF_VEC_SIZE
    return (experts, math.ceil(rows / ROW_TILE), math.ceil(sf_cols / 4), 32, 4, 4)


def atom_offsets(rows: int, sf_cols: int) -> np.ndarray:
    m_idx = np.arange(rows, dtype=np.int64).reshape(rows, 1)
    k_idx = np.arange(sf_cols, dtype=np.int64).reshape(1, sf_cols)
    inner_k = k_idx % 4
    inner_m = (m_idx % ROW_TILE) // 32
    outer_m = m_idx % 32
    k_tile = k_idx // 4
    num_k_tiles = (sf_cols + 3) // 4
    m_tile = m_idx // ROW_TILE
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


def fp8_bytes_to_float(raw: np.ndarray) -> np.ndarray:
    import torch

    raw_u8 = np.ascontiguousarray(raw, dtype=np.uint8)
    return torch.from_numpy(raw_u8).view(torch.float8_e4m3fn).float().numpy()


def float_to_fp8_bytes(values: np.ndarray | list[float]) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)
    return tensor.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()


def make_scale_tensor(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    *,
    non_uniform: bool,
) -> np.ndarray:
    shape = scale_shape(rows, cols)
    scale_values = np.asarray([0.00390625, 0.005859375, 0.0078125, 0.01171875], dtype=np.float32)
    scale_bytes = float_to_fp8_bytes(scale_values)
    if not non_uniform:
        out = np.empty(shape, dtype=np.uint8)
        out.fill(int(scale_bytes[2]))
        return out.view(np.int8)
    indices = rng.integers(0, len(scale_bytes), size=shape, dtype=np.uint8)
    return scale_bytes[indices].astype(np.uint8, copy=False).view(np.int8)


def make_qweights(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Random bytes intentionally exercise all FP4 nibbles while keeping memory
    # bounded. The reference dequantizes the exact selected bytes that TRT sees.
    return rng.integers(0, 256, size=shape, dtype=np.uint8).view(np.int8)


def topk_logits_from_scores(expert_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    logits = np.full((expert_ids.shape[0], SM110_NUM_EXPERTS), -120.0, dtype=np.float32)
    for token in range(expert_ids.shape[0]):
        for slot in range(SM110_TOP_K):
            score = max(float(scores[token, slot]), 1e-12)
            logits[token, int(expert_ids[token, slot])] = np.float32(math.log(score) + 25.0)
    return logits


def topk_softmax(router_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(router_logits, dtype=np.float32)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True), dtype=np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True)
    top_ids = np.zeros((logits.shape[0], SM110_TOP_K), dtype=np.int32)
    top_weights = np.zeros((logits.shape[0], SM110_TOP_K), dtype=np.float32)
    for token in range(logits.shape[0]):
        used = np.zeros((SM110_NUM_EXPERTS,), dtype=bool)
        for slot in range(SM110_TOP_K):
            best_expert = -1
            best_prob = -1.0
            for expert in range(SM110_NUM_EXPERTS):
                if used[expert]:
                    continue
                prob = float(probs[token, expert])
                if best_expert < 0 or prob > best_prob or (prob == best_prob and expert < best_expert):
                    best_expert = expert
                    best_prob = prob
            used[best_expert] = True
            top_ids[token, slot] = best_expert
            top_weights[token, slot] = best_prob
    top_weights /= np.sum(top_weights, axis=1, keepdims=True) + np.float32(1e-20)
    return top_ids, top_weights


def round_fp4(values: np.ndarray) -> np.ndarray:
    values_f32 = np.asarray(values, dtype=np.float32)
    distances = np.abs(values_f32[..., None] - FP4_LEVELS.reshape((1,) * values_f32.ndim + (16,)))
    return np.argmin(distances, axis=-1).astype(np.uint8)


def fp4_roundtrip_linear_sf(values: np.ndarray, global_scale: float) -> np.ndarray:
    return (fp4_roundtrip_linear_sf_raw(values, global_scale) * np.float32(global_scale)).astype(np.float32)


def fp4_roundtrip_linear_sf_raw(values: np.ndarray, global_scale: float) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    if arr.shape[-1] % SF_VEC_SIZE != 0:
        raise ValueError(f"last dimension must be divisible by {SF_VEC_SIZE}, got {arr.shape[-1]}")
    out = np.zeros_like(arr, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    scale = max(float(global_scale), 1e-12)
    for row in range(flat.shape[0]):
        for begin in range(0, flat.shape[1], SF_VEC_SIZE):
            end = begin + SF_VEC_SIZE
            block = flat[row, begin:end]
            vec_max = float(np.max(np.abs(block)))
            if vec_max == 0.0:
                continue
            sf_value = (vec_max / 6.0) / scale
            sf_back = float(fp8_bytes_to_float(float_to_fp8_bytes([sf_value]))[0])
            effective_scale = sf_back * scale
            if effective_scale <= 0.0 or not np.isfinite(effective_scale):
                continue
            codes = round_fp4(block / effective_scale)
            out.reshape(-1, arr.shape[-1])[row, begin:end] = FP4_LEVELS[codes] * sf_back
    return out


def dequant_weight(
    qweights: np.ndarray,
    scale6d: np.ndarray,
    expert: int,
    *,
    rows: int,
    cols: int,
    alpha: float,
) -> np.ndarray:
    packed = np.ascontiguousarray(qweights[expert], dtype=np.int8).view(np.uint8)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibbles = np.empty((rows, cols), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    values = FP4_LEVELS[nibbles].astype(np.float32, copy=False)

    sf_cols = cols // SF_VEC_SIZE
    offsets = atom_offsets(rows, sf_cols)
    sf_bytes = np.ascontiguousarray(scale6d[expert], dtype=np.int8).view(np.uint8).reshape(-1)[offsets]
    scales = np.repeat(fp8_bytes_to_float(sf_bytes), SF_VEC_SIZE, axis=1)[:, :cols]
    return (values * scales * np.float32(alpha)).astype(np.float32)


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values, dtype=np.float32))


def apply_swiglu_interleaved(projection: np.ndarray) -> np.ndarray:
    if projection.shape[0] % 128 != 0:
        raise ValueError(f"SwiGLU projection must be 128-interleaved, got {projection.shape[0]}")
    chunks = projection.reshape(-1, 128)
    up = chunks[:, :64]
    gate = chunks[:, 64:]
    return (up * silu(gate)).reshape(-1).astype(np.float32)


def fc1_input_n(intermediate_size: int, activation_type: int) -> int:
    return 2 * intermediate_size if int(activation_type) == 2 else intermediate_size


def apply_fc1_activation(projection: np.ndarray, activation_type: int) -> np.ndarray:
    if int(activation_type) == 2:
        return apply_swiglu_interleaved(projection)
    if int(activation_type) == 4:
        return np.square(np.maximum(projection, 0.0)).astype(np.float32)
    raise ValueError(f"unsupported SM110 activation_type: {activation_type}")


def compute_reference(case: Sm110Case) -> np.ndarray:
    cfg = case.config
    hidden = np.asarray(case.hidden_states, dtype=np.float16).reshape(-1, cfg.hidden_size).astype(np.float32)
    top_ids, top_weights = topk_softmax(case.router_logits)

    fc1_cache: dict[int, np.ndarray] = {}
    fc2_cache: dict[int, np.ndarray] = {}
    active_slots: list[tuple[int, int, float, np.ndarray]] = []
    n1 = fc1_input_n(cfg.intermediate_size, cfg.activation_type)

    for token in range(hidden.shape[0]):
        for slot in range(SM110_TOP_K):
            expert = int(top_ids[token, slot])
            if expert not in fc1_cache:
                fc1_cache[expert] = dequant_weight(
                    case.fc1_qweights,
                    case.fc1_blocks_scale,
                    expert,
                    rows=n1,
                    cols=cfg.hidden_size,
                    alpha=float(case.fc1_alpha[expert]),
                )
            hidden_raw = fp4_roundtrip_linear_sf_raw(
                hidden[token:token + 1], float(case.input_global_scale[expert])
            )[0]
            projection = fc1_cache[expert] @ hidden_raw
            projection *= np.float32(case.input_global_scale[expert])
            activated = apply_fc1_activation(projection, cfg.activation_type)
            active_slots.append((token, expert, float(top_weights[token, slot]), activated))

    output = np.zeros((hidden.shape[0], cfg.hidden_size), dtype=np.float32)
    for token, expert, router_weight, activated in active_slots:
        activated_raw = fp4_roundtrip_linear_sf_raw(activated[None, :], float(case.down_input_scale[expert]))[0]
        if expert not in fc2_cache:
            fc2_cache[expert] = dequant_weight(
                case.fc2_qweights,
                case.fc2_blocks_scale,
                expert,
                rows=cfg.hidden_size,
                cols=cfg.intermediate_size,
                alpha=float(case.fc2_alpha[expert]),
            )
        output[token] += (
            np.float32(router_weight)
            * np.float32(case.down_input_scale[expert])
            * (fc2_cache[expert] @ activated_raw)
        )
    return output.reshape(1, hidden.shape[0], cfg.hidden_size)


def make_case(config: Sm110CaseConfig) -> Sm110Case:
    rng = np.random.default_rng(config.seed)
    num_tokens = max(1, int(config.num_tokens))
    if num_tokens == 1:
        # Hand-picked spread covers low/mid/high expert indices for the
        # single-token decode path.
        selected = np.asarray([[0, 7, 19, 31, 47, 64, 96, 127]], dtype=np.int32)
        scores = np.asarray(
            [[0.28, 0.21, 0.16, 0.12, 0.09, 0.06, 0.045, 0.035]],
            dtype=np.float32,
        )
    else:
        selected = np.zeros((num_tokens, SM110_TOP_K), dtype=np.int32)
        for token in range(num_tokens):
            selected[token] = rng.choice(SM110_NUM_EXPERTS, size=SM110_TOP_K, replace=False)
        base_scores = np.asarray(
            [0.28, 0.21, 0.16, 0.12, 0.09, 0.06, 0.045, 0.035], dtype=np.float32,
        )
        jitter = rng.uniform(0.9, 1.1, size=(num_tokens, SM110_TOP_K)).astype(np.float32)
        scores = base_scores * jitter
    router_logits = topk_logits_from_scores(selected, scores)
    hidden = rng.normal(0.0, 0.05, size=(1, num_tokens, config.hidden_size)).astype(np.float16)
    n1 = fc1_input_n(config.intermediate_size, config.activation_type)
    fc1_q = make_qweights(rng, (SM110_NUM_EXPERTS, n1, config.hidden_size // 2))
    fc2_q = make_qweights(rng, (SM110_NUM_EXPERTS, config.hidden_size, config.intermediate_size // 2))
    fc1_scale = make_scale_tensor(rng, n1, config.hidden_size, non_uniform=config.non_uniform_scales)
    fc2_scale = make_scale_tensor(
        rng, config.hidden_size, config.intermediate_size, non_uniform=config.non_uniform_scales)
    if config.non_uniform_scales:
        fc1_alpha = np.linspace(0.55, 1.35, SM110_NUM_EXPERTS, dtype=np.float32)
        fc2_alpha = np.linspace(1.25, 0.65, SM110_NUM_EXPERTS, dtype=np.float32)
        input_scale = np.linspace(1.0e-4, 8.0e-4, SM110_NUM_EXPERTS, dtype=np.float32)
        down_scale = np.linspace(8.0e-4, 1.0e-4, SM110_NUM_EXPERTS, dtype=np.float32)
    else:
        fc1_alpha = np.full((SM110_NUM_EXPERTS,), 0.85, dtype=np.float32)
        fc2_alpha = np.full((SM110_NUM_EXPERTS,), 0.75, dtype=np.float32)
        input_scale = np.full((SM110_NUM_EXPERTS,), 1.0e-4, dtype=np.float32)
        down_scale = np.full((SM110_NUM_EXPERTS,), 1.0e-4, dtype=np.float32)
    if config.scale_mode == "all_ones":
        fc1_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        input_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "input_scale_only":
        fc1_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "down_scale_only":
        fc1_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        input_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "alpha_only":
        input_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode not in {"config", "full_non_uniform"}:
        raise ValueError(f"unknown SM110 scale_mode: {config.scale_mode}")

    if config.scale_mode in {"config", "full_non_uniform"} and not config.non_uniform_scales:
        input_floor = max(float(np.max(np.abs(hidden.astype(np.float32)))) / (448.0 * 6.0), 1e-12)
        input_scale.fill(np.float32(input_floor))
    # The unified Nvfp4MoePlugin requires an e_score_correction_bias input even
    # in softmax-topk mode (the bias is added to the router logits before top-k).
    # A zero bias is a no-op for the routing selection and keeps the numpy
    # reference in compute_reference() consistent with the plugin.
    e_score_correction_bias = np.zeros((SM110_NUM_EXPERTS,), dtype=np.float32)
    return Sm110Case(
        config=config,
        router_logits=np.ascontiguousarray(router_logits),
        hidden_states=np.ascontiguousarray(hidden),
        fc1_qweights=np.ascontiguousarray(fc1_q),
        fc1_blocks_scale=np.ascontiguousarray(fc1_scale),
        fc1_alpha=np.ascontiguousarray(fc1_alpha),
        fc2_qweights=np.ascontiguousarray(fc2_q),
        fc2_blocks_scale=np.ascontiguousarray(fc2_scale),
        fc2_alpha=np.ascontiguousarray(fc2_alpha),
        input_global_scale=np.ascontiguousarray(input_scale),
        down_input_scale=np.ascontiguousarray(down_scale),
        e_score_correction_bias=np.ascontiguousarray(e_score_correction_bias),
    )


def preload_libnvinfer() -> None:
    trt_dir = os.environ.get("TRT_PACKAGE_DIR", "").strip()
    candidates: list[Path] = []
    if trt_dir:
        candidates.extend(Path(trt_dir).expanduser().glob("lib/libnvinfer.so*"))
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry:
            candidates.extend(Path(entry).expanduser().glob("libnvinfer.so*"))
    for candidate in candidates:
        if candidate.is_file():
            try:
                ctypes.CDLL(os.fspath(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                return
            except OSError:
                continue


def diagnose_dlopen(plugin_so: Path) -> str:
    lines: list[str] = []
    try:
        ctypes.CDLL(os.fspath(plugin_so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        lines.append("ctypes.CDLL(plugin, RTLD_GLOBAL) succeeded")
    except OSError as exc:
        lines.append(f"ctypes.CDLL(plugin, RTLD_GLOBAL) failed: {exc}")
    if sys.platform.startswith("linux") and shutil.which("ldd"):
        proc = subprocess.run(["ldd", os.fspath(plugin_so)], capture_output=True, text=True, check=False)
        lines.append(proc.stdout.rstrip())
        if proc.stderr:
            lines.append(proc.stderr.rstrip())
    return "\n".join(line for line in lines if line)


def load_plugin(trt: Any, logger: Any, plugin_so: Path) -> None:
    preload_libnvinfer()
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    loaded = bool(registry.load_library(os.fspath(plugin_so)))
    if not loaded:
        try:
            ctypes.CDLL(os.fspath(plugin_so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError as exc:
            raise RuntimeError(f"failed to load {plugin_so}: {exc}\n{diagnose_dlopen(plugin_so)}") from exc
    if get_creator(trt) is None:
        raise RuntimeError(f"Nvfp4MoePlugin creator not registered after loading {plugin_so}")


def get_creator(trt: Any) -> Any | None:
    registry = trt.get_plugin_registry()
    for getter_name in ("get_creator", "get_plugin_creator", "getPluginCreator"):
        getter = getattr(registry, getter_name, None)
        if not callable(getter):
            continue
        for namespace in ("", "trt"):
            try:
                creator = getter("Nvfp4MoePlugin", "1", namespace)
            except TypeError:
                creator = getter("Nvfp4MoePlugin", "1")
            if creator is not None:
                return creator
    return None


def plugin_fields(trt: Any, case: Sm110Case) -> Any:
    cfg = case.config
    fields = {
        "num_experts": np.asarray([SM110_NUM_EXPERTS], dtype=np.int32),
        "top_k": np.asarray([SM110_TOP_K], dtype=np.int32),
        "hidden_size": np.asarray([cfg.hidden_size], dtype=np.int32),
        "moe_inter_size": np.asarray([cfg.intermediate_size], dtype=np.int32),
        "activation_type": np.asarray([cfg.activation_type], dtype=np.int32),
        "backend": np.asarray([0], dtype=np.int32),
        "max_routed_rows": np.asarray([max(1, cfg.num_tokens) * SM110_TOP_K], dtype=np.int32),
        "io_dtype": np.asarray([1], dtype=np.int32),
    }
    case._plugin_field_backing = fields  # type: ignore[attr-defined]
    return trt.PluginFieldCollection([
        trt.PluginField(name, value, trt.PluginFieldType.INT32) for name, value in fields.items()
    ])


def build_engine(trt: Any, logger: Any, case: Sm110Case) -> bytes:
    cfg = case.config
    creator = get_creator(trt)
    if creator is None:
        raise RuntimeError("Nvfp4MoePlugin creator not found")
    pfc = plugin_fields(trt, case)
    try:
        plugin = creator.create_plugin("sm110_nvfp4_moe", pfc, trt.TensorRTPhase.BUILD)
    except TypeError:
        plugin = creator.create_plugin("sm110_nvfp4_moe", pfc)

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    n1 = fc1_input_n(cfg.intermediate_size, cfg.activation_type)
    inputs = [
        network.add_input("router_logits", trt.float32, tuple(case.router_logits.shape)),
        network.add_input("hidden_states", trt.float16, tuple(case.hidden_states.shape)),
        network.add_input("fc1_qweights", trt.int8, (SM110_NUM_EXPERTS, n1, cfg.hidden_size // 2)),
        network.add_input("fc1_blocks_scale", trt.int8, scale_shape(n1, cfg.hidden_size)),
        network.add_input("fc1_alpha", trt.float32, (SM110_NUM_EXPERTS,)),
        network.add_input("fc2_qweights", trt.int8, (SM110_NUM_EXPERTS, cfg.hidden_size, cfg.intermediate_size // 2)),
        network.add_input("fc2_blocks_scale", trt.int8, scale_shape(cfg.hidden_size, cfg.intermediate_size)),
        network.add_input("fc2_alpha", trt.float32, (SM110_NUM_EXPERTS,)),
        network.add_input("input_global_scale", trt.float32, (SM110_NUM_EXPERTS,)),
        network.add_input("down_input_scale", trt.float32, (SM110_NUM_EXPERTS,)),
        network.add_input("e_score_correction_bias", trt.float32, (SM110_NUM_EXPERTS,)),
    ]
    layer = network.add_plugin_v3(inputs, [], plugin)
    out = layer.get_output(0)
    out.name = "output"
    network.mark_output(out)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    return bytes(serialized)


def trt_to_torch_dtype(trt: Any, dtype: Any) -> Any:
    import torch

    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.int32:
        return torch.int32
    raise ValueError(f"unsupported TensorRT dtype: {dtype}")


def execute_engine(torch: Any, trt: Any, serialized: bytes, inputs: dict[str, np.ndarray]) -> np.ndarray:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None")
    for name, arr in inputs.items():
        setter = getattr(context, "set_input_shape", None)
        if callable(setter):
            ok = setter(name, tuple(arr.shape))
            if ok is False:
                raise RuntimeError(f"set_input_shape({name}, {arr.shape}) failed")
    bindings: dict[str, Any] = {}
    outputs: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        dtype = trt_to_torch_dtype(trt, engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        bindings[name] = torch.empty(shape, dtype=dtype, device=device)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    if not outputs:
        raise RuntimeError("engine has no output")
    with torch.cuda.stream(stream):
        for name, arr in inputs.items():
            host = torch.from_numpy(np.ascontiguousarray(arr))
            bindings[name].copy_(host.to(device=device, dtype=bindings[name].dtype), non_blocking=False)
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            ok = context.set_tensor_address(name, bindings[name].data_ptr())
            if ok is False:
                raise RuntimeError(f"set_tensor_address({name}) failed")
        ok = context.execute_async_v3(stream.cuda_stream)
        if ok is False:
            raise RuntimeError("execute_async_v3 failed")
    stream.synchronize()
    output_name = "output" if "output" in outputs else outputs[0]
    return np.array(bindings[output_name].detach().cpu().numpy(), copy=True)


def summarize_output(case: Sm110Case, trt_output: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    got = np.asarray(trt_output, dtype=np.float32).reshape(reference.shape)
    ref = np.asarray(reference, dtype=np.float32)
    assert np.all(np.isfinite(got)), f"{case.config.name}: TRT output contains non-finite values"
    assert np.all(np.isfinite(ref)), f"{case.config.name}: reference contains non-finite values"
    got_flat = got.reshape(got.shape[0] * got.shape[1], -1).astype(np.float64)
    ref_flat = ref.reshape(ref.shape[0] * ref.shape[1], -1).astype(np.float64)
    denom = np.linalg.norm(got_flat, axis=1) * np.linalg.norm(ref_flat, axis=1)
    cosine = np.where(denom > 0.0, np.sum(got_flat * ref_flat, axis=1) / np.maximum(denom, 1e-30), 1.0)
    median_cosine = float(np.median(cosine[np.isfinite(cosine)]))
    mag_ratio = float(np.linalg.norm(got_flat) / max(np.linalg.norm(ref_flat), 1e-30))
    diff = np.abs(got - ref)
    return {
        "name": case.config.name,
        "scale_mode": case.config.scale_mode,
        "median_cosine": median_cosine,
        "mag_ratio": mag_ratio,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
    }


def assert_output_close(case: Sm110Case, trt_output: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    summary = summarize_output(case, trt_output, reference)
    print(
        f"[{case.config.name}] median_cos={summary['median_cosine']:.6f} "
        f"mag_ratio={summary['mag_ratio']:.6f} max_abs={summary['max_abs']:.6g} "
        f"mean_abs={summary['mean_abs']:.6g} scale_mode={case.config.scale_mode}"
    )
    assert summary["median_cosine"] >= case.config.min_cosine, (
        f"{case.config.name}: median cosine {summary['median_cosine']:.6f} < {case.config.min_cosine:.6f}"
    )
    assert case.config.min_mag_ratio <= summary["mag_ratio"] <= case.config.max_mag_ratio, (
        f"{case.config.name}: magnitude ratio {summary['mag_ratio']:.6f} outside "
        f"[{case.config.min_mag_ratio:.6f}, {case.config.max_mag_ratio:.6f}]"
    )
    return summary


def run_case(config: Sm110CaseConfig) -> dict[str, float | str]:
    torch, trt = check_requirements()
    logger = trt.Logger(trt.Logger.WARNING)
    plugin_so = resolve_plugin_so()
    load_plugin(trt, logger, plugin_so)
    case = make_case(config)
    t0 = time.perf_counter()
    reference = compute_reference(case)
    print(f"[{config.name}] reference ready in {time.perf_counter() - t0:.2f}s")
    serialized = build_engine(trt, logger, case)
    print(f"[{config.name}] engine bytes={len(serialized)}")
    trt_output = execute_engine(
        torch,
        trt,
        serialized,
        {
            "router_logits": case.router_logits,
            "hidden_states": case.hidden_states,
            "fc1_qweights": case.fc1_qweights,
            "fc1_blocks_scale": case.fc1_blocks_scale,
            "fc1_alpha": case.fc1_alpha,
            "fc2_qweights": case.fc2_qweights,
            "fc2_blocks_scale": case.fc2_blocks_scale,
            "fc2_alpha": case.fc2_alpha,
            "input_global_scale": case.input_global_scale,
            "down_input_scale": case.down_input_scale,
            "e_score_correction_bias": case.e_score_correction_bias,
        },
    )
    return assert_output_close(case, trt_output, reference)


# --- Per-scope scale-mode accuracy tests (h=256, i=128, e=128, top_k=8).
# Each test isolates one of the four scaling tensors in the Nvfp4MoePlugin
# (fc1_alpha/fc2_alpha, input_global_scale, down_input_scale) so a regression
# fails the specific scope test that points at it. Tolerance bands are
# uniform for now (matching the pre-existing full_non_uniform test). Tighten
# the tighter-scope ones (all_ones, *_only) once real Thor numbers are in
# hand -- "all_ones" in particular should empirically be near cosine=1.0.


def test_sm110_plugin_scale_all_ones_accuracy() -> None:
    # All four scaling tensors are 1.0 -- isolates FP4 quantization rounding
    # only. Catches plugin paths that mis-apply scales when they should be no-ops.
    run_case(
        Sm110CaseConfig(
            name="sm110_scale_all_ones_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11021,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
            scale_mode="all_ones",
        )
    )


def test_sm110_plugin_scale_input_only_accuracy() -> None:
    # Only FC1 input_global_scale is non-uniform per expert; everything else
    # is 1.0. Isolates the FC1 activation global-scale path.
    run_case(
        Sm110CaseConfig(
            name="sm110_scale_input_only_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11022,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
            scale_mode="input_scale_only",
        )
    )


def test_sm110_plugin_scale_down_only_accuracy() -> None:
    # Only FC2 down_input_scale is non-uniform per expert; everything else
    # is 1.0. Isolates the FC2 activation global-scale path.
    run_case(
        Sm110CaseConfig(
            name="sm110_scale_down_only_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11023,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
            scale_mode="down_scale_only",
        )
    )


def test_sm110_plugin_scale_alpha_only_accuracy() -> None:
    # Only per-expert fc1_alpha/fc2_alpha (weight scales) are non-uniform;
    # activation scales are 1.0. Isolates the per-expert weight-scale path.
    run_case(
        Sm110CaseConfig(
            name="sm110_scale_alpha_only_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11024,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
            scale_mode="alpha_only",
        )
    )


def test_sm110_plugin_scale_full_non_uniform_accuracy() -> None:
    # All four scaling tensors non-uniform per expert. Exercises the same
    # path the deployed plugin uses end-to-end on a small shape.
    run_case(
        Sm110CaseConfig(
            name="sm110_scale_full_non_uniform_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11001,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
            scale_mode="full_non_uniform",
        )
    )


# Prefill token count for the per-model prefill tests. Kept modest because the
# numpy reference does (num_tokens * top_k) dequant + matmul iterations on CPU,
# so larger values mainly slow the host reference, not the SM110 kernel.
PREFILL_NUM_TOKENS = 8


# --- nvidia/Qwen3-30B-A3B-NVFP4 routed MoE (H=2048, I=768, E=128, top_k=8) ---
# Decode hits the fused fast-decode setup (useFastDecodeSetup requires H==2048).


def test_sm110_plugin_qwen3_decode_accuracy() -> None:
    run_case(
        Sm110CaseConfig(
            name="sm110_qwen3_decode_h2048_i768_e128_topk8",
            hidden_size=2048,
            intermediate_size=768,
            num_tokens=1,
            seed=11003,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )


def test_sm110_plugin_qwen3_prefill_accuracy() -> None:
    run_case(
        Sm110CaseConfig(
            name=f"sm110_qwen3_prefill_h2048_i768_e128_topk8_nt{PREFILL_NUM_TOKENS}",
            hidden_size=2048,
            intermediate_size=768,
            num_tokens=PREFILL_NUM_TOKENS,
            seed=11004,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )


# --- nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 routed MoE
#     (H=2688, I=1856, E=128, top_k=8). Both decode and prefill go through the
#     general gather + grouped-GEMM path because H != 2048.


def test_sm110_plugin_nemotron_decode_accuracy() -> None:
    run_case(
        Sm110CaseConfig(
            name="sm110_nemotron_decode_h2688_i1856_e128_topk8",
            hidden_size=2688,
            intermediate_size=1856,
            num_tokens=1,
            seed=11002,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )


def test_sm110_plugin_nemotron_prefill_accuracy() -> None:
    run_case(
        Sm110CaseConfig(
            name=f"sm110_nemotron_prefill_h2688_i1856_e128_topk8_nt{PREFILL_NUM_TOKENS}",
            hidden_size=2688,
            intermediate_size=1856,
            num_tokens=PREFILL_NUM_TOKENS,
            seed=11005,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )
