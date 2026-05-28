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

The tests are opt-in because they require a Thor SM110 system, TensorRT Python,
and a plugin built with ``-DENABLE_CUTE_DSL=nvfp4_moe``.
Set ``EDGELLM_RUN_SM110_PLUGIN_ACCURACY=1`` and
``EDGELLM_NVFP4_MOE_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so`` to run.
"""

from __future__ import annotations

import ctypes
import json
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_ENV = "EDGELLM_RUN_SM110_PLUGIN_ACCURACY"
_PLUGIN_ENV = "EDGELLM_NVFP4_MOE_PLUGIN_SO"
_SCALE_SWEEP_JSON_ENV = "EDGELLM_SM110_SCALE_SWEEP_JSON"
_SF_VEC_SIZE = 16
_ROW_TILE = 128
_FP4_LEVELS = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
_SM110_NUM_EXPERTS = 128
_SM110_TOP_K = 8

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_ENV, "0") != "1",
    reason=f"set {_RUN_ENV}=1 on Thor to run SM110 plugin accuracy",
)


@dataclass(frozen=True)
class _Sm110CaseConfig:
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


@dataclass
class _Sm110Case:
    config: _Sm110CaseConfig
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


def _resolve_plugin_so() -> Path:
    env = os.environ.get(_PLUGIN_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        _REPO_ROOT / "build_thor_sm110" / "libNvInfer_edgellm_plugin.so",
        _REPO_ROOT / "build" / "cpp" / "libNvInfer_edgellm_plugin.so",
        _REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _check_requirements() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch", reason="torch required for CUDA execution and FP8 reference")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip(f"Thor SM110 required, got compute capability {torch.cuda.get_device_capability()}")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn required for FP8 scale reference")
    trt = pytest.importorskip("tensorrt", reason="TensorRT Python required")
    plugin_so = _resolve_plugin_so()
    if not plugin_so.is_file():
        pytest.skip(f"missing plugin library: {plugin_so}")
    return torch, trt


def _scale_shape(rows: int, cols: int, experts: int = _SM110_NUM_EXPERTS) -> tuple[int, int, int, int, int, int]:
    sf_cols = cols // _SF_VEC_SIZE
    return (experts, math.ceil(rows / _ROW_TILE), math.ceil(sf_cols / 4), 32, 4, 4)


def _atom_offsets(rows: int, sf_cols: int) -> np.ndarray:
    m_idx = np.arange(rows, dtype=np.int64).reshape(rows, 1)
    k_idx = np.arange(sf_cols, dtype=np.int64).reshape(1, sf_cols)
    inner_k = k_idx % 4
    inner_m = (m_idx % _ROW_TILE) // 32
    outer_m = m_idx % 32
    k_tile = k_idx // 4
    num_k_tiles = (sf_cols + 3) // 4
    m_tile = m_idx // _ROW_TILE
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


def _fp8_bytes_to_float(raw: np.ndarray) -> np.ndarray:
    import torch

    raw_u8 = np.ascontiguousarray(raw, dtype=np.uint8)
    return torch.from_numpy(raw_u8).view(torch.float8_e4m3fn).float().numpy()


def _float_to_fp8_bytes(values: np.ndarray | list[float]) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)
    return tensor.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()


def _make_scale_tensor(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    *,
    non_uniform: bool,
) -> np.ndarray:
    shape = _scale_shape(rows, cols)
    scale_values = np.asarray([0.00390625, 0.005859375, 0.0078125, 0.01171875], dtype=np.float32)
    scale_bytes = _float_to_fp8_bytes(scale_values)
    if not non_uniform:
        out = np.empty(shape, dtype=np.uint8)
        out.fill(int(scale_bytes[2]))
        return out.view(np.int8)
    indices = rng.integers(0, len(scale_bytes), size=shape, dtype=np.uint8)
    return scale_bytes[indices].astype(np.uint8, copy=False).view(np.int8)


def _make_qweights(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Random bytes intentionally exercise all FP4 nibbles while keeping memory
    # bounded. The reference dequantizes the exact selected bytes that TRT sees.
    return rng.integers(0, 256, size=shape, dtype=np.uint8).view(np.int8)


def _topk_logits_from_scores(expert_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    logits = np.full((expert_ids.shape[0], _SM110_NUM_EXPERTS), -120.0, dtype=np.float32)
    for token in range(expert_ids.shape[0]):
        for slot in range(_SM110_TOP_K):
            score = max(float(scores[token, slot]), 1e-12)
            logits[token, int(expert_ids[token, slot])] = np.float32(math.log(score) + 25.0)
    return logits


def _topk_softmax(router_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(router_logits, dtype=np.float32)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True), dtype=np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True)
    top_ids = np.zeros((logits.shape[0], _SM110_TOP_K), dtype=np.int32)
    top_weights = np.zeros((logits.shape[0], _SM110_TOP_K), dtype=np.float32)
    for token in range(logits.shape[0]):
        used = np.zeros((_SM110_NUM_EXPERTS,), dtype=bool)
        for slot in range(_SM110_TOP_K):
            best_expert = -1
            best_prob = -1.0
            for expert in range(_SM110_NUM_EXPERTS):
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


def _round_fp4(values: np.ndarray) -> np.ndarray:
    values_f32 = np.asarray(values, dtype=np.float32)
    distances = np.abs(values_f32[..., None] - _FP4_LEVELS.reshape((1,) * values_f32.ndim + (16,)))
    return np.argmin(distances, axis=-1).astype(np.uint8)


def _fp4_roundtrip_linear_sf(values: np.ndarray, global_scale: float) -> np.ndarray:
    return (_fp4_roundtrip_linear_sf_raw(values, global_scale) * np.float32(global_scale)).astype(np.float32)


def _fp4_roundtrip_linear_sf_raw(values: np.ndarray, global_scale: float) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    if arr.shape[-1] % _SF_VEC_SIZE != 0:
        raise ValueError(f"last dimension must be divisible by {_SF_VEC_SIZE}, got {arr.shape[-1]}")
    out = np.zeros_like(arr, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    scale = max(float(global_scale), 1e-12)
    for row in range(flat.shape[0]):
        for begin in range(0, flat.shape[1], _SF_VEC_SIZE):
            end = begin + _SF_VEC_SIZE
            block = flat[row, begin:end]
            vec_max = float(np.max(np.abs(block)))
            if vec_max == 0.0:
                continue
            sf_value = (vec_max / 6.0) / scale
            sf_back = float(_fp8_bytes_to_float(_float_to_fp8_bytes([sf_value]))[0])
            effective_scale = sf_back * scale
            if effective_scale <= 0.0 or not np.isfinite(effective_scale):
                continue
            codes = _round_fp4(block / effective_scale)
            out.reshape(-1, arr.shape[-1])[row, begin:end] = _FP4_LEVELS[codes] * sf_back
    return out


def _dequant_weight(
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
    values = _FP4_LEVELS[nibbles].astype(np.float32, copy=False)

    sf_cols = cols // _SF_VEC_SIZE
    offsets = _atom_offsets(rows, sf_cols)
    sf_bytes = np.ascontiguousarray(scale6d[expert], dtype=np.int8).view(np.uint8).reshape(-1)[offsets]
    scales = np.repeat(_fp8_bytes_to_float(sf_bytes), _SF_VEC_SIZE, axis=1)[:, :cols]
    return (values * scales * np.float32(alpha)).astype(np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values, dtype=np.float32))


def _apply_swiglu_interleaved(projection: np.ndarray) -> np.ndarray:
    if projection.shape[0] % 128 != 0:
        raise ValueError(f"SwiGLU projection must be 128-interleaved, got {projection.shape[0]}")
    chunks = projection.reshape(-1, 128)
    up = chunks[:, :64]
    gate = chunks[:, 64:]
    return (up * _silu(gate)).reshape(-1).astype(np.float32)


def _fc1_input_n(intermediate_size: int, activation_type: int) -> int:
    return 2 * intermediate_size if int(activation_type) == 2 else intermediate_size


def _apply_fc1_activation(projection: np.ndarray, activation_type: int) -> np.ndarray:
    if int(activation_type) == 2:
        return _apply_swiglu_interleaved(projection)
    if int(activation_type) == 4:
        return np.square(np.maximum(projection, 0.0)).astype(np.float32)
    raise ValueError(f"unsupported SM110 activation_type: {activation_type}")


def _compute_reference(case: _Sm110Case) -> np.ndarray:
    cfg = case.config
    hidden = np.asarray(case.hidden_states, dtype=np.float16).reshape(-1, cfg.hidden_size).astype(np.float32)
    top_ids, top_weights = _topk_softmax(case.router_logits)

    fc1_cache: dict[int, np.ndarray] = {}
    fc2_cache: dict[int, np.ndarray] = {}
    active_slots: list[tuple[int, int, float, np.ndarray]] = []
    n1 = _fc1_input_n(cfg.intermediate_size, cfg.activation_type)

    for token in range(hidden.shape[0]):
        for slot in range(_SM110_TOP_K):
            expert = int(top_ids[token, slot])
            if expert not in fc1_cache:
                fc1_cache[expert] = _dequant_weight(
                    case.fc1_qweights,
                    case.fc1_blocks_scale,
                    expert,
                    rows=n1,
                    cols=cfg.hidden_size,
                    alpha=float(case.fc1_alpha[expert]),
                )
            hidden_raw = _fp4_roundtrip_linear_sf_raw(
                hidden[token:token + 1], float(case.input_global_scale[expert])
            )[0]
            projection = fc1_cache[expert] @ hidden_raw
            projection *= np.float32(case.input_global_scale[expert])
            activated = _apply_fc1_activation(projection, cfg.activation_type)
            active_slots.append((token, expert, float(top_weights[token, slot]), activated))

    output = np.zeros((hidden.shape[0], cfg.hidden_size), dtype=np.float32)
    for token, expert, router_weight, activated in active_slots:
        activated_raw = _fp4_roundtrip_linear_sf_raw(activated[None, :], float(case.down_input_scale[expert]))[0]
        if expert not in fc2_cache:
            fc2_cache[expert] = _dequant_weight(
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


def _make_case(config: _Sm110CaseConfig) -> _Sm110Case:
    rng = np.random.default_rng(config.seed)
    selected = np.asarray([[0, 7, 19, 31, 47, 64, 96, 127]], dtype=np.int32)
    scores = np.asarray([[0.28, 0.21, 0.16, 0.12, 0.09, 0.06, 0.045, 0.035]], dtype=np.float32)
    router_logits = _topk_logits_from_scores(selected, scores)
    hidden = rng.normal(0.0, 0.05, size=(1, 1, config.hidden_size)).astype(np.float16)
    n1 = _fc1_input_n(config.intermediate_size, config.activation_type)
    fc1_q = _make_qweights(rng, (_SM110_NUM_EXPERTS, n1, config.hidden_size // 2))
    fc2_q = _make_qweights(rng, (_SM110_NUM_EXPERTS, config.hidden_size, config.intermediate_size // 2))
    fc1_scale = _make_scale_tensor(rng, n1, config.hidden_size, non_uniform=config.non_uniform_scales)
    fc2_scale = _make_scale_tensor(
        rng, config.hidden_size, config.intermediate_size, non_uniform=config.non_uniform_scales)
    if config.non_uniform_scales:
        fc1_alpha = np.linspace(0.55, 1.35, _SM110_NUM_EXPERTS, dtype=np.float32)
        fc2_alpha = np.linspace(1.25, 0.65, _SM110_NUM_EXPERTS, dtype=np.float32)
        input_scale = np.linspace(1.0e-4, 8.0e-4, _SM110_NUM_EXPERTS, dtype=np.float32)
        down_scale = np.linspace(8.0e-4, 1.0e-4, _SM110_NUM_EXPERTS, dtype=np.float32)
    else:
        fc1_alpha = np.full((_SM110_NUM_EXPERTS,), 0.85, dtype=np.float32)
        fc2_alpha = np.full((_SM110_NUM_EXPERTS,), 0.75, dtype=np.float32)
        input_scale = np.full((_SM110_NUM_EXPERTS,), 1.0e-4, dtype=np.float32)
        down_scale = np.full((_SM110_NUM_EXPERTS,), 1.0e-4, dtype=np.float32)
    if config.scale_mode == "all_ones":
        fc1_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        input_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "input_scale_only":
        fc1_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "down_scale_only":
        fc1_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        fc2_alpha = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        input_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode == "alpha_only":
        input_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
        down_scale = np.ones((_SM110_NUM_EXPERTS,), dtype=np.float32)
    elif config.scale_mode not in {"config", "full_non_uniform"}:
        raise ValueError(f"unknown SM110 scale_mode: {config.scale_mode}")

    if config.scale_mode in {"config", "full_non_uniform"} and not config.non_uniform_scales:
        input_floor = max(float(np.max(np.abs(hidden.astype(np.float32)))) / (448.0 * 6.0), 1e-12)
        input_scale.fill(np.float32(input_floor))
    return _Sm110Case(
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
    )


def _preload_libnvinfer() -> None:
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


def _diagnose_dlopen(plugin_so: Path) -> str:
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


def _load_plugin(trt: Any, logger: Any, plugin_so: Path) -> None:
    _preload_libnvinfer()
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    loaded = bool(registry.load_library(os.fspath(plugin_so)))
    if not loaded:
        try:
            ctypes.CDLL(os.fspath(plugin_so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError as exc:
            raise RuntimeError(f"failed to load {plugin_so}: {exc}\n{_diagnose_dlopen(plugin_so)}") from exc
    if _get_creator(trt) is None:
        raise RuntimeError(f"Nvfp4MoePlugin creator not registered after loading {plugin_so}")


def _get_creator(trt: Any) -> Any | None:
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


def _plugin_fields(trt: Any, case: _Sm110Case) -> Any:
    cfg = case.config
    fields = {
        "num_experts": np.asarray([_SM110_NUM_EXPERTS], dtype=np.int32),
        "top_k": np.asarray([_SM110_TOP_K], dtype=np.int32),
        "hidden_size": np.asarray([cfg.hidden_size], dtype=np.int32),
        "moe_inter_size": np.asarray([cfg.intermediate_size], dtype=np.int32),
        "activation_type": np.asarray([cfg.activation_type], dtype=np.int32),
        "backend": np.asarray([0], dtype=np.int32),
        "max_routed_rows": np.asarray([_SM110_TOP_K], dtype=np.int32),
        "io_dtype": np.asarray([1], dtype=np.int32),
    }
    case._plugin_field_backing = fields  # type: ignore[attr-defined]
    return trt.PluginFieldCollection([
        trt.PluginField(name, value, trt.PluginFieldType.INT32) for name, value in fields.items()
    ])


def _build_engine(trt: Any, logger: Any, case: _Sm110Case) -> bytes:
    cfg = case.config
    creator = _get_creator(trt)
    if creator is None:
        raise RuntimeError("Nvfp4MoePlugin creator not found")
    pfc = _plugin_fields(trt, case)
    try:
        plugin = creator.create_plugin("sm110_nvfp4_moe", pfc, trt.TensorRTPhase.BUILD)
    except TypeError:
        plugin = creator.create_plugin("sm110_nvfp4_moe", pfc)

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    n1 = _fc1_input_n(cfg.intermediate_size, cfg.activation_type)
    inputs = [
        network.add_input("router_logits", trt.float32, tuple(case.router_logits.shape)),
        network.add_input("hidden_states", trt.float16, tuple(case.hidden_states.shape)),
        network.add_input("fc1_qweights", trt.int8, (_SM110_NUM_EXPERTS, n1, cfg.hidden_size // 2)),
        network.add_input("fc1_blocks_scale", trt.int8, _scale_shape(n1, cfg.hidden_size)),
        network.add_input("fc1_alpha", trt.float32, (_SM110_NUM_EXPERTS,)),
        network.add_input("fc2_qweights", trt.int8, (_SM110_NUM_EXPERTS, cfg.hidden_size, cfg.intermediate_size // 2)),
        network.add_input("fc2_blocks_scale", trt.int8, _scale_shape(cfg.hidden_size, cfg.intermediate_size)),
        network.add_input("fc2_alpha", trt.float32, (_SM110_NUM_EXPERTS,)),
        network.add_input("input_global_scale", trt.float32, (_SM110_NUM_EXPERTS,)),
        network.add_input("down_input_scale", trt.float32, (_SM110_NUM_EXPERTS,)),
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


def _trt_to_torch_dtype(trt: Any, dtype: Any) -> Any:
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


def _execute_engine(torch: Any, trt: Any, serialized: bytes, inputs: dict[str, np.ndarray]) -> np.ndarray:
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
        dtype = _trt_to_torch_dtype(trt, engine.get_tensor_dtype(name))
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


def _summarize_output(case: _Sm110Case, trt_output: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
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


def _assert_output_close(case: _Sm110Case, trt_output: np.ndarray, reference: np.ndarray) -> dict[str, float | str]:
    summary = _summarize_output(case, trt_output, reference)
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


def _run_case(config: _Sm110CaseConfig, *, check: bool = True) -> dict[str, float | str]:
    torch, trt = _check_requirements()
    logger = trt.Logger(trt.Logger.WARNING)
    plugin_so = _resolve_plugin_so()
    _load_plugin(trt, logger, plugin_so)
    case = _make_case(config)
    t0 = time.perf_counter()
    reference = _compute_reference(case)
    print(f"[{config.name}] reference ready in {time.perf_counter() - t0:.2f}s")
    serialized = _build_engine(trt, logger, case)
    print(f"[{config.name}] engine bytes={len(serialized)}")
    trt_output = _execute_engine(
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
        },
    )
    if check:
        return _assert_output_close(case, trt_output, reference)
    summary = _summarize_output(case, trt_output, reference)
    print(
        f"[{config.name}] median_cos={summary['median_cosine']:.6f} "
        f"mag_ratio={summary['mag_ratio']:.6f} scale_mode={config.scale_mode}"
    )
    return summary


def test_sm110_plugin_non_uniform_scale_accuracy() -> None:
    _run_case(
        _Sm110CaseConfig(
            name="sm110_non_uniform_scale_h256_i128_e128_topk8",
            hidden_size=256,
            intermediate_size=128,
            seed=11001,
            non_uniform_scales=True,
            min_cosine=0.97,
            min_mag_ratio=0.40,
            max_mag_ratio=2.50,
        )
    )


def test_sm110_plugin_scale_sweep_debug() -> None:
    summary_path = os.environ.get(_SCALE_SWEEP_JSON_ENV, "").strip()
    if not summary_path:
        pytest.skip(f"set {_SCALE_SWEEP_JSON_ENV} to write SM110 scale sweep diagnostics")

    summaries = []
    for mode in ("all_ones", "input_scale_only", "down_scale_only", "alpha_only", "full_non_uniform"):
        summaries.append(
            _run_case(
                _Sm110CaseConfig(
                    name=f"sm110_scale_sweep_{mode}",
                    hidden_size=256,
                    intermediate_size=128,
                    seed=11011,
                    non_uniform_scales=True,
                    min_cosine=0.0,
                    min_mag_ratio=0.0,
                    max_mag_ratio=float("inf"),
                    scale_mode=mode,
                ),
                check=False,
            )
        )
    path = Path(summary_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[scale_sweep] wrote {path}")


def test_sm110_plugin_qwen_style_accuracy() -> None:
    _run_case(
        _Sm110CaseConfig(
            name="sm110_qwen_style_h2688_i1856_e128_topk8",
            hidden_size=2688,
            intermediate_size=1856,
            seed=11002,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )


def test_sm110_plugin_qwen3_split_accuracy() -> None:
    _run_case(
        _Sm110CaseConfig(
            name="sm110_qwen3_split_h2048_i768_e128_topk8",
            hidden_size=2048,
            intermediate_size=768,
            seed=11003,
            non_uniform_scales=False,
            min_cosine=0.94,
            min_mag_ratio=0.25,
            max_mag_ratio=3.00,
        )
    )
