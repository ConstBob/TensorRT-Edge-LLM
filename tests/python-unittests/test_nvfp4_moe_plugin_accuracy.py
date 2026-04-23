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
r"""
Accuracy tests for TensorRT ``Nvfp4MoePlugin`` (:class:`NemotronHMoEW4A4Plugin`).

Compares TRT plugin output against HF ``NemotronHMoE`` routed-expert output on identical inputs and
dequantized NVFP4 weights (see :func:`_replace_hf_expert_weights_with_dequantized_nvfp4`).  This
isolates the comparison to MoE computation logic (routing, expert dispatch, activation, accumulation)
rather than NVFP4 weight quantization error.  Accuracy is measured by cosine similarity and relative
L2 norm (see :meth:`NemotronHMoEReference.assert_trt_vs_hf_cosine_similarity`); detailed per-element
statistics are printed via :meth:`NemotronHMoEReference.print_cross_check_output_distribution`.

Test methodology
----------------

Both paths share a common setup:

1. Create a toy ``NemotronHMoE`` with random weights, then pack to NVFP4 Marlin format.
2. Dequantize the packed NVFP4 weights (block scales x global scales) back to FP32 and
   overwrite HF ``experts.up_proj`` / ``down_proj`` so that HF and TRT use identical weights.
3. Build a TRT engine with gate matmul baked into the graph (router logits computed in-engine).
4. Run both TRT and HF on the same hidden states; compare outputs.

**W4A16 (FP16 hidden states):**

::

    hidden_states (FP16)
         |
         +---> TRT graph: gate FC --> sigmoid group top-k --> Nvfp4MoePlugin (NVFP4 expert weights)
         |         |
         |         +--> trt_out [B,S,H]
         |
         +---> HF NemotronHMoE: gate FC --> route_tokens --> experts (dequantized NVFP4 weights)
                   |
                   +--> hf_ref [B,S,H]

    assert cosine(trt_out, hf_ref) >= 0.9999  and  rel_L2 <= 0.01

**W4A4 (INT8 NVFP4-packed hidden states):**

::

    hidden_states (FP16) --> pack NVFP4 --> dequant FP16 (hidden_ref)
         |                                       |
         |   +---- hidden_states_fp16 -----------+  (gate matmul input)
         |   |                                   |
         |   |                                   +---> HF NemotronHMoE (dequantized NVFP4 weights)
         |   |                                              |
         |   |                                              +--> hf_ref [B,S,H]
         |   |
         +---+--> TRT graph:
                    hidden_states_fp16 --> gate FC --> sigmoid group top-k
                    hidden_states (INT8) + scales --> Nvfp4MoePlugin
                         |
                         +--> trt_out [B,S,H]

    assert cosine(trt_out, hf_ref) >= 0.9999  and  rel_L2 <= 0.01

**Needs:** CUDA, ``tensorrt`` >= 10.15, Nemotron-H in ``transformers``, ``torch.float8_e4m3fn`` or
``float8_e4m3``, built ``libNvInfer_edgellm_plugin.so`` (``EDGELLM_NVFP4_MOE_PLUGIN_SO`` or default
``build/`` paths).

**Run:** ``pytest tests/python-unittests/test_nvfp4_moe_plugin_accuracy.py -v``.
Skips use :func:`check_requirements` (importorskip / pytest.skip, no module ``pytestmark``).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pytest
import test_attention_utils as attn_utils
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_edgellm_nvfp4_moe_plugin_so() -> Path:
    """``EDGELLM_NVFP4_MOE_PLUGIN_SO`` if set, else ``build/cpp/`` or ``build/`` ``libNvInfer_edgellm_plugin.so``."""
    env = os.environ.get("EDGELLM_NVFP4_MOE_PLUGIN_SO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        _REPO_ROOT / "build" / "cpp" / "libNvInfer_edgellm_plugin.so",
        _REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[-1].resolve()


_PLUGIN_SO = resolve_edgellm_nvfp4_moe_plugin_so()

# Populated by :func:`check_requirements` before Nemotron-H / TensorRT test paths run.
NemotronHConfig: type | None = None
NemotronHMoE: type | None = None
trt: object | None = None


def check_requirements(*, _module_import_guard: bool = False) -> None:
    """Skip via ``pytest`` if deps are missing or out of range.

    Verifies ``transformers`` loads using the same imports as
    ``tensorrt_edgellm/quantization/llm_quantization.py`` (entry points that pull hub, tokenizers, and model classes).

    With ``_module_import_guard=True``, only that step runs and ``pytest.skip`` uses ``allow_module_level`` so
    collection skips instead of erroring before ``tensorrt_edgellm`` is imported.

    Otherwise: Nemotron-H, TensorRT 10.15+, plugin ``.so``, CUDA, FP8 dtypes.
    """
    try:
        from transformers import (  # noqa: F401  — mirror llm_quantization (validates HF stack)
            AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)
    except ImportError as exc:
        pytest.skip(
            f"transformers imports failed (match tensorrt_edgellm.quantization.llm_quantization): {exc}",
            allow_module_level=_module_import_guard,
        )
    try:
        from modelopt.onnx.quantization.qdq_utils import \
            fp4qdq_to_2dq  # noqa: F401
    except ImportError as exc:
        pytest.skip(
            f"modelopt does not meet requirement (fp4qdq_to_2dq): {exc}",
            allow_module_level=_module_import_guard,
        )
    if _module_import_guard:
        return

    global NemotronHConfig, NemotronHMoE, trt
    if NemotronHConfig is not None and NemotronHMoE is not None and trt is not None:
        return
    pytest.importorskip(
        "transformers.models.nemotron_h.modeling_nemotron_h",
        reason="transformers Nemotron-H MoE not available",
    )
    from transformers.models.nemotron_h.configuration_nemotron_h import \
        NemotronHConfig as _cfg
    from transformers.models.nemotron_h.modeling_nemotron_h import \
        NemotronHMoE as _moe

    NemotronHConfig = _cfg
    NemotronHMoE = _moe

    if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3")):
        pytest.skip(
            "NVFP4 Marlin scales need torch.float8_e4m3fn or float8_e4m3")

    trt = pytest.importorskip("tensorrt", reason="tensorrt not installed")
    version_ok, version_msg = attn_utils.check_tensorrt_version(trt, 10, 15)
    if not version_ok:
        pytest.skip(version_msg)
    if not _PLUGIN_SO.is_file():
        pytest.skip(f"missing plugin library: {_PLUGIN_SO}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


check_requirements(_module_import_guard=True)

from tensorrt_edgellm.llm_models.layers.nvfp4_moe_plugin import \
    NemotronHMoEW4A4Plugin  # noqa: E402
from tensorrt_edgellm.llm_models.marlin_converter import \
    MarlinConverter  # noqa: E402

# If True, stderr cross-check stats for TRT vs refs (see :meth:`NemotronHMoEReference.print_cross_check_output_distribution`).
PRINT_TRT_VS_TORCH_DIST = False


@dataclass(frozen=True)
class CrossCheckPrintConfig:
    """Gates sections of :meth:`NemotronHMoEReference.print_cross_check_output_distribution`."""

    print_header: bool = True
    print_finite_summary: bool = True
    show_hf_dense_histograms: bool = True
    show_hf_dense_pair_norms: bool = True
    show_trt_numpy_marlin: bool = True
    show_torch_hf_fp16_block: bool = True
    show_marlin_torch_block: bool = True
    show_marlin_pair_norms: bool = True
    show_interpret_footer: bool = True
    log_tag: str = "PRINT_TRT_VS_TORCH_DIST"


# Default for ``print_config=None`` in :meth:`NemotronHMoEReference.print_cross_check_output_distribution`.
PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG: CrossCheckPrintConfig | None = None

# --- Nemotron-H fixtures ---


class WeightsGenerator:
    """Deterministic gate + expert weight init for test MoEs."""

    @staticmethod
    def gate_test_weight_std(fan_in: int) -> float:
        if fan_in <= 0:
            return 0.06
        s = 1.0 / math.sqrt(float(fan_in))
        return float(min(max(s, 0.06), 0.2))

    @staticmethod
    def expert_linear_std(fan_in: int, initializer_range: float) -> float:
        """Std dev for expert Linear: at least ``initializer_range`` and He-style width; floored for small dims."""
        fi = max(int(fan_in), 1)
        he = math.sqrt(2.0 / float(fi))
        base = float(initializer_range)
        floor = 1.0 / float(fi)
        width_min = 0.0
        if fi <= 512:
            width_min = 0.32
        elif fi <= 2048:
            width_min = max(0.18, 3.0 / math.sqrt(float(fi)))
        return float(max(base, he, floor, width_min))

    @staticmethod
    def random_dense_weights_ehi_eih(
        *,
        num_experts: int,
        expert_input_dim: int,
        moe_inter_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
        initializer_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random ``w_up_ehi`` ``[E,H,I]``, ``w_down_eih`` ``[E,I,H]`` (plugin layout)."""
        e, h_in, inter = int(num_experts), int(expert_input_dim), int(
            moe_inter_size)
        std_up = WeightsGenerator.expert_linear_std(h_in, initializer_range)
        std_dn = WeightsGenerator.expert_linear_std(inter, initializer_range)
        w_up_ehi = torch.randn(
            e, h_in, inter, device=device, dtype=dtype,
            generator=generator) * std_up
        w_down_eih = torch.randn(
            e, inter, h_in, device=device, dtype=dtype,
            generator=generator) * std_dn
        return w_up_ehi, w_down_eih

    @staticmethod
    def random_fill_nemotron_h_moe_params(
        moe: "NemotronHMoE",
        *,
        seed: int = 44,
        fill_hf_expert_params: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-init gate + experts; returns dense ``(w_up_ehi, w_down_eih)`` for packing."""
        wdev = moe.gate.weight.device
        gen = torch.Generator(
            device=wdev) if wdev.type == "cuda" else torch.Generator()
        gen.manual_seed(seed)
        cfg = moe.config
        std_gate = WeightsGenerator.gate_test_weight_std(int(cfg.hidden_size))
        std_corr = std_gate * 0.3
        nn.init.normal_(moe.gate.weight, mean=0.0, std=std_gate, generator=gen)
        nn.init.normal_(moe.gate.e_score_correction_bias,
                        mean=0.0,
                        std=std_corr,
                        generator=gen)

        gen_w = torch.Generator(
            device=wdev) if wdev.type == "cuda" else torch.Generator()
        gen_w.manual_seed(seed + 101)
        expert_in = int(
            cfg.moe_latent_size) if cfg.moe_latent_size is not None else int(
                cfg.hidden_size)
        w_up_ehi, w_down_eih = WeightsGenerator.random_dense_weights_ehi_eih(
            num_experts=int(moe.n_routed_experts),
            expert_input_dim=expert_in,
            moe_inter_size=int(cfg.moe_intermediate_size),
            device=wdev,
            dtype=torch.float32,
            generator=gen_w,
            initializer_range=float(cfg.initializer_range),
        )
        up_dt = moe.experts.up_proj.dtype
        dn_dt = moe.experts.down_proj.dtype
        if fill_hf_expert_params:
            i_sz = int(cfg.moe_intermediate_size)
            up = moe.experts.up_proj
            dn = moe.experts.down_proj
            up_hw = (int(up.shape[1]), int(up.shape[2]))
            dn_hw = (int(dn.shape[1]), int(dn.shape[2]))
            exp_tr = {(expert_in, i_sz), (i_sz, expert_in)}
            if up_hw not in exp_tr or dn_hw not in exp_tr:
                raise ValueError(
                    f"experts.up_proj / down_proj trailing shapes {up_hw} / {dn_hw} do not match "
                    f"expert_input_dim={expert_in} (moe_latent_size or hidden_size) and "
                    f"moe_intermediate_size={i_sz}")
            w_u = w_up_ehi.to(up_dt)
            w_d = w_down_eih.to(dn_dt)
            if up_hw == (expert_in, i_sz):
                up.data.copy_(w_u)
            else:
                up.data.copy_(w_u.transpose(1, 2))
            if dn_hw == (i_sz, expert_in):
                dn.data.copy_(w_d)
            else:
                dn.data.copy_(w_d.transpose(1, 2))
        else:
            nn.init.zeros_(moe.experts.up_proj)
            nn.init.zeros_(moe.experts.down_proj)
        return w_up_ehi, w_down_eih


def create_nemotron_h_moe_from_config(
    cfg: "NemotronHConfig",
    *,
    layer_idx: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    eval_mode: bool = True,
    random_fill_weights: bool = False,
    random_fill_seed: int = 44,
    fill_hf_expert_params: bool = True,
) -> "NemotronHMoE":
    """Build ``NemotronHMoE``; optional :meth:`WeightsGenerator.random_fill_nemotron_h_moe_params`."""
    assert NemotronHMoE is not None
    moe = NemotronHMoE(cfg, layer_idx=int(layer_idx))
    if eval_mode:
        moe.eval()
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        moe = moe.to(device=dev, dtype=dtype)
    else:
        moe = moe.to(dtype=dtype)
    if random_fill_weights:
        WeightsGenerator.random_fill_nemotron_h_moe_params(
            moe,
            seed=random_fill_seed,
            fill_hf_expert_params=fill_hf_expert_params,
        )
    return moe


# --- Accuracy scenario dims + activations (see :func:`run_nvfp4_w4a16_moe_plugin_accuracy_case`) ---

_NVFP4_MOE_FAST_BATCH = 2
_NVFP4_MOE_FAST_HIDDEN = 384
_NVFP4_MOE_FAST_INTER = 768
_NVFP4_MOE_FAST_EXPERTS = 8

# W4A4 packed activations + FP32 decode GEMV: keep toy magnitudes moderate so intermediates stay finite (no FP32 inf).
_NVFP4_MOE_W4A4_HIDDEN_AMP = 0.1
_NVFP4_MOE_W4A4_ROUTED_SCALING = 1.0
_NVFP4_MOE_W4A4_EXPERT_WEIGHT_SCALE = 0.1

_TOY_HIDDEN_SIN_AMPLITUDE = 1.0
_TOY_HIDDEN_SIN_FREQ = 0.031


def create_toy_moe(
    device: torch.device,
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
    routed_scaling_factor: float = 2.5,
) -> "NemotronHMoE":
    """Tiny routed Nemotron-H MoE for plugin accuracy."""
    assert NemotronHConfig is not None
    inter = int(moe_inter_size)
    cfg = NemotronHConfig(
        vocab_size=128,
        hidden_size=int(hidden_size),
        layers_block_type=["moe"],
        moe_intermediate_size=inter,
        n_routed_experts=int(num_experts),
        num_experts_per_tok=int(top_k),
        mlp_hidden_act="relu2",
        routed_scaling_factor=float(routed_scaling_factor),
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        moe_latent_size=None,
        initializer_range=0.08,
        moe_shared_expert_intermediate_size=inter,
    )
    torch.manual_seed(seed)
    return create_nemotron_h_moe_from_config(
        cfg,
        layer_idx=0,
        device=device,
        dtype=torch.float16,
        eval_mode=True,
        random_fill_weights=True,
        random_fill_seed=seed + 101,
    )


def structured_noise_hidden_states_bsh(
    batch: int,
    seq: int,
    hidden_size: int,
    *,
    seed: int,
    amp: float = 2.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """FP16 ``[B,S,H]`` structured + noise; optional ``device`` move after CPU RNG."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    bh = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
    sj = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    hh = torch.arange(hidden_size, dtype=torch.float32).view(1, 1, hidden_size)
    structural = float(amp) * torch.sin(_TOY_HIDDEN_SIN_FREQ * hh + 0.07 *
                                        (bh + sj))
    noise = torch.randn(
        batch, seq, hidden_size, generator=g,
        dtype=torch.float32) * (float(amp) * 0.12)
    out = (structural + noise).to(torch.float16)
    if device is not None:
        out = out.to(device)
    return out


# --- Nemotron-H MoE reference (NumPy + torch, TRT checks) ---


class NemotronHMoEReference:
    """NumPy + torch refs for ``Nvfp4MoePlugin``: routing, Marlin unpack, dense MoE, TRT tolerance."""

    @staticmethod
    def print_cross_check_output_distribution(
        trt_out: np.ndarray,
        torch_ref_fp32_bsh: np.ndarray,
        numpy_hf_ref_fp32_bsh: np.ndarray,
        case: str,
        *,
        numpy_marlin_unpack_ref_fp32: np.ndarray | None = None,
        torch_marlin_unpack_ref_fp32: np.ndarray | None = None,
        torch_hf_fp16_ref_fp32_bsh: np.ndarray | None = None,
        print_config: CrossCheckPrintConfig | None = None,
        file: TextIO | None = None,
        w4a4_dequant_hidden: bool = False,
    ) -> None:
        """Debug stats: TRT vs HF reference(s); optional Marlin blocks (see :class:`CrossCheckPrintConfig`).

        When ``torch_ref_fp32_bsh is numpy_hf_ref_fp32_bsh`` (single HF reference, typical for the
        dequantized-NVFP4-weights flow), redundant cross-ref sections are collapsed to a clean
        ``TRT vs HF dequant-NVFP4`` summary.
        """
        cfg = (print_config if print_config is not None else
               (PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG
                or CrossCheckPrintConfig()))
        out = file if file is not None else sys.stderr
        single_hf_ref = torch_ref_fp32_bsh is numpy_hf_ref_fp32_bsh
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(torch_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        n = np.asarray(numpy_hf_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        m_arr = (None if numpy_marlin_unpack_ref_fp32 is None else np.asarray(
            numpy_marlin_unpack_ref_fp32, dtype=np.float32).reshape(-1))
        tm_arr = (None if torch_marlin_unpack_ref_fp32 is None else np.asarray(
            torch_marlin_unpack_ref_fp32, dtype=np.float32).reshape(-1))
        extra = ""
        if m_arr is not None or tm_arr is not None:
            extra = " + Marlin CPU ref"
            if m_arr is not None and tm_arr is not None:
                extra += " (NumPy decode + torch matmul)"
            elif m_arr is not None:
                extra += " (NumPy decode)"
            else:
                extra += " (torch matmul)"
        tag = cfg.log_tag
        if cfg.print_header:
            if single_hf_ref:
                print(
                    f"\n[{tag}] cross-check (TRT vs HF dequant-NVFP4)  case={case!r}\n"
                    f"  shape_trt={tuple(np.asarray(trt_out).shape)} "
                    f"shape_hf_ref={tuple(np.asarray(torch_ref_fp32_bsh).shape)} nelems={t.size}",
                    file=out,
                    flush=True,
                )
            else:
                print(
                    f"\n[{tag}] cross-check (TRT / torch / NumPy HF{extra})  case={case!r}\n"
                    f"  shape_trt={tuple(np.asarray(trt_out).shape)} shape_torch={tuple(np.asarray(torch_ref_fp32_bsh).shape)} "
                    f"shape_numpy_hf={tuple(np.asarray(numpy_hf_ref_fp32_bsh).shape)} nelems={t.size}",
                    file=out,
                    flush=True,
                )
            if w4a4_dequant_hidden:
                print(
                    "  note: HF ref uses CPU-dequantized NVFP4 activations (W4A4 packed path).",
                    file=out,
                    flush=True,
                )
        if not (t.size == r.size == n.size):
            print(
                f"  ERROR: size mismatch trt={t.size} torch={r.size} numpy_hf={n.size}",
                file=out,
                flush=True,
            )
            return
        if m_arr is not None and m_arr.size != t.size:
            print(f"  ERROR: marlin ref size {m_arr.size} != trt {t.size}",
                  file=out,
                  flush=True)
            return
        if tm_arr is not None and tm_arr.size != t.size:
            print(
                f"  ERROR: torch marlin ref size {tm_arr.size} != trt {t.size}",
                file=out,
                flush=True,
            )
            return

        t64 = t.astype(np.float64)
        r64 = r.astype(np.float64)
        n64 = n.astype(np.float64)
        fin = np.isfinite(t64) & np.isfinite(r64) & np.isfinite(n64)
        if m_arr is not None:
            m64 = m_arr.astype(np.float64)
            fin = fin & np.isfinite(m64)
        else:
            m64 = None
        if tm_arr is not None:
            tm64 = tm_arr.astype(np.float64)
            fin = fin & np.isfinite(tm64)
        else:
            tm64 = None

        n_fin = int(np.sum(fin))
        nf_m = int(np.sum(~np.isfinite(m64))) if m64 is not None else 0
        nf_tm = int(np.sum(~np.isfinite(tm64))) if tm64 is not None else 0
        if cfg.print_finite_summary:
            if single_hf_ref:
                print(
                    f"  finite=pairs={n_fin}/{t.size}  nonfinite_trt={int(np.sum(~np.isfinite(t64)))} "
                    f"nonfinite_hf={int(np.sum(~np.isfinite(r64)))}",
                    file=out,
                    flush=True,
                )
            else:
                nway = 3 + (1 if m64 is not None else 0) + (1 if tm64
                                                            is not None else 0)
                way = {
                    3: "triples",
                    4: "quads",
                    5: "quints"
                }.get(nway, f"{nway}-way")
                print(
                    f"  finite={way}={n_fin}/{t.size}  nonfinite_trt={int(np.sum(~np.isfinite(t64)))} "
                    f"nonfinite_torch={int(np.sum(~np.isfinite(r64)))} nonfinite_numpy_hf={int(np.sum(~np.isfinite(n64)))}"
                    + (f" nonfinite_numpy_marlin={nf_m}" if m64 is not None
                       else "") + (f" nonfinite_torch_marlin={nf_tm}"
                                   if tm64 is not None else ""),
                    file=out,
                    flush=True,
                )
        if n_fin == 0:
            return

        tf = t64[fin]
        rf = r64[fin]
        nf = n64[fin]
        mf = m64[fin] if m64 is not None else None
        tmf = tm64[fin] if tm64 is not None else None

        def _ln(label: str, arr: np.ndarray) -> None:
            print(
                f"  {label}: min={float(np.min(arr)):.6g} max={float(np.max(arr)):.6g} "
                f"mean={float(np.mean(arr)):.6g} std={float(np.std(arr)):.6g} "
                f"p50={float(np.percentile(arr, 50)):.6g} p90={float(np.percentile(arr, 90)):.6g} "
                f"p99={float(np.percentile(arr, 99)):.6g}",
                file=out,
                flush=True,
            )

        def _pair_line(name: str, a: np.ndarray, b: np.ndarray) -> None:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            nd = float(np.linalg.norm(a - b))
            cos = float(np.dot(a, b) / max(na * nb, 1e-20))
            print(
                f"  {name}: ||a||_2={na:.6g} ||b||_2={nb:.6g} ||a-b||_2={nd:.6g} "
                f"rel_l2_vs_b={nd / max(nb, 1e-20):.6g}  cosine={cos:.9f}",
                file=out,
                flush=True,
            )

        if cfg.show_hf_dense_histograms:
            if single_hf_ref:
                _ln("trt_out       ", tf)
                _ln("hf_ref        ", rf)
                _ln("|trt-hf_ref|  ", np.abs(tf - rf))
                denom = np.maximum(np.abs(rf), 1e-12)
                _ln("|trt-hf_ref|/|hf_ref|", np.abs(tf - rf) / denom)
            else:
                _ln("trt_out       ", tf)
                _ln("torch_ref     ", rf)
                _ln("numpy_hf_ref  ", nf)
                _ln("|trt-torch|   ", np.abs(tf - rf))
                _ln("|trt-numpy_hf|", np.abs(tf - nf))
                _ln("|numpy_hf-torch|", np.abs(nf - rf))
                denom_t = np.maximum(np.abs(rf), 1e-12)
                _ln("|trt-torch|/|torch|", np.abs(tf - rf) / denom_t)
                denom_n = np.maximum(np.abs(nf), 1e-12)
                _ln("|trt-numpy_hf|/|numpy_hf|", np.abs(tf - nf) / denom_n)
            peak_t = float(np.max(np.abs(tf)))
            peak_r = float(np.max(np.abs(rf)))
            if peak_t < 0.01 and peak_r > 1.0:
                print(
                    "  NOTE: max|TRT| << max|ref| → per-element |trt-ref|/|ref| is ~1 everywhere (relative to ref), "
                    "not a small relative error; use pair norms below.",
                    file=out,
                    flush=True,
                )
        if cfg.show_hf_dense_pair_norms:
            if single_hf_ref:
                _pair_line("pair trt ↔ hf_ref", tf, rf)
            else:
                _pair_line("pair trt ↔ torch", tf, rf)
                _pair_line("pair trt ↔ numpy_hf", tf, nf)
                _pair_line("pair numpy_hf ↔ torch", nf, rf)

        if mf is not None and cfg.show_trt_numpy_marlin:
            print(
                "  --- TRT vs NumPy Marlin unpack (plugin primary baseline) ---",
                file=out,
                flush=True,
            )
            _ln("|trt-numpy_marlin|", np.abs(tf - mf))
            _pair_line("pair trt ↔ numpy_marlin", tf, mf)

        if cfg.show_torch_hf_fp16_block and torch_hf_fp16_ref_fp32_bsh is not None:
            h16 = np.asarray(torch_hf_fp16_ref_fp32_bsh,
                             dtype=np.float32).reshape(-1)
            if h16.size == t.size:
                print(
                    "  --- torch HF FP16 expert matmul (diagnostic only; TRT assert uses Marlin unpack, not this ref) ---",
                    file=out,
                    flush=True,
                )
                h128 = h16.astype(np.float64)
                fin_h = np.isfinite(t64) & np.isfinite(h128)
                if int(np.sum(fin_h)) > 0:
                    th = t64[fin_h]
                    hh = h128[fin_h]
                    _ln("torch_hf_fp16_ref (as FP32 view)", hh)
                    _ln("|trt - torch_hf_fp16|", np.abs(th - hh))
                    _pair_line("pair trt ↔ torch_hf_fp16", th, hh)

        if mf is not None:
            if tmf is not None and cfg.show_marlin_torch_block:
                print(
                    "  --- torch Marlin unpack ref (same unpacked weights as NumPy Marlin; torch matmul on CPU FP32) ---",
                    file=out,
                    flush=True,
                )
                _ln("torch_marlin_ref ", tmf)
                _ln("|torch_marlin-numpy_marlin|", np.abs(tmf - mf))
                _ln("|trt-torch_marlin|", np.abs(tf - tmf))
                if cfg.show_marlin_pair_norms:
                    _pair_line("pair numpy_marlin ↔ torch_marlin", mf, tmf)
                    _pair_line("pair trt ↔ torch_marlin", tf, tmf)
            if cfg.show_interpret_footer:
                print(
                    "  Interpret:  TRT≈numpy_marlin & both≪HF/torch → NVFP4 decode+math agrees with CPU ref; gap vs HF is quantization.\n"
                    "              numpy_marlin≈HF but TRT≪numpy_marlin → TRT/plugin kernel or graph (not host pack/unpack).\n"
                    "              numpy_marlin≪HF with sane HF → suspect Marlin tile/FP8 block-scale decode vs packer (incl. non-OCP e4m3).",
                    file=out,
                    flush=True,
                )
        if single_hf_ref and mf is None and cfg.show_interpret_footer:
            print(
                "  Interpret:  cosine≈1 & small rel_l2 → TRT plugin matches HF routed-expert (dequant-NVFP4 weights).\n"
                "              Large gap → debug plugin kernel, gate matmul, or routing logic.",
                file=out,
                flush=True,
            )

    @staticmethod
    def sigmoid_group_topk_numpy(
        router_logits: np.ndarray,
        top_k: int,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Same as :meth:`NemotronHMoEW4A4Plugin.sigmoid_group_topk_numpy` (CUDA ``moeSigmoidGroupTopk`` parity)."""
        return NemotronHMoEW4A4Plugin.sigmoid_group_topk_numpy(
            router_logits, top_k, n_group, topk_group, norm_topk_prob,
            routed_scaling_factor, correction_bias)

    @staticmethod
    def sort_topk_slots_descending(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        top_k: int,
    ) -> None:
        """In-place bubble sort per token: descending score; ties → lower expert id first (matches CUDA kernel)."""
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        for t in range(num_tokens):
            row_e = e[t]
            row_s = s[t]
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    wi = float(row_s[i])
                    wj = float(row_s[j])
                    ei = int(row_e[i])
                    ej = int(row_e[j])
                    j_better = (wj > wi) or (wj == wi and ej < ei)
                    if j_better:
                        row_s[i], row_s[j] = row_s[j], row_s[i]
                        row_e[i], row_e[j] = row_e[j], row_e[i]

    @staticmethod
    def topk_weights_indices_from_sorted_desired_slots(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Renormalized ``(topw, topi)`` from sorted slots (after :meth:`router_logits_from_desired_topk`)."""
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        denom = np.sum(s, axis=-1, keepdims=True) + np.float32(1e-20)
        topw = (s / denom).astype(np.float32)
        return topw, e

    @staticmethod
    def router_logits_from_desired_topk(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        num_experts: int,
        top_k: int,
    ) -> np.ndarray:
        """FP32 logits that produce desired expert selections under sigmoid group top-k routing.

        For selected experts with desired relative weight ``w`` (in (0,1)), uses the sigmoid inverse
        (logit function) ``log(w / (1-w))``. Unselected experts get large negative logits so that
        ``sigmoid(logit) ≈ 0``. With ``n_group=1, topk_group=1`` (toy config) and ``norm_topk_prob=True``,
        the renormalized weights preserve the relative proportions of the input scores.
        """
        NemotronHMoEReference.sort_topk_slots_descending(expert,
                                                         score,
                                                         num_tokens=num_tokens,
                                                         top_k=top_k)
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        logits = np.full((num_tokens, num_experts), -120.0, dtype=np.float32)
        for t in range(num_tokens):
            for k in range(top_k):
                ex = int(e[t, k])
                w = float(s[t, k])
                w = max(min(w, 1.0 - 1e-7), 1e-7)
                logits[t, ex] = np.float32(np.log(w / (1.0 - w)))
        return logits

    @staticmethod
    def trt_output_elem_acceptable(got: float, ref_fp32: float) -> bool:
        """Match ``trtNvfp4MoeOutputElemAcceptable`` in ``experiment/test_nvfp4_moe_trt_plugin_accuracy.cu``."""
        if not math.isfinite(got) or not math.isfinite(ref_fp32):
            return False

        ref_h = float(np.float16(ref_fp32))
        err = min(abs(got - ref_fp32), abs(got - ref_h))
        scale = max(1.0, abs(ref_fp32))
        tol = max(1.4, 0.25 + 0.6 * scale)
        return err <= tol

    @staticmethod
    def assert_non_degenerate_output_magnitudes(
        trt_out: np.ndarray,
        ref_fp32: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_p90_abs: float = 5e-6,
    ) -> None:
        """Assert TRT and ref outputs have non-trivial peak and p90 |x|."""
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(ref_fp32, dtype=np.float32).reshape(-1)
        assert t.size == r.size, f"{case}: magnitude check shape mismatch"
        assert np.all(
            np.isfinite(t)), f"{case}: TRT output has non-finite values"
        assert np.all(
            np.isfinite(r)), f"{case}: reference has non-finite values"
        at = np.abs(t.astype(np.float64))
        ar = np.abs(r.astype(np.float64))
        peak_t = float(np.max(at))
        peak_r = float(np.max(ar))
        p90_t = float(np.percentile(at, 90))
        p90_r = float(np.percentile(ar, 90))
        assert peak_t > min_peak_abs, (
            f"{case}: TRT max|out|={peak_t:.6g} <= {min_peak_abs} (expected non-trivial peak magnitude)"
        )
        assert peak_r > min_peak_abs, (
            f"{case}: ref max|out|={peak_r:.6g} <= {min_peak_abs} (expected non-trivial peak magnitude)"
        )
        assert p90_t > min_p90_abs, (
            f"{case}: TRT p90|out|={p90_t:.6g} <= {min_p90_abs} (bulk of elements near zero)"
        )
        assert p90_r > min_p90_abs, (
            f"{case}: ref p90|out|={p90_r:.6g} <= {min_p90_abs} (bulk of elements near zero)"
        )

    @staticmethod
    def assert_trt_matches_reference(
        trt_out: np.ndarray,
        ref_fp32: np.ndarray,
        case: str,
        *,
        tol_scale: float = 1.0,
        use_fp16_rounded_baseline: bool = True,
        max_outlier_frac: float = 0.0,
    ) -> None:
        """Per-element tolerance matching C++ ``trtNvfp4MoeOutputElemAcceptable`` (vectorized).

        ``max_outlier_frac`` (default 0) allows up to that fraction of elements to exceed the
        per-element tolerance without failing.  Set to e.g. 0.003 for 0.3% outlier budget.
        """
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(ref_fp32, dtype=np.float32).reshape(-1)
        assert t.size == r.size, f"{case}: shape mismatch"

        # Vectorized equivalent of :meth:`trt_output_elem_acceptable` (for reporting + pass/fail).
        # Use float64 for err/tol so results match the Python scalar reference (float32 intermediates can differ).
        t64 = t.astype(np.float64)
        r64 = r.astype(np.float64)
        if use_fp16_rounded_baseline:
            # Match :meth:`trt_output_elem_acceptable` / scalar path: min(|trt-fp32_ref|, |trt-fp16_roundtrip_ref|).
            ref_h = r.astype(np.float16).astype(np.float32)
            ref_h128 = ref_h.astype(np.float64)
            err_min = np.minimum(np.abs(t64 - r64), np.abs(t64 - ref_h128))
        else:
            ref_h = r
            err_min = np.abs(t64 - r64)
        raw_abs = np.abs(t - r)
        scale = np.maximum(1.0, np.abs(r64))
        ts = float(tol_scale)
        tol = np.maximum(1.4, 0.25 + 0.6 * scale) * ts
        finite = np.isfinite(t64) & np.isfinite(r64)
        ok = finite & (err_min <= tol)
        n_bad = int(np.sum(~ok))

        if n_bad == 0:
            return
        if t.size > 0 and float(max_outlier_frac) > 0.0:
            if n_bad / t.size <= float(max_outlier_frac):
                return

        # Allow a small fraction of outliers (FP4 quantization noise).
        if max_outlier_frac > 0.0 and t.size > 0:
            bad_frac = n_bad / t.size
            if bad_frac <= max_outlier_frac:
                print(
                    f"{case}: {n_bad}/{t.size} elements ({bad_frac*100:.4f}%) exceed tolerance "
                    f"(within allowed {max_outlier_frac*100:.4f}% outlier budget, PASS)."
                )
                return

        max_abs = float(np.max(raw_abs)) if t.size else 0.0
        n_nonfinite = int(np.sum(~finite))
        ids = np.flatnonzero(~ok)
        err_bad = err_min[ids]
        tol_bad = tol[ids]
        raw_bad = raw_abs[ids]
        excess = err_bad - tol_bad
        order = np.argsort(-excess)
        topk = min(16, len(ids))

        lines: list[str] = [
            f"{case}: {n_bad}/{t.size} elements fail TRT vs ref "
            f"(experiment/test_nvfp4_moe_trt_plugin_accuracy.cu / trtNvfp4MoeOutputElemAcceptable, tol_scale={ts}"
            f"{'' if use_fp16_rounded_baseline else ', FP32 ref only (no FP16-rounded baseline)'}).",
            f"  max_raw_abs(trt - ref)={max_abs}",
        ]
        if n_nonfinite:
            lines.append(f"  non-finite (trt or ref): {n_nonfinite}")
        if use_fp16_rounded_baseline:
            closer_fp32 = np.abs(t64[ids] - r64[ids]) <= np.abs(t64[ids] -
                                                                ref_h128[ids])
            n_closer_fp32 = int(np.sum(closer_fp32))
            lines.extend([
                "  failing elements: err_min=min(|trt-fp32_ref|, |trt-fp16rounded_ref|) vs tol=max(1.4, 0.25+0.6*|ref|).",
                f"  err_min percentiles (bad only): p50={float(np.percentile(err_bad, 50)):.6g} "
                f"p90={float(np.percentile(err_bad, 90)):.6g} max={float(np.max(err_bad)):.6g}",
                f"  tol percentiles (bad only): p50={float(np.percentile(tol_bad, 50)):.6g} "
                f"p90={float(np.percentile(tol_bad, 90)):.6g}",
                f"  excess=err_min-tol (bad only): max={float(np.max(excess)):.6g} "
                f"p90={float(np.percentile(excess, 90)):.6g}",
                f"  bad elems closer to fp32 ref than fp16-rounded ref: {n_closer_fp32}/{len(ids)} "
                f"(remainder dominated by |trt-fp16_ref| arm)",
                "  worst indices (idx: trt, fp32_ref, fp16rt_ref, err_min, tol, excess, raw_abs):",
            ])
            for rank in range(topk):
                j = int(order[rank])
                i = int(ids[j])
                lines.append(
                    f"    {i}: trt={float(t[i]):.8g} fp32_ref={float(r[i]):.8g} fp16rt_ref={float(ref_h[i]):.8g} "
                    f"err_min={float(err_bad[j]):.6g} tol={float(tol_bad[j]):.6g} "
                    f"excess={float(excess[j]):.6g} raw_abs={float(raw_bad[j]):.6g}"
                )
        else:
            lines.extend([
                "  failing elements: err=|trt-ref| vs tol=max(1.4, 0.25+0.6*|ref|).",
                f"  err percentiles (bad only): p50={float(np.percentile(err_bad, 50)):.6g} "
                f"p90={float(np.percentile(err_bad, 90)):.6g} max={float(np.max(err_bad)):.6g}",
                f"  tol percentiles (bad only): p50={float(np.percentile(tol_bad, 50)):.6g} "
                f"p90={float(np.percentile(tol_bad, 90)):.6g}",
                f"  excess=err-tol (bad only): max={float(np.max(excess)):.6g} "
                f"p90={float(np.percentile(excess, 90)):.6g}",
                "  worst indices (idx: trt, ref, err, tol, excess, raw_abs):",
            ])
            for rank in range(topk):
                j = int(order[rank])
                i = int(ids[j])
                lines.append(
                    f"    {i}: trt={float(t[i]):.8g} ref={float(r[i]):.8g} "
                    f"err={float(err_bad[j]):.6g} tol={float(tol_bad[j]):.6g} "
                    f"excess={float(excess[j]):.6g} raw_abs={float(raw_bad[j]):.6g}"
                )

        msg = "\n".join(lines)
        print(msg, file=sys.stderr)
        raise AssertionError(msg)

    @staticmethod
    def assert_trt_output_matches_nvfp4_marlin_numpy_reference(
        trt_out: np.ndarray,
        numpy_marlin_dense_ref_fp32: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_p90_abs: float = 5e-6,
        tol_scale: float = 1.0,
        max_outlier_frac: float = 0.0,
    ) -> None:
        """TRT vs NumPy Marlin-unpacked ref (:meth:`reference_from_packed_plugin_state`).

        Plugin tensors are FP16; per-element error uses ``min(|trt-fp32_ref|, |trt-fp16_ref|)``.
        ``tol_scale`` multiplies the C++ ``trtNvfp4MoeOutputElemAcceptable`` tolerance (see
        ``experiment/test_nvfp4_moe_trt_plugin_accuracy.cu``).
        ``max_outlier_frac`` (default 0) allows up to that fraction of elements to exceed the
        per-element tolerance without failing.
        """
        t = np.asarray(trt_out)
        r = np.asarray(numpy_marlin_dense_ref_fp32, dtype=np.float32)
        if r.shape != t.shape:
            if r.size != t.size:
                raise AssertionError(
                    f"{case}: TRT shape {tuple(t.shape)} vs numpy ref shape {tuple(r.shape)} (size mismatch)"
                )
            r = r.reshape(t.shape)
        NemotronHMoEReference.assert_non_degenerate_output_magnitudes(
            trt_out,
            r,
            case,
            min_peak_abs=min_peak_abs,
            min_p90_abs=min_p90_abs)
        NemotronHMoEReference.assert_trt_matches_reference(
            trt_out,
            r,
            case,
            use_fp16_rounded_baseline=True,
            tol_scale=tol_scale,
            max_outlier_frac=max_outlier_frac,
        )

    @staticmethod
    def _fp4_nibble_to_float(nib: int) -> float:
        nib = int(nib) & 0xF
        mag_i = nib & 7
        sign = -1.0 if (nib & 8) else 1.0
        levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        return sign * levels[mag_i]

    @staticmethod
    def _unpack_fp8_e4m3_int32_word_to_float4(word: int) -> np.ndarray:
        """Four FP32 scales from one Marlin block-scale int32 (CUDA/host decode, not torch float8 cast)."""
        q = int(word) & 0xFFFFFFFF
        out1 = (q & 0xFF00FF00) >> 1
        qs = (q << 8) & 0xFFFFFFFF
        out2 = (qs & 0xFF00FF00) >> 1
        u1 = np.uint32(out1)
        u2 = np.uint32(out2)
        f1 = np.frombuffer(u1.tobytes(), dtype=np.float16).astype(np.float32)
        f2 = np.frombuffer(u2.tobytes(), dtype=np.float16).astype(np.float32)
        return np.array([f2[0], f2[1], f1[0], f1[1]], dtype=np.float32)

    @staticmethod
    def unpack_nvfp4_marlin_tile_64(
        payload_i32: np.ndarray,
        block_scale_i32: int,
    ) -> np.ndarray:
        """One 64-lane Marlin tile → ``float32[64]`` (8 int32 payload + block scales)."""
        pl = np.asarray(payload_i32, dtype=np.int32).reshape(8)
        scales4 = NemotronHMoEReference._unpack_fp8_e4m3_int32_word_to_float4(
            int(block_scale_i32))
        nib = np.empty(64, dtype=np.int32)
        for lane in range(8):
            w = int(pl[lane]) & 0xFFFFFFFF
            for i in range(4):
                b = (w >> (8 * i)) & 0xFF
                lo = int(b & 0xF)
                hi = int((b >> 4) & 0xF)
                base = lane * 8 + i * 2
                nib[base] = lo
                nib[base + 1] = hi
        out = np.empty(64, dtype=np.float32)
        for i in range(64):
            g = i // 16
            out[i] = NemotronHMoEReference._fp4_nibble_to_float(int(
                nib[i])) * float(scales4[g])
        return out

    @staticmethod
    def _read_atom_scale_word(buf_u8: np.ndarray, m_idx: int, k_tile: int,
                              num_sf_cols: int) -> int:
        """Read 4 raw FP8 bytes from atom-layout positions and return Marlin-packed int32 scale word."""
        raw = np.zeros(4, dtype=np.uint8)
        for g in range(4):
            off = MarlinConverter.atom_sf_offset(m_idx, k_tile * 4 + g,
                                                 num_sf_cols)
            raw[g] = buf_u8[off]
        # Atom stores {s0,s1,s2,s3}; Marlin expects {s0,s2,s1,s3}
        marlin = np.array([raw[0], raw[2], raw[1], raw[3]], dtype=np.uint8)
        return int(np.frombuffer(marlin.tobytes(), dtype=np.int32)[0])

    @staticmethod
    def dense_weights_from_nvfp4_plugin_buffers(
        fc_up_qweights: np.ndarray,
        fc_up_blocks_scale: np.ndarray,
        fc_down_qweights: np.ndarray,
        fc_down_blocks_scale: np.ndarray,
        *,
        num_experts: int,
        hidden_size: int,
        moe_inter_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """INT8 plugin buffers → dense ``w_up_ehi`` / ``w_down_eih`` (no global scales; kernel applies those)."""
        e = int(num_experts)
        h = int(hidden_size)
        inter = int(moe_inter_size)
        nic = inter // 64
        n_h_chunks = h // 64
        assert nic * 64 == inter
        assert n_h_chunks * 64 == h

        up_q8 = np.ascontiguousarray(fc_up_qweights, dtype=np.int8)
        up_bs8 = np.ascontiguousarray(fc_up_blocks_scale, dtype=np.int8)
        dn_q8 = np.ascontiguousarray(fc_down_qweights, dtype=np.int8)
        dn_bs8 = np.ascontiguousarray(fc_down_blocks_scale, dtype=np.int8)
        assert up_q8.shape == (e, h // 2, inter)
        # SF M-dimension is padded to multiple of 128 for Cutlass Atom layout.
        up_sf_m_padded = ((h + 127) // 128) * 8
        dn_sf_m_padded = ((inter + 127) // 128) * 128
        assert up_bs8.shape == (e, up_sf_m_padded, inter)
        assert dn_q8.shape == (e, inter, h // 2)
        assert dn_bs8.shape == (e, dn_sf_m_padded, h // 16)

        num_sf_cols_up = inter // 16
        num_sf_cols_dn = h // 16

        w_up = np.zeros((e, h, inter), dtype=np.float32)
        w_down = np.zeros((e, inter, h), dtype=np.float32)
        for ex in range(e):
            up_flat = up_q8[ex].reshape(-1)
            up_bs_u8 = up_bs8[ex].view(np.uint8).reshape(-1)
            for jj in range(h):
                for c in range(nic):
                    tile_u = jj * nic + c
                    pl = up_flat[tile_u * 32:(tile_u + 1) * 32].view(
                        np.int32).reshape(8)
                    bs_i32 = NemotronHMoEReference._read_atom_scale_word(
                        up_bs_u8, jj, c, num_sf_cols_up)
                    w_up[ex, jj, c * 64:(c + 1) * 64] = (
                        NemotronHMoEReference.unpack_nvfp4_marlin_tile_64(
                            pl, bs_i32))
            dn_bs_u8 = dn_bs8[ex].view(np.uint8).reshape(-1)
            dn_q_flat = dn_q8[ex].reshape(-1)
            for j in range(inter):
                for c in range(n_h_chunks):
                    tile_d_idx = j * n_h_chunks + c
                    pl_d = dn_q_flat[tile_d_idx * 32:(tile_d_idx + 1) *
                                     32].view(np.int32).reshape(8)
                    bs_d_i32 = NemotronHMoEReference._read_atom_scale_word(
                        dn_bs_u8, j, c, num_sf_cols_dn)
                    w_down[ex, j, c * 64:(c + 1) * 64] = (
                        NemotronHMoEReference.unpack_nvfp4_marlin_tile_64(
                            pl_d, bs_d_i32))
        return w_up, w_down

    @staticmethod
    def dense_forward_from_topk(
        x_bh: np.ndarray,
        topk_weights: np.ndarray,
        topk_indices: np.ndarray,
        w_up_ehi: np.ndarray,
        w_down_eih: np.ndarray,
        *,
        activation_type: int = 0,
    ) -> np.ndarray:
        """Dense MoE: weighted ``down(act(up @ x))`` per slot; ``activation_type`` 0=ReLU², 1=SiLU; out ``[B,1,H]``."""
        x_bh = np.asarray(x_bh, dtype=np.float32)
        topk_weights = np.asarray(topk_weights, dtype=np.float32)
        topk_indices = np.asarray(topk_indices, dtype=np.int32)
        w_up_ehi = np.asarray(w_up_ehi, dtype=np.float32)
        w_down_eih = np.asarray(w_down_eih, dtype=np.float32)

        b, h = x_bh.shape
        e, h2, inter = w_up_ehi.shape
        assert h2 == h
        assert w_down_eih.shape == (e, inter, h)
        bk, k = topk_weights.shape
        assert bk == b
        assert topk_indices.shape == (b, k)

        out = np.zeros((b, h), dtype=np.float32)
        for bb in range(b):
            for slot in range(k):
                ex = int(topk_indices[bb, slot])
                s = float(topk_weights[bb, slot])
                if s == 0.0 or ex < 0 or ex >= e:
                    continue
                z = x_bh[bb] @ w_up_ehi[ex]
                act = NemotronHMoEW4A4Plugin.moe_activation_numpy(
                    z, activation_type)
                t = act * s
                out[bb] += t @ w_down_eih[ex]
        return out.reshape(b, 1, h)

    @staticmethod
    def hf_expert_weights_numpy_ehi_eih(
            moe: "NemotronHMoE") -> tuple[np.ndarray, np.ndarray]:
        """HF expert weights as ``w_up_ehi`` / ``w_down_eih`` (CPU, via FP16 round-trip)."""
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers import failed).")
        up = moe.experts.up_proj.detach().cpu().half().float().numpy()
        dn = moe.experts.down_proj.detach().cpu().half().float().numpy()
        # HF: up (E, inter, hidden_in), down (E, hidden_in, inter)
        w_up_ehi = np.ascontiguousarray(np.transpose(up, (0, 2, 1)))
        w_down_eih = np.ascontiguousarray(np.transpose(dn, (0, 2, 1)))
        return w_up_ehi, w_down_eih

    @staticmethod
    def reference_dense_from_hf_moe(
        moe: "NemotronHMoE",
        hidden_states_fp16_bsh: np.ndarray,
        topk_weights: np.ndarray,
        topk_indices: np.ndarray,
        *,
        hidden_size: int,
        activation_type: int,
    ) -> np.ndarray:
        """NumPy dense MoE with HF weights (FP16 activations); aligns with torch HF ref for same top-k."""
        w_up, w_down = NemotronHMoEReference.hf_expert_weights_numpy_ehi_eih(
            moe)
        x = np.asarray(hidden_states_fp16_bsh, dtype=np.float16).reshape(
            -1, int(hidden_size)).astype(np.float32)
        return NemotronHMoEReference.dense_forward_from_topk(
            x,
            topk_weights,
            topk_indices,
            w_up,
            w_down,
            activation_type=int(activation_type),
        )

    @staticmethod
    def assert_numpy_dense_matches_torch_reference(
        ref_np_bsh: np.ndarray,
        torch_ref_fp32_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 1e-3,
        atol: float = 2.0,
    ) -> None:
        """NumPy HF dense vs torch dense FP32 ref (loose ``allclose``; CPU vs NumPy ordering)."""
        a = np.asarray(ref_np_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(torch_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(
                f"{case}: numpy vs torch ref size mismatch {a.size} vs {b.size}"
            )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(
                f"{case}: non-finite numpy or torch reference")
        ok = np.allclose(a, b, rtol=float(rtol), atol=float(atol))
        if ok:
            return
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"{case}: NumPy HF dense ref diverges from torch FP32 ref (rtol={rtol}, atol={atol}); "
            f"max_abs_diff={float(np.max(diff)):.6g} worst_idx={worst} np={float(a[worst]):.8g} torch={float(b[worst]):.8g}"
        )

    @staticmethod
    def assert_marlin_unpacked_numpy_matches_torch(
        numpy_marlin_bsh: np.ndarray,
        torch_marlin_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 1e-3,
        atol: float = 2e-2,
    ) -> None:
        """Require Marlin-unpacked dense outputs from NumPy ``@`` vs torch CPU ``matmul`` to agree.

        Cross-check uses ``|a-b| <= atol + rtol * max(|a|,|b|)`` (float64 diff). These two CPU refs
        differ only in GEMM ordering / accumulation; a few hundred ULPs at mid-range magnitudes is
        expected, so ``rtol=1e-3`` (0.1%) is used here — **not** the same bar as TRT vs NumPy Marlin.
        """
        a = np.asarray(numpy_marlin_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(torch_marlin_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(
                f"{case}: numpy vs torch marlin ref size mismatch {a.size} vs {b.size}"
            )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(
                f"{case}: non-finite numpy or torch Marlin unpack ref")
        af = a.astype(np.float64)
        bf = b.astype(np.float64)
        diff = np.abs(af - bf)
        scale = np.maximum(np.abs(af), np.abs(bf))
        bound = float(atol) + float(rtol) * scale
        if np.all(diff <= bound):
            return
        excess = diff - bound
        fail_idx = int(np.argmax(excess)) if diff.size else 0
        peak = float(np.max(diff)) if diff.size else 0.0
        raise AssertionError(
            f"{case}: NumPy Marlin unpack ref vs torch Marlin unpack ref diverges "
            f"(symmetric rtol={rtol:g}, atol={atol:g}); max_abs={peak:.6g}; "
            f"largest_violation_idx={fail_idx} diff={float(diff[fail_idx]):.6g} "
            f"bound={float(bound[fail_idx]):.6g} "
            f"np={float(a[fail_idx]):.8g} torch={float(b[fail_idx]):.8g}")

    @staticmethod
    def _scaled_dense_moe_tensors_from_packed_plugin_state(
        hidden_states_fp16_bsh: np.ndarray,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        *,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack NVFP4 buffers, apply per-expert global scales, FP32 activations ``x`` ``[num_tokens, H]``."""
        w_up, w_down = NemotronHMoEReference.dense_weights_from_nvfp4_plugin_buffers(
            fc_up_q,
            fc_up_bs,
            fc_dn_q,
            fc_dn_bs,
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_inter_size=moe_inter_size,
        )
        up_gs = np.asarray(fc_up_gs, dtype=np.float32).reshape(-1)
        dn_gs = np.asarray(fc_dn_gs, dtype=np.float32).reshape(-1)
        for ex in range(int(num_experts)):
            w_up[ex] *= float(up_gs[ex])
            w_down[ex] *= float(dn_gs[ex])
        x = np.asarray(hidden_states_fp16_bsh, dtype=np.float16).reshape(
            -1, hidden_size).astype(np.float32)
        return x, w_up, w_down, up_gs, dn_gs

    @staticmethod
    def reference_from_packed_plugin_state(
        hidden_states_fp16_bsh: np.ndarray,
        topk_weights_be: np.ndarray,
        topk_indices_be: np.ndarray,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        *,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
        activation_type: int = 0,
    ) -> np.ndarray:
        """Packed plugin state → scaled dense MoE output (same top-k as engine)."""
        x, w_up, w_down, up_gs, dn_gs = (
            NemotronHMoEReference.
            _scaled_dense_moe_tensors_from_packed_plugin_state(
                hidden_states_fp16_bsh,
                fc_up_q,
                fc_up_bs,
                fc_dn_q,
                fc_dn_bs,
                fc_up_gs,
                fc_dn_gs,
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
            ))
        out = NemotronHMoEReference.dense_forward_from_topk(
            x,
            topk_weights_be,
            topk_indices_be,
            w_up,
            w_down,
            activation_type=int(activation_type),
        )
        return out

    @staticmethod
    def assert_topk_matches_numpy_reference(
        router_logits: torch.Tensor,
        top_k: int,
        topw_torch: torch.Tensor,
        topi_torch: torch.Tensor,
        case: str,
        *,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: np.ndarray | None = None,
    ) -> None:
        """Require plugin-parity top-k tensors to match :meth:`NemotronHMoEW4A4Plugin.sigmoid_group_topk_numpy`."""
        L = np.ascontiguousarray(router_logits.detach().float().cpu().numpy())
        nw, ni = NemotronHMoEW4A4Plugin.sigmoid_group_topk_numpy(
            L, int(top_k), n_group, topk_group, norm_topk_prob,
            routed_scaling_factor, correction_bias)
        tw = topw_torch.detach().float().cpu().numpy()
        ti = topi_torch.detach().cpu().numpy().astype(np.int32)
        if not np.array_equal(ti, ni):
            raise AssertionError(
                f"{case}: top-k expert indices differ from NumPy sigmoid_group_topk"
            )
        if not np.allclose(tw, nw, rtol=1e-6, atol=1e-7):
            max_d = float(np.max(np.abs(tw - nw))) if tw.size else 0.0
            raise AssertionError(
                f"{case}: top-k weights differ from NumPy sigmoid_group_topk (max_abs_delta={max_d:.6g})"
            )

    @staticmethod
    def assert_moe_output_non_degenerate(
        out_fp32_bsh: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_fraction_abs_gt: float = 0.01,
        abs_gt_eps: float = 1e-6,
    ) -> None:
        """Torch MoE ref: finite, peak magnitude, fraction of elems above ``abs_gt_eps``."""
        a = np.asarray(out_fp32_bsh, dtype=np.float32).reshape(-1)
        if a.size == 0:
            raise AssertionError(f"{case}: empty torch MoE reference output")
        if not np.all(np.isfinite(a)):
            raise AssertionError(
                f"{case}: torch MoE reference has non-finite values")
        am = np.abs(a.astype(np.float64))
        peak = float(np.max(am))
        if peak <= min_peak_abs:
            raise AssertionError(
                f"{case}: torch MoE ref max|out|={peak:.6g} <= {min_peak_abs} (expected non-trivial peak)"
            )
        frac = float(np.mean(am > float(abs_gt_eps)))
        if frac < float(min_fraction_abs_gt):
            raise AssertionError(
                f"{case}: torch MoE ref only {frac * 100:.2f}% of elems have |x| > {abs_gt_eps:g} "
                f"(require >= {min_fraction_abs_gt * 100:.2f}%)")

    @staticmethod
    def assert_fp16_ref_tracks_fp32_ref(
        ref_fp32_bsh: np.ndarray,
        ref_fp16_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 0.08,
        atol: float = 0.35,
    ) -> None:
        """FP16-matmul expert path should stay in the ballpark of the FP32 path (same routing, same HF weights)."""
        a = np.asarray(ref_fp32_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(ref_fp16_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(f"{case}: fp32 vs fp16 ref shape mismatch")
        ok = np.isfinite(a) & np.isfinite(b)
        if not np.all(ok):
            raise AssertionError(
                f"{case}: non-finite values in fp32/fp16 ref pair")
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.maximum(np.abs(a.astype(np.float64)), 1.0)
        bound = float(atol) + float(rtol) * scale
        bad = diff > bound
        n_bad = int(np.sum(bad))
        if n_bad == 0:
            return
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"{case}: fp16 torch ref diverges from fp32 ref on {n_bad}/{a.size} elems "
            f"(rtol={rtol}, atol={atol}); worst idx {worst}: fp32={float(a[worst]):.6g} fp16_as_f32={float(b[worst]):.6g} "
            f"diff={float(diff[worst]):.6g} bound={float(bound[worst]):.6g}")

    @staticmethod
    def sigmoid_group_topk_torch_plugin_parity(
        router_logits: torch.Tensor,
        top_k: int,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plugin-parity sigmoid group top-k on CPU float (tie-break: lower expert id)."""
        return NemotronHMoEW4A4Plugin.sigmoid_group_topk_torch(
            router_logits, top_k, n_group, topk_group, norm_topk_prob,
            routed_scaling_factor, correction_bias)

    @staticmethod
    def routed_experts_from_router_logits_via_hf_experts(
        moe: "NemotronHMoE",
        hidden_bsh: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        *,
        compute_dtype: torch.dtype,
        activation_type: int,
        case: str,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """HF expert weights + plugin-parity top-k; dense FP32 matmul matching :meth:`reference_dense_from_hf_moe`.

        Does not call ``moe.experts`` in FP16: that path can overflow for large activations (e.g. W4A4 dequant
        hiddens) while the NumPy dense reference uses FP32 activations after FP16 storage round-trip.
        ``compute_dtype`` selects FP32 vs FP16 routing masses (same as the old HF forward).
        """
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers import failed).")
        if not isinstance(moe, NemotronHMoE):
            raise TypeError(
                f"moe must be NemotronHMoE, got {type(moe).__name__!r}")

        h_sz = int(moe.config.hidden_size)
        # Match ``NemotronHExperts`` / ``up_proj`` device — ``hidden_bsh`` may be CPU (e.g. host-only fixtures)
        # while the MoE module lives on CUDA.
        exp_dev = moe.experts.up_proj.device
        logits = router_logits.to(device=exp_dev, dtype=torch.float32)
        cb_t = torch.from_numpy(correction_bias).to(
            device=exp_dev,
            dtype=torch.float32) if correction_bias is not None else None
        topw, topi = NemotronHMoEReference.sigmoid_group_topk_torch_plugin_parity(
            logits, int(top_k), n_group, topk_group, norm_topk_prob,
            routed_scaling_factor, cb_t)
        NemotronHMoEReference.assert_topk_matches_numpy_reference(
            logits,
            int(top_k),
            topw,
            topi,
            case,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias)
        # Match :meth:`reference_dense_from_hf_moe`: FP16-stored activations, then FP32 ``@`` with HF weights.
        x_np = np.asarray(
            hidden_bsh.detach().cpu().numpy(),
            dtype=np.float16,
        ).reshape(-1, h_sz).astype(np.float32)
        topw_cpu = topw.float().cpu()
        if compute_dtype == torch.float16:
            topw_cpu = topw_cpu.to(torch.float16).to(torch.float32)
        w_up, w_down = NemotronHMoEReference.hf_expert_weights_numpy_ehi_eih(
            moe)
        out_t = NemotronHMoEReference.dense_forward_from_topk_torch(
            torch.from_numpy(x_np),
            topw_cpu,
            topi.long().cpu(),
            torch.from_numpy(w_up),
            torch.from_numpy(w_down),
            activation_type=int(activation_type),
        )
        out_np = np.asarray(out_t.detach().numpy(),
                            dtype=np.float32).reshape(tuple(hidden_bsh.shape))
        NemotronHMoEReference.assert_moe_output_non_degenerate(out_np, case)
        return out_np

    @staticmethod
    def moe_activation_torch(z: torch.Tensor,
                             activation_type: int) -> torch.Tensor:
        """FP32 expert nonlinearity matching :meth:`NemotronHMoEW4A4Plugin.moe_activation_numpy`."""
        z = z.float()
        if int(activation_type) == 1:
            zc = z.clamp(-50.0, 50.0)
            return zc / (1.0 + torch.exp(-zc))
        t = torch.clamp(z, min=0.0)
        return t * t

    @staticmethod
    def dense_forward_from_topk_torch(
        x_bh: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        w_up_ehi: torch.Tensor,
        w_down_eih: torch.Tensor,
        *,
        activation_type: int = 0,
    ) -> torch.Tensor:
        """CPU torch ``matmul`` version of :meth:`dense_forward_from_topk`."""
        x_bh = x_bh.float().cpu()
        topk_weights = topk_weights.float().cpu()
        topk_indices = topk_indices.long().cpu()
        w_up_ehi = w_up_ehi.float().cpu()
        w_down_eih = w_down_eih.float().cpu()
        b, h = int(x_bh.shape[0]), int(x_bh.shape[1])
        e, h2, inter = int(w_up_ehi.shape[0]), int(w_up_ehi.shape[1]), int(
            w_up_ehi.shape[2])
        assert h2 == h
        assert tuple(w_down_eih.shape) == (e, inter, h)
        bk, k = int(topk_weights.shape[0]), int(topk_weights.shape[1])
        assert bk == b
        out = torch.zeros(b, h, dtype=torch.float32)
        for bb in range(b):
            for slot in range(k):
                ex = int(topk_indices[bb, slot].item())
                s = float(topk_weights[bb, slot].item())
                if s == 0.0 or ex < 0 or ex >= e:
                    continue
                z = x_bh[bb] @ w_up_ehi[ex]
                act = NemotronHMoEReference.moe_activation_torch(
                    z, activation_type)
                t = act * s
                out[bb] = out[bb] + t @ w_down_eih[ex]
        return out.reshape(b, 1, h)

    @staticmethod
    def routed_experts_from_router_logits_via_marlin_unpacked_weights(
        hidden_bsh: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        *,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
        activation_type: int,
        case: str,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Marlin-unpacked weights + CPU torch matmul; same routing checks as HF path."""
        hidden_np = np.ascontiguousarray(hidden_bsh.detach().cpu().numpy())
        x, w_up, w_down, _, _ = NemotronHMoEReference._scaled_dense_moe_tensors_from_packed_plugin_state(
            hidden_np,
            fc_up_q,
            fc_up_bs,
            fc_dn_q,
            fc_dn_bs,
            fc_up_gs,
            fc_dn_gs,
            hidden_size=int(hidden_size),
            moe_inter_size=int(moe_inter_size),
            num_experts=int(num_experts),
        )
        exp_dev = router_logits.device
        logits = router_logits.to(device=exp_dev, dtype=torch.float32)
        cb_t = torch.from_numpy(correction_bias).to(
            device=exp_dev,
            dtype=torch.float32) if correction_bias is not None else None
        topw, topi = NemotronHMoEReference.sigmoid_group_topk_torch_plugin_parity(
            logits, int(top_k), n_group, topk_group, norm_topk_prob,
            routed_scaling_factor, cb_t)
        NemotronHMoEReference.assert_topk_matches_numpy_reference(
            logits,
            int(top_k),
            topw,
            topi,
            case,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias)
        out_t = NemotronHMoEReference.dense_forward_from_topk_torch(
            torch.from_numpy(x),
            topw,
            topi,
            torch.from_numpy(w_up),
            torch.from_numpy(w_down),
            activation_type=int(activation_type),
        )
        return np.asarray(out_t.detach().numpy(),
                          dtype=np.float32).reshape(tuple(hidden_bsh.shape))

    @staticmethod
    def hf_routed_expert_output(
        moe: "NemotronHMoE",
        hidden_bsh: torch.Tensor,
    ) -> np.ndarray:
        """``gate → route → fc1_latent → experts → fc2_latent`` (no shared experts) → FP32 numpy ``[B,S,H]``.

        Runs the full HF ``NemotronHMoE`` routed-expert path (excluding shared experts) so that
        the gate linear computes router logits from ``hidden_bsh``.  Used as end-to-end reference
        for TRT accuracy tests.
        """
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers import failed).")
        with torch.no_grad():
            orig_shape = hidden_bsh.shape
            dev = moe.gate.weight.device
            h = hidden_bsh.to(device=dev)
            router_logits = moe.gate(h)
            topk_indices, topk_weights = moe.route_tokens_to_experts(
                router_logits)
            flat = h.view(-1, h.shape[-1])
            flat = moe.fc1_latent_proj(flat)
            out = moe.experts(flat, topk_indices, topk_weights)
            out = moe.fc2_latent_proj(out)
            out = out.view(*orig_shape)
        return out.float().cpu().numpy().astype(np.float32)

    @staticmethod
    def assert_trt_vs_hf_cosine_similarity(
        trt_out: np.ndarray,
        hf_ref_fp32: np.ndarray,
        case: str,
        *,
        min_cosine: float = 0.9999,
        max_rel_l2: float = 0.01,
        min_peak_abs: float = 1e-4,
    ) -> None:
        """Cosine + relative L2 check: TRT (gate + plugin) vs HF ``NemotronHMoE`` routed expert output.

        Both TRT and HF use dequantized NVFP4 expert weights (see
        :func:`_replace_hf_expert_weights_with_dequantized_nvfp4`), so differences arise only from
        FP16-vs-FP32 accumulation in the gate/routing/expert paths.

        Verifies non-degenerate magnitudes, then requires ``cosine(trt, hf) >= min_cosine`` and
        ``||trt - hf||_2 / ||hf||_2 <= max_rel_l2``.
        """
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(hf_ref_fp32, dtype=np.float32).reshape(-1)
        assert t.size == r.size, (
            f"{case}: shape mismatch trt {t.size} vs hf {r.size}")
        assert np.all(
            np.isfinite(t)), f"{case}: TRT output has non-finite values"
        assert np.all(
            np.isfinite(r)), f"{case}: HF reference has non-finite values"
        peak_t = float(np.max(np.abs(t)))
        peak_r = float(np.max(np.abs(r)))
        assert peak_t > min_peak_abs, (
            f"{case}: TRT max|out|={peak_t:.6g} (degenerate)")
        assert peak_r > min_peak_abs, (
            f"{case}: HF max|out|={peak_r:.6g} (degenerate)")
        t64 = t.astype(np.float64)
        r64 = r.astype(np.float64)
        norm_t = float(np.linalg.norm(t64))
        norm_r = float(np.linalg.norm(r64))
        cos = float(np.dot(t64, r64)) / max(norm_t * norm_r, 1e-20)
        rel_l2 = float(np.linalg.norm(t64 - r64)) / max(norm_r, 1e-20)
        assert cos >= min_cosine, (
            f"{case}: TRT-vs-HF cosine {cos:.6f} < {min_cosine} "
            f"(rel_l2={rel_l2:.6g}, ||trt||={norm_t:.6g}, ||hf||={norm_r:.6g})"
        )
        assert rel_l2 <= max_rel_l2, (
            f"{case}: TRT-vs-HF rel_l2 {rel_l2:.6g} > {max_rel_l2} "
            f"(cosine={cos:.6f}, ||trt||={norm_t:.6g}, ||hf||={norm_r:.6g})")


# --- TensorRT: load plugin + run engine ---


def _require_trt_ok(ok: bool) -> None:
    if not ok:
        raise RuntimeError("TensorRT API call failed")


def _resolve_edgellm_trt_plugin_path(plugin_so: Path) -> Path:
    """Real path for plugin .so (symlink-safe); glob parent if basename missing."""
    p = plugin_so.expanduser()
    if p.is_file():
        return p.resolve()
    parent = p.parent
    if parent.is_dir():
        for cand in sorted(parent.glob("libNvInfer_edgellm_plugin.so*"),
                           key=lambda x: -len(x.name)):
            if cand.is_file():
                return cand.resolve()
    return p.resolve(strict=False)


def _dedupe_existing_lib_dirs(dirs: list[Path]) -> list[Path]:
    """Resolve, keep existing directories only, preserve order, drop duplicates."""
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        try:
            r = d.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _tensorrt_lib_dirs_from_trt_package_and_ld_path() -> list[Path]:
    """``TRT_PACKAGE_DIR/lib`` plus ``LD_LIBRARY_PATH`` entries (unordered, may contain duplicates)."""
    dirs: list[Path] = []
    trt_pkg = os.environ.get("TRT_PACKAGE_DIR", "").strip()
    if trt_pkg:
        dirs.append(Path(trt_pkg) / "lib")
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        e = entry.strip()
        if e:
            dirs.append(Path(e))
    return dirs


def _tensorrt_lib_dirs_env_ld_only() -> list[Path]:
    """``TRT_PACKAGE_DIR/lib`` and ``LD_LIBRARY_PATH`` entries (no Python tensorrt path)."""
    return _dedupe_existing_lib_dirs(
        _tensorrt_lib_dirs_from_trt_package_and_ld_path())


def _tensorrt_lib_search_dirs() -> list[Path]:
    """Directories that may contain ``libnvinfer.so*`` (for RTLD_GLOBAL preload)."""
    dirs = _tensorrt_lib_dirs_from_trt_package_and_ld_path()
    if trt is not None:
        dirs.append(Path(
            trt.__file__).resolve().parent)  # type: ignore[union-attr]
    else:
        try:
            import tensorrt as trt_mod

            dirs.append(Path(trt_mod.__file__).resolve().parent)
        except ImportError:
            pass
    return _dedupe_existing_lib_dirs(dirs)


_libnvinfer_rtld_global_path: Path | None = None


def _ctypes_preload_libnvinfer_from_dirs(dirs: list[Path], *,
                                         verbose: bool) -> Path | None:
    global _libnvinfer_rtld_global_path
    import ctypes

    if not hasattr(ctypes, "RTLD_GLOBAL"):
        return None
    for d in dirs:
        candidates = sorted(d.glob("libnvinfer.so*"),
                            key=lambda p: len(p.name),
                            reverse=True)
        for so in candidates:
            if not so.is_file():
                continue
            try:
                ctypes.CDLL(os.fspath(so), mode=ctypes.RTLD_GLOBAL)
                if _libnvinfer_rtld_global_path is None:
                    _libnvinfer_rtld_global_path = so.resolve()
                    if verbose:
                        print(
                            f"[TRT] Preloaded libnvinfer (RTLD_GLOBAL): {so}")
                return so
            except OSError:
                continue
    return None


def _preload_libnvinfer_rtld_global(verbose: bool) -> Path | None:
    p = _ctypes_preload_libnvinfer_from_dirs(_tensorrt_lib_dirs_env_ld_only(),
                                             verbose=verbose)
    if p is not None:
        return p
    p = _ctypes_preload_libnvinfer_from_dirs(_tensorrt_lib_search_dirs(),
                                             verbose=verbose)
    if p is None and verbose:
        print(
            "[TRT] Note: could not preload libnvinfer.so* from TRT_PACKAGE_DIR, LD_LIBRARY_PATH, or "
            "the tensorrt package directory. Set TRT_PACKAGE_DIR to the TensorRT tree used to build "
            "the plugin if load_library fails.")
    return p


def _get_nvfp4_moe_plugin_creator(registry) -> object | None:
    getters: list[tuple[str, object]] = []
    for gname in ("get_creator", "get_plugin_creator", "getPluginCreator"):
        g = getattr(registry, gname, None)
        if callable(g):
            getters.append((gname, g))
    if not getters:
        return None
    versions = ("1", "1.0")
    namespaces = ("", "trt")
    for _gname, getter in getters:
        for ver in versions:
            for ns in namespaces:
                try:
                    c = getter("Nvfp4MoePlugin", ver, ns)
                except TypeError:
                    c = None
                except Exception:
                    c = None
                if c is not None:
                    return c
        for ver in versions:
            try:
                c = getter("Nvfp4MoePlugin", ver)
            except TypeError:
                c = None
            except Exception:
                c = None
            else:
                if c is not None:
                    return c
    return None


def _plugin_dlopen_diagnosis(plugin_path: Path) -> str:
    import ctypes

    lines: list[str] = []
    try:
        ctypes.CDLL(os.fspath(plugin_path),
                    mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        lines.append("ctypes.CDLL(plugin, RTLD_GLOBAL) succeeded.")
    except OSError as exc:
        lines.append(f"ctypes.CDLL(plugin): {exc}")

    if sys.platform.startswith("linux") and shutil.which("ldd"):
        try:
            proc = subprocess.run(
                ["ldd", os.fspath(plugin_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if out.strip():
                lines.append("ldd output:")
                lines.append(out.rstrip())
        except (subprocess.TimeoutExpired, OSError) as exc:
            lines.append(f"ldd failed: {exc}")

    return "\n".join(lines)


_PLUGIN_LOADED: bool = False


def load_nvfp4_moe_edge_llm_plugins(logger,
                                    plugin_so: Path,
                                    verbose: bool = True) -> None:
    global _PLUGIN_LOADED
    import ctypes

    assert trt is not None

    # Guard against loading the plugin .so more than once per process. TRT's
    # registry.load_library segfaults on the second call for libraries that use
    # REGISTER_TENSORRT_PLUGIN (no getCreators symbol).
    if _PLUGIN_LOADED:
        if verbose:
            print("[TRT] Edge-LLM plugins already loaded — skipping.")
        return

    lib_path = _resolve_edgellm_trt_plugin_path(plugin_so)
    path_str = os.fspath(lib_path)
    if verbose:
        print(f"[TRT] Loading Edge-LLM plugins from:\n      {lib_path}")

    _preload_libnvinfer_rtld_global(verbose=verbose)

    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    loaded_via_registry = bool(registry.load_library(path_str))

    def _try_ctypes_plugin() -> None:
        if not hasattr(ctypes, "RTLD_GLOBAL"):
            raise RuntimeError(
                "ctypes.RTLD_GLOBAL not available on this platform")
        ctypes.CDLL(path_str, mode=ctypes.RTLD_GLOBAL)

    if not loaded_via_registry:
        if verbose:
            print(
                "[TRT] load_library returned False; opening plugin with ctypes RTLD_GLOBAL "
                "(runs static ctors / plugin registration) …")
        try:
            _try_ctypes_plugin()
        except OSError as exc:
            diag = _plugin_dlopen_diagnosis(lib_path)
            raise RuntimeError("Could not load Edge-LLM plugin DSO.\n"
                               f"  Path: {lib_path}\n"
                               f"  ctypes: {exc}\n\n"
                               f"Diagnostics:\n{diag}") from exc

    creator = _get_nvfp4_moe_plugin_creator(registry)
    if creator is None and loaded_via_registry:
        if verbose:
            print(
                "[TRT] Nvfp4MoePlugin not visible after load_library; retrying ctypes RTLD_GLOBAL …"
            )
        try:
            _try_ctypes_plugin()
        except OSError:
            pass
        creator = _get_nvfp4_moe_plugin_creator(registry)

    if creator is None:
        diag = _plugin_dlopen_diagnosis(lib_path)
        raise RuntimeError(
            "Edge-LLM plugin library loaded but Nvfp4MoePlugin is not in the TensorRT plugin registry.\n"
            f"  Path: {lib_path}\n"
            "Rebuild the plugin against the same TensorRT as the Python ``tensorrt`` package "
            "(set TRT_PACKAGE_DIR when running CMake).\n\n"
            f"Diagnostics:\n{diag}")

    _PLUGIN_LOADED = True

    if verbose:
        if loaded_via_registry:
            print(
                "[TRT] Nvfp4MoePlugin is registered (load_library returned True)."
            )
        else:
            print(
                "[TRT] Nvfp4MoePlugin is registered via ctypes RTLD_GLOBAL "
                "(TensorRT load_library returned False; this is a known quirk on some 10.x builds)."
            )


def _trt_torch_dtype(trt_dtype):
    assert trt is not None
    if trt_dtype == trt.float16:
        return torch.float16
    if trt_dtype == trt.float32:
        return torch.float32
    if trt_dtype == trt.int32:
        return torch.int32
    if trt_dtype == trt.int8:
        return torch.int8
    raise ValueError(f"unsupported TensorRT dtype: {trt_dtype}")


def execute_trt_engine(
    serialized: bytes,
    inputs: dict[str, np.ndarray],
    stream=None,
    device=None,
    *,
    verbose: bool = True,
    label: str = "",
) -> np.ndarray:
    assert trt is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dev = device or torch.device("cuda", torch.cuda.current_device())
    if stream is None:
        stream = torch.cuda.Stream(device=dev)

    tag = f" {label}" if label else ""
    if verbose:
        print(
            f"[exec{tag}] CUDA device={dev}  engine blob={len(serialized)} bytes\n"
            f"[exec{tag}] Host inputs:")
        for name, arr in inputs.items():
            print(
                f"[exec{tag}]   {name}: shape={tuple(arr.shape)} dtype={arr.dtype}"
            )

    t0 = time.perf_counter()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None")
    if verbose:
        print(
            f"[exec{tag}] deserialize + create_execution_context in {time.perf_counter() - t0:.3f}s"
        )

    for name, arr in inputs.items():
        _require_trt_ok(context.set_input_shape(name, tuple(arr.shape)))

    bindings: dict[str, torch.Tensor] = {}
    output_names: list[str] = []
    if verbose:
        print(f"[exec{tag}] I/O tensors ({engine.num_io_tensors}):")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(context.get_tensor_shape(name))
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = _trt_torch_dtype(trt_dtype)
        bindings[name] = torch.empty(shape, dtype=torch_dtype, device=dev)
        if verbose:
            kind = "OUTPUT" if mode == trt.TensorIOMode.OUTPUT else "INPUT"
            print(
                f"[exec{tag}]   [{i}] {kind} {name}: shape={shape} trt_dtype={trt_dtype}"
            )
        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if not output_names:
        raise RuntimeError("TensorRT engine has no output tensors")
    # Prefer the Nvfp4MoePlugin output (``mark_output`` name in verify_nvfp4_moe_trt_engine.py); TensorRT may list
    # multiple OUTPUT tensors or order them such that the first is not the plugin dense MoE tensor — reading the
    # wrong binding yields uninitialized (NaN) FP16.
    output_name = "output" if "output" in output_names else output_names[0]
    for on in output_names:
        bindings[on].zero_()

    if verbose:
        print(f"[exec{tag}] H2D copy + execute_async_v3 …")
    t1 = time.perf_counter()
    with torch.cuda.stream(stream):
        for name, arr in inputs.items():
            host = torch.from_numpy(np.ascontiguousarray(arr))
            # Pageable CPU numpy → GPU: non_blocking requires pinned host memory; async copies can race and
            # corrupt INT8/FP32 plugin inputs (downstream MoE output all-NaN).
            bindings[name].copy_(
                host.to(device=dev,
                        dtype=bindings[name].dtype,
                        non_blocking=False),
                non_blocking=False,
            )
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            _require_trt_ok(
                context.set_tensor_address(name, bindings[name].data_ptr()))
        _require_trt_ok(context.execute_async_v3(stream.cuda_stream))
    stream.synchronize()
    if verbose:
        print(
            f"[exec{tag}] GPU kernel finished in {time.perf_counter() - t1:.3f}s "
            f"(wall incl. sync)\n[exec{tag}] D2H output '{output_name}' shape={tuple(bindings[output_name].shape)}"
        )

    return np.array(bindings[output_name].detach().cpu().numpy(), copy=True)


# --- Build engine + accuracy driver ---


def build_nvfp4_moe_engine_trt_api(
    module: object,
    dummy_hidden_fp16: Any,
    logger,
    *,
    verbose: bool = True,
    label: str = "",
    w4a4_activation: bool = False,
) -> bytes:
    """
    Build a serialized TensorRT engine for ``Nvfp4MoePlugin`` using the network API (no ONNX).

    ``module`` must be a :class:`NemotronHMoEW4A4Plugin` on CPU with NVFP4 buffers already filled
    (or zeroed). ``dummy_hidden_fp16`` defines static ``(batch, seq, hidden)`` layout (used for
    optimization profile bounds); it must match ``module.hidden_size``.

    **W4A16:** ``hidden_states`` is FP16 ``(B,S,H)`` used for both the gate matmul (router logits)
    and the MoE plugin expert computation.

    **W4A4:** ``hidden_states`` is INT8 ``(B,S,H/2)`` NVFP4-packed activations for expert computation,
    with ``hidden_block_scale`` / ``hidden_global_scale`` as runtime inputs (Marlin tile scales).
    A separate FP16 ``hidden_states_fp16`` ``(B,S,H)`` input drives the gate matmul to compute
    router logits inside the TRT graph (INT8 activations cannot drive the gate matmul directly).
    """
    import tensorrt as trt

    if dummy_hidden_fp16.dim() != 3:
        raise ValueError(
            f"dummy_hidden_fp16 must be (B,S,H), got shape {tuple(dummy_hidden_fp16.shape)}"
        )
    b, s, h = (int(dummy_hidden_fp16.shape[i]) for i in range(3))
    if int(getattr(module, "hidden_size")) != h:
        raise ValueError(
            f"dummy hidden {h} != module.hidden_size {getattr(module, 'hidden_size')}"
        )

    tag = f" {label}" if label else ""
    if verbose:
        if w4a4_activation:
            hid_mode = f"INT8 NVFP4 ({b},{s},{h // 2}) + FP16 gate input ({b},{s},{h})"
        else:
            hid_mode = f"FP16 ({b},{s},{h})"
        print(
            f"[TRT API{tag}] Building engine (STRONGLY_TYPED, add_plugin_v3)  "
            f"hidden_states={hid_mode}  router=from gate matmul in graph")

    registry = trt.get_plugin_registry()
    creator = _get_nvfp4_moe_plugin_creator(registry)
    if creator is None:
        raise RuntimeError(
            "Nvfp4MoePlugin creator not found (load Edge-LLM plugin DSO first)."
        )

    e_ct = int(getattr(module, "num_experts"))
    top_k = int(getattr(module, "top_k"))
    inter = int(getattr(module, "moe_inter_size"))
    f_act = int(getattr(module, "activation_type", 0))
    f_qgs = int(getattr(module, "quantization_group_size", 16))
    f_ng = int(getattr(module, "n_group", 1))
    f_tkg = int(getattr(module, "topk_group", 1))
    f_ntp = int(getattr(module, "norm_topk_prob", 1))
    f_rsf = float(getattr(module, "routed_scaling_factor", 1.0))
    # NemotronHMoEW4A4Plugin drives the sigmoid + grouped top-k path (routing_mode=1).
    f_rm = int(getattr(module, "routing_mode", 1))
    # Keep numpy arrays alive until after create_plugin — trt.PluginField
    # stores a raw data pointer without preventing GC of the backing array.
    _pf_bufs = [
        np.array([e_ct], dtype=np.int32),
        np.array([top_k], dtype=np.int32),
        np.array([h], dtype=np.int32),
        np.array([inter], dtype=np.int32),
        np.array([f_act], dtype=np.int32),
        np.array([f_qgs], dtype=np.int32),
        np.array([f_ng], dtype=np.int32),
        np.array([f_tkg], dtype=np.int32),
        np.array([f_ntp], dtype=np.int32),
        np.array([f_rsf], dtype=np.float32),
        np.array([f_rm], dtype=np.int32),
    ]
    pfc = trt.PluginFieldCollection([
        trt.PluginField("num_experts", _pf_bufs[0], trt.PluginFieldType.INT32),
        trt.PluginField("top_k", _pf_bufs[1], trt.PluginFieldType.INT32),
        trt.PluginField("hidden_size", _pf_bufs[2], trt.PluginFieldType.INT32),
        trt.PluginField("moe_inter_size", _pf_bufs[3],
                        trt.PluginFieldType.INT32),
        trt.PluginField("activation_type", _pf_bufs[4],
                        trt.PluginFieldType.INT32),
        trt.PluginField("quantization_group_size", _pf_bufs[5],
                        trt.PluginFieldType.INT32),
        trt.PluginField("n_group", _pf_bufs[6], trt.PluginFieldType.INT32),
        trt.PluginField("topk_group", _pf_bufs[7], trt.PluginFieldType.INT32),
        trt.PluginField("norm_topk_prob", _pf_bufs[8],
                        trt.PluginFieldType.INT32),
        trt.PluginField("routed_scaling_factor", _pf_bufs[9],
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("routing_mode", _pf_bufs[10],
                        trt.PluginFieldType.INT32),
    ])
    try:
        plugin = creator.create_plugin("Nvfp4MoePlugin", pfc,
                                       trt.TensorRTPhase.BUILD)
    except TypeError:
        plugin = creator.create_plugin("Nvfp4MoePlugin", pfc)
    del _pf_bufs  # safe to release after create_plugin has parsed fields

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    if w4a4_activation:
        hidden_in = network.add_input("hidden_states", trt.int8,
                                      (b, s, h // 2))
        hidden_bs_in = network.add_input("hidden_block_scale", trt.int8,
                                         (b, s, h // 16))
        hidden_gs_in = network.add_input("hidden_global_scale", trt.float32,
                                         (1, ))
    else:
        hidden_in = network.add_input("hidden_states", trt.float16, (b, s, h))

    # Gate matmul: compute router_logits from FP16 hidden states inside the TRT graph.
    # W4A4 uses a separate FP16 input (INT8 packed activations can't drive the gate).
    if w4a4_activation:
        hidden_fp16_in = network.add_input("hidden_states_fp16", trt.float16,
                                           (b, s, h))
        gate_source = hidden_fp16_in
    else:
        gate_source = hidden_in

    flat = network.add_shuffle(gate_source)
    flat.reshape_dims = (b * s, h)

    router_f32 = network.add_cast(flat.get_output(0), trt.float32)

    # Match ``F.linear(x, W)`` / ``nn.Linear``: ``W`` is ``(E, H)``; logits are ``x @ W.T`` i.e. ``(T,H) @ (H,E)``.
    # Bake ``W.T`` as a contiguous ``(H, E)`` constant and use ``NONE`` on both operands so the GEMM does not rely
    # on ``MatrixOperation.TRANSPOSE`` for the weight tensor (avoids STRONGLY_TYPED / constant-layer quirks that can
    # corrupt logits and yield NaNs → plugin decode skips all experts, ~0 output).
    gate_w = getattr(module, "gate").weight.detach().float().cpu().numpy()
    if gate_w.shape != (e_ct, h):
        raise ValueError(
            f"gate.weight shape {gate_w.shape} expected ({e_ct}, {h})")
    gate_w_T = np.ascontiguousarray(gate_w.astype(np.float32, copy=False).T)
    gate_const = network.add_constant((h, e_ct), gate_w_T)
    mm = network.add_matrix_multiply(
        router_f32.get_output(0),
        trt.MatrixOperation.NONE,
        gate_const.get_output(0),
        trt.MatrixOperation.NONE,
    )
    router_logits = mm.get_output(0)
    bias = getattr(module, "gate").bias
    if bias is not None:
        b_np = np.ascontiguousarray(
            bias.detach().float().cpu().numpy().reshape(1, e_ct).astype(
                np.float32))
        bias_const = network.add_constant((1, e_ct), b_np)
        router_logits = network.add_elementwise(
            router_logits, bias_const.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

    up_q = np.ascontiguousarray(
        module.fc_up_qweights.detach().cpu().numpy().astype(np.int8,
                                                            copy=False))
    up_bs = np.ascontiguousarray(
        module.fc_up_blocks_scale.detach().cpu().numpy().astype(np.int8,
                                                                copy=False))
    up_gs = np.ascontiguousarray(
        module.fc_up_global_scale.detach().cpu().numpy().astype(np.float32,
                                                                copy=False))
    dn_q = np.ascontiguousarray(
        module.fc_down_qweights.detach().cpu().numpy().astype(np.int8,
                                                              copy=False))
    dn_bs = np.ascontiguousarray(
        module.fc_down_blocks_scale.detach().cpu().numpy().astype(np.int8,
                                                                  copy=False))
    dn_gs = np.ascontiguousarray(
        module.fc_down_global_scale.detach().cpu().numpy().astype(np.float32,
                                                                  copy=False))

    c_up_q = network.add_constant(tuple(up_q.shape), up_q).get_output(0)
    c_up_bs = network.add_constant(tuple(up_bs.shape), up_bs).get_output(0)
    c_up_gs = network.add_constant(tuple(up_gs.shape), up_gs).get_output(0)
    c_dn_q = network.add_constant(tuple(dn_q.shape), dn_q).get_output(0)
    c_dn_bs = network.add_constant(tuple(dn_bs.shape), dn_bs).get_output(0)
    c_dn_gs = network.add_constant(tuple(dn_gs.shape), dn_gs).get_output(0)

    # e_score_correction_bias [E] FP32 — input [10] for sigmoid group top-k routing.
    corr_bias_np = np.ascontiguousarray(
        module.e_score_correction_bias.detach().cpu().numpy().astype(
            np.float32, copy=False))
    c_corr_bias = network.add_constant(tuple(corr_bias_np.shape),
                                       corr_bias_np).get_output(0)

    if w4a4_activation:
        plugin_inputs = [
            router_logits,
            hidden_in,
            hidden_bs_in,
            hidden_gs_in,
            c_up_q,
            c_up_bs,
            c_up_gs,
            c_dn_q,
            c_dn_bs,
            c_dn_gs,
            c_corr_bias,
        ]
    else:
        hb_dummy = np.zeros((1, 1, 1), dtype=np.int8)
        hg_dummy = np.ones((1, ), dtype=np.float32)
        c_hb = network.add_constant(tuple(hb_dummy.shape),
                                    hb_dummy).get_output(0)
        c_hg = network.add_constant(tuple(hg_dummy.shape),
                                    hg_dummy).get_output(0)

        plugin_inputs = [
            router_logits,
            hidden_in,
            c_hb,
            c_hg,
            c_up_q,
            c_up_bs,
            c_up_gs,
            c_dn_q,
            c_dn_bs,
            c_dn_gs,
            c_corr_bias,
        ]
    # TensorRT 10.x: ``add_plugin_v3(inputs, shape_inputs, plugin)`` — no shape tensors for Nvfp4MoePlugin.
    moe_layer = network.add_plugin_v3(plugin_inputs, [], plugin)

    out_t = moe_layer.get_output(0)
    out_t.name = "output"
    network.mark_output(out_t)

    profile = builder.create_optimization_profile()
    if w4a4_activation:
        profile.set_shape("hidden_states", (b, s, h // 2), (b, s, h // 2),
                          (b, s, h // 2))
        profile.set_shape("hidden_block_scale", (b, s, h // 16),
                          (b, s, h // 16), (b, s, h // 16))
        profile.set_shape("hidden_global_scale", (1, ), (1, ), (1, ))
        profile.set_shape("hidden_states_fp16", (b, s, h), (b, s, h),
                          (b, s, h))
    else:
        profile.set_shape("hidden_states", (b, s, h), (b, s, h), (b, s, h))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.add_optimization_profile(profile)

    if verbose:
        print(
            f"[TRT API{tag}] build_serialized_network (workspace limit 1 GiB) …"
        )
    t1 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    if not isinstance(serialized, (bytes, bytearray)):
        serialized = bytes(serialized)
    if verbose:
        mib = len(serialized) / (1024 * 1024)
        print(
            f"[TRT API{tag}] Engine ready: {len(serialized)} bytes ({mib:.2f} MiB) "
            f"in {time.perf_counter() - t1:.2f}s")
    return serialized


def pack_w4a4_activation_nvfp4_bsh(
    hidden_fp16_bsh: np.ndarray,
    *,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack FP16 ``[B,S,H]`` activations to Marlin NVFP4 tiles (same rules as :class:`MarlinConverter`).

    Returns:
        ``payload_int8`` ``[B,S,H/2]``, ``block_scale_int8`` ``[B,S,H/16]`` (four bytes per 64-lane tile).
    """
    x = np.asarray(hidden_fp16_bsh, dtype=np.float16)
    if x.ndim != 3 or int(x.shape[2]) != int(hidden_size):
        raise ValueError(
            f"hidden_fp16_bsh must be [B,S,H] with H={hidden_size}, got {tuple(x.shape)}"
        )
    b, s, h = (int(x.shape[i]) for i in range(3))
    if h % 64 != 0:
        raise ValueError(f"hidden_size {h} must be a multiple of 64")
    c_tiles = h // 64
    pl = np.zeros((b, s, h // 2), dtype=np.int8)
    bs = np.zeros((b, s, h // 16), dtype=np.int8)
    for bi in range(b):
        for sj in range(s):
            for c in range(c_tiles):
                vec64 = x[bi, sj, c * 64:(c + 1) * 64].astype(np.float32)
                pl_i32, scale_word = (
                    MarlinConverter.
                    quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
                        vec64,
                        expert_block_scale_max_fp32=1.0,
                    ))
                pl[bi, sj, c * 32:(c + 1) * 32] = np.frombuffer(
                    np.ascontiguousarray(pl_i32).tobytes(),
                    dtype=np.int8,
                )
                # scale_word is a u32 bit pattern (may exceed signed int32 range).
                bs[bi, sj, c * 4:(c + 1) * 4] = np.frombuffer(
                    np.uint32(int(scale_word) & 0xFFFFFFFF).tobytes(),
                    dtype=np.int8,
                )
    return pl, bs


def fp16_hidden_from_dequantized_nvfp4_activation(
    payload_int8_bsh2: np.ndarray,
    block_scale_int8_bsh16: np.ndarray,
    global_scale_fp32: np.ndarray,
    *,
    hidden_size: int,
) -> np.ndarray:
    """Decode packed NVFP4 activations (+ Marlin FP8 block scales × global scale) to FP16 ``[B,S,H]`` for MoE ref."""
    pl = np.asarray(payload_int8_bsh2, dtype=np.int8)
    bs = np.asarray(block_scale_int8_bsh16, dtype=np.int8)
    g0 = float(np.asarray(global_scale_fp32, dtype=np.float32).reshape(-1)[0])
    if pl.ndim != 3 or int(pl.shape[2]) != int(hidden_size) // 2:
        raise ValueError(
            f"payload must be [B,S,H/2], got {tuple(pl.shape)} for H={hidden_size}"
        )
    b, s, _ = (int(pl.shape[i]) for i in range(3))
    h = int(hidden_size)
    c_tiles = h // 64
    out = np.zeros((b, s, h), dtype=np.float16)
    for bi in range(b):
        for sj in range(s):
            for c in range(c_tiles):
                raw_pl = pl[bi, sj, c * 32:(c + 1) * 32]
                pl_i32 = np.frombuffer(raw_pl.tobytes(), dtype=np.int32)
                raw_sq = bs[bi, sj, c * 4:(c + 1) * 4]
                scale_word = int(
                    np.frombuffer(raw_sq.tobytes(), dtype=np.int32)[0])
                dec = NemotronHMoEReference.unpack_nvfp4_marlin_tile_64(
                    pl_i32, scale_word)
                dec = dec.astype(np.float32) * g0
                out[bi, sj, c * 64:(c + 1) * 64] = dec.astype(np.float16)
    return out


def _replace_hf_expert_weights_with_dequantized_nvfp4(
    moe: "NemotronHMoE",
    mod_cpu: "NemotronHMoEW4A4Plugin",
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
) -> None:
    """Overwrite ``moe.experts.up_proj`` / ``down_proj`` with weights dequantized from NVFP4 Marlin buffers.

    This ensures the HF reference uses the same quantization-lossy weights as TRT, isolating the
    accuracy comparison to the MoE computation logic rather than NVFP4 weight quantization error.
    """
    w_up, w_down = NemotronHMoEReference.dense_weights_from_nvfp4_plugin_buffers(
        mod_cpu.fc_up_qweights.numpy(),
        mod_cpu.fc_up_blocks_scale.numpy(),
        mod_cpu.fc_down_qweights.numpy(),
        mod_cpu.fc_down_blocks_scale.numpy(),
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
    )
    # Apply per-expert global scales (same as plugin kernel)
    up_gs = mod_cpu.fc_up_global_scale.numpy().astype(np.float32).reshape(-1)
    dn_gs = mod_cpu.fc_down_global_scale.numpy().astype(np.float32).reshape(-1)
    for ex in range(num_experts):
        w_up[ex] *= float(up_gs[ex])
        w_down[ex] *= float(dn_gs[ex])
    # w_up is [E,H,I], HF up_proj is [E,I,H] → transpose (0,2,1)
    # w_down is [E,I,H], HF down_proj is [E,H,I] → transpose (0,2,1)
    dev = moe.experts.up_proj.device
    dtype = moe.experts.up_proj.dtype
    with torch.no_grad():
        moe.experts.up_proj.data.copy_(
            torch.from_numpy(np.ascontiguousarray(w_up.transpose(0, 2, 1))).to(
                device=dev, dtype=dtype))
        moe.experts.down_proj.data.copy_(
            torch.from_numpy(np.ascontiguousarray(w_down.transpose(
                0, 2, 1))).to(device=dev, dtype=dtype))


def run_nvfp4_w4a16_moe_plugin_accuracy_case(
    *,
    device: torch.device,
    top_k: int,
    moe_seed: int,
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
    batch: int = _NVFP4_MOE_FAST_BATCH,
    seq: int = 1,
    hidden_size: int = _NVFP4_MOE_FAST_HIDDEN,
    moe_inter_size: int = _NVFP4_MOE_FAST_INTER,
    num_experts: int = _NVFP4_MOE_FAST_EXPERTS,
) -> None:
    """W4A16 accuracy case: TRT (gate FC + MoE plugin) vs HF routed experts on dequantized NVFP4 weights.

    Steps:
        1. Create toy MoE, pack weights to NVFP4 Marlin, dequantize back into HF expert params.
        2. Build TRT engine with gate matmul in graph; HF runs ``hf_routed_expert_output``.
        3. Both receive the same FP16 ``hidden_states`` and use the same dequantized weights.
        4. Print detailed accuracy stats (element-wise distribution, pair norms, cosine).
        5. Assert ``cosine >= 0.9999`` and ``rel_L2 <= 0.01``.
    """
    check_requirements()
    assert trt is not None
    tk = int(top_k)
    h, inter, e_ct = hidden_size, moe_inter_size, num_experts

    moe = create_toy_moe(
        device,
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        top_k=tk,
        seed=moe_seed,
    )
    mod = NemotronHMoEW4A4Plugin(moe)
    mod.eval().to(device)
    mod.pack_experts_weights_to_marlin(moe)

    hidden_dev = device if hidden_on_cuda else None
    hidden_bsh = structured_noise_hidden_states_bsh(batch,
                                                    seq,
                                                    h,
                                                    seed=hidden_seed,
                                                    amp=2.5,
                                                    device=hidden_dev)

    mod_cpu = mod.cpu()
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())

    # Replace HF expert weights with dequantized NVFP4 weights so both TRT and HF
    # use the same quantization-lossy weights (isolates comparison to MoE computation).
    _replace_hf_expert_weights_with_dequantized_nvfp4(moe,
                                                      mod_cpu,
                                                      hidden_size=h,
                                                      moe_inter_size=inter,
                                                      num_experts=e_ct)

    # HF reference: full routed expert path (gate → route → experts) with dequantized weights
    hf_ref_fp32 = NemotronHMoEReference.hf_routed_expert_output(
        moe, hidden_bsh)

    # TRT engine: gate matmul + MoE plugin in a single graph
    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = build_nvfp4_moe_engine_trt_api(
        mod_cpu,
        torch.zeros(batch, seq, h, dtype=torch.float16, device="cpu"),
        logger,
        verbose=False,
        label="nvfp4-moe-gate-in-graph",
    )

    stream = torch.cuda.Stream(device=device)
    trt_out = execute_trt_engine(
        eng,
        {"hidden_states": np.ascontiguousarray(hidden_np.astype(np.float16))},
        stream,
        device=device,
        verbose=False,
        label=trt_execute_label,
    )

    if PRINT_TRT_VS_TORCH_DIST:
        NemotronHMoEReference.print_cross_check_output_distribution(
            trt_out,
            hf_ref_fp32,
            hf_ref_fp32,
            case,
            print_config=PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG,
        )

    NemotronHMoEReference.assert_trt_vs_hf_cosine_similarity(
        trt_out, hf_ref_fp32, f"{case}_trt_gate_vs_hf")


def run_nvfp4_w4a4_moe_plugin_accuracy_case(
    *,
    device: torch.device,
    top_k: int,
    moe_seed: int,
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
    batch: int = _NVFP4_MOE_FAST_BATCH,
    seq: int = 1,
    hidden_size: int = _NVFP4_MOE_FAST_HIDDEN,
    moe_inter_size: int = _NVFP4_MOE_FAST_INTER,
    num_experts: int = _NVFP4_MOE_FAST_EXPERTS,
) -> None:
    """W4A4 accuracy case: TRT (gate FC + MoE plugin) vs HF routed experts on dequantized NVFP4 weights.

    Steps:
        1. Create toy MoE, pack weights to NVFP4 Marlin, dequantize back into HF expert params.
        2. Pack FP16 hidden states to NVFP4, then dequantize to FP16 (``hidden_ref``).
        3. Build TRT engine with gate matmul from ``hidden_states_fp16`` (dequantized FP16)
           and expert computation from INT8 NVFP4-packed ``hidden_states`` + block/global scales.
        4. HF runs ``hf_routed_expert_output(moe, hidden_ref)`` with dequantized weights.
        5. Print detailed accuracy stats; assert ``cosine >= 0.9999`` and ``rel_L2 <= 0.01``.
    """
    check_requirements()
    assert trt is not None
    tk = int(top_k)
    h, inter, e_ct = hidden_size, moe_inter_size, num_experts

    moe = create_toy_moe(
        device,
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        top_k=tk,
        seed=moe_seed,
        routed_scaling_factor=_NVFP4_MOE_W4A4_ROUTED_SCALING,
    )
    with torch.no_grad():
        moe.experts.up_proj.data.mul_(_NVFP4_MOE_W4A4_EXPERT_WEIGHT_SCALE)
        moe.experts.down_proj.data.mul_(_NVFP4_MOE_W4A4_EXPERT_WEIGHT_SCALE)
    mod = NemotronHMoEW4A4Plugin(moe)
    mod.eval().to(device)
    mod.pack_experts_weights_to_marlin(moe)

    hidden_dev = device if hidden_on_cuda else None
    hidden_bsh = structured_noise_hidden_states_bsh(
        batch,
        seq,
        h,
        seed=hidden_seed,
        amp=_NVFP4_MOE_W4A4_HIDDEN_AMP,
        device=hidden_dev)

    mod_cpu = mod.cpu()
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())

    # Pack activations to NVFP4 and dequantize back to FP16 for reference
    act_pl, act_bs = pack_w4a4_activation_nvfp4_bsh(hidden_np, hidden_size=h)
    act_gs = np.ones(1, dtype=np.float32)

    hidden_ref_fp16 = fp16_hidden_from_dequantized_nvfp4_activation(
        act_pl,
        act_bs,
        act_gs,
        hidden_size=h,
    )
    hidden_ref_np = np.ascontiguousarray(hidden_ref_fp16)
    hidden_ref_t = torch.as_tensor(hidden_ref_np,
                                   device=device,
                                   dtype=torch.float16)

    # Replace HF expert weights with dequantized NVFP4 weights so both TRT and HF
    # use the same quantization-lossy weights (isolates comparison to MoE computation).
    _replace_hf_expert_weights_with_dequantized_nvfp4(moe,
                                                      mod_cpu,
                                                      hidden_size=h,
                                                      moe_inter_size=inter,
                                                      num_experts=e_ct)

    # HF reference: full routed expert path on dequantized hidden states + dequantized weights
    hf_ref_fp32 = NemotronHMoEReference.hf_routed_expert_output(
        moe, hidden_ref_t)

    # TRT engine: gate matmul (from hidden_states_fp16) + MoE plugin (INT8 packed hidden_states)
    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = build_nvfp4_moe_engine_trt_api(
        mod_cpu,
        torch.zeros(batch, seq, h, dtype=torch.float16, device="cpu"),
        logger,
        verbose=False,
        label="nvfp4-moe-w4a4-gate-in-graph",
        w4a4_activation=True,
    )

    stream = torch.cuda.Stream(device=device)
    trt_out = execute_trt_engine(
        eng,
        {
            "hidden_states":
            np.ascontiguousarray(act_pl),
            "hidden_block_scale":
            np.ascontiguousarray(act_bs),
            "hidden_global_scale":
            np.ascontiguousarray(act_gs),
            "hidden_states_fp16":
            np.ascontiguousarray(hidden_ref_np.astype(np.float16)),
        },
        stream,
        device=device,
        verbose=False,
        label=trt_execute_label,
    )

    if PRINT_TRT_VS_TORCH_DIST:
        NemotronHMoEReference.print_cross_check_output_distribution(
            trt_out,
            hf_ref_fp32,
            hf_ref_fp32,
            case,
            print_config=PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG,
            w4a4_dequant_hidden=True,
        )

    NemotronHMoEReference.assert_trt_vs_hf_cosine_similarity(
        trt_out, hf_ref_fp32, f"{case}_trt_gate_vs_hf")


@pytest.mark.parametrize(
    (
        "top_k",
        "moe_seed",
        "hidden_seed",
        "hidden_on_cuda",
        "trt_execute_label",
        "case",
        "batch",
        "seq",
        "hidden_size",
        "moe_inter_size",
        "num_experts",
    ),
    [
        pytest.param(
            1,
            2026,
            91021,
            True,
            "nvfp4-moe-realistic-b2s1-tk1",
            "realistic_b2s1_h384_e8_topk1",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk1",
        ),
        pytest.param(
            2,
            2027,
            91022,
            False,
            "nvfp4-moe-realistic-b2s1-tk2",
            "realistic_b2s1_h384_e8_topk2",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk2",
        ),
        pytest.param(
            4,
            2028,
            91023,
            True,
            "nvfp4-moe-realistic-b2s1-tk4",
            "realistic_b2s1_h384_e8_topk4",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk4",
        ),
        pytest.param(
            2,
            2029,
            91024,
            False,
            "nvfp4-moe-realistic-b2s2-tk2",
            "realistic_b2s2_h384_e8_topk2",
            _NVFP4_MOE_FAST_BATCH,
            2,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s2_topk2",
        ),
        pytest.param(
            6,
            2030,
            91025,
            True,
            "nvfp4-moe-b1s1-h64-e128-tk6",
            "b1s1_h64_e128_topk6",
            1,
            1,
            64,
            128,
            128,
            id="b1s1_h64_e128_topk6",
        ),
    ],
)
def test_nvfp4_w4a16_moe_plugin_accuracy(
    top_k: int,
    moe_seed: int,
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
    batch: int,
    seq: int,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
) -> None:
    """W4A16: TRT gate-in-graph + MoE plugin vs HF routed experts (dequantized NVFP4 weights)."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    run_nvfp4_w4a16_moe_plugin_accuracy_case(
        device=dev,
        top_k=int(top_k),
        moe_seed=int(moe_seed),
        hidden_seed=int(hidden_seed),
        hidden_on_cuda=bool(hidden_on_cuda),
        trt_execute_label=trt_execute_label,
        case=case,
        batch=int(batch),
        seq=int(seq),
        hidden_size=int(hidden_size),
        moe_inter_size=int(moe_inter_size),
        num_experts=int(num_experts),
    )


def test_nvfp4_w4a16_moe_plugin_accuracy_sf_padding_non128_aligned() -> None:
    """Regression: hidden_size=1856 (not 128-aligned) exercises Cutlass Atom SF padding."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    run_nvfp4_w4a16_moe_plugin_accuracy_case(
        device=dev,
        top_k=1,
        moe_seed=5050,
        hidden_seed=91050,
        hidden_on_cuda=True,
        trt_execute_label="nvfp4-moe-sf-padding-non128",
        case="sf_padding_non128_h1856_i2688",
        batch=1,
        seq=1,
        hidden_size=1856,  # NOT a multiple of 128 (1856 = 128*14 + 64)
        moe_inter_size=2688,
        num_experts=2,
    )


@pytest.mark.parametrize(
    (
        "top_k",
        "moe_seed",
        "hidden_seed",
        "hidden_on_cuda",
        "trt_execute_label",
        "case",
        "batch",
        "seq",
        "hidden_size",
        "moe_inter_size",
        "num_experts",
    ),
    [
        pytest.param(
            1,
            2026,
            91021,
            True,
            "nvfp4-moe-w4a4-realistic-b2s1-tk1",
            "realistic_b2s1_h384_e8_topk1",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk1",
        ),
        pytest.param(
            2,
            2027,
            91022,
            False,
            "nvfp4-moe-w4a4-realistic-b2s1-tk2",
            "realistic_b2s1_h384_e8_topk2",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk2",
        ),
        pytest.param(
            4,
            2028,
            91023,
            True,
            "nvfp4-moe-w4a4-realistic-b2s1-tk4",
            "realistic_b2s1_h384_e8_topk4",
            _NVFP4_MOE_FAST_BATCH,
            1,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s1_topk4",
        ),
        pytest.param(
            2,
            2029,
            91024,
            False,
            "nvfp4-moe-w4a4-realistic-b2s2-tk2",
            "realistic_b2s2_h384_e8_topk2",
            _NVFP4_MOE_FAST_BATCH,
            2,
            _NVFP4_MOE_FAST_HIDDEN,
            _NVFP4_MOE_FAST_INTER,
            _NVFP4_MOE_FAST_EXPERTS,
            id="realistic_b2s2_topk2",
        ),
        pytest.param(
            6,
            2030,
            91025,
            True,
            "nvfp4-moe-w4a4-b1s1-h64-e128-tk6",
            "b1s1_h64_e128_topk6",
            1,
            1,
            64,
            128,
            128,
            id="b1s1_h64_e128_topk6",
        ),
    ],
)
def test_nvfp4_w4a4_moe_plugin_accuracy(
    top_k: int,
    moe_seed: int,
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
    batch: int,
    seq: int,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
) -> None:
    """W4A4: TRT gate-in-graph + MoE plugin vs HF routed experts (dequantized NVFP4 weights)."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    run_nvfp4_w4a4_moe_plugin_accuracy_case(
        device=dev,
        top_k=int(top_k),
        moe_seed=int(moe_seed),
        hidden_seed=int(hidden_seed),
        hidden_on_cuda=bool(hidden_on_cuda),
        trt_execute_label=trt_execute_label,
        case=case,
        batch=int(batch),
        seq=int(seq),
        hidden_size=int(hidden_size),
        moe_inter_size=int(moe_inter_size),
        num_experts=int(num_experts),
    )
