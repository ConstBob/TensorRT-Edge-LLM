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
"""Official *-NVFP4 lm_head promotion to NVFP4 W4A16 (issue 703)."""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm import config


def test_promote_official_nvfp4_lm_head():
    quant = config.QuantConfig(quant_type=config.QUANT_NVFP4,
                               group_size=16,
                               excluded=["lm_head", "visual"])
    promoted = config._promote_official_nvfp4_lm_head_to_a16(
        "/unused", {
            "hidden_size": 2048,
            "vocab_size": 248320,
            "tie_word_embeddings": False
        }, quant)
    assert "lm_head" not in promoted.excluded
    assert "visual" in promoted.excluded
    assert promoted.layer_overrides["lm_head"] == config.QUANT_NVFP4_A16


def test_skip_tied_and_already_quantized_lm_head():
    tied = config._promote_official_nvfp4_lm_head_to_a16(
        "/unused", {
            "hidden_size": 2048,
            "tie_word_embeddings": True
        },
        config.QuantConfig(quant_type=config.QUANT_NVFP4,
                           excluded=["lm_head"]))
    assert tied.layer_overrides.get("lm_head") is None
    assert "lm_head" in tied.excluded

    explicit = config._promote_official_nvfp4_lm_head_to_a16(
        "/unused", {"hidden_size": 2048},
        config.QuantConfig(quant_type=config.QUANT_NVFP4,
                           excluded=["lm_head"],
                           layer_overrides={"lm_head": config.QUANT_NVFP4}))
    assert explicit.layer_overrides["lm_head"] == config.QUANT_NVFP4


def test_promote_mixed_precision_unlisted_lm_head():
    quant = config.QuantConfig(quant_type=config.QUANT_NVFP4_A16,
                               group_size=16,
                               is_mixed_precision=True,
                               layer_overrides={"layers.0.mlp.experts":
                                                config.QUANT_NVFP4_A16})
    promoted = config._promote_official_nvfp4_lm_head_to_a16(
        "/unused", {"hidden_size": 2048}, quant)
    assert promoted.layer_overrides["lm_head"] == config.QUANT_NVFP4_A16


def test_keep_official_nvfp4_lm_head_env(monkeypatch):
    monkeypatch.setenv("EDGELLM_KEEP_OFFICIAL_NVFP4_LM_HEAD", "1")
    quant = config.QuantConfig(quant_type=config.QUANT_NVFP4,
                               excluded=["lm_head"])
    kept = config._promote_official_nvfp4_lm_head_to_a16(
        "/unused", {"hidden_size": 2048}, quant)
    assert "lm_head" in kept.excluded
    assert kept.layer_overrides.get("lm_head") is None


def test_quantize_fp16_lm_head_roundtrip():
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")  # noqa: F841
    from tensorrt_edgellm.checkpoint.repacking import (
        decode_modelopt_nvfp4, quantize_fp16_to_modelopt_nvfp4)

    torch.manual_seed(0)
    weight = torch.randn(128, 64, dtype=torch.float16)
    packed, scale, scale2 = quantize_fp16_to_modelopt_nvfp4(weight)
    assert packed.shape == (128, 32)
    assert scale.shape == (128, 4)
    recovered = torch.from_numpy(
        decode_modelopt_nvfp4(packed, scale, scale2, group_size=16))
    # Group-16 E2M1 is coarse; check the reconstruction stays close.
    rel = (recovered - weight.float()).abs() / weight.float().abs().clamp_min(
        1e-3)
    assert float(rel.median()) < 0.15


def test_repack_nvfp4_a16_gated_moe_shapes():
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import (
        quantize_fp16_to_modelopt_nvfp4,
        repack_nvfp4_a16_marlin_gated_moe_experts)

    torch.manual_seed(1)
    hidden, inter, experts = 128, 128, 2
    gate, up, down = [], [], []
    for _ in range(experts):
        g = quantize_fp16_to_modelopt_nvfp4(
            torch.randn(inter, hidden, dtype=torch.float16))
        u = quantize_fp16_to_modelopt_nvfp4(
            torch.randn(inter, hidden, dtype=torch.float16))
        # Plugin takes one FC1 global; keep gate/up identical.
        u = (u[0], u[1], g[2])
        d = quantize_fp16_to_modelopt_nvfp4(
            torch.randn(hidden, inter, dtype=torch.float16))
        gate.append(g)
        up.append(u)
        down.append(d)
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = (
        repack_nvfp4_a16_marlin_gated_moe_experts(
            [t[0] for t in gate], [t[1] for t in gate], [t[2] for t in gate],
            [t[0] for t in up], [t[1] for t in up], [t[2] for t in up],
            [t[0] for t in down], [t[1] for t in down], [t[2] for t in down],
            inter))
    assert tuple(fc1_q.shape) == (experts, hidden // 16, 8 * 2 * inter)
    assert tuple(fc1_s.shape) == (experts, hidden // 16, 2 * inter)
    assert tuple(fc1_g.shape) == (experts, )
    assert tuple(fc2_q.shape) == (experts, inter // 16, 8 * hidden)
    assert tuple(fc2_g.shape) == (experts, )
