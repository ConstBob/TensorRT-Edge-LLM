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
"""Focused tests for Qwen3 NVFP4 MoE repacking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorrt_edgellm.checkpoint import repacking  # noqa: E402
from tensorrt_edgellm.models.linear import NVFP4Linear  # noqa: E402


def _make_expert(hidden_size: int,
                 moe_inter_size: int,
                 group_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        gate_proj=NVFP4Linear(hidden_size, moe_inter_size, group_size),
        up_proj=NVFP4Linear(hidden_size, moe_inter_size, group_size),
        down_proj=NVFP4Linear(moe_inter_size, hidden_size, group_size),
    )


def _dense_values(rows: int, cols: int, offset: float) -> np.ndarray:
    return (np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) +
            offset)


def _patch_decode(monkeypatch: pytest.MonkeyPatch, expert: SimpleNamespace,
                  gate_dense: np.ndarray, up_dense: np.ndarray,
                  down_dense: np.ndarray) -> None:
    dense_by_weight_id = {
        id(expert.gate_proj.weight): gate_dense,
        id(expert.up_proj.weight): up_dense,
        id(expert.down_proj.weight): down_dense,
    }

    def fake_decode_modelopt_nvfp4(weight, weight_scale, weight_scale_2,
                                   group_size):
        del weight_scale, weight_scale_2, group_size
        return dense_by_weight_id[id(weight)]

    monkeypatch.setattr(repacking, "decode_modelopt_nvfp4",
                        fake_decode_modelopt_nvfp4)


def test_qwen3_fc1_interleaves_64_row_up_gate_chunks(
        monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_size = 16
    moe_inter_size = 128
    expert = _make_expert(hidden_size, moe_inter_size)

    gate_dense = _dense_values(moe_inter_size, hidden_size, offset=10_000.0)
    up_dense = _dense_values(moe_inter_size, hidden_size, offset=20_000.0)
    down_dense = _dense_values(hidden_size, moe_inter_size, offset=30_000.0)
    _patch_decode(monkeypatch, expert, gate_dense, up_dense, down_dense)

    captured_dense = []

    def fake_pack_nvfp4_moe_weight(dense_w_mk, group_size=16):
        del group_size
        captured_dense.append(dense_w_mk.copy())
        return (torch.zeros((dense_w_mk.shape[0], dense_w_mk.shape[1] // 2),
                            dtype=torch.int8),
                torch.zeros((1, 1, 1, 1, 1), dtype=torch.int8))

    monkeypatch.setattr(repacking, "_pack_nvfp4_moe_weight",
                        fake_pack_nvfp4_moe_weight)

    repacking.repack_nvfp4_qwen3_moe_experts([expert], hidden_size,
                                                     moe_inter_size)

    expected_fc1 = np.stack(
        [
            up_dense.reshape(2, 64, hidden_size),
            gate_dense.reshape(2, 64, hidden_size),
        ],
        axis=1,
    ).reshape(2 * moe_inter_size, hidden_size)

    assert len(captured_dense) == 2
    np.testing.assert_array_equal(captured_dense[0], expected_fc1)


def test_qwen3_fc1_requires_64_row_aligned_intermediate_size(
        monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_size = 16
    moe_inter_size = 96
    expert = _make_expert(hidden_size, moe_inter_size)

    gate_dense = _dense_values(moe_inter_size, hidden_size, offset=10_000.0)
    up_dense = _dense_values(moe_inter_size, hidden_size, offset=20_000.0)
    down_dense = _dense_values(hidden_size, moe_inter_size, offset=30_000.0)
    _patch_decode(monkeypatch, expert, gate_dense, up_dense, down_dense)

    def fail_pack_nvfp4_moe_weight(dense_w_mk, group_size=16):
        del dense_w_mk, group_size
        pytest.fail("packing should not run for non-64-aligned FC1")

    monkeypatch.setattr(repacking, "_pack_nvfp4_moe_weight",
                        fail_pack_nvfp4_moe_weight)

    with pytest.raises(ValueError, match="multiple of 64"):
        repacking.repack_nvfp4_qwen3_moe_experts([expert],
                                                         hidden_size,
                                                         moe_inter_size)
