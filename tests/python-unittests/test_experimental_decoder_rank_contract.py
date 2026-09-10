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
"""Rank propagation contracts for token-major experimental decoders."""

from types import SimpleNamespace

import pytest

trt = pytest.importorskip("tensorrt")

from experimental.builder.models.dflash.modeling_dflash_draft import \
    DFlashTargetProjection
from experimental.builder.models.dspark.modeling_dspark_draft import \
    DSparkTargetProjection
from experimental.builder.models.phi4mm.modeling_phi4mm_text import \
    Phi4MultimodalMLP
from experimental.builder.ops import functional as F
from experimental.builder.ops.functional import core as functional_core
from experimental.builder.ops.linear import Linear
from experimental.builder.ops.module import BuildContext, BuildOptions
from experimental.builder.ops.normalization import RMSNorm


class _Tensor:

    def __init__(self, ndim):
        self.ndim = ndim
        self.dtype = trt.float16


class _SliceTensor(_Tensor):

    def __init__(self, ndim, ranks):
        super().__init__(ndim)
        self.ranks = ranks

    def slice_last_dim(self, _offset, _size, rank):
        self.ranks.append(rank)
        return self

    def activation(self, _name):
        return self

    def __mul__(self, _other):
        return self


def _context():
    weights = SimpleNamespace(
        module_quant_type=lambda *args, **kwargs: "fp16",
        linear_descriptor=lambda *args, **kwargs: SimpleNamespace(
            in_features=16, out_features=16),
        shard_linear=lambda descriptor, *args: descriptor,
        fp16_parameter=lambda *args: SimpleNamespace(shape=(16, )),
        f16=lambda *args: None,
        linear_adapter=lambda *args: None,
    )
    cfg = SimpleNamespace(tp_size=1,
                          tp_rank=0,
                          rms_norm_eps=1e-6,
                          tie_word_embeddings=False)
    return BuildContext(net=SimpleNamespace(),
                        cfg=cfg,
                        weights=weights,
                        options=BuildOptions(),
                        bundle=SimpleNamespace(),
                        args=SimpleNamespace())


def test_linear_uses_the_actual_activation_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_from_weights",
        lambda hidden, descriptor, rank, name="": ranks.append(rank) or hidden)

    Linear(_context(), "model.layers.0.self_attn.q_proj")(_Tensor(2))

    assert ranks == [2]


def test_linear_preserves_explicit_encoder_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_from_weights",
        lambda hidden, descriptor, rank, name="": ranks.append(rank) or hidden)

    Linear(_context(), "visual.proj", rank=3,
           tensor_parallel=False)(_Tensor(2))

    assert ranks == [3]


def test_direct_weight_projection_infers_token_major_rank(monkeypatch):
    ranks = []
    backend = SimpleNamespace(linear_from_weights=lambda hidden, weights, rank,
                              name="": ranks.append(rank) or hidden)
    monkeypatch.setattr(functional_core, "current_net", lambda: backend)
    monkeypatch.setattr(functional_core, "tensor", lambda value: value)

    F.linear_from_weights(_Tensor(2), object(), name="fused_qkv")

    assert ranks == [2]


def test_direct_weight_projection_preserves_explicit_rank(monkeypatch):
    ranks = []
    backend = SimpleNamespace(linear_from_weights=lambda hidden, weights, rank,
                              name="": ranks.append(rank) or hidden)
    monkeypatch.setattr(functional_core, "current_net", lambda: backend)
    monkeypatch.setattr(functional_core, "tensor", lambda value: value)

    F.linear_from_weights(_Tensor(2), object(), rank=3, name="vision_proj")

    assert ranks == [3]


def test_rmsnorm_uses_the_actual_activation_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F, "rms_norm", lambda hidden, weight, eps, rank, **kwargs: ranks.
        append(rank) or hidden)

    RMSNorm(_context(), "model.layers.0.input_layernorm")(_Tensor(2))

    assert ranks == [2]


def test_rmsnorm_preserves_explicit_encoder_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F, "rms_norm", lambda hidden, weight, eps, rank, **kwargs: ranks.
        append(rank) or hidden)

    RMSNorm(_context(), "visual.norm", rank=3)(_Tensor(2))

    assert ranks == [3]


@pytest.mark.parametrize("projection",
                         (DFlashTargetProjection, DSparkTargetProjection))
def test_target_projection_uses_token_major_rank(monkeypatch, projection):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_f32_from_weights",
        lambda hidden, descriptor, name, rank=3: ranks.append(rank) or hidden)

    projection(_context(), "fc")(_Tensor(2))

    assert ranks == [2]


def test_phi4_multimodal_mlp_slices_token_major_activations():
    ranks = []
    mlp = object.__new__(Phi4MultimodalMLP)
    mlp.ctx = SimpleNamespace(
        cfg=SimpleNamespace(intermediate_size=16, hidden_act="silu"))
    mlp.gate_up_proj = lambda _hidden: _SliceTensor(2, ranks)
    mlp.down_proj = lambda hidden: hidden

    mlp.forward(_Tensor(2))

    assert ranks == [2, 2]
