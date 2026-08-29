# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
QsaAttentionPlugin (Qwen Sparse Attention, prefill v1) unit tests.

Two layers of coverage:

1. ``QsaTorchReference`` self-tests (pure torch, CPU, no TensorRT / CUDA):
   the dense-equivalence invariant (for S <= 2051 the sparse selection is
   exactly the full causal prefix, so sparse output == causal SDPA) and the
   structural invariants of the produced index lists.

2. Plugin tests driving the TRT ``QsaAttentionPlugin`` through the
   ``PluginRunner`` harness on CUDA: dense-equivalence prefill, sparse
   prefill vs the full FP32 QSA torch reference, paged-KV write-through,
   decode-shape rejection, and determinism.

Algorithm contract under test (frozen with the C++ plugin, M3):

* Main path: split packed qkv [B,S,(Hq+2*Hkv)*D] -> per-head Gemma qk-norm
  (``normalize(x) * gamma`` with gamma PRE-FOLDED as (1+w)) -> partial neox
  RoPE (rotary dim 64 of head size 256, theta=1e7) -> paged KV write
  (roped K, raw V).
* Indexer (NH=4, D_idx=128, ratio=4, budget=512, width=2051): q Gemma-norm
  with RAW w (kernel computes 1+w internally) + rope@t; compressed keys per
  complete block g: fp32 mean of 4 raw keys (fixed order) -> cast fp16 ->
  Gemma-norm -> rope@4g; scores = sum_h relu(q_h . kbar_g) / sqrt(128) in
  FP32, visible g < (t+1)//4; top-512 blocks -> expand blk*4+[0..3] ->
  ALWAYS append tail [4*n_vis .. t] ((t+1)%4 tokens) -> -1 pad to 2051.
* Sparse GQA: attend only listed tokens; FP32 softmax at 1/sqrt(256) applied
  on FP32 scores after the QK dot; -1 masked; all-invalid row -> zeros; rows
  past context_lengths -> exact zeros.

Top-k tie-break caveat: block scores are compared in FP32, and the plugin's
warp-reduction accumulation order differs from torch's, so blocks whose
scores sit exactly at the top-512 boundary may be swapped between plugin and
reference for S > 2051. All tests use continuous random inputs (ties are
measure-zero) and the sparse comparisons use fp16 tolerances plus a cosine
gate instead of exact index-list equality.

Run:
    python3 -m pytest tests/python-unittests/test_qsa_attention_plugin.py -v
Reference-only (no GPU needed):
    python3 -m pytest tests/python-unittests/test_qsa_attention_plugin.py -k reference -v
"""

from __future__ import annotations

import math

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, _device_sm, _get_logger,
                              assert_close, cosine_sim, find_plugin_library,
                              load_edgellm_plugins, pf_float32, pf_int32,
                              poison_padding)

# The reference self-tests only need torch (CPU is fine); the plugin tests
# additionally need TensorRT + CUDA (DEPENDENCIES_AVAILABLE from the harness).
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt

DEV = "cuda"

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason=f"torch not available: {TORCH_IMPORT_ERROR}")
requires_gpu = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

# --------------------------------------------------------------------------- #
# Frozen QSA model constants (Qwen3.8-Flash-Next qwen4_exp)
# --------------------------------------------------------------------------- #
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_SIZE = 256
INDEXER_N_HEADS = 4
INDEXER_HEAD_DIM = 128
INDEXER_BUDGET = 512
INDEXER_COMPRESS_RATIO = 4
# Widest possible index row: budget complete blocks + a (ratio-1)-token tail.
INDEX_WIDTH = INDEXER_BUDGET * INDEXER_COMPRESS_RATIO + (
    INDEXER_COMPRESS_RATIO - 1)  # 2051
ROTARY_DIM = 64  # partial rope: first 64 dims of each head
ROPE_THETA = 1.0e7
RMS_NORM_EPS = 1e-6
# Tokens per page of the paged-KV pool. Must match kTOKENS_PER_PAGE.
PAGE_SIZE = 128

QKV_PACKED = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_SIZE  # 7168
INDEX_QK_PACKED = (INDEXER_N_HEADS + 1) * INDEXER_HEAD_DIM  # 640

# QSA CuTe-DSL sparse kernel variants are registered for these SMs only.
QSA_SMS = frozenset({110, 120, 121})


# --------------------------------------------------------------------------- #
# RoPE table + reference math helpers
# --------------------------------------------------------------------------- #
def make_qsa_rope_table(max_pos: int, device="cpu") -> "torch.Tensor":
    """Build the shared QSA cos/sin table [max_pos, 64] (FP32).

    Layout matches the plugin binding (and test_attention_plugin's
    ``_make_rope``): cos in [0:32], sin in [32:64];
    ``inv_freq[i] = theta**(-i/half)`` with theta=1e7, half=32.
    """
    half = ROTARY_DIM // 2
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)[:, None]
    inv_freq = 1.0 / (ROPE_THETA**(torch.arange(
        0, half, dtype=torch.float32, device=device) / half))[None, :]
    ang = pos * inv_freq  # [max_pos, half]
    table = torch.zeros((max_pos, ROTARY_DIM),
                        dtype=torch.float32,
                        device=device)
    table[:, :half] = torch.cos(ang)
    table[:, half:] = torch.sin(ang)
    return table


def apply_partial_rope(x32: "torch.Tensor",
                       table: "torch.Tensor",
                       positions: "torch.Tensor",
                       interleaved: bool = False) -> "torch.Tensor":
    """Partial RoPE on the first ROTARY_DIM dims of the last axis (FP32).

    ``x32``: [N, ..., Dh] with the position axis first; ``positions``: [N]
    int64; ``table``: [max_pos, 64] from :func:`make_qsa_rope_table`.

    ``interleaved=False`` (default) is the neox rotate-half convention:
    pairs are (dim i, dim i+32) within the first 64 dims. ``interleaved=True``
    pairs (dim 2i, dim 2i+1) — kept as a one-line flip in case the HF
    qwen4_exp cross-check pins the other layout.
    """
    half = ROTARY_DIM // 2
    cos = table[positions, :half].float()
    sin = table[positions, half:].float()
    bshape = [x32.shape[0]] + [1] * (x32.dim() - 2) + [half]
    cos = cos.reshape(bshape)
    sin = sin.reshape(bshape)
    rot = x32[..., :ROTARY_DIM]
    if interleaved:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
    else:
        x1, x2 = rot[..., :half], rot[..., half:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    if interleaved:
        roped = torch.stack((r1, r2), dim=-1).reshape(rot.shape)
    else:
        roped = torch.cat((r1, r2), dim=-1)
    return torch.cat((roped, x32[..., ROTARY_DIM:]), dim=-1)


def gemma_rms_norm(x32: "torch.Tensor", gamma32: "torch.Tensor",
                   eps: float) -> "torch.Tensor":
    """FP32 RMSNorm ``x * rsqrt(mean(x^2) + eps) * gamma`` (llama convention).

    The Gemma ``(1+w)`` semantics live in the caller: main-path gammas arrive
    pre-folded, indexer gammas arrive raw and the caller adds the 1.
    """
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gamma32


# --------------------------------------------------------------------------- #
# Pure-torch FP32 QSA reference
# --------------------------------------------------------------------------- #
class QsaTorchReference:
    """FP32 torch reference for the full QSA plugin math (prefill).

    Follows the frozen cast chain: indexer q/kbar are stored through an FP16
    cast (like the CUDA kernels) before the FP32 score dot; compressed keys
    take the fixed-order fp32 mean -> fp16 cast -> Gemma-norm -> rope@4g
    chain. The main path keeps FP32 throughout (the plugin's fp16 rounding is
    covered by the comparison tolerances).
    """

    def __init__(self,
                 num_q_heads: int = NUM_Q_HEADS,
                 num_kv_heads: int = NUM_KV_HEADS,
                 head_size: int = HEAD_SIZE,
                 indexer_n_heads: int = INDEXER_N_HEADS,
                 indexer_head_dim: int = INDEXER_HEAD_DIM,
                 indexer_budget: int = INDEXER_BUDGET,
                 indexer_compress_ratio: int = INDEXER_COMPRESS_RATIO,
                 attention_scale: float = 0.0,
                 rms_norm_eps: float = RMS_NORM_EPS,
                 interleaved: bool = False):
        assert num_q_heads % num_kv_heads == 0
        self.hq = num_q_heads
        self.hkv = num_kv_heads
        self.d = head_size
        self.nh = indexer_n_heads
        self.di = indexer_head_dim
        self.budget = indexer_budget
        self.ratio = indexer_compress_ratio
        self.width = indexer_budget * indexer_compress_ratio + (
            indexer_compress_ratio - 1)
        # attention_scale == 0.0 selects the plugin default 1/sqrt(head_size).
        self.scale = (attention_scale if attention_scale != 0.0 else 1.0 /
                      math.sqrt(head_size))
        self.eps = rms_norm_eps
        self.interleaved = interleaved

    # -- main path ---------------------------------------------------------- #
    def main_qkv(self, qkv, q_gamma_folded, k_gamma_folded, table):
        """Split packed qkv, qk-norm (pre-folded gammas), partial rope.

        qkv [B,S,(Hq+2*Hkv)*D] (any float dtype); gammas [D] hold (1+w).
        Returns FP32 (q_roped, k_roped, v) each [B,S,H,D].
        """
        b, s, _ = qkv.shape
        qh = self.hq * self.d
        kvh = self.hkv * self.d
        q = qkv[..., :qh].reshape(b, s, self.hq, self.d).float()
        k = qkv[..., qh:qh + kvh].reshape(b, s, self.hkv, self.d).float()
        v = qkv[..., qh + kvh:].reshape(b, s, self.hkv, self.d).float()
        q = gemma_rms_norm(q, q_gamma_folded.float(), self.eps)
        k = gemma_rms_norm(k, k_gamma_folded.float(), self.eps)
        positions = torch.arange(s, device=qkv.device, dtype=torch.long)
        q = torch.stack([
            apply_partial_rope(q[bi], table, positions, self.interleaved)
            for bi in range(b)
        ])
        k = torch.stack([
            apply_partial_rope(k[bi], table, positions, self.interleaved)
            for bi in range(b)
        ])
        return q, k, v

    # -- indexer ------------------------------------------------------------ #
    def build_indices(self, index_qk, iq_gamma_raw, ik_gamma_raw, table,
                      context_lengths):
        """Full indexer pipeline -> int32 index lists [B, S, width] (-1 pad).

        ``iq_gamma_raw`` / ``ik_gamma_raw`` [D_idx] hold RAW w; the effective
        Gemma gamma is (1 + w) computed here in FP32 (as the CUDA kernel does).
        Rows past a request's context length are all -1.
        """
        b, s, _ = index_qk.shape
        dev = index_qk.device
        out = torch.full((b, s, self.width), -1, dtype=torch.int32, device=dev)
        iq_gamma = 1.0 + iq_gamma_raw.float()
        ik_gamma = 1.0 + ik_gamma_raw.float()
        ar_ratio = torch.arange(self.ratio, device=dev, dtype=torch.int64)
        for bi in range(b):
            length = int(context_lengths[bi])
            if length <= 0:
                continue
            q_raw = index_qk[bi, :length, :self.nh * self.di].reshape(
                length, self.nh, self.di).float()
            k_raw = index_qk[bi, :length,
                             self.nh * self.di:].reshape(length,
                                                         self.di).float()
            positions = torch.arange(length, device=dev, dtype=torch.long)
            # q: gemma-norm (1 + w_raw) -> rope@t -> fp16 storage cast.
            q_idx = gemma_rms_norm(q_raw, iq_gamma, self.eps)
            q_idx = apply_partial_rope(q_idx, table, positions,
                                       self.interleaved)
            q_idx = q_idx.to(torch.float16).float()
            num_blocks = length // self.ratio  # complete blocks only
            n_vis = (positions + 1) // self.ratio  # [length]
            top_idx = None
            if num_blocks > 0:
                # Compressed keys: FIXED-ORDER fp32 mean (((k0+k1)+k2)+k3)*0.25
                # -> cast fp16 -> gemma-norm -> rope@4g (bit-compat-critical).
                kg = k_raw[:num_blocks * self.ratio].reshape(
                    num_blocks, self.ratio, self.di)
                mean = kg[:, 0]
                for j in range(1, self.ratio):
                    mean = mean + kg[:, j]
                mean = mean * (1.0 / self.ratio)
                mean = mean.to(torch.float16).float()
                kbar = gemma_rms_norm(mean, ik_gamma, self.eps)
                block_pos = torch.arange(
                    num_blocks, device=dev, dtype=torch.long) * self.ratio
                kbar = apply_partial_rope(kbar, table, block_pos,
                                          self.interleaved)
                kbar = kbar.to(torch.float16).float()
                # scores[t, g] = sum_h relu(q[t,h] . kbar[g]) / sqrt(D_idx):
                # relu per head, THEN sum over heads, THEN scale — all FP32.
                dots = torch.einsum("thd,gd->thg", q_idx, kbar)
                scores = torch.relu(dots).sum(dim=1) / math.sqrt(self.di)
                visible = (torch.arange(num_blocks, device=dev)[None, :]
                           < n_vis[:, None])
                scores = scores.masked_fill(~visible, float("-inf"))
                k_top = min(self.budget, num_blocks)
                # topk sorts descending; -inf (masked) blocks rank after every
                # finite score, so slicing to n_sel keeps only visible blocks.
                top_idx = torch.topk(scores, k_top, dim=-1).indices
            for t in range(length):
                nv = int(n_vis[t])
                n_sel = min(self.budget, nv)
                parts = []
                if n_sel > 0:
                    blocks = top_idx[t, :n_sel].to(torch.int64)
                    parts.append(
                        (blocks[:, None] * self.ratio + ar_ratio).reshape(-1))
                tail_start = nv * self.ratio
                if tail_start <= t:  # (t+1) % ratio tokens, ALWAYS appended
                    parts.append(
                        torch.arange(tail_start,
                                     t + 1,
                                     device=dev,
                                     dtype=torch.int64))
                row = torch.cat(parts) if parts else None
                if row is not None and row.numel() > 0:
                    out[bi, t, :row.numel()] = row.to(torch.int32)
        return out

    # -- sparse attention ---------------------------------------------------- #
    def sparse_attention(self,
                         q,
                         k,
                         v,
                         indices,
                         context_lengths,
                         row_chunk: int = 64):
        """Sparse GQA over the listed tokens only (no causal mask in here).

        q [B,S,Hq,D], k/v [B,S,Hkv,D] FP32; indices [B,S,width] int32 (-1 =
        masked). FP32 softmax; scale applied on the FP32 scores after the QK
        dot. All-invalid rows and rows past context_lengths -> exact zeros.
        """
        b, s, hq, d = q.shape
        group = self.hq // self.hkv
        out = torch.zeros((b, s, hq, d), dtype=torch.float32, device=q.device)
        for bi in range(b):
            length = int(context_lengths[bi])
            for r0 in range(0, length, row_chunk):
                r1 = min(r0 + row_chunk, length)
                idx = indices[bi, r0:r1].to(torch.int64)  # [R, W]
                valid = idx >= 0
                idx_cl = idx.clamp(min=0)
                kg = k[bi][idx_cl]  # [R, W, Hkv, D]
                vg = v[bi][idx_cl]
                qc = q[bi, r0:r1].reshape(r1 - r0, self.hkv, group, d)
                scores = torch.einsum("rkgd,rwkd->rkgw", qc, kg) * self.scale
                scores = scores.masked_fill(~valid[:, None, None, :],
                                            float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                # All-invalid rows softmax to NaN -> force exact zero output.
                probs = torch.nan_to_num(probs, nan=0.0)
                out[bi, r0:r1] = torch.einsum("rkgw,rwkd->rkgd", probs,
                                              vg).reshape(r1 - r0, hq, d)
        return out

    def dense_attention(self, q, k, v, context_lengths):
        """Causal dense SDPA on the same post-norm/rope q/k/v (FP32).

        The dense-equivalence variant: for context lengths <= width the
        sparse selection is exactly the causal prefix, so this must match
        :meth:`sparse_attention`. Padding rows -> exact zeros.
        """
        b, s, hq, d = q.shape
        group = self.hq // self.hkv
        out = torch.zeros((b, s, hq, d), dtype=torch.float32, device=q.device)
        for bi in range(b):
            length = int(context_lengths[bi])
            if length <= 0:
                continue
            qb = q[bi, :length].permute(1, 0, 2)  # [Hq, L, D]
            kb = k[bi, :length].permute(1, 0, 2).repeat_interleave(group,
                                                                   dim=0)
            vb = v[bi, :length].permute(1, 0, 2).repeat_interleave(group,
                                                                   dim=0)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * self.scale
            causal = torch.ones((length, length),
                                dtype=torch.bool,
                                device=q.device).tril()
            scores = scores.masked_fill(~causal[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[bi, :length] = torch.matmul(probs, vb).permute(1, 0, 2)
        return out

    # -- full forward -------------------------------------------------------- #
    def forward(self, qkv, index_qk, q_gamma_folded, k_gamma_folded,
                iq_gamma_raw, ik_gamma_raw, table, context_lengths):
        """Full QSA forward. Returns (out, indices, k_roped, v) — k/v FP32
        [B,S,Hkv,D] as the plugin stores them (roped K, raw V)."""
        q, k, v = self.main_qkv(qkv, q_gamma_folded, k_gamma_folded, table)
        indices = self.build_indices(index_qk, iq_gamma_raw, ik_gamma_raw,
                                     table, context_lengths)
        out = self.sparse_attention(q, k, v, indices, context_lengths)
        return out, indices, k, v

    def dense_forward(self, qkv, q_gamma_folded, k_gamma_folded, table,
                      context_lengths):
        """Causal-dense variant of :meth:`forward` (no indexer)."""
        q, k, v = self.main_qkv(qkv, q_gamma_folded, k_gamma_folded, table)
        return self.dense_attention(q, k, v, context_lengths)


# --------------------------------------------------------------------------- #
# Shared input builders
# --------------------------------------------------------------------------- #
def _make_gammas(gen, device="cpu"):
    """Random Gemma weights. Returns (q_folded, k_folded, iq_raw, ik_raw) FP16.

    Main-path gammas are PRE-FOLDED (1 + w) — exactly what the plugin (and
    reference) multiply by. Indexer gammas are RAW w — the kernel (and
    reference) compute (1 + w) internally.
    """
    w_q = 0.25 * torch.randn(HEAD_SIZE, generator=gen, dtype=torch.float32)
    w_k = 0.25 * torch.randn(HEAD_SIZE, generator=gen, dtype=torch.float32)
    w_iq = 0.25 * torch.randn(
        INDEXER_HEAD_DIM, generator=gen, dtype=torch.float32)
    w_ik = 0.25 * torch.randn(
        INDEXER_HEAD_DIM, generator=gen, dtype=torch.float32)
    return ((1.0 + w_q).to(torch.float16).to(device), (1.0 + w_k).to(
        torch.float16).to(device), w_iq.to(torch.float16).to(device),
            w_ik.to(torch.float16).to(device))


def _make_activations(batch, seq, gen, device="cpu"):
    """Random packed qkv / index_qk (FP16), generated on CPU for determinism."""
    qkv = torch.randn((batch, seq, QKV_PACKED),
                      generator=gen,
                      dtype=torch.float32).to(device).to(torch.float16)
    index_qk = torch.randn((batch, seq, INDEX_QK_PACKED),
                           generator=gen,
                           dtype=torch.float32).to(device).to(torch.float16)
    return qkv, index_qk


# =========================================================================== #
# A/B. Reference self-tests — pure torch on CPU, no plugin / CUDA required
# =========================================================================== #
@requires_torch
@pytest.mark.parametrize("seq_lens", [[1], [4], [33], [129, 87]],
                         ids=lambda sl: "L" + "_".join(map(str, sl)))
def test_reference_dense_equivalence(seq_lens):
    """Invariant: for L <= 2051, sparse reference == causal SDPA reference.

    n_vis = (t+1)//4 <= 512 for every t < 2051, so top-512 keeps ALL visible
    blocks and expansion + tail reproduce exactly the causal prefix {0..t}.
    """
    torch.manual_seed(20260829)
    gen = torch.Generator().manual_seed(20260829)
    batch = len(seq_lens)
    seq = max(seq_lens)
    ref = QsaTorchReference()
    q_g, k_g, iq_g, ik_g = _make_gammas(gen)
    qkv, index_qk = _make_activations(batch, seq, gen)
    ctx = torch.tensor(seq_lens, dtype=torch.int32)
    table = make_qsa_rope_table(max(seq, 64))

    sparse_out, indices, _, _ = ref.forward(qkv, index_qk, q_g, k_g, iq_g,
                                            ik_g, table, ctx)
    dense_out = ref.dense_forward(qkv, q_g, k_g, table, ctx)

    # The token sets are identical; only the softmax/matmul accumulation
    # order differs (the sparse list is unsorted), hence the tiny fp32 slack.
    torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-5)

    # Short-seq index rows must be exactly the causal prefix (as a set).
    for bi, length in enumerate(seq_lens):
        for t in (0, length // 2, length - 1):
            row = indices[bi, t]
            valid = row[row >= 0].tolist()
            assert sorted(valid) == list(range(t + 1)), \
                f"row (b={bi}, t={t}) is not the causal prefix"


@requires_torch
def test_reference_index_list_invariants():
    """Structural invariants of the index lists, incl. the sparse regime.

    Uses S=2178 (S % 4 == 2) so rows cover n_vis <= 512 (dense-equivalent)
    and n_vis > 512 (true top-k), plus a short ragged second request.
    """
    gen = torch.Generator().manual_seed(4242)
    batch_lengths = [2178, 517]
    seq = max(batch_lengths)
    ref = QsaTorchReference()
    _, _, iq_g, ik_g = _make_gammas(gen)
    _, index_qk = _make_activations(len(batch_lengths), seq, gen)
    ctx = torch.tensor(batch_lengths, dtype=torch.int32)
    table = make_qsa_rope_table(seq)

    indices = ref.build_indices(index_qk, iq_g, ik_g, table, ctx)

    assert indices.shape == (len(batch_lengths), seq, INDEX_WIDTH)
    assert INDEX_WIDTH == 2051

    ratio, budget = INDEXER_COMPRESS_RATIO, INDEXER_BUDGET
    for bi, length in enumerate(batch_lengths):
        # Padding rows are all -1.
        assert (indices[bi, length:] == -1).all(), "padding rows must be -1"
        sample_rows = sorted(
            set([0, 1, 2, 3, 4, 7, length // 3, length // 2, length - 1] +
                list(range(2044, min(length, 2060)))))
        for t in sample_rows:
            row = indices[bi, t]
            valid = row[row >= 0]
            n_vis = (t + 1) // ratio
            n_sel = min(budget, n_vis)
            expected_count = n_sel * ratio + (t + 1) - n_vis * ratio
            # Valid entries are packed to the front; the rest is -1 padding.
            assert (row[expected_count:] == -1).all(), \
                f"(b={bi}, t={t}) expected -1 padding after {expected_count}"
            assert valid.numel() == expected_count, \
                f"(b={bi}, t={t}) count {valid.numel()} != {expected_count}"
            # Distinct, in-range (< t+1 keeps causality inside the list).
            assert valid.unique().numel() == valid.numel(), \
                f"(b={bi}, t={t}) duplicate token indices"
            assert int(valid.min()) >= 0 and int(valid.max()) <= t, \
                f"(b={bi}, t={t}) token index out of causal range"
            # Tail [4*n_vis .. t] is ALWAYS present ((t+1)%4 tokens).
            tail = set(range(n_vis * ratio, t + 1))
            assert tail.issubset(set(valid.tolist())), \
                f"(b={bi}, t={t}) tail tokens missing"
            # Selected blocks are expanded as complete runs of 4.
            block_tokens = [int(x) for x in valid if x < n_vis * ratio]
            blocks = set(tok // ratio for tok in block_tokens)
            assert len(block_tokens) == ratio * len(blocks), \
                f"(b={bi}, t={t}) partial block expansion"
            assert len(blocks) == n_sel
            # Short-seq rows are exactly the causal prefix.
            if n_vis <= budget:
                assert sorted(valid.tolist()) == list(range(t + 1)), \
                    f"(b={bi}, t={t}) short row is not the causal prefix"


@requires_torch
def test_reference_sparse_regime_differs_from_dense():
    """Sanity: past 2051 tokens the sparse output actually drops tokens
    (guards against a reference that silently stays dense)."""
    gen = torch.Generator().manual_seed(77)
    length = 2100  # n_vis = 525 > 512 for the last rows
    ref = QsaTorchReference()
    _, _, iq_g, ik_g = _make_gammas(gen)
    _, index_qk = _make_activations(1, length, gen)
    ctx = torch.tensor([length], dtype=torch.int32)
    table = make_qsa_rope_table(length)
    indices = ref.build_indices(index_qk, iq_g, ik_g, table, ctx)
    t = length - 1
    valid = indices[0, t][indices[0, t] >= 0]
    n_vis = (t + 1) // INDEXER_COMPRESS_RATIO
    assert n_vis > INDEXER_BUDGET
    assert valid.numel() == INDEXER_BUDGET * INDEXER_COMPRESS_RATIO + (
        t + 1) % INDEXER_COMPRESS_RATIO
    assert valid.numel() < t + 1  # tokens were dropped


# =========================================================================== #
# C. Plugin tests (TensorRT engine on CUDA)
# =========================================================================== #
def _require_qsa_plugin():
    """Skip (not fail) when the QSA plugin cannot run here.

    Unlike the mature plugins (which fail loudly on a missing build), the QSA
    C++ plugin (M3) lands in parallel with these tests, so a missing library
    or unregistered creator is an expected state and skips cleanly.
    """
    if _device_sm() not in QSA_SMS:
        pytest.skip(f"QSA kernels not registered for SM{_device_sm()} "
                    f"(supported: {sorted(QSA_SMS)})")
    if find_plugin_library() is None:
        pytest.skip("libNvInfer_edgellm_plugin.so not built")
    try:
        load_edgellm_plugins(_get_logger(False))
    except RuntimeError as e:
        pytest.skip(f"could not load Edge-LLM plugins: {e}")
    creator = trt.get_plugin_registry().get_creator("QsaAttentionPlugin", "1",
                                                    "")
    if creator is None:
        pytest.skip("QsaAttentionPlugin not registered in this plugin build")


class QsaPluginRunner:
    """Builds + runs a QsaAttentionPlugin engine for prefill testing.

    The 11-input plugin contract (all required):
    qkv, index_qk, past_key_value (paged pool [2, numPages, PAGE_SIZE, Hkv,
    D]), context_lengths, rope_rotary_cos_sin [1, maxPos, 64],
    kvcache_start_index (runtime shape [0] = prefill), kv_page_table
    [B, 2, mpps], and the four FP16 gamma constants (engine weights).
    """

    def __init__(self,
                 *,
                 batch_size: int,
                 seq_len: int,
                 max_position_embeddings: int,
                 q_gamma: "torch.Tensor",
                 k_gamma: "torch.Tensor",
                 iq_gamma: "torch.Tensor",
                 ik_gamma: "torch.Tensor",
                 attention_scale: float = 0.0,
                 rms_norm_eps: float = RMS_NORM_EPS,
                 flip_page_table: bool = False):
        self.batch = batch_size
        self.seq = seq_len
        self.mpe = max_position_embeddings
        self.flip_page_table = flip_page_table
        self.cap = -(-seq_len // PAGE_SIZE) * PAGE_SIZE
        self.mpps = self.cap // PAGE_SIZE
        self.num_pages = batch_size * self.mpps
        self._pool = None
        self._page_table = None

        pool_shape = (2, self.num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        rope_shape = (1, self.mpe, ROTARY_DIM)
        input_specs = [
            ("qkv", trt.float16, (-1, -1, QKV_PACKED)),
            ("index_qk", trt.float16, (-1, -1, INDEX_QK_PACKED)),
            ("past_key_value", trt.float16, (2, -1, PAGE_SIZE, NUM_KV_HEADS,
                                             HEAD_SIZE)),
            ("context_lengths", trt.int32, (-1, )),
            ("rope_rotary_cos_sin", trt.float32, rope_shape),
            ("kvcache_start_index", trt.int32, (-1, )),
            ("kv_page_table", trt.int32, (-1, 2, self.mpps)),
        ]
        profiles = {
            "qkv": ((1, 1, QKV_PACKED), (batch_size, seq_len, QKV_PACKED),
                    (batch_size, seq_len, QKV_PACKED)),
            "index_qk":
            ((1, 1, INDEX_QK_PACKED), (batch_size, seq_len, INDEX_QK_PACKED),
             (batch_size, seq_len, INDEX_QK_PACKED)),
            # The pool never resizes: numPages fixed per engine.
            "past_key_value": (pool_shape, pool_shape, pool_shape),
            "context_lengths": ((1, ), (batch_size, ), (batch_size, )),
            "rope_rotary_cos_sin": (rope_shape, rope_shape, rope_shape),
            # [0] is the prefill sentinel; max allows binding [B] so the
            # decode-shape rejection test exercises the runtime guard.
            "kvcache_start_index": ((0, ), (0, ), (batch_size, )),
            "kv_page_table": ((1, 2, self.mpps), (batch_size, 2, self.mpps),
                              (batch_size, 2, self.mpps)),
        }
        constant_specs = [
            ("q_norm_gamma", trt.float16, (HEAD_SIZE, ), q_gamma),
            ("k_norm_gamma", trt.float16, (HEAD_SIZE, ), k_gamma),
            ("indexer_q_norm_gamma", trt.float16, (INDEXER_HEAD_DIM, ),
             iq_gamma),
            ("indexer_k_norm_gamma", trt.float16, (INDEXER_HEAD_DIM, ),
             ik_gamma),
        ]
        # Frozen plugin input order (must match the C++ plugin contract).
        plugin_input_order = [
            "qkv", "index_qk", "past_key_value", "context_lengths",
            "rope_rotary_cos_sin", "kvcache_start_index", "kv_page_table",
            "q_norm_gamma", "k_norm_gamma", "indexer_q_norm_gamma",
            "indexer_k_norm_gamma"
        ]
        fields = [
            pf_int32("num_q_heads", NUM_Q_HEADS),
            pf_int32("num_kv_heads", NUM_KV_HEADS),
            pf_int32("head_size", HEAD_SIZE),
            pf_int32("indexer_n_heads", INDEXER_N_HEADS),
            pf_int32("indexer_head_dim", INDEXER_HEAD_DIM),
            pf_int32("indexer_budget", INDEXER_BUDGET),
            pf_int32("indexer_compress_ratio", INDEXER_COMPRESS_RATIO),
            pf_float32("attention_scale", attention_scale),
            pf_float32("rms_norm_eps", rms_norm_eps),
        ]
        self.runner = PluginRunner()
        self.runner.build(
            input_specs=input_specs,
            output_names=["attention_output", "present_key_value"],
            plugin_name="QsaAttentionPlugin",
            plugin_version="1",
            plugin_fields=fields,
            profiles=profiles,
            constant_specs=constant_specs,
            plugin_input_order=plugin_input_order,
            # Indexer workspace (sort buffers + [B,S,2051] outIdx) exceeds 1GB
            # only far past these test shapes; 2GB gives headroom.
            workspace_bytes=2 << 30,
        )

    def _get_pool(self):
        if self._pool is None:
            self._pool = torch.zeros(
                (2, self.num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_SIZE),
                dtype=torch.float16,
                device=DEV)
        return self._pool

    def _make_page_table(self, batch):
        if self._page_table is None or self._page_table.shape[0] != batch:
            k_ids = torch.arange(batch * self.mpps,
                                 dtype=torch.int32,
                                 device=DEV)
            if self.flip_page_table:
                k_ids = k_ids.flip(0)
            k_ids = k_ids.reshape(batch, self.mpps)
            self._page_table = torch.stack((k_ids, k_ids + self.num_pages),
                                           dim=1).contiguous()
        return self._page_table

    def run(self,
            qkv,
            index_qk,
            context_lengths,
            rope_table,
            kv_start_shape=(0, ),
            attention_output=None):
        """Execute one prefill. Returns the [B, S, Hq, D] FP16 output.

        ``kv_start_shape=(0,)`` binds the prefill sentinel; passing ``(B,)``
        exercises the decode-shape rejection path. The pool is zeroed before
        each run so KV comparisons never see stale pages.
        """
        batch, seq, _ = qkv.shape
        pool = self._get_pool()
        pool.zero_()
        page_table = self._make_page_table(batch)
        kv_start = torch.zeros((self.batch, ), dtype=torch.int32, device=DEV)
        attn_out = attention_output
        if attn_out is None:
            attn_out = torch.empty((batch, seq, NUM_Q_HEADS, HEAD_SIZE),
                                   dtype=torch.float16,
                                   device=DEV)
        tensors = {
            "qkv": qkv,
            "index_qk": index_qk,
            "past_key_value": pool,
            "context_lengths": context_lengths,
            "rope_rotary_cos_sin": rope_table,
            "kvcache_start_index": kv_start,
            "kv_page_table": page_table,
            "attention_output": attn_out,
            "present_key_value": pool,  # aliased in-place to the pool binding
        }
        self.runner.execute(tensors,
                            {"kvcache_start_index": tuple(kv_start_shape)})
        return attn_out

    def gather_kv(self, batch, length):
        """Gather the paged pool through the page table into token-major K/V.

        Returns (k, v) FP16 [batch, length, Hkv, D] — roped K and raw V as
        the plugin stores them.
        """
        pool = self._get_pool()
        page_table = self._make_page_table(batch)
        k = torch.zeros((batch, self.cap, NUM_KV_HEADS, HEAD_SIZE),
                        dtype=torch.float16,
                        device=DEV)
        v = torch.zeros_like(k)
        for bi in range(batch):
            for lp in range(self.mpps):
                k_page = int(page_table[bi, 0, lp])
                v_page = int(page_table[bi, 1, lp]) - self.num_pages
                begin = lp * PAGE_SIZE
                k[bi, begin:begin + PAGE_SIZE] = pool[0, k_page]
                v[bi, begin:begin + PAGE_SIZE] = pool[1, v_page]
        return k[:, :length], v[:, :length]


def _plugin_case(batch,
                 seq,
                 seed,
                 *,
                 attention_scale=0.0,
                 flip_page_table=False):
    """Build runner + reference + matched random inputs for one config."""
    gen = torch.Generator().manual_seed(seed)
    q_g, k_g, iq_g, ik_g = _make_gammas(gen, device=DEV)
    qkv, index_qk = _make_activations(batch, seq, gen, device=DEV)
    mpe = max(seq, ROTARY_DIM)
    table = make_qsa_rope_table(mpe, device=DEV)
    rope_binding = table[None]  # [1, mpe, 64]
    runner = QsaPluginRunner(batch_size=batch,
                             seq_len=seq,
                             max_position_embeddings=mpe,
                             q_gamma=q_g,
                             k_gamma=k_g,
                             iq_gamma=iq_g,
                             ik_gamma=ik_g,
                             attention_scale=attention_scale,
                             flip_page_table=flip_page_table)
    ref = QsaTorchReference(attention_scale=attention_scale)
    return runner, ref, qkv, index_qk, (q_g, k_g, iq_g, ik_g), table, \
        rope_binding


@requires_gpu
@pytest.mark.parametrize("seq_len", [33, 384])
def test_plugin_dense_equivalence_prefill(seq_len):
    """For S <= 2051 the plugin must equal the causal-dense SDPA reference
    (with qk-norm + partial rope): the index list is exactly {0..t}."""
    _require_qsa_plugin()
    batch = 1
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(batch,
                                                                   seq_len,
                                                                   seed=1000 +
                                                                   seq_len)
    q_g, k_g, _, _ = gammas
    ctx = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEV)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out = ref.dense_forward(qkv, q_g, k_g, table, ctx)
    assert_close(f"qsa_dense_equiv[s{seq_len}]", ref_out, out)


@requires_gpu
@pytest.mark.parametrize("context_lengths", [[4096], [4096, 2700]],
                         ids=["bs1_s4096", "bs2_ragged"])
def test_plugin_sparse_prefill(context_lengths):
    """S=4096 sparse prefill vs the full FP32 QSA torch reference.

    fp16 tolerances + a cosine gate (not exact index equality): FP32 score
    accumulation order differs between the warp kernel and torch, so blocks
    at the top-512 boundary may swap (see the module docstring caveat).
    """
    _require_qsa_plugin()
    batch = len(context_lengths)
    seq = max(context_lengths)
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(batch,
                                                                   seq,
                                                                   seed=2026)
    q_g, k_g, iq_g, ik_g = gammas
    ctx = torch.tensor(context_lengths, dtype=torch.int32, device=DEV)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out, _, _, _ = ref.forward(qkv, index_qk, q_g, k_g, iq_g, ik_g, table,
                                   ctx)
    for bi, length in enumerate(context_lengths):
        case = f"qsa_sparse[b{bi} L{length}]"
        assert_close(case,
                     ref_out[bi, :length],
                     out[bi, :length],
                     cos_threshold=0.999)
        assert cosine_sim(ref_out[bi, :length], out[bi, :length]) > 0.999
        # Rows past the context length must be exact zeros.
        if length < seq:
            assert (out[bi, length:] == 0).all(), \
                f"{case}: padding rows must be exactly zero"


@requires_gpu
def test_plugin_kv_write_through():
    """Paged-KV write-through: pool gathered via a flipped page table equals
    the roped-K / raw-V reference; poisoned padding must not leak anywhere."""
    _require_qsa_plugin()
    context_lengths = [384, 200]
    batch, seq = len(context_lengths), max(context_lengths)
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(
        batch, seq, seed=333, flip_page_table=True)
    q_g, k_g, iq_g, ik_g = gammas
    ctx = torch.tensor(context_lengths, dtype=torch.int32, device=DEV)
    # Any kernel that reads a row's padding region now corrupts the result.
    poison_padding([qkv, index_qk], context_lengths)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out, _, ref_k, ref_v = ref.forward(qkv, index_qk, q_g, k_g, iq_g, ik_g,
                                           table, ctx)
    plugin_k, plugin_v = runner.gather_kv(batch, seq)
    for bi, length in enumerate(context_lengths):
        case = f"b{bi} L{length}"
        assert_close(f"qsa_kv_k[{case}]", ref_k[bi, :length],
                     plugin_k[bi, :length])
        assert_close(f"qsa_kv_v[{case}]", ref_v[bi, :length],
                     plugin_v[bi, :length])
        assert_close(f"qsa_attn_poisoned[{case}]", ref_out[bi, :length],
                     out[bi, :length])
        if length < seq:
            assert (out[bi, length:] == 0).all(), \
                f"qsa_attn_poisoned[{case}]: padding rows must be zero"


@requires_gpu
def test_plugin_decode_shape_rejected():
    """v1 is prefill-only: a decode-shaped kvcache_start_index ([B] instead
    of the [0] prefill sentinel) must fail the enqueue cleanly."""
    _require_qsa_plugin()
    batch, seq = 1, 16
    runner, _, qkv, index_qk, _, _, rope = _plugin_case(batch, seq, seed=555)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        runner.run(qkv, index_qk, ctx, rope, kv_start_shape=(batch, ))


@requires_gpu
def test_plugin_determinism():
    """Two identical sparse prefills produce bitwise-identical outputs and KV
    (cub sort + gather kernels must be run-to-run deterministic)."""
    _require_qsa_plugin()
    batch, seq = 1, 4096
    runner, _, qkv, index_qk, _, _, rope = _plugin_case(batch, seq, seed=777)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)

    out1 = runner.run(qkv.clone(), index_qk.clone(), ctx, rope).clone()
    k1, v1 = runner.gather_kv(batch, seq)
    k1, v1 = k1.clone(), v1.clone()
    out2 = runner.run(qkv.clone(), index_qk.clone(), ctx, rope)
    k2, v2 = runner.gather_kv(batch, seq)

    assert torch.equal(out1.view(torch.int16), out2.view(torch.int16)), \
        "attention output is not deterministic"
    assert torch.equal(k1.view(torch.int16), k2.view(torch.int16)), \
        "K cache write is not deterministic"
    assert torch.equal(v1.view(torch.int16), v2.view(torch.int16)), \
        "V cache write is not deterministic"
