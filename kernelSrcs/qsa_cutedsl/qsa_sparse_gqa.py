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

# Origin: Structural fork of kernelSrcs/fmha_v2_cutedsl/fmha.py (FMHA-v2
# Ampere-floor forward kernel), which is itself adapted from
# xlite-dev/ffpa-attn (Apache-2.0).  The MMA/softmax core copies that file;
# the sparse index-gather traversal is this variant's own.

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Callable, Type

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import numpy as np
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


"""QSA sparse-GQA prefill kernel (CuTe DSL), AOT-export build.

Qwen Sparse Attention (Qwen3.8-Flash-Next): every query token carries an
int32 list of selected KV token indices produced by the QSA indexer
(top-k blocks of 4 tokens, expanded, plus an always-attended tail;
``-1``-padded on the right, unsorted, distinct, all ``< t + 1``).  The
kernel attends ONLY to the listed tokens — there is no causal or
positional mask in-kernel; causality lives entirely in the index list.

Layout: Q / K / V / O are dense padded ``BSND = (batch, seq, num_head,
head_dim)``, fp16 or bf16.  ``indices`` is ``(B, S, topk)`` Int32 and
``context_lengths`` is ``(B,)`` Int32 (live length per padded batch row).

Per-CTA tile: grid ``(S, H_kv, B)`` — one CTA per (query token, KV head).
The M tile is the token's GQA head group (``group = H_q / H_kv`` rows,
zero-padded to ``m_block_size``); the index list is walked in
``n_block_size`` chunks with a per-row cp.async gather (one warp-wave per
K/V row).  FA2-style online softmax (exp2, scale folded at exponent time —
i.e. applied to the FP32 scores, never pre-scaled into the fp16/bf16 Q).

Guarantees:
  * ``-1`` entries: K/V smem rows zero-filled AND scores forced to -inf
    (NaN hardening — masked columns stay inert even if memory holds NaN).
  * padding query rows (``t >= context_lengths[b]``): zero chunks run and
    the epilogue stores exact zeros.
  * all-invalid rows: ``row_sum == 0`` guard in normalize keeps zeros.
  * head-padding rows (``row >= group``): never loaded, never stored.

Variant axes baked at compile time: ``head_dim`` and the
``(Br, Bc, threads)`` tuning.  Runtime-dynamic: batch, seq, H_q, H_kv
(GQA group size), topk, strides, softmax scale.
"""

QSA_DEFAULT_M_BLOCK = 16
QSA_DEFAULT_N_BLOCK = 16
QSA_DEFAULT_THREADS = 32


class QSASparseGQAPrefill:

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = QSA_DEFAULT_M_BLOCK,
        n_block_size: int = QSA_DEFAULT_N_BLOCK,
        num_threads: int = QSA_DEFAULT_THREADS,
    ):
        """Initialize the QSA sparse-GQA prefill kernel.

        ``head_dim`` must be a multiple of 8 (16-byte alignment of the
        contiguous mode).  ``m_block_size`` bounds the GQA group size
        (``H_q / H_kv <= m_block_size``); the production shape is 12 -> 16.
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        # cp.async row gather: one warp-wave per K/V row, 128-bit per lane.
        self._async_load_cache_mode = cpasync.LoadCacheMode.GLOBAL

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads
    ) -> bool:
        """Check whether the (dtype, tile, threads) combo is implementable.

        Each warp owns 16 query rows in the MMA layout, so the warp count
        must tile ``m_block_size`` exactly.  The row-gather copy uses one
        full warp per K/V row (32 lanes x 8 elems = 256), so ``head_dim``
        must be a multiple of ``32 * 8 = 256 / (16 / dtype_bytes)`` — for
        the 16-bit dtypes this means ``head_dim % 256 == 0`` when the whole
        row is one wave; smaller head dims would need a different value
        layout, deferred until a variant needs it.
        """
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        # One warp-wave covers num_threads lanes * 8 halves; the per-row
        # gather and the row-wise Q/O copies require exactly one wave per
        # 16-bit row, so the CTA width is pinned to the head dim.
        if head_dim != num_threads * 8:
            return False

        head_dim_padded = (head_dim + 31) // 32 * 32
        smem_usage = (
            m_block_size * head_dim_padded
            + n_block_size * head_dim_padded * 2
        ) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        indices: cute.Tensor,
        context_lengths: cute.Tensor,
        attention_scale: cutlass.Float32,
        sm_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Configure SMEM / tiled-copy / tiled-mma and launch the kernel.

        ``q_tensor``/``o_tensor`` are ``(B, S, H_q, D)``; ``k_tensor``/
        ``v_tensor`` are ``(B, S, H_kv, D)``; all share dtype (fp16/bf16)
        with contiguous ``D``-innermost packing.  ``indices`` is
        ``(B, S, topk)`` Int32, ``context_lengths`` is ``(B,)`` Int32.
        ``H_q % H_kv == 0`` and ``H_q / H_kv <= m_block_size`` are caller
        contracts.  ``sm_count`` is ABI symmetry with the FMHA-v2 wrappers
        and is unused.
        """
        if cutlass.const_expr(
            not (
                q_tensor.element_type
                == k_tensor.element_type
                == v_tensor.element_type
                == o_tensor.element_type
            )
        ):
            raise TypeError("Q, K, V, and O must have the same data type")
        if cutlass.const_expr(
            not (
                q_tensor.element_type == cutlass.Float16
                or q_tensor.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = q_tensor.element_type

        # ///////////////////////////////////////////////////////////////////
        # Shared memory layouts (same swizzled atoms as FMHA-v2).
        # ///////////////////////////////////////////////////////////////////
        smem_k_block_size = (
            64
            if self._head_dim_padded % 64 == 0
            else 32
            if self._head_dim_padded % 32 == 0
            else 16
        )
        swizzle_bits = (
            3 if smem_k_block_size == 64 else 2 if smem_k_block_size == 32 else 1
        )
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )
        sV_layout = sKV_layout
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
            ]
            # Ping-pong staging of the current / next chunk's token indices:
            # written during the K gather, read by the V gather and the
            # sparse score mask.
            sIdx: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 2 * self._n_block_size], 16
            ]

        # ///////////////////////////////////////////////////////////////////
        # GMEM copies.  The row gather uses a (1, 32) x (1, 8) tiled copy:
        # one warp-wave moves one whole 256-element K/V/Q row (32 lanes x
        # 128 bits), so a gathered row is a single fully-coalesced wave.
        # ///////////////////////////////////////////////////////////////////
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self._async_load_cache_mode),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        row_thr_layout = cute.make_layout(
            (1, self._num_threads), stride=(self._num_threads, 1)
        )
        row_val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_row = cute.make_tiled_copy_tv(
            atom_async_copy, row_thr_layout, row_val_layout
        )
        gmem_tiled_copy_row_O = cute.make_tiled_copy_tv(
            atom_universal_copy, row_thr_layout, row_val_layout
        )

        # ///////////////////////////////////////////////////////////////////
        # Tiled mma (same m16n8k16 arrangement as FMHA-v2).
        # ///////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid: (query token, KV head, batch) — adjacent CTAs share L2 for
        # overlapping gathered rows of neighbouring tokens.
        grid_dim = (
            cute.size(q_tensor.shape[1]),
            cute.size(k_tensor.shape[2]),
            cute.size(q_tensor.shape[0]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = attention_scale * LOG2_E
        _ = sm_count
        self.kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            indices,
            context_lengths,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_row,
            gmem_tiled_copy_row_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mIdx: cute.Tensor,
        mCtxLen: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_tiled_copy_row_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Sparse FA2 body: prologue cp.async Q rows + gathered K chunk 0,
        chunk loop of ``compute_one_chunk`` (BMM1 -> sparse mask + online
        softmax -> BMM2), epilogue rmem -> smem (aliased over sQ) -> gmem
        for the ``row < group`` rows only.
        """
        tidx, _, _ = cute.arch.thread_idx()
        token, kv_head, batch = cute.arch.block_idx()

        num_q_heads = mQ.shape[2]
        num_kv_heads = mK.shape[2]
        group_size = num_q_heads // num_kv_heads
        head_base = kv_head * group_size

        ctx_len = mCtxLen[batch]
        topk = mIdx.shape[2]

        # Chunk bound: never more useful entries than (token + 1) causal
        # candidates; correctness never depends on this (entries past the
        # valid prefix are -1 and masked), only wasted work does.
        row_limit = cutlass.min(topk, token + 1)
        n_chunks = cute.ceil_div(row_limit, self._n_block_size)
        # Padding query rows: run zero chunks; acc_O keeps its zero fill and
        # the row_sum == 0 guard in normalize_softmax stores exact zeros.
        n_chunks = 0 if token >= ctx_len else n_chunks

        # ///////////////////////////////////////////////////////////////////
        # Strides recomputed from shapes (not read from the layout) so the
        # divby facts survive dynamic-shape marking — same trick as the
        # FMHA-v2 packed-varlen path.
        # ///////////////////////////////////////////////////////////////////
        q_seq_stride = num_q_heads * self._head_dim
        q_seq_stride = cute.assume(q_seq_stride, divby=8)
        q_batch_stride = mQ.shape[1] * q_seq_stride
        kv_seq_stride = num_kv_heads * self._head_dim
        kv_seq_stride = cute.assume(kv_seq_stride, divby=8)
        kv_batch_stride = mK.shape[1] * kv_seq_stride

        q_row_base = (
            batch * q_batch_stride
            + token * q_seq_stride
            + head_base * self._head_dim
        )
        kv_row_base = batch * kv_batch_stride + kv_head * self._head_dim

        # ///////////////////////////////////////////////////////////////////
        # Shared memory.
        # ///////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sIdx = storage.sIdx.get_tensor(
            cute.make_layout((2, self._n_block_size), stride=(self._n_block_size, 1))
        )

        # Transposed view of V for BMM2.
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_row = gmem_tiled_copy_row.get_slice(tidx)
        gmem_thr_copy_row_O = gmem_tiled_copy_row_O.get_slice(tidx)
        # Row-tiled destination partitions: mode 1 walks the tile rows.
        tQsQ = gmem_thr_copy_row.partition_D(sQ)
        tKsK = gmem_thr_copy_row.partition_D(sK)
        tVsV = gmem_thr_copy_row.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////
        # MMA partitions and accumulators.
        # ///////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////
        # Prologue: Q rows (row < group only; head-padding rows are
        # zero-filled and never read from gmem — for the last token of the
        # last batch they would fall outside the allocation) + gathered K
        # chunk 0.
        #
        # Padding tokens (n_chunks == 0) must issue NO cp.async at all: the
        # chunk loop — the only place that waits on the async groups — never
        # runs for them, and an in-flight Q load could otherwise land in sQ
        # after the epilogue has overwritten it as sO (the padding rows must
        # store exact zeros). Their sQ contents are never read.
        # ///////////////////////////////////////////////////////////////////
        if n_chunks > 0:
            for row in cutlass.range_constexpr(self._m_block_size):
                if row < group_size:
                    gRow = self._gmem_row_view(mQ, q_row_base + row * self._head_dim)
                    tRow = gmem_thr_copy_row.partition_S(gRow)
                    tRowAligned = cute.make_tensor(tRow.iterator.align(16), tRow.layout)
                    cute.copy(
                        gmem_tiled_copy_row,
                        tRowAligned[None, 0, 0],
                        tQsQ[None, row, 0],
                    )
                else:
                    tQsQ[None, row, 0].fill(0)
            self._gather_chunk(
                mK,
                mIdx,
                sIdx,
                tKsK,
                gmem_tiled_copy_row,
                gmem_thr_copy_row,
                batch,
                token,
                kv_row_base,
                kv_seq_stride,
                cutlass.Int32(0),
                topk,
                load_indices=True,
            )
        cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////
        # Online-softmax state.
        # ///////////////////////////////////////////////////////////////////
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        basic_params = SimpleNamespace(
            batch=batch,
            token=token,
            kv_row_base=kv_row_base,
            kv_seq_stride=kv_seq_stride,
            topk=topk,
            n_chunks=n_chunks,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_row=gmem_tiled_copy_row,
            gmem_thr_copy_row=gmem_thr_copy_row,
            tKsK=tKsK,
            tVsV=tVsV,
            mK=mK,
            mV=mV,
            mIdx=mIdx,
            sIdx=sIdx,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Chunk 0 peeled for the is_first constexpr split, exactly like the
        # FMHA-v2 mask-step peeling.
        if n_chunks > 0:
            self.compute_one_chunk(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                cutlass.Int32(0),
                is_first_chunk=True,
            )
        for chunk in range(1, n_chunks, 1):
            self.compute_one_chunk(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                chunk,
                is_first_chunk=False,
            )

        # ///////////////////////////////////////////////////////////////////
        # Epilogue: normalize, rmem -> smem (aliased over sQ) -> gmem for
        # the live group rows only.
        # ///////////////////////////////////////////////////////////////////
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        self.cta_sync_barrier.arrive_and_wait()

        o_seq_stride = mO.shape[2] * self._head_dim
        o_seq_stride = cute.assume(o_seq_stride, divby=8)
        o_batch_stride = mO.shape[1] * o_seq_stride
        o_row_base = (
            batch * o_batch_stride
            + token * o_seq_stride
            + head_base * self._head_dim
        )
        tOsO = gmem_thr_copy_row_O.partition_S(sO)
        for row in cutlass.range_constexpr(self._m_block_size):
            if row < group_size:
                gRow = self._gmem_row_view(mO, o_row_base + row * self._head_dim)
                tRowO = gmem_thr_copy_row_O.partition_D(gRow)
                tRowOAligned = cute.make_tensor(tRowO.iterator.align(16), tRowO.layout)
                cute.copy(
                    gmem_tiled_copy_row_O,
                    tOsO[None, row, 0],
                    tRowOAligned[None, 0, 0],
                )

    @cute.jit
    def _gmem_row_view(self, mT: cute.Tensor, element_offset: cutlass.Int32):
        """A ``(1, head_dim)`` gmem row view at a dynamic element offset.

        Every contributing offset term is a multiple of 8 elements (16
        bytes at 16-bit dtypes) — batch/seq strides carry the static
        ``head_dim`` factor and the head offset is ``row * head_dim`` —
        so the 128-bit cp.async alignment is provable.
        """
        raw_ptr = mT.iterator + element_offset
        aligned_ptr = cute.make_ptr(
            mT.element_type,
            raw_ptr.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        return cute.make_tensor(
            aligned_ptr,
            cute.make_layout((1, self._head_dim), stride=(self._head_dim, 1)),
        )

    @cute.jit
    def _gather_chunk(
        self,
        mKV: cute.Tensor,
        mIdx: cute.Tensor,
        sIdx: cute.Tensor,
        dst,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_thr_copy_row,
        batch: cutlass.Int32,
        token: cutlass.Int32,
        kv_row_base: cutlass.Int32,
        kv_seq_stride: cutlass.Int32,
        chunk: cutlass.Int32,
        topk: cutlass.Int32,
        load_indices: cutlass.Constexpr,
    ):
        """Gather one ``n_block_size``-row K or V chunk by token index.

        One warp-wave per row: the whole warp cp.async-copies one 256-element
        row (32 lanes x 128 bits, fully coalesced within the row).  Invalid
        (``-1`` or beyond-topk) rows are zero-filled so masked columns stay
        inert in BMM2 even against NaN-poisoned memory.

        ``load_indices=True`` (K pass) reads ``mIdx`` from gmem and stages
        the chunk's indices into the ``chunk % 2`` half of the ``sIdx``
        ping-pong buffer; ``False`` (V pass) replays the staged indices.
        """
        buf = chunk % 2
        for w in cutlass.range_constexpr(self._n_block_size):
            pos = chunk * self._n_block_size + w
            idx = cutlass.Int32(-1)
            if cutlass.const_expr(load_indices):
                if pos < topk:
                    idx = mIdx[batch, token, pos]
                # All lanes store the same value — benign, keeps the wave
                # uniform without a lane predicate.
                sIdx[buf, w] = idx
            else:
                idx = sIdx[buf, w]
            if idx >= 0:
                gRow = self._gmem_row_view(
                    mKV, kv_row_base + idx * kv_seq_stride
                )
                tRow = gmem_thr_copy_row.partition_S(gRow)
                tRowAligned = cute.make_tensor(tRow.iterator.align(16), tRow.layout)
                cute.copy(
                    gmem_tiled_copy_row,
                    tRowAligned[None, 0, 0],
                    dst[None, w, 0],
                )
            else:
                dst[None, w, 0].fill(0)

    @cute.jit
    def compute_one_chunk(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        chunk: cutlass.Int32,
        is_first_chunk: cutlass.Constexpr,
    ):
        """One index chunk: BMM1 (Q @ gathered-K^T) -> sparse mask + online
        softmax -> BMM2 (P @ gathered-V).  cp.async commit/wait/barrier
        structure is one-to-one with FMHA-v2's ``compute_one_n_block``.
        """
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # Wait for this chunk's K (and, on chunk 0, the Q rows).
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # V gather of the current chunk replays the staged indices.
        self._gather_chunk(
            gmem_copy_params.mV,
            gmem_copy_params.mIdx,
            gmem_copy_params.sIdx,
            gmem_copy_params.tVsV,
            gmem_copy_params.gmem_tiled_copy_row,
            gmem_copy_params.gmem_thr_copy_row,
            basic_params.batch,
            basic_params.token,
            basic_params.kv_row_base,
            basic_params.kv_seq_stride,
            chunk,
            basic_params.topk,
            load_indices=False,
        )
        cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////
        # S = Q @ K^T  (BMM1), ldmatrix double-buffered over the k mode.
        # ///////////////////////////////////////////////////////////////////
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Prefetch next chunk's K into the other sIdx half while softmax and
        # BMM2 run.
        if chunk + 1 < basic_params.n_chunks:
            self._gather_chunk(
                gmem_copy_params.mK,
                gmem_copy_params.mIdx,
                gmem_copy_params.sIdx,
                gmem_copy_params.tKsK,
                gmem_copy_params.gmem_tiled_copy_row,
                gmem_copy_params.gmem_thr_copy_row,
                basic_params.batch,
                basic_params.token,
                basic_params.kv_row_base,
                basic_params.kv_seq_stride,
                chunk + 1,
                basic_params.topk,
                load_indices=True,
            )
            cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////
        # Sparse mask + online softmax.
        # ///////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            mma_params,
            softmax_params,
            gmem_copy_params.sIdx,
            chunk,
            acc_S,
            is_first_chunk,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////
        # O += P @ V  (BMM2)
        # ///////////////////////////////////////////////////////////////////
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        sIdx: cute.Tensor,
        chunk: cutlass.Int32,
        acc_S: cute.Tensor,
        is_first_chunk: cutlass.Constexpr,
    ):
        """Apply the sparse-index mask and online softmax to ``acc_S``.

        The only mask is index validity: staged ``sIdx`` entries < 0 force
        the column to -inf for every row.  There is no positional masking —
        causality is baked into the index list by the indexer.  The scale
        is applied to the FP32 scores at exp2 time (row_max tracked on raw
        scores), never pre-folded into the low-precision Q.
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_chunk):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)

        buf = chunk % 2
        mcS = cute.make_identity_tensor(
            (self._m_block_size, self._n_block_size)
        )
        tScS = mma_params.thr_mma.partition_C(mcS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)
        # Per-thread column validity, hoisted out of the row loop (each
        # thread sees the same columns for both of its accumulator rows).
        col_valid = cute.make_rmem_tensor(
            cute.make_layout(cute.size(tScS_mn.shape[1])), cutlass.Boolean
        )
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            col_idx = tScS_mn[0, c][1]
            col_valid[c] = sIdx[buf, col_idx] >= 0

        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                if not col_valid[c]:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            if cutlass.const_expr(not is_first_chunk):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            # Keep -inf in the running max until a valid score arrives, but
            # use a finite max for the exponent arithmetic so fully-masked
            # rows stay at exp2(-inf) = 0 instead of NaN.
            row_max_safe_row = (
                0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
            )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_safe_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            if cutlass.const_expr(not is_first_chunk):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_safe_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Final softmax normalisation with the zero-row guard.

        ``row_sum == 0`` (padding token rows and all-invalid index rows)
        keeps scale = 1 so the zero-initialised accumulator stores exact
        zeros — never NaN.
        """
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret a ``(MMA, MMA_M, MMA_N)`` accumulator as ``(M, N)``."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """Reduce across the four threads holding the same column of an MMA fragment."""
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)


# ---------------------------------------------------------------------------
# Host-side helpers (CuPy/NumPy; no torch dependency)
# ---------------------------------------------------------------------------


def _cutlass_to_cupy_dtype(cutlass_dtype):
    if cutlass_dtype == cutlass.Float16:
        return cp.float16
    if cutlass_dtype == cutlass.BFloat16:
        # CuPy lacks a native bf16 dtype; store raw uint16 bytes — `from_dlpack`
        # plus `element_type = cutlass.BFloat16` interpret it correctly.
        return cp.uint16
    raise ValueError(f"Unsupported cutlass dtype for CuPy: {cutlass_dtype}")


def _to_float32(arr: cp.ndarray, dtype) -> cp.ndarray:
    """View a stored fp16/bf16 CuPy array as fp32 values."""
    if dtype == cutlass.Float16:
        return arr.astype(cp.float32)
    return (arr.astype(cp.uint32) << 16).view(cp.float32).reshape(arr.shape)


def _nan_pattern(dtype):
    if dtype == cutlass.Float16:
        return cp.float16(cp.nan)
    return cp.uint16(0x7FC0)  # bf16 quiet NaN


def _create_bsnd_tensor(
    b: int,
    s: int,
    h: int,
    d: int,
    dtype: Type[cutlass.Numeric],
    *,
    fill_random: bool,
):
    """Allocate a BSND CuPy tensor and return the ``cute.Tensor`` wrapper.

    B / S / H are runtime-dynamic; D is compile-time-known.  We deliberately
    do NOT call ``mark_layout_dynamic`` — see the FMHA-v2 comment: it would
    strip the static-D alignment fact the IR verifier needs for the 128-bit
    cp.async source pointers.
    """
    shape = (b, s, h, d)
    cp_dtype = _cutlass_to_cupy_dtype(dtype)
    if fill_random:
        if dtype == cutlass.Float16:
            arr = cp.random.uniform(-1.0, 1.0, shape).astype(cp_dtype)
        else:
            f32 = cp.random.uniform(-1.0, 1.0, shape).astype(cp.float32)
            arr = cp.ascontiguousarray(
                (f32.view(cp.uint32) >> 16).astype(cp.uint16)
            )
    else:
        arr = cp.zeros(shape, dtype=cp_dtype)

    t = from_dlpack(arr, assumed_align=16)
    t.element_type = dtype
    so = (0, 1, 2, 3)
    t = (
        t.mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )
    return t, arr


def _wrap_indices_tensor(arr: cp.ndarray):
    """Wrap a contiguous ``(B, S, topk)`` Int32 index tensor (all modes dynamic)."""
    t = from_dlpack(arr, assumed_align=16)
    so = (0, 1, 2)
    t = (
        t.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )
    return t


def _wrap_ctx_lengths_tensor(arr: cp.ndarray):
    """Wrap a ``(B,)`` Int32 context-lengths tensor."""
    t = from_dlpack(arr, assumed_align=16)
    return t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )


def _generate_qsa_indices(
    batch_size: int,
    seqlen: int,
    topk: int,
    ctx_lengths,
    *,
    compress_ratio: int = 4,
    block_topk: int = 512,
    seed: int = 20260829,
) -> np.ndarray:
    """Generate index lists matching the QSA indexer output contract.

    Per valid row t: ``min(block_topk, (t+1)//ratio)`` distinct blocks
    (random subset standing in for the score top-k), expanded to runs of
    ``ratio`` consecutive token ids and packed to the front in shuffled
    block order, followed by the ``(t+1) % ratio`` tail tokens (always
    present), ``-1``-padded to ``topk``.  Padding rows are all ``-1``.
    """
    rng = np.random.default_rng(seed)
    idx = np.full((batch_size, seqlen, topk), -1, dtype=np.int32)
    for b in range(batch_size):
        L = int(ctx_lengths[b])
        for t in range(L):
            n_vis = (t + 1) // compress_ratio
            k_sel = min(block_topk, n_vis)
            if k_sel > 0:
                blocks = rng.choice(n_vis, size=k_sel, replace=False)
                toks = (
                    blocks[:, None] * compress_ratio
                    + np.arange(compress_ratio)[None, :]
                ).reshape(-1)
            else:
                toks = np.empty(0, dtype=np.int64)
            tail = np.arange(n_vis * compress_ratio, t + 1)
            row = np.concatenate([toks, tail]).astype(np.int32)
            assert len(row) <= topk, (t, len(row), topk)
            idx[b, t, : len(row)] = row
    return idx


def _qsa_sparse_reference(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    indices: np.ndarray,
    ctx_lengths,
    softmax_scale: float,
    dtype,
) -> cp.ndarray:
    """FP32 CuPy oracle: gather-by-index softmax attention per query row.

    Rows with no valid index (and padding rows) produce exact zeros.
    """
    batch_size, seqlen, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group_size = num_q_heads // num_kv_heads
    reference = cp.zeros(q.shape, dtype=cp.float32)
    q_f32 = _to_float32(q, dtype)
    k_f32 = _to_float32(k, dtype)
    v_f32 = _to_float32(v, dtype)
    idx_gpu = cp.asarray(indices)
    for b in range(batch_size):
        L = int(ctx_lengths[b])
        for t in range(L):
            row = idx_gpu[b, t]
            valid = row[row >= 0]
            if valid.size == 0:
                continue
            for g in range(num_kv_heads):
                k_sel = k_f32[b, valid, g, :]  # (n, D)
                v_sel = v_f32[b, valid, g, :]
                q_grp = q_f32[b, t, g * group_size : (g + 1) * group_size, :]
                scores = q_grp @ k_sel.T * softmax_scale
                scores -= cp.max(scores, axis=1, keepdims=True)
                p = cp.exp(scores)
                p /= cp.sum(p, axis=1, keepdims=True)
                reference[b, t, g * group_size : (g + 1) * group_size, :] = (
                    p @ v_sel
                )
    return reference


def _report_reference_error(tag: str, actual: cp.ndarray, reference: cp.ndarray):
    """Print and enforce the standalone low-precision-vs-FP32 gate."""
    error = cp.abs(actual - reference)
    max_abs = float(cp.max(error).get()) if error.size else 0.0
    close_rate = float(cp.mean(error <= 5.0e-2).get()) if error.size else 1.0
    print(f"{tag} reference max_abs={max_abs:.6f}, close_rate@0.05={close_rate:.6f}")
    if max_abs > 1.0e-1 or close_rate < 0.999:
        raise RuntimeError(
            f"{tag} reference check failed: max_abs={max_abs:.6f}, "
            f"close_rate@0.05={close_rate:.6f}"
        )


# ---------------------------------------------------------------------------
# run(): test + AOT export entry point
# ---------------------------------------------------------------------------


def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen: int,
    num_head: int,
    head_dim: int,
    kv_group_size: int = 12,
    topk: int = 2051,
    softmax_scale: float = 0.0,
    m_block_size: int = QSA_DEFAULT_M_BLOCK,
    n_block_size: int = QSA_DEFAULT_N_BLOCK,
    num_threads: int = QSA_DEFAULT_THREADS,
    ragged: bool = False,
    warmup_iterations: int = 3,
    iterations: int = 10,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./qsa_sparse_aot_artifacts",
    file_name: str = "qsa_sparse",
    function_prefix: str = "qsa_sparse",
    **kwargs,
):
    """Compile, test, benchmark, or export the QSA sparse-GQA kernel.

    AOT export uses dummy placeholder shapes; only ``head_dim`` and the
    ``(Br, Bc, threads)`` tuning are baked at compile time — batch, seq,
    H_q, H_kv, topk, strides and the softmax scale stay runtime-dynamic.
    """
    _tag = f"[{file_name}]"

    if not QSASparseGQAPrefill.can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads
    ):
        raise ValueError(
            f"{_tag} Unsupported config: dtype={dtype}, head_dim={head_dim}, "
            f"Br={m_block_size}, Bc={n_block_size}, threads={num_threads}"
        )
    if num_head % kv_group_size != 0:
        raise ValueError(
            f"{_tag} num_head ({num_head}) must be divisible by "
            f"kv_group_size ({kv_group_size})"
        )
    if kv_group_size > m_block_size:
        raise ValueError(
            f"{_tag} kv_group_size ({kv_group_size}) must be <= "
            f"m_block_size ({m_block_size})"
        )

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this kernel.")

    if softmax_scale <= 0.0:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    h_q = num_head
    h_kv = h_q // kv_group_size

    if export_only:
        print(
            f"{_tag} Compiling QSA sparse-GQA CuTe DSL: dtype={dtype}, "
            f"head_dim={head_dim}, Br={m_block_size}, Bc={n_block_size}, "
            f"threads={num_threads}"
        )
    else:
        print(f"{_tag} Running QSA sparse-GQA CuTe DSL forward with:")
        print(f"{_tag}   dtype={dtype}, head_dim={head_dim}, topk={topk}")
        print(
            f"{_tag}   B={batch_size}, S={seqlen}, H_q={h_q}, H_kv={h_kv}, "
            f"group={kv_group_size}, ragged={ragged}"
        )
        print(f"{_tag}   softmax_scale={softmax_scale}")
        print(f"{_tag}   Br={m_block_size}, Bc={n_block_size}, threads={num_threads}")
        cp.random.seed(20260829)
        print(f"{_tag}   CuPy random seed=20260829")

    q_dyn, q_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_q, head_dim, dtype, fill_random=not export_only
    )
    k_dyn, k_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_kv, head_dim, dtype, fill_random=not export_only
    )
    v_dyn, v_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_kv, head_dim, dtype, fill_random=not export_only
    )
    o_dyn, o_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_q, head_dim, dtype, fill_random=False
    )

    if ragged and batch_size > 1:
        rng = np.random.default_rng(20260829)
        ctx_host = np.sort(
            rng.integers(1, seqlen + 1, size=batch_size).astype(np.int32)
        )[::-1].copy()
        ctx_host[0] = seqlen
    else:
        ctx_host = np.full(batch_size, seqlen, dtype=np.int32)
    ctx_arr = cp.asarray(ctx_host)
    ctx_dyn = _wrap_ctx_lengths_tensor(ctx_arr)

    idx_host = _generate_qsa_indices(batch_size, seqlen, topk, ctx_host)
    idx_arr = cp.asarray(idx_host)
    idx_dyn = _wrap_indices_tensor(idx_arr)

    if not export_only:
        # Poison everything the kernel must never read: K/V rows at or past
        # the live length (NaN), Q rows of padding tokens (NaN), and the
        # output buffer (sentinel — padding rows must come back as zeros).
        nan_val = _nan_pattern(dtype)
        for b in range(batch_size):
            L = int(ctx_host[b])
            if L < seqlen:
                k_arr[b, L:] = nan_val
                v_arr[b, L:] = nan_val
                q_arr[b, L:] = nan_val
        if dtype == cutlass.Float16:
            o_arr.fill(cp.float16(123.0))
        else:
            o_arr.fill(cp.uint16(0x42F6))  # bf16 123.0

    qsa_fwd = QSASparseGQAPrefill(
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
    )

    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    _ptx_parts = []
    if os.getenv("QSA_PTXAS_VERBOSE"):
        _ptx_parts.append("-v")
    _extra_ptx = os.getenv("QSA_PTXAS_OPTS", "")
    if _extra_ptx:
        _ptx_parts.append(_extra_ptx)
    compile_options = (
        {"options": f"--ptxas-options={','.join(_ptx_parts)}"} if _ptx_parts else {}
    )

    print(f"{_tag} Compiling kernel...")
    t0 = time.time()
    compiled_qsa = cute.compile(
        qsa_fwd,
        q_dyn,
        k_dyn,
        v_dyn,
        o_dyn,
        idx_dyn,
        ctx_dyn,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count()),
        current_stream,
        **compile_options,
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_qsa.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None

    compiled_qsa(
        q_dyn,
        k_dyn,
        v_dyn,
        o_dyn,
        idx_dyn,
        ctx_dyn,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count()),
        current_stream,
    )
    cp.cuda.Device().synchronize()

    if not skip_ref_check:
        reference = _qsa_sparse_reference(
            q_arr, k_arr, v_arr, idx_host, ctx_host, softmax_scale, dtype
        )
        o_f32 = _to_float32(o_arr, dtype)
        # Padding token rows must be EXACT zeros (sentinel-poisoned output
        # buffer proves the kernel actually stored them).
        for b in range(batch_size):
            L = int(ctx_host[b])
            if L < seqlen:
                pad = o_f32[b, L:]
                if float(cp.max(cp.abs(pad)).get()) != 0.0:
                    raise RuntimeError(
                        f"{_tag} padding rows of batch {b} are not exact zeros"
                    )
        # Gate only the live rows: padded rows are exactly zero on both sides
        # and would otherwise inflate close_rate on ragged runs.
        valid_actual = cp.concatenate(
            [o_f32[b, : int(ctx_host[b])].ravel() for b in range(batch_size)]
        )
        valid_reference = cp.concatenate(
            [reference[b, : int(ctx_host[b])].ravel() for b in range(batch_size)]
        )
        _report_reference_error(_tag, valid_actual, valid_reference)
        print(f"{_tag} SPARSE_PASS: matched the FP32 gather-softmax reference")

    def _bench():
        t_start = time.time()
        for _ in range(iterations):
            compiled_qsa(
                q_dyn,
                k_dyn,
                v_dyn,
                o_dyn,
                idx_dyn,
                ctx_dyn,
                cutlass.Float32(softmax_scale),
                cutlass.Int32(
                    utils.HardwareInfo().get_device_multiprocessor_count()
                ),
                current_stream,
            )
        cp.cuda.Device().synchronize()
        return (time.time() - t_start) / iterations * 1e6

    for _ in range(warmup_iterations):
        compiled_qsa(
            q_dyn,
            k_dyn,
            v_dyn,
            o_dyn,
            idx_dyn,
            ctx_dyn,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count()),
            current_stream,
        )
    cp.cuda.Device().synchronize()
    avg_time_us = _bench()

    # Sparse FMHA FLOPs: 4 * H_q * D * sum over live rows of min(topk, t+1).
    flops = 0.0
    for b in range(batch_size):
        live = int(ctx_host[b])
        active = np.minimum(np.arange(1, live + 1), topk)
        flops += 4.0 * h_q * float(active.sum()) * head_dim
    tflops = flops / (avg_time_us * 1e-6) / 1e12
    print(f"{_tag} avg_time_us: {avg_time_us:.2f}  |  {tflops:.2f} TFLOPS ({dtype})")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="QSA sparse-GQA CuTe DSL prefill kernel: test and AOT export."
    )
    p.add_argument("--dtype", type=cutlass.dtype, default=cutlass.Float16,
                   help="Input/output dtype: Float16 or BFloat16 (default: Float16).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=384)
    p.add_argument("--num_head", type=int, default=24,
                   help="Number of Q heads (H_q). Must be divisible by kv_group_size.")
    p.add_argument("--head_dim", type=int, default=256)
    p.add_argument("--kv_group_size", type=int, default=12,
                   help="GQA group size H_q / H_kv (default 12 = Qwen3.8-Flash-Next).")
    p.add_argument("--topk", type=int, default=2051,
                   help="Index-list width (default 2051 = 512*4 + 3 tail).")
    p.add_argument("--softmax_scale", type=float, default=0.0,
                   help="Softmax scale; 0 (default) => 1 / sqrt(head_dim).")
    p.add_argument("--m_block_size", type=int, default=QSA_DEFAULT_M_BLOCK)
    p.add_argument("--n_block_size", type=int, default=QSA_DEFAULT_N_BLOCK)
    p.add_argument("--num_threads", type=int, default=QSA_DEFAULT_THREADS)
    p.add_argument("--ragged", action="store_true",
                   help="Use random per-batch context lengths (< seqlen).")
    p.add_argument("--warmup_iterations", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export only; skip reference check and benchmark.")
    p.add_argument("--output_dir", type=str, default="./qsa_sparse_aot_artifacts",
                   help="Output directory for AOT artifacts (<file_name>.{h,o}).")
    p.add_argument("--file_name", type=str, default="qsa_sparse",
                   help="Base file name for exported artifacts.")
    p.add_argument("--function_prefix", type=str, default="qsa_sparse",
                   help="Function prefix for exported C symbols.")
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    run(
        dtype=args.dtype,
        batch_size=args.batch_size,
        seqlen=args.seqlen,
        num_head=args.num_head,
        head_dim=args.head_dim,
        kv_group_size=args.kv_group_size,
        topk=args.topk,
        softmax_scale=args.softmax_scale,
        m_block_size=args.m_block_size,
        n_block_size=args.n_block_size,
        num_threads=args.num_threads,
        ragged=args.ragged,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
