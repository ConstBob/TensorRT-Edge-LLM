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
"""Marlin NVFP4 tile packing for ``trt_edgellm::Nvfp4MoePlugin`` MoE expert weights: dequantize ModelOpt NVFP4 → dense fp32 → Marlin tile-packed payload + atom-layout FP8 block scales. Port of ``tensorrt_edgellm/llm_models/layers/nvfp4_moe_plugin.py::_pack_one_expert``."""

from __future__ import annotations

import numpy as np
import torch

FP8_MAX = 448.0

_FP4_E2M1_POSITIVE_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                     dtype=np.float32)
# Midpoints between consecutive E2M1 levels (for searchsorted-based quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)


def decode_modelopt_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int = 16,
) -> np.ndarray:
    """Dequantize one ModelOpt NVFP4 weight tensor to dense fp32 ``[out, in]``.

    ``weight`` is ``[out, in//2]`` int8/uint8 with two FP4 E2M1 nibbles per
    byte (low nibble = even index).  ``weight_scale`` is ``[out, in//group_size]``
    FP8 E4M3 (accepts ``float8_e4m3fn``, an int8 view of it, or a float cast).
    ``weight_scale_2`` is ``[1]`` fp32 per-tensor scale-of-scale.
    """
    w = weight.detach().cpu().numpy()
    if w.dtype == np.int8:
        w = w.view(np.uint8)
    if w.dtype != np.uint8:
        raise TypeError(f"unexpected weight dtype {w.dtype}")
    out_f, half = w.shape

    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, half * 2), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    sign = (nibbles & np.uint8(0x08)) != 0
    magnitude = nibbles & np.uint8(0x07)
    values = _FP4_E2M1_POSITIVE_LEVELS[magnitude]
    values = np.where(sign, -values, values).astype(np.float32)

    if weight_scale.dtype == torch.float8_e4m3fn:
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    elif weight_scale.dtype == torch.int8:
        ws_fp32 = (weight_scale.detach().view(torch.float8_e4m3fn).to(
            torch.float32).cpu().numpy())
    elif weight_scale.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    else:
        raise TypeError(f"unsupported weight_scale dtype {weight_scale.dtype}")

    ws2 = float(weight_scale_2.detach().reshape(-1)[0].item())

    num_groups = ws_fp32.shape[-1]
    in_f = num_groups * group_size
    if values.shape != (out_f, in_f):
        raise ValueError(f"nibble shape {values.shape} does not match "
                         f"(out={out_f}, num_groups*group_size={in_f})")
    values_grouped = values.reshape(out_f, num_groups, group_size)
    dense = values_grouped * ws_fp32[..., np.newaxis]
    dense = dense.reshape(out_f, in_f)
    dense *= ws2
    return dense.astype(np.float32)


def _atom_sf_offsets(M: int, num_sf_cols: int) -> np.ndarray:
    """Byte offsets for the 128x4 atom-layout scale-factor swizzle.

    Returns an ``[M, num_sf_cols]`` int64 array into the flat per-expert scale
    buffer.  Matches ``MarlinConverter.atom_sf_offset`` in
    ``tensorrt_edgellm/llm_models/marlin_converter.py``.
    """
    m_idx = np.arange(M, dtype=np.int64)[:, None]
    k_idx = np.arange(num_sf_cols, dtype=np.int64)[None, :]
    inner_k = k_idx % 4
    inner_m = (m_idx % 128) // 32
    outer_m = m_idx % 32
    k_tile = k_idx // 4
    num_k_tiles = (num_sf_cols + 3) // 4
    m_tile = m_idx // 128
    return (m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 +
            inner_m * 4 + inner_k)


def _pack_marlin_tiles(
    w_ah: np.ndarray,
    s_max_ex: float,
    n_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marlin tile-pack one ``[A, B]`` fp32 matrix (``B == n_chunks * 64``) to ``(payload_i8 [A, n_chunks, 32], block_scale_u8 [A, 4*n_chunks])``.

    The B axis is split into ``n_chunks`` 64-lane tiles, each further split
    into 4 sub-groups of 16 lanes sharing an FP8 E4M3 block scale.  Sub-group
    scales pass through the Marlin ``f16 → f8`` bit projection (``(fp16 << 1)
    >> 8`` top byte); values off the reachable manifold snap (e.g. ``100 → 96``).
    Scales are returned in natural sub-group order; the caller scatters them
    into the atom-layout buffer via :func:`_atom_sf_offsets`.
    """
    A, B = w_ah.shape
    if B != n_chunks * 64:
        raise ValueError(f"B ({B}) must equal n_chunks ({n_chunks}) * 64")

    w_tiles = w_ah.reshape(A, n_chunks, 4, 16)

    group_max = np.abs(w_tiles).max(axis=-1)
    group_scales = np.maximum(group_max / 6.0, 1e-12)
    w_scaled = (w_tiles / group_scales[..., np.newaxis]).clip(-6.0, 6.0)

    x_abs = np.abs(w_scaled)
    best_idx = np.searchsorted(_E2M1_BOUNDS, x_abs).astype(np.uint8)
    sign_bits = (w_scaled < 0).astype(np.uint8) << np.uint8(3)
    nibbles = (best_idx | sign_bits).reshape(A, n_chunks, 64)

    lo = nibbles[..., ::2] & np.uint8(0xF)
    hi = nibbles[..., 1::2] & np.uint8(0xF)
    payload_i8 = (lo | (hi << np.uint8(4))).astype(np.uint8).view(np.int8)

    scales_norm = (group_scales / s_max_ex * FP8_MAX).astype(np.float16)
    # FP16 → FP8 E4M3: shift left 1 to drop sign, take upper byte.
    h16 = scales_norm.view(np.uint16).astype(np.uint32)
    fp8_bytes = ((
        (h16 << np.uint32(1)) & np.uint32(0xFF00)) >> np.uint32(8)).astype(
            np.uint8)  # [A, n_chunks, 4]
    # Flatten sub-group axis so output is [A, num_sf_cols=4*n_chunks].
    block_scale_u8 = fp8_bytes.reshape(A, 4 * n_chunks)

    return payload_i8, block_scale_u8


def _scatter_atom_layout(block_scale_mk: np.ndarray,
                         total_bytes: int) -> np.ndarray:
    """Scatter a natural-order ``[M, num_sf_cols]`` uint8 plane into a flat atom-layout byte buffer of ``total_bytes`` bytes."""
    M, num_sf_cols = block_scale_mk.shape
    flat = np.zeros(total_bytes, dtype=np.uint8)
    offsets = _atom_sf_offsets(M, num_sf_cols)
    if offsets.max() >= total_bytes:
        raise ValueError(
            f"atom-layout offset {int(offsets.max())} exceeds buffer "
            f"{total_bytes} bytes (M={M}, num_sf_cols={num_sf_cols})")
    flat[offsets] = block_scale_mk
    return flat


def marlin_pack_expert_up(
    dense_w_hi: np.ndarray, ) -> tuple[np.ndarray, np.ndarray, float]:
    """Pack dense up_proj weight ``[H, I]`` fp32 to ``(payload_i8 [H//2, I], block_scale_i8 [padded_H, I//16], global_scale_f32)``.

    ``dense_w_hi`` uses the old-pipeline ``[H, I]`` convention — the transpose
    of PyTorch's ``weight[out=I, in=H]``.  ``block_scale`` is the atom-layout
    swizzle over (M=H, K_sf=I/16) with M padded up to a multiple of 128 so
    the last partial m-tile is fully addressable.
    """
    H, I = dense_w_hi.shape
    if H % 64 or I % 64:
        raise ValueError(f"H ({H}) and I ({I}) must both be multiples of 64")
    n_chunks = I // 64
    s_max = max(float(np.abs(dense_w_hi).max()) / 6.0, 1e-12)
    pl, bs_mk = _pack_marlin_tiles(np.ascontiguousarray(dense_w_hi), s_max,
                                   n_chunks)
    num_sf_cols = I // 16
    padded_M = ((H + 127) // 128) * 128
    padded_sf_cols = ((num_sf_cols + 3) // 4) * 4
    flat = _scatter_atom_layout(bs_mk, padded_M * padded_sf_cols)
    payload = pl.reshape(H // 2, I).copy()
    block_scale = flat.view(np.int8).reshape(padded_M, padded_sf_cols).copy()
    return payload, block_scale, float(s_max / FP8_MAX)


def marlin_pack_expert_down(
    dense_w_ih: np.ndarray, ) -> tuple[np.ndarray, np.ndarray, float]:
    """Pack dense down_proj weight ``[I, H]`` fp32 to ``(payload_i8 [I, H//2], block_scale_i8 [padded_I, H//16], global_scale_f32)``.

    ``dense_w_ih`` uses the old-pipeline ``[I, H]`` convention.  ``block_scale``
    is the atom-layout swizzle over (M=I, K_sf=H/16) with M padded up to a
    multiple of 128.
    """
    I, H = dense_w_ih.shape
    if H % 64 or I % 64:
        raise ValueError(f"H ({H}) and I ({I}) must both be multiples of 64")
    n_chunks = H // 64
    s_max = max(float(np.abs(dense_w_ih).max()) / 6.0, 1e-12)
    pl, bs_mk = _pack_marlin_tiles(np.ascontiguousarray(dense_w_ih), s_max,
                                   n_chunks)
    num_sf_cols = H // 16
    padded_M = ((I + 127) // 128) * 128
    padded_sf_cols = ((num_sf_cols + 3) // 4) * 4
    flat = _scatter_atom_layout(bs_mk, padded_M * padded_sf_cols)
    payload = pl.reshape(I, H // 2).copy()
    block_scale = flat.view(np.int8).reshape(padded_M, padded_sf_cols).copy()
    return payload, block_scale, float(s_max / FP8_MAX)
