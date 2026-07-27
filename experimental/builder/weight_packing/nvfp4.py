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
"""NumPy helpers for decoding checkpoint NVFP4 tensors and packing MoE
weights into FC1/FC2 FP4 buffers and six-dimensional block-scale layouts."""

import numpy as np

from ..core.numpy_dtypes import (FP4_E2M1_LEVELS, f32_to_fp8_e4m3_bytes,
                                 fp8_e4m3_bytes_to_f32, round_f32_to_bf16)

__all__ = [
    "decode_modelopt_nvfp4",
    "swizzle_nvfp4_mma_scales",
    "pack_nvfp4_moe_weight",
]

# Midpoints between consecutive E2M1 levels (for searchsorted quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)


def decode_modelopt_nvfp4(packed: np.ndarray,
                          scale_fp8_bytes: np.ndarray,
                          weight_scale_2: float,
                          group_size: int = 16) -> np.ndarray:
    """Dequantize one ModelOpt NVFP4 weight tensor to dense fp32 ``[out, in]``.

    * ``packed`` -- uint8 ``[out, in//2]`` (two FP4 E2M1 nibbles per byte).
    * ``scale_fp8_bytes`` -- raw FP8 E4M3 bytes ``[out, in//group_size]``.
    * ``weight_scale_2`` -- per-tensor scalar (fp32).
    """
    w = np.ascontiguousarray(packed, dtype=np.uint8)
    out_f, half = w.shape
    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, half * 2), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    sign = (nibbles & np.uint8(0x08)) != 0
    magnitude = nibbles & np.uint8(0x07)
    values = FP4_E2M1_LEVELS[magnitude]
    values = np.where(sign, -values, values).astype(np.float32)

    ws_fp32 = fp8_e4m3_bytes_to_f32(
        np.ascontiguousarray(scale_fp8_bytes, dtype=np.uint8))
    num_groups = ws_fp32.shape[-1]
    in_f = num_groups * group_size
    if values.shape != (out_f, in_f):
        raise ValueError(f"nibble shape {values.shape} != (out={out_f}, "
                         f"num_groups*group_size={in_f})")
    grouped = values.reshape(out_f, num_groups, group_size)
    dense = grouped * ws_fp32[..., np.newaxis]
    dense = dense.reshape(out_f, in_f) * float(weight_scale_2)
    return dense.astype(np.float32)


def swizzle_nvfp4_mma_scales(scale_bytes: np.ndarray, m_dim: int,
                             k_sf_dim: int) -> np.ndarray:
    """Swizzle linear FP8 block scales to the six-dimensional MMA layout."""
    sf = np.ascontiguousarray(scale_bytes, dtype=np.uint8)
    if sf.shape != (m_dim, k_sf_dim):
        raise ValueError(f"scale shape {sf.shape} != ({m_dim}, {k_sf_dim})")
    m_tiles = (m_dim + 127) // 128
    k_tiles = (k_sf_dim + 3) // 4
    padded_m = m_tiles * 128
    padded_k_sf = k_tiles * 4
    sf_padded = np.zeros((padded_m, padded_k_sf), dtype=np.uint8)
    sf_padded[:m_dim, :k_sf_dim] = sf
    sf_5d = sf_padded.reshape(m_tiles, 4, 32, k_tiles, 4)
    return sf_5d.transpose(0, 3, 2, 1, 4).copy().view(np.int8)


def pack_nvfp4_moe_weight(dense_w_mk: np.ndarray, group_size: int = 16):
    """Pack dense ``[M, K]`` fp32 weights for ``Nvfp4MoePlugin``.

    Returns ``(qweights int8 [M, K/2], blocks_scale int8 [m_tiles,k_tiles,32,4,4])``.
    """
    if group_size != 16:
        raise NotImplementedError("Nvfp4MoePlugin requires group_size=16")
    m_dim, k_dim = dense_w_mk.shape
    if k_dim % group_size != 0 or k_dim % 2 != 0:
        raise ValueError(f"K ({k_dim}) must be a multiple of {group_size} "
                         "and even")
    dense = round_f32_to_bf16(np.ascontiguousarray(dense_w_mk, np.float32))
    k_sf_dim = k_dim // group_size
    dense_blocks = dense.reshape(m_dim, k_sf_dim, group_size)
    block_scales = np.maximum(np.abs(dense_blocks).max(axis=-1) / 6.0,
                              1e-12).astype(np.float32)

    scaled = (dense_blocks / block_scales[..., np.newaxis]).clip(-6.0, 6.0)
    abs_idx = np.searchsorted(_E2M1_BOUNDS, np.abs(scaled)).astype(np.uint8)
    sign_bit = (scaled < 0).astype(np.uint8) << np.uint8(3)
    nibbles = (abs_idx | sign_bit).reshape(m_dim, k_dim)

    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    qweights = (lo | (hi << np.uint8(4))).astype(np.uint8).view(np.int8)

    sf_bytes = f32_to_fp8_e4m3_bytes(block_scales)
    blocks_scale = swizzle_nvfp4_mma_scales(sf_bytes, m_dim, k_sf_dim)
    return np.ascontiguousarray(qweights), np.ascontiguousarray(blocks_scale)


# Re-expose decoders used elsewhere.
_ = fp8_e4m3_bytes_to_f32
