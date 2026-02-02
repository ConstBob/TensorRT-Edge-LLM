# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
INT4 MoE Plugin utilities for AWQ Marlin weight packing.
"""

from typing import Tuple

import numpy as np
import torch

# Pre-computed Marlin tensor core layout indices (128 threads, 8 values each)
# fmt: off
_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)

_MARLIN_OUT_IDX = np.array([
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127
], dtype=np.int32)

# ROW_IDX: 4 unique patterns repeated 32 times each
_ROW_PATTERN = np.array([
    [0, 1, 8, 9, 0, 1, 8, 9],
    [2, 3, 10, 11, 2, 3, 10, 11],
    [4, 5, 12, 13, 4, 5, 12, 13],
    [6, 7, 14, 15, 6, 7, 14, 15]
], dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_ROW_PATTERN, (32, 1))

# COL_IDX: columns follow pattern based on warp_id and thread_id
_MARLIN_COL_IDX = np.array([
    [0, 0, 0, 0, 8, 8, 8, 8], [0, 0, 0, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 8, 8, 8], [0, 0, 0, 0, 8, 8, 8, 8],
    [1, 1, 1, 1, 9, 9, 9, 9], [1, 1, 1, 1, 9, 9, 9, 9],
    [1, 1, 1, 1, 9, 9, 9, 9], [1, 1, 1, 1, 9, 9, 9, 9],
    [2, 2, 2, 2, 10, 10, 10, 10], [2, 2, 2, 2, 10, 10, 10, 10],
    [2, 2, 2, 2, 10, 10, 10, 10], [2, 2, 2, 2, 10, 10, 10, 10],
    [3, 3, 3, 3, 11, 11, 11, 11], [3, 3, 3, 3, 11, 11, 11, 11],
    [3, 3, 3, 3, 11, 11, 11, 11], [3, 3, 3, 3, 11, 11, 11, 11],
    [4, 4, 4, 4, 12, 12, 12, 12], [4, 4, 4, 4, 12, 12, 12, 12],
    [4, 4, 4, 4, 12, 12, 12, 12], [4, 4, 4, 4, 12, 12, 12, 12],
    [5, 5, 5, 5, 13, 13, 13, 13], [5, 5, 5, 5, 13, 13, 13, 13],
    [5, 5, 5, 5, 13, 13, 13, 13], [5, 5, 5, 5, 13, 13, 13, 13],
    [6, 6, 6, 6, 14, 14, 14, 14], [6, 6, 6, 6, 14, 14, 14, 14],
    [6, 6, 6, 6, 14, 14, 14, 14], [6, 6, 6, 6, 14, 14, 14, 14],
    [7, 7, 7, 7, 15, 15, 15, 15], [7, 7, 7, 7, 15, 15, 15, 15],
    [7, 7, 7, 7, 15, 15, 15, 15], [7, 7, 7, 7, 15, 15, 15, 15],
    [16, 16, 16, 16, 24, 24, 24, 24], [16, 16, 16, 16, 24, 24, 24, 24],
    [16, 16, 16, 16, 24, 24, 24, 24], [16, 16, 16, 16, 24, 24, 24, 24],
    [17, 17, 17, 17, 25, 25, 25, 25], [17, 17, 17, 17, 25, 25, 25, 25],
    [17, 17, 17, 17, 25, 25, 25, 25], [17, 17, 17, 17, 25, 25, 25, 25],
    [18, 18, 18, 18, 26, 26, 26, 26], [18, 18, 18, 18, 26, 26, 26, 26],
    [18, 18, 18, 18, 26, 26, 26, 26], [18, 18, 18, 18, 26, 26, 26, 26],
    [19, 19, 19, 19, 27, 27, 27, 27], [19, 19, 19, 19, 27, 27, 27, 27],
    [19, 19, 19, 19, 27, 27, 27, 27], [19, 19, 19, 19, 27, 27, 27, 27],
    [20, 20, 20, 20, 28, 28, 28, 28], [20, 20, 20, 20, 28, 28, 28, 28],
    [20, 20, 20, 20, 28, 28, 28, 28], [20, 20, 20, 20, 28, 28, 28, 28],
    [21, 21, 21, 21, 29, 29, 29, 29], [21, 21, 21, 21, 29, 29, 29, 29],
    [21, 21, 21, 21, 29, 29, 29, 29], [21, 21, 21, 21, 29, 29, 29, 29],
    [22, 22, 22, 22, 30, 30, 30, 30], [22, 22, 22, 22, 30, 30, 30, 30],
    [22, 22, 22, 22, 30, 30, 30, 30], [22, 22, 22, 22, 30, 30, 30, 30],
    [23, 23, 23, 23, 31, 31, 31, 31], [23, 23, 23, 23, 31, 31, 31, 31],
    [23, 23, 23, 23, 31, 31, 31, 31], [23, 23, 23, 23, 31, 31, 31, 31],
    [32, 32, 32, 32, 40, 40, 40, 40], [32, 32, 32, 32, 40, 40, 40, 40],
    [32, 32, 32, 32, 40, 40, 40, 40], [32, 32, 32, 32, 40, 40, 40, 40],
    [33, 33, 33, 33, 41, 41, 41, 41], [33, 33, 33, 33, 41, 41, 41, 41],
    [33, 33, 33, 33, 41, 41, 41, 41], [33, 33, 33, 33, 41, 41, 41, 41],
    [34, 34, 34, 34, 42, 42, 42, 42], [34, 34, 34, 34, 42, 42, 42, 42],
    [34, 34, 34, 34, 42, 42, 42, 42], [34, 34, 34, 34, 42, 42, 42, 42],
    [35, 35, 35, 35, 43, 43, 43, 43], [35, 35, 35, 35, 43, 43, 43, 43],
    [35, 35, 35, 35, 43, 43, 43, 43], [35, 35, 35, 35, 43, 43, 43, 43],
    [36, 36, 36, 36, 44, 44, 44, 44], [36, 36, 36, 36, 44, 44, 44, 44],
    [36, 36, 36, 36, 44, 44, 44, 44], [36, 36, 36, 36, 44, 44, 44, 44],
    [37, 37, 37, 37, 45, 45, 45, 45], [37, 37, 37, 37, 45, 45, 45, 45],
    [37, 37, 37, 37, 45, 45, 45, 45], [37, 37, 37, 37, 45, 45, 45, 45],
    [38, 38, 38, 38, 46, 46, 46, 46], [38, 38, 38, 38, 46, 46, 46, 46],
    [38, 38, 38, 38, 46, 46, 46, 46], [38, 38, 38, 38, 46, 46, 46, 46],
    [39, 39, 39, 39, 47, 47, 47, 47], [39, 39, 39, 39, 47, 47, 47, 47],
    [39, 39, 39, 39, 47, 47, 47, 47], [39, 39, 39, 39, 47, 47, 47, 47],
    [48, 48, 48, 48, 56, 56, 56, 56], [48, 48, 48, 48, 56, 56, 56, 56],
    [48, 48, 48, 48, 56, 56, 56, 56], [48, 48, 48, 48, 56, 56, 56, 56],
    [49, 49, 49, 49, 57, 57, 57, 57], [49, 49, 49, 49, 57, 57, 57, 57],
    [49, 49, 49, 49, 57, 57, 57, 57], [49, 49, 49, 49, 57, 57, 57, 57],
    [50, 50, 50, 50, 58, 58, 58, 58], [50, 50, 50, 50, 58, 58, 58, 58],
    [50, 50, 50, 50, 58, 58, 58, 58], [50, 50, 50, 50, 58, 58, 58, 58],
    [51, 51, 51, 51, 59, 59, 59, 59], [51, 51, 51, 51, 59, 59, 59, 59],
    [51, 51, 51, 51, 59, 59, 59, 59], [51, 51, 51, 51, 59, 59, 59, 59],
    [52, 52, 52, 52, 60, 60, 60, 60], [52, 52, 52, 52, 60, 60, 60, 60],
    [52, 52, 52, 52, 60, 60, 60, 60], [52, 52, 52, 52, 60, 60, 60, 60],
    [53, 53, 53, 53, 61, 61, 61, 61], [53, 53, 53, 53, 61, 61, 61, 61],
    [53, 53, 53, 53, 61, 61, 61, 61], [53, 53, 53, 53, 61, 61, 61, 61],
    [54, 54, 54, 54, 62, 62, 62, 62], [54, 54, 54, 54, 62, 62, 62, 62],
    [54, 54, 54, 54, 62, 62, 62, 62], [54, 54, 54, 54, 62, 62, 62, 62],
    [55, 55, 55, 55, 63, 63, 63, 63], [55, 55, 55, 55, 63, 63, 63, 63],
    [55, 55, 55, 55, 63, 63, 63, 63], [55, 55, 55, 55, 63, 63, 63, 63],
], dtype=np.int32)
# fmt: on


def pack_int4_awq_marlin(
        weights_q: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack INT4 weights (stored as int16) and scales into AWQ Marlin format.
    
    Args:
        weights_q: INT4 weights as int16 tensor, shape [E, N, K] with values in [0, 15].
            - E: number of experts
            - N: output features
            - K: input features (must be divisible by 16)
            - N must be divisible by 64
        scales: Scale tensor, shape [E, N, num_groups] where num_groups = K // group_size.
        group_size: Quantization group size (default: 128).
    
    Returns:
        Tuple of:
        - weights_marlin: Marlin-formatted weights [E, K//16, 2*N] as int32.
        - scales_marlin: Permuted scales [E, num_groups, N] for Marlin kernel.
    """
    num_experts, N, K = weights_q.shape
    assert K % 16 == 0, f"K={K} must be divisible by 16"
    assert N % 64 == 0, f"N={N} must be divisible by 64"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    num_groups = K // group_size
    assert scales.shape == (num_experts, N, num_groups), \
        f"scales shape {scales.shape} must be [{num_experts}, {N}, {num_groups}]"

    device = weights_q.device
    weights_marlin_list = []
    pack_order = [0, 4, 1, 5, 2, 6, 3, 7]

    for expert_id in range(num_experts):
        # Transpose to [K, N]
        w = weights_q[expert_id].transpose(0, 1).contiguous()

        # Pack 8 int4 values into int32 with AWQ interleaved order
        packed = torch.zeros(K, N // 8, dtype=torch.int32, device=device)
        for i in range(8):
            packed |= (w[:, i::8].to(torch.int32) << (pack_order[i] * 4))

        # Repack to Marlin layout
        b_np = packed.cpu().numpy().astype(np.uint32)
        undo_pack = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32)
        unpacked = np.zeros((K, N), dtype=np.uint32)
        for i in range(8):
            unpacked[:, i::8] = (b_np >> (undo_pack[i] * 4)) & 0xF

        k_tiles, n_tiles = K // 16, N // 64
        tiles = unpacked.reshape(k_tiles, 16, n_tiles,
                                 64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, _MARLIN_ROW_IDX,
                         _MARLIN_COL_IDX][:, :, :,
                                          _MARLIN_PACK_IDX].astype(np.uint32)

        packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4) |
                      (gathered[:, :, :, 2] << 8) |
                      (gathered[:, :, :, 3] << 12) |
                      (gathered[:, :, :, 4] << 16) |
                      (gathered[:, :, :, 5] << 20) |
                      (gathered[:, :, :, 6] << 24) |
                      (gathered[:, :, :, 7] << 28))

        out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for n_tile_id in range(n_tiles):
            out[:,
                n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                                n_tile_id, :]

        weights_marlin_list.append(
            torch.from_numpy(out.view(np.int32)).to(device))

    weights_marlin = torch.stack(weights_marlin_list, dim=0)

    # Permute scales for Marlin kernel: [E, N, num_groups] -> [E, num_groups, N]
    scales_marlin = scales.transpose(1, 2).contiguous()

    return weights_marlin, scales_marlin
