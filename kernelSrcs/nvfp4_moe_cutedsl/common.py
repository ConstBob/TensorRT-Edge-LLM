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

"""Shared helpers for NvFP4 MoE AOT export scripts."""


def compute_sf_buffer_size(m, n, sf_vec_size=16):
    """Compute atom-layout scale factor buffer size in bytes.

    This matches the swizzled atom layout produced by swizzle_sf() and
    expected by the blockscaled GEMM kernels.
    """
    scale_k = n // sf_vec_size
    padded_m_tiles = (m + 127) // 128
    padded_scale_k_groups = (scale_k + 3) // 4
    return 32 * 4 * padded_m_tiles * 4 * padded_scale_k_groups


def create_dummy_pointers(sf_vec_size=16,
                          dummy_m=128, dummy_n=1856, dummy_k=2688, dummy_l=16,
                          is_swiglu=False, include_fc2_extras=False,
                          dummy_num_tokens=128, dummy_top_k=6):
    """Create typed GPU pointers for AOT export compilation.

    Returns a dict of (name -> typed_ptr) plus backing buffers kept alive
    in a list. The caller must keep the returned buffers alive until after
    ``cute.compile`` finishes.

    Parameters
    ----------
    include_fc2_extras : bool
        If True, also creates permuted_idx and token_scales pointers
        needed by the FC2 finalize kernel.
    """
    import cupy as cp
    import cutlass
    import cutlass.cute as cute
    from utils import make_ptr

    dummy_n_out = dummy_n // 2 if is_swiglu else dummy_n
    bufs = []  # prevent GC

    def _alloc(shape, dtype):
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)
        buf = cp.zeros(shape, dtype=dtype)
        bufs.append(buf)
        return buf

    # FP4 input A: [m, K/2] bytes
    a_buf = _alloc((dummy_m, dummy_k // 2), cp.uint8)
    a_ptr = make_ptr(cutlass.Float4E2M1FN, a_buf.data.ptr,
                     cute.AddressSpace.gmem, assumed_align=32)

    # FP4 weights B: [L, N, K/2] bytes
    b_buf = _alloc((dummy_l, dummy_n, dummy_k // 2), cp.uint8)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, b_buf.data.ptr,
                     cute.AddressSpace.gmem, assumed_align=32)

    # Scale factors A: atom-layout
    a_sf_size = compute_sf_buffer_size(dummy_m, dummy_k, sf_vec_size)
    a_sf_buf = _alloc(a_sf_size, cp.uint8)
    a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, a_sf_buf.data.ptr,
                        cute.AddressSpace.gmem, assumed_align=16)

    # Scale factors B: atom-layout (per-expert)
    padded_n_tiles = (dummy_n + 127) // 128
    padded_scale_k_groups = (dummy_k // sf_vec_size + 3) // 4
    b_sf_size = 32 * 4 * padded_n_tiles * 4 * padded_scale_k_groups * dummy_l
    b_sf_buf = _alloc(b_sf_size, cp.uint8)
    b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, b_sf_buf.data.ptr,
                        cute.AddressSpace.gmem, assumed_align=16)

    # Output C
    if include_fc2_extras:
        # FC2: output is [num_tokens, N]
        c_buf = _alloc((dummy_num_tokens, dummy_n), cp.float16)
    else:
        c_buf = _alloc((dummy_m, dummy_n_out), cp.float16)
    c_ptr = make_ptr(cutlass.BFloat16, c_buf.data.ptr,
                     cute.AddressSpace.gmem, assumed_align=32)

    # Alpha: [L] float32
    alpha_buf = _alloc(dummy_l, cp.float32)
    alpha_ptr = make_ptr(cutlass.Float32, alpha_buf.data.ptr,
                         cute.AddressSpace.gmem, assumed_align=16)

    # Tile metadata
    dummy_num_tiles = dummy_m // 128
    tile_group_buf = _alloc(dummy_num_tiles, cp.int32)
    tile_group_ptr = make_ptr(cutlass.Int32, tile_group_buf.data.ptr,
                              cute.AddressSpace.gmem)
    tile_mn_buf = _alloc(dummy_num_tiles, cp.int32)
    tile_mn_ptr = make_ptr(cutlass.Int32, tile_mn_buf.data.ptr,
                           cute.AddressSpace.gmem)

    ptrs = dict(
        a_ptr=a_ptr, b_ptr=b_ptr, a_sf_ptr=a_sf_ptr, b_sf_ptr=b_sf_ptr,
        c_ptr=c_ptr, alpha_ptr=alpha_ptr,
        tile_group_ptr=tile_group_ptr, tile_mn_ptr=tile_mn_ptr,
    )

    if include_fc2_extras:
        # permuted_idx_to_expanded_idx: [m] int32
        perm_buf = _alloc(dummy_m, cp.int32)
        perm_ptr = make_ptr(cutlass.Int32, perm_buf.data.ptr,
                            cute.AddressSpace.gmem)
        ptrs["perm_ptr"] = perm_ptr

    # num_non_exiting_tiles: [1] int32
    num_tiles_buf = _alloc(1, cp.int32)
    num_tiles_ptr = make_ptr(cutlass.Int32, num_tiles_buf.data.ptr,
                             cute.AddressSpace.gmem)
    ptrs["num_tiles_ptr"] = num_tiles_ptr

    if include_fc2_extras:
        # token_final_scales: [num_tokens, top_k] float32
        ts_buf = _alloc((dummy_num_tokens, dummy_top_k), cp.float32)
        ts_ptr = make_ptr(cutlass.Float32, ts_buf.data.ptr,
                          cute.AddressSpace.gmem, assumed_align=16)
        ptrs["token_scales_ptr"] = ts_ptr

    return ptrs, bufs


def get_max_active_clusters(cluster_size=1):
    """Query hardware for max active clusters."""
    import cutlass
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
