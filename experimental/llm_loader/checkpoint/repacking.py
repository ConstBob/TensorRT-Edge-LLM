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
Post-load weight repacking for quantized linear layers.

Transforms checkpoint weight formats (AWQ column-packed int32, GPTQ row-packed
int32, ModelOpt uint8) into the int4 GEMM plugin layout expected by TensorRT.
All functions operate in-place on ``module._buffers`` or return new tensors;
they are called by :func:`~loader.load_weights` after all checkpoint tensors
have been assigned.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "repack_awq_to_plugin",
    "repack_gptq_to_plugin",
]

# ---------------------------------------------------------------------------
# AWQ weight swizzle
# ---------------------------------------------------------------------------


def repack_awq_to_plugin(qweight: torch.Tensor,
                         qzeros: torch.Tensor) -> torch.Tensor:
    """Repack AWQ qweight from [in, out//8] int32 to [out//2, in] int8.

    AWQ packs 8 int4 nibbles per int32 along the output axis::

        int32 = (n7 << 28) | (n6 << 24) | ... | (n0 << 0)
        where n_k = output channel (8*col + k), value in [0, 15]

    The int4 GEMM kernel uses ``(nibble - 8) * scale``.
    AWQ dequantizes as ``(nibble - qzero) * scale``.
    So we adjust each nibble: ``adjusted = nibble - qzero + 8``, baking the
    per-group zero-point into the weights before packing.

    Output ``[out//2, in]`` int8: K-block permute, even/odd shuffle within 8,
    N-row interleave, then four nibbles per int16 (viewed as two int8 rows).
    """
    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # AutoAWQ packs 8 nibbles per int32 in non-sequential output-channel order.
    # Bit position k within each packed int32 stores the value for output channel
    # _AWQ_BIT_TO_CH[k] within that group of 8, derived from AutoAWQ's
    # AWQ_REVERSE_ORDER = [0,4,1,5,2,6,3,7] (packing_utils.py): the inverse
    # permutation gives the output channel encoded at each bit position.
    # Without this reorder, output channels within each group of 8 are scrambled.
    _AWQ_BIT_TO_CH = [0, 2, 4, 6, 1, 3, 5, 7]

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[:, _AWQ_BIT_TO_CH[k]::8] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[in//g, out] = uint4 in [0, 15]
    zeros = torch.zeros(in_features // group_size,
                        out_features,
                        dtype=torch.int32)
    for k in range(8):
        zeros[:, _AWQ_BIT_TO_CH[k]::8] = (qz >> (4 * k)) & 0xF

    # Expand zeros from [in//g, out] -> [in, out] by repeating each row group_size times
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)  # [in, out]

    # Adjust nibbles: kernel does (nibble - 8) * scale; AWQ does (nibble - qzero) * scale
    # So adjusted = nibble - qzero + 8 -> kernel result = (adjusted - 8) = (nibble - qzero)
    nibbles = (nibbles - zeros_expanded + 8).clamp(0, 15)

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)  # [N, K]

    packed_int16 = _pack_intweights(nibbles_nk)  # [N//4, K] int16
    packed_int8 = packed_int16.view(np.int8).reshape(
        packed_int16.shape[0] * 2, packed_int16.shape[1])  # [N//2, K]

    return torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)


def _pack_intweights(unpacked_qweight: np.ndarray) -> np.ndarray:
    """Pack nibbles ``[N, K]`` int16 in ``[0, 15]`` to ``[N//4, K]`` int16 (int4 GEMM layout).

    Steps: permute within each 32-wide K block; even/odd reorder within each 8;
    interleave every four N rows across 64-wide K stripes; pack four nibbles per int16.
    """
    interleave = 4
    kstride = 64
    N, K = unpacked_qweight.shape

    # Step 1: Permute within K-blocks of 32
    # np.arange(32).reshape(4,4,2).transpose(1,0,2) -> [0,1,8,9,16,17,24,25,...]
    pk = unpacked_qweight.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    pk = pk.reshape(N, K // 32, 32)

    # Step 2: Within each group of 8, reorder [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]
    pk = pk.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    pk = pk.reshape(N, K)

    # Step 3: Interleave every 4 rows (N dimension) across K-blocks of 64
    pk = pk.reshape(N // interleave, interleave, K // kstride, kstride)
    pk = pk.transpose(0, 2, 1, 3)  # [N//4, K//64, 4, 64]
    pk = pk.reshape(N // interleave, K // kstride, kstride, interleave)

    # Step 4: Pack 4 nibbles per int16 (little-endian nibble order)
    pk = (pk[..., 0]
          | (pk[..., 1] << 4)
          | (pk[..., 2] << 8)
          | (pk[..., 3] << 12))
    return pk.reshape(N // interleave, K).astype(np.int16)


def _gather_rows_by_gidx_order(
    weight: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rows of ``weight`` (K major) so channels with the same ``g_idx`` group are contiguous."""
    group_num = int(weight.shape[0] / group_size)
    gmax = int(torch.max(g_idx).item())
    assert group_num == gmax + 1, (
        f"Group number {group_num} != max(g_idx)+1 ({gmax + 1})")
    indices_list = []
    for i in range(group_num):
        indices = torch.nonzero(g_idx == i, as_tuple=False).squeeze(1)
        indices_list.append(indices)
    permute_idx = torch.cat(indices_list, dim=0)
    new_weight = weight.index_select(0, permute_idx)
    assert new_weight.shape[0] == weight.shape[0]
    return new_weight, permute_idx


# ---------------------------------------------------------------------------
# GPTQ weight swizzle
# ---------------------------------------------------------------------------


def repack_gptq_to_plugin(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack GPTQ ``qweight`` ``[in//8, out]`` int32 to plugin ``[out//2, in]`` int8.

    Unpacks eight nibbles per int32 along K, applies GPTQ zero-point offset,
    optionally reorders K rows by ``g_idx`` (``desc_act``), transposes to ``[N, K]``,
    then :func:`_pack_intweights`.

    Returns:
        ``(qweight_out, int4_act_perm)`` — permute activations with
        ``x.index_select(-1, int4_act_perm)`` before the int4 GEMM op when non-trivial.
    """
    in_div8, out_features = qweight.shape
    in_features = in_div8 * 8
    num_groups = qzeros.shape[0]
    group_size = in_features // num_groups

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    # GPTQ row-packs: bit k of column `in` is in row `in//8`, bit position 4*k
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[k::8, :] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[group, out] = uint4 in [0, 15]
    # qzeros is [num_groups, out//8] -- same column packing as AWQ qzeros
    zeros = torch.zeros(num_groups, out_features, dtype=torch.int32)
    for k in range(8):
        zeros[:, k::8] = (qz >> (4 * k)) & 0xF

    if g_idx is None:
        g_idx_t = torch.arange(in_features, dtype=torch.int32) // group_size
    else:
        g_idx_t = g_idx.cpu().to(torch.int32)
    # Expand zeros from [num_groups, out] -> [in, out] using per-channel group ids.
    zeros_expanded = zeros[g_idx_t.to(torch.int64)]  # [in, out]

    # GPTQ stores (zero_point - 1) in qzeros, so actual zero_point = stored + 1.
    # Adjust nibbles: kernel does (nibble - 8) * scale; GPTQ does (nibble - actual_zero) * scale
    # -> repacked = nibble - (stored_zero + 1) + 8
    nibbles = (nibbles - zeros_expanded - 1 + 8).clamp(0, 15)

    # Gather K rows by group (identity order when ``g_idx`` is sequential).
    nibbles, permute_idx = _gather_rows_by_gidx_order(nibbles, g_idx_t,
                                                      group_size)

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)
    packed_int16 = _pack_intweights(nibbles_nk)
    packed_int8 = packed_int16.view(np.int8).reshape(packed_int16.shape[0] * 2,
                                                     packed_int16.shape[1])

    qw_out = torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)
    perm = permute_idx.to(torch.int64)
    return qw_out, perm


# ---------------------------------------------------------------------------
# Post-load cast / format fixups (called by load_weights)
# ---------------------------------------------------------------------------


def apply_all_repacking(model: nn.Module) -> None:
    """Apply all quantization repacking passes after checkpoint load."""
    _repack_awq_weights(model)
    _repack_gptq_weights(model)
    _cast_modelopt_awq_prepacked(model)
    _cast_fp8_linear_scales(model)
    _cast_nvfp4_weights(model)


def _cast_modelopt_awq_prepacked(model: nn.Module) -> None:
    """Post-process W4A16 prepacked AWQ linear buffers after load.

    1. Unpack ``[N//2, K] uint8`` (two nibbles per byte) to nibbles, apply
       :func:`_pack_intweights`, store ``[N//2, K] int8``.
    2. Cast optional ``pre_quant_scale`` to float16 so forward() Mul stays in fp16.
    3. Transpose scales to ``[K//g, N]`` float16 for the int4 GEMM custom op.
    """
    from ..models.linear import ModelOptAWQPrepackedLinear  # local import
    for module in model.modules():
        if isinstance(module, ModelOptAWQPrepackedLinear):
            # 1. Repack weight: ModelOpt uint8[N//2, K] -> swizzled int8[N//2, K]
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                w_cpu = w.cpu()
                N_half, K = w_cpu.shape
                N = N_half * 2
                # Unpack 2 nibbles per byte: low nibble -> even N rows, high -> odd N rows
                # ModelOpt pack_int4_in_uint8 stores weights using two's complement masking:
                # s in [-8,7] -> u = s & 0xF (so s=-8 -> u=8, s=0 -> u=0, s=7 -> u=7)
                # The plugin kernel uses (nibble - 8) * scale, so nibble must be s+8 in [0,15].
                # Convert: plugin_nibble = (u + 8) % 16
                w_i16 = w_cpu.to(torch.int16)
                nibbles = torch.zeros(N, K, dtype=torch.int16)
                nibbles[0::2] = w_i16 & 0xF  # even N channels = low nibble
                nibbles[1::2] = (
                    w_i16 >> 4) & 0xF  # odd N channels = high nibble
                nibbles = (nibbles +
                           8) % 16  # two's complement -> plugin convention
                nibbles_np = nibbles.numpy().astype(np.int16)
                packed_int16 = _pack_intweights(nibbles_np)  # [N//4, K] int16
                packed_int8 = packed_int16.view(np.int8).reshape(
                    packed_int16.shape[0] * 2, packed_int16.shape[1])
                module._buffers["weight"] = torch.tensor(packed_int8,
                                                         dtype=torch.int8).to(
                                                             w.device)

            sc = module._buffers.get("weight_scale")
            if sc is None:
                continue

            # 2. Cast pre_quant_scale to float16 so forward() Mul stays in fp16.
            # pre_quant_scale is an AWQ activation smoothing scale applied as
            # x_smooth = x * pqs before the GEMM (matches the reference pipeline
            # where DQ+MatMul patterns include a leading Mul(x, pqs) node).
            pqs = module._buffers.get("pre_quant_scale")
            sc_f32 = sc.to(torch.float32)  # work in fp32 for precision
            if pqs is not None and pqs.dtype != torch.float16:
                module._buffers["pre_quant_scale"] = pqs.to(torch.float16)

            # 3. Transpose [N, K//g] -> [K//g, N] and cast to float16
            module._buffers["weight_scale"] = sc_f32.t().contiguous().to(
                torch.float16)


def _cast_fp8_linear_scales(model: nn.Module) -> None:
    """Cast FP8Linear ``input_scale`` / ``weight_scale`` to float16 if needed."""
    from ..models.linear import FP8Linear  # local import to avoid circular dep
    for module in model.modules():
        if not isinstance(module, FP8Linear):
            continue
        for name in ("weight_scale", "input_scale"):
            t = module._buffers.get(name)
            if t is None or t.dtype == torch.float16:
                continue
            module._buffers[name] = t.to(torch.float16)


def _cast_nvfp4_weights(model: nn.Module) -> None:
    """View-cast NVFP4Linear weight buffers from uint8 to int8 in-place.

    Packed FP4 nibbles have the same bit pattern in both types.
    Some ONNX importers mishandle UINT8 weight initializers for block DQ; int8 works.
    """
    from ..models.linear import \
        NVFP4Linear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                module._buffers["weight"] = w.view(torch.int8)


def _repack_awq_weights(model: nn.Module) -> None:
    """Swizzle ``AWQLinear.qweight`` after load (fold zeros; pack to int8 layout).

    Scales should already be ``[K//g, N]``; cast to float16 if needed.
    """
    from ..models.linear import AWQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, AWQLinear):
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                module._buffers["qweight"] = repack_awq_to_plugin(qw, qz)
                logger.debug("Repacked AWQ qweight: %s -> %s", list(qw.shape),
                             list(module._buffers["qweight"].shape))
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                module._buffers["scales"] = sc.to(torch.float16)


def _repack_gptq_weights(model: nn.Module) -> None:
    """Swizzle ``GPTQLinear.qweight`` after load; set ``int4_act_perm`` for ``desc_act``."""
    from ..models.linear import \
        GPTQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, GPTQLinear):
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                g_idx_buf = module._buffers.get("g_idx")
                packed, perm = repack_gptq_to_plugin(qw, qz, g_idx_buf)
                module._buffers["qweight"] = packed
                module._buffers["int4_act_perm"] = perm
                logger.debug("Repacked GPTQ qweight: %s -> %s", list(qw.shape),
                             list(packed.shape))
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                module._buffers["scales"] = sc.to(torch.float16)
