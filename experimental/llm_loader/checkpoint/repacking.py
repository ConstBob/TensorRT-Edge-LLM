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
    """Apply all quantization repacking passes after checkpoint load.

    MoE expert stacking runs FIRST because it needs the original GPTQ int32
    weights (before regular repacking converts them to the swizzled plugin
    format).  After stacking, the per-expert GPTQLinear modules have their
    qweight set to None so ``_repack_gptq_weights`` skips them.
    """
    _stack_moe_experts(model)
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
    logger.info("Repacked GPTQ weights")


def _stack_moe_experts(model: nn.Module) -> None:
    """Stack per-expert weights into Marlin-packed 3-D tensors for Int4MoePlugin.

    Must run BEFORE ``_repack_gptq_weights`` because it needs the original
    GPTQ int32 packed weights.  After extracting, per-expert qweight buffers
    are set to ``None`` so the regular GPTQ repacking skips them.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "_prepare_moe_weights"):
            module._prepare_moe_weights()
            count += 1
    if count:
        logger.info("Marlin-packed expert weights for %d MoE blocks", count)


# ---------------------------------------------------------------------------
# Marlin INT4 repacking for MoE experts
# ---------------------------------------------------------------------------
# Adapted from tensorrt_edgellm/llm_models/layers/int4_moe_plugin.py.
# These functions convert GPTQ int32-packed weights → Marlin layout consumed
# by trt_edgellm::Int4MoePlugin.
# ---------------------------------------------------------------------------


def _unpack_int4_gptq(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ ``[K//8, N]`` int32 → ``[K, N]`` int16 nibbles."""
    pack_factor = 8
    wf = torch.tensor(list(range(0, 32, 4)),
                      dtype=torch.int32).unsqueeze(0).to(qweight.device)
    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
            wf.unsqueeze(-1).to(qweight.device)).to(torch.int16), 15)
    return weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])


def _unpack_qzeros_moe(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros ``[num_groups, N//8]`` → ``[num_groups, N]``."""
    device = qzeros.device
    wf = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                      dtype=torch.int64,
                      device=device).view(1, 1, -1)
    z = qzeros.unsqueeze(2).expand(-1, -1, 8).to(torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(z, wf),
                             15).reshape(qzeros.shape[0], -1)


def _extract_gptq_for_marlin(
        proj: nn.Module, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(weights [N, K] int16, scales [N, num_groups] fp16)`` from a
    GPTQ linear module, remapping zero-points so Marlin's ``(q - 8) * scale``
    equals GPTQ's ``(q - zero) * scale``.

    GPTQ v1 checkpoints store ``zero_point - 1`` in qzeros, so the actual
    zero-point is ``stored + 1``.  The adjustment is therefore::

        q_marlin = q - (stored_zero + 1) + 8 = q - stored_zero - 1 + 8
    """
    unpacked = _unpack_int4_gptq(proj.qweight)  # [K, N]

    if hasattr(proj, "qzeros") and proj.qzeros is not None:
        zeros = _unpack_qzeros_moe(proj.qzeros)  # [num_groups, N]
        K, N = unpacked.shape
        group_ids = torch.arange(K, device=unpacked.device) // group_size
        zeros_expanded = zeros[group_ids.clamp(max=zeros.shape[0] - 1)]
        # GPTQ v1: actual_zero = stored_zero + 1
        unpacked = torch.clamp(
            unpacked.to(torch.int32) - zeros_expanded.to(torch.int32) - 1 + 8,
            0, 15).to(torch.int16)

    weights = unpacked.transpose(0, 1).contiguous()  # [N, K]
    scales = proj.scales.data.to(torch.float16).transpose(0, 1).contiguous()
    return weights, scales


# Pre-computed Marlin tensor core layout indices (from int4_moe_plugin.py).
_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)

# fmt: off
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

_ROW_PATTERN = np.array([
    [0, 1, 8, 9, 0, 1, 8, 9], [2, 3, 10, 11, 2, 3, 10, 11],
    [4, 5, 12, 13, 4, 5, 12, 13], [6, 7, 14, 15, 6, 7, 14, 15]
], dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_ROW_PATTERN, (32, 1))

_MARLIN_COL_IDX = np.array([
    [0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],
    [1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],
    [2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],
    [3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],
    [4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],
    [5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],
    [6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],
    [7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],
    [16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],
    [17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],
    [18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],
    [19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],
    [20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],
    [21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],
    [22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],
    [23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],
    [32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],
    [33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],
    [34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],
    [35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],
    [36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],
    [37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],
    [38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],
    [39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],
    [48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],
    [49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],
    [50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],
    [51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],
    [52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],
    [53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],
    [54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],
    [55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],
], dtype=np.int32)
# fmt: on


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scale columns for Marlin kernel shared-memory read pattern."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def pack_int4_awq_marlin(
    weights_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack INT4 ``[E, N, K]`` weights + ``[E, N, num_groups]`` scales to Marlin.

    Returns ``(weights_marlin [E, K//16, 2*N] int32,
               scales_marlin  [E, num_groups, N] fp16)``.
    """
    num_experts, N, K = weights_q.shape
    device = weights_q.device
    weights_marlin_list = []

    for expert_id in range(num_experts):
        w_np = weights_q[expert_id].transpose(
            0, 1).contiguous().cpu().numpy().astype(np.uint32)  # [K, N]

        k_tiles, n_tiles = K // 16, N // 64
        tiles = w_np.reshape(k_tiles, 16, n_tiles, 64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, _MARLIN_ROW_IDX,
                         _MARLIN_COL_IDX][:, :, :,
                                          _MARLIN_PACK_IDX].astype(np.uint32)

        packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4)
                      | (gathered[:, :, :, 2] << 8)
                      | (gathered[:, :, :, 3] << 12)
                      | (gathered[:, :, :, 4] << 16)
                      | (gathered[:, :, :, 5] << 20)
                      | (gathered[:, :, :, 6] << 24)
                      | (gathered[:, :, :, 7] << 28))

        out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for n_tile_id in range(n_tiles):
            out[:,
                n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                                n_tile_id, :]
        weights_marlin_list.append(
            torch.from_numpy(out.view(np.int32)).to(device))

    weights_marlin = torch.stack(weights_marlin_list, dim=0)

    scales_marlin = scales.transpose(1, 2).contiguous()  # [E, num_groups, N]
    for e in range(num_experts):
        scales_marlin[e] = _marlin_permute_scales(scales_marlin[e], K, N,
                                                  group_size)

    return weights_marlin, scales_marlin
