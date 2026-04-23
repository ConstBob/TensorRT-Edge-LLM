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
ONNX / PyTorch integration for TensorRT ``Nvfp4MoePlugin`` (W4A16 decode: FP16 activations, NVFP4 weights).

Mirrors ``int4_moe_plugin.py``: ONNX ``OpSchema``, symbolic export, ``torch.library.custom_op`` stub,
and ``nn.Module`` wrapper. TensorRT engine build and inference live in tests (see
``tests/python-unittests/test_moe_w4an_plugin.py``).

The C++ plugin uses router logits ``[batch * seq_len, E]`` and FP16 hidden states
``[batch, seq_len, hidden_size]`` (``seq_len`` may be ``> 1``). Routing inside the plugin is
``kernel::moeSigmoidGroupTopk`` (NemotronH sigmoid + grouped top-k routing); see
:meth:`NemotronHMoEW4A4Plugin.sigmoid_group_topk_torch`. ``hidden_size`` and ``moe_inter_size`` must be
multiples of 64 (decode GEMV / Marlin tile chunks). Expert up
quantized weights are INT8 ``[E, hidden_size/2, moe_inter_size]`` (two NVFP4 nibbles per byte) with
INT8 Marlin block scales ``[E, hidden_size/16, moe_inter_size]``. Down weights are INT8
``[E, moe_inter_size, hidden_size/2]`` with block scales ``[E, moe_inter_size, hidden_size/16]``.
Activations are dense FP16 (W4A16).
Plugin output tensors are FP16.

HuggingFace ``NemotronHMoE`` (``transformers.models.nemotron_h.modeling_nemotron_h``) uses
``NemotronHTopkRouter`` / ``DeepseekV3TopkRouter`` for **pre-routing logits** (FP32 matmul), then
``route_tokens_to_experts`` (sigmoid, grouped top-k, correction bias). That post-processing is now
implemented inside the TRT plugin via ``kernel::moeSigmoidGroupTopk``.
:class:`NemotronHMoEW4A4Plugin` computes router logits via
:meth:`NemotronHMoEW4A4Plugin.nemotron_h_plugin_router_logits`.

**Expert weight layout:** HuggingFace ``NemotronHExperts`` stores ``up_proj`` as ``[E, I, H_in]`` and
``down_proj`` as ``[E, H_in, I]`` (``I`` = ``moe_intermediate_size``, ``H_in`` = ``moe_latent_size`` or
``hidden_size``). The TensorRT ``Nvfp4MoePlugin`` / Marlin pack path uses ``w_up_ehi`` ``[E, H_in, I]`` and
``w_down_eih`` ``[E, I, H_in]``; use ``experts.up_proj.data.transpose(1, 2)`` /
``experts.down_proj.data.transpose(1, 2)`` into Marlin layout (see :meth:`NemotronHMoEW4A4Plugin.pack_experts_weights_to_marlin`).
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION
from ..marlin_converter import MarlinConverter

try:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE
except Exception:  # pragma: no cover - optional dependency surface
    NemotronHMoE = None  # type: ignore[misc, assignment]

nvfp4_moe_plugin_schema = OpSchema(
    name="Nvfp4MoePlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    ("Custom TensorRT Nvfp4MoePlugin: Top-K routing + W4A16 decode GEMV (FP16 hidden states, NVFP4 weights). "
     "FP32 router logits; up qweights INT8 [E, K/2, inter] + block scales [E, K/16, inter]; "
     "down qweights INT8 [E, inter, K/2] + block scales [E, inter, K/16] (K=hidden); "
     "FP32 per-expert scales for up and down."),
    inputs=[
        OpSchema.FormalParameter(
            name="router_logits",
            description=
            ("Router logits (batch * seq_len, E) FP32 before plugin routing; Nvfp4MoePlugin applies "
             "sigmoid + grouped top-k routing (moeSigmoidGroupTopk) internally."
             ),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="hidden_states",
            description=
            "FP16 activations (batch, seq_len, hidden_size) for W4A16, or INT8 NVFP4-packed (batch, seq_len, "
            "hidden_size/2) for W4A4.",
            type_str="tensor(float16)",
        ),
        OpSchema.FormalParameter(
            name="hidden_block_scale",
            description=
            "W4A4: INT8 Marlin block scales (batch, seq_len, hidden_size/16). W4A16: unused; supply a 1×1×1 dummy.",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="hidden_global_scale",
            description=
            "W4A4: FP32 length-1 tensor (activation.global_scale[0]). W4A16: unused dummy.",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_qweights",
            description=
            "NVFP4 up-proj quantized payload (E, K/2, inter) INT8 with K=hidden_size (two FP4 per byte).",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_blocks_scale",
            description=
            "Up-proj Marlin block scales (E, K/16, inter) INT8 with K=hidden_size (four FP8 bytes per tile word).",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_global_scale",
            description=
            ("Per-expert FP32 scale for up weights (E): S_max/448 with S_max the max FP32 block scale "
             "(max|w|/6 per 16-lane group) for that expert; FP8 block scales store (s/S_max)*448."
             ),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_qweights",
            description=
            "NVFP4 down-proj quantized payload (E, inter, K/2) INT8 with K=hidden_size (two FP4 per byte).",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_blocks_scale",
            description=
            "Down-proj Marlin block scales (E, inter, K/16) INT8 with K=hidden_size (four FP8 bytes per tile word).",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_global_scale",
            description=
            ("Per-expert FP32 scale for down weights (E); same convention as fc_up_global_scale "
             "(S_max/448)."),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="e_score_correction_bias",
            description=
            "NemotronH expert load-balancing correction bias (E) FP32. Zeros if not available.",
            type_str="tensor(float)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description=
            "MoE output (batch, seq_len, hidden_size) FP16, same shape as hidden_states.",
            type_str="tensor(float16)",
        ),
    ],
    type_constraints=[],
    attributes=[
        OpSchema.Attribute(
            name="num_experts",
            type=OpSchema.AttrType.INT,
            description="Number of experts E.",
            required=True,
        ),
        OpSchema.Attribute(
            name="top_k",
            type=OpSchema.AttrType.INT,
            description="Top-K experts per token.",
            required=True,
        ),
        OpSchema.Attribute(
            name="hidden_size",
            type=OpSchema.AttrType.INT,
            description="Hidden dimension (multiple of 64).",
            required=True,
        ),
        OpSchema.Attribute(
            name="moe_inter_size",
            type=OpSchema.AttrType.INT,
            description=
            "MoE intermediate size (multiple of 64; decode kernel strip tiling).",
            required=True,
        ),
        OpSchema.Attribute(
            name="activation_type",
            type=OpSchema.AttrType.INT,
            description=
            ("Expert MLP nonlinearity after up-proj: 0 = ReLU², 1 = SiLU "
             "(MoEActivationKind; see nvfp4MoePlugin.cpp nvfp4StoredActivationToKernelKind)."
             ),
            required=True,
        ),
        OpSchema.Attribute(
            name="quantization_group_size",
            type=OpSchema.AttrType.INT,
            description=
            "Marlin NVFP4 block scale group size along hidden_size (must be 16 for Nvfp4MoePlugin).",
            required=True,
        ),
        OpSchema.Attribute(
            name="n_group",
            type=OpSchema.AttrType.INT,
            description=
            "Number of expert groups for NemotronH grouped top-k routing.",
            required=True,
        ),
        OpSchema.Attribute(
            name="topk_group",
            type=OpSchema.AttrType.INT,
            description=
            "Number of groups to select in NemotronH grouped top-k routing.",
            required=True,
        ),
        OpSchema.Attribute(
            name="norm_topk_prob",
            type=OpSchema.AttrType.INT,
            description=
            "Whether to renormalize top-k weights to sum to 1 (0 or 1).",
            required=True,
        ),
        OpSchema.Attribute(
            name="routed_scaling_factor",
            type=OpSchema.AttrType.FLOAT,
            description="Scaling factor applied to final top-k weights.",
            required=True,
        ),
        OpSchema.Attribute(
            name="routing_mode",
            type=OpSchema.AttrType.INT,
            description=
            ("Router selection kernel: 0 = softmax + flat top-k (moeTopkSoftmax; default), "
             "1 = sigmoid + grouped top-k (moeSigmoidGroupTopk; NemotronH)."),
            required=True,
        ),
    ],
)
onnx.defs.register_schema(nvfp4_moe_plugin_schema)


@symbolic_helper.parse_args(
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "f",
    "i",
)
def symbolic_nvfp4_moe_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    router_logits: torch._C.Value,
    hidden_states: torch._C.Value,
    hidden_block_scale: torch._C.Value,
    hidden_global_scale: torch._C.Value,
    fc_up_qweights: torch._C.Value,
    fc_up_blocks_scale: torch._C.Value,
    fc_up_global_scale: torch._C.Value,
    fc_down_qweights: torch._C.Value,
    fc_down_blocks_scale: torch._C.Value,
    fc_down_global_scale: torch._C.Value,
    e_score_correction_bias: torch._C.Value,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
):
    output = g.op(
        "trt::Nvfp4MoePlugin",
        router_logits,
        hidden_states,
        hidden_block_scale,
        hidden_global_scale,
        fc_up_qweights,
        fc_up_blocks_scale,
        fc_up_global_scale,
        fc_down_qweights,
        fc_down_blocks_scale,
        fc_down_global_scale,
        e_score_correction_bias,
        num_experts_i=num_experts,
        top_k_i=top_k,
        hidden_size_i=hidden_size,
        moe_inter_size_i=moe_inter_size,
        activation_type_i=activation_type,
        quantization_group_size_i=quantization_group_size,
        n_group_i=n_group,
        topk_group_i=topk_group,
        norm_topk_prob_i=norm_topk_prob,
        routed_scaling_factor_f=routed_scaling_factor,
        routing_mode_i=routing_mode,
    )
    hs_sizes = _get_tensor_sizes(hidden_states)
    output.setType(router_logits.type().with_dtype(torch.float16).with_sizes(
        [hs_sizes[0], hs_sizes[1], hs_sizes[2]]))
    return output


@torch.library.custom_op("trt::nvfp4_moe_plugin", mutates_args=())
def nvfp4_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_block_scale: torch.Tensor,
    hidden_global_scale: torch.Tensor,
    fc_up_qweights: torch.Tensor,
    fc_up_blocks_scale: torch.Tensor,
    fc_up_global_scale: torch.Tensor,
    fc_down_qweights: torch.Tensor,
    fc_down_blocks_scale: torch.Tensor,
    fc_down_global_scale: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
) -> torch.Tensor:
    """
    Placeholder for ONNX tracing; TensorRT ``Nvfp4MoePlugin`` executes the real path. The ``routing_mode``
    attribute selects between ``moeTopkSoftmax`` (0, default) and ``moeSigmoidGroupTopk`` (1, NemotronH).
    """
    del (
        hidden_block_scale,
        hidden_global_scale,
        fc_up_qweights,
        fc_up_blocks_scale,
        fc_up_global_scale,
        fc_down_qweights,
        fc_down_blocks_scale,
        fc_down_global_scale,
        e_score_correction_bias,
        num_experts,
        top_k,
        moe_inter_size,
        activation_type,
        quantization_group_size,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        routing_mode,
    )
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be (batch, seq_len, hidden_size), got shape {tuple(hidden_states.shape)}"
        )
    b, s, hd = hidden_states.shape
    if int(hd) != int(hidden_size):
        raise ValueError(
            f"hidden_states last dim {hd} != hidden_size {hidden_size}")
    num_tokens = int(b) * int(s)
    if int(router_logits.shape[0]) != num_tokens:
        raise ValueError(
            f"router_logits leading dim {int(router_logits.shape[0])} must equal batch*seq_len ({num_tokens})"
        )
    return torch.zeros(
        b,
        s,
        hidden_size,
        dtype=torch.float16,
        device=router_logits.device,
    )


class NemotronHMoEW4A4Plugin(nn.Module):
    """
    ONNX export wrapper for TensorRT ``Nvfp4MoePlugin`` (Nemotron-style routed experts only).

    Experts follow **Nemotron-H** layout: ``down( act( up(x) ) )`` with no fused Qwen3-style
    ``gate_up_proj`` / GLU. Only :class:`NemotronHMoE` is supported (``NemotronHMoE(config, layer_idx)``).

    Router logits are computed with :meth:`nemotron_h_plugin_router_logits` so they match
    ``NemotronHTopkRouter``’s FP32 linear (see ``modeling_nemotron_h.py``). The TRT plugin then runs
    ``moeSigmoidGroupTopk`` — the NemotronH grouped top-k routing algorithm
    (sigmoid, ``e_score_correction_bias``, grouped top-k, renormalize, scale).

    Router logits are ``(num_tokens, num_experts)`` with ``num_tokens = batch * seq_len``; FP16
    ``hidden_states`` are ``(batch, seq_len, hidden_size)`` (same linear memory as a flattened
    ``(num_tokens, hidden_size)`` view). The custom op returns ``(batch, seq_len, hidden_size)`` in FP16,
    matching HF layout.

    Packed expert weights follow plugin layout ``w_up_ehi`` ``[E,H,I]`` / ``w_down_eih`` ``[E,I,H]``;
    HF ``NemotronHExperts`` stores ``up_proj`` / ``down_proj`` as ``[E,I,H]`` / ``[E,H,I]`` (transpose
    last two axes for Marlin packing).
    """

    @staticmethod
    def sigmoid_group_topk_torch(
        router_logits: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch equivalent of TensorRT ``kernel::moeSigmoidGroupTopk`` matching
        HuggingFace ``NemotronHMoE.route_tokens_to_experts``.
        """
        logits = router_logits.float()
        scores = torch.sigmoid(logits)
        biased = scores.clone()
        if correction_bias is not None:
            biased = biased + correction_bias.float()
        num_experts = logits.shape[-1]
        experts_per_group = num_experts // n_group
        # Top-2 per group → group scores
        grouped = biased.view(-1, n_group, experts_per_group)
        group_top2, _ = grouped.topk(2, dim=-1)
        group_scores = group_top2.sum(dim=-1)  # [T, nGroup]
        # Select topk_group groups
        _, group_idx = group_scores.topk(topk_group, dim=-1)  # [T, topk_group]
        # Build group mask
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        expert_mask = group_mask.unsqueeze(-1).expand(
            -1, -1, experts_per_group).reshape_as(biased)
        biased[~expert_mask] = float("-inf")
        # Top-K from masked biased scores
        _, topk_idx = biased.topk(top_k, dim=-1)
        # Gather weights from ORIGINAL sigmoid scores
        topk_weights = scores.gather(-1, topk_idx)
        if norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_idx.to(torch.int32)

    @staticmethod
    def moe_activation_numpy(z: np.ndarray,
                             activation_type: int) -> np.ndarray:
        """
        FP32 expert nonlinearity matching plugin / C++ ``referenceMoeActivation``: ``0`` = ReLU², ``1`` = SiLU
        (clipped to [-50, 50] before ``sigmoid`` form).
        """
        z = np.asarray(z, dtype=np.float32)
        if int(activation_type) == 1:
            zc = np.clip(z, -50.0, 50.0)
            return (zc / (1.0 + np.exp(-zc))).astype(np.float32)
        t = np.maximum(z, 0.0)
        return (t * t).astype(np.float32)

    @staticmethod
    def sigmoid_group_topk_numpy(
        router_logits: np.ndarray,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NumPy sigmoid + grouped top-k aligned with CUDA ``moeSigmoidGroupTopk``: iterative argmax with
        masking, **lower expert id wins on equal score** (same as the kernel's CUB reduce).
        """
        logits = np.asarray(router_logits, dtype=np.float32)
        num_tokens, num_experts = logits.shape
        sigmoid_scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        biased = sigmoid_scores.copy()
        if correction_bias is not None:
            biased = biased + np.asarray(correction_bias, dtype=np.float32)
        experts_per_group = num_experts // n_group
        k = min(int(top_k), int(num_experts))
        topw = np.zeros((num_tokens, k), dtype=np.float32)
        topi = np.zeros((num_tokens, k), dtype=np.int32)
        for t in range(int(num_tokens)):
            b_row = biased[t].copy()
            s_row = sigmoid_scores[t]
            # Top-2 per group → group scores
            group_scores = np.zeros(n_group, dtype=np.float32)
            for g in range(n_group):
                gs = g * experts_per_group
                vals = sorted(b_row[gs:gs + experts_per_group], reverse=True)
                group_scores[g] = vals[0] + (vals[1] if len(vals) > 1 else 0.0)
            # Select topk_group groups
            group_selected = np.zeros(n_group, dtype=bool)
            for _ in range(topk_group):
                best_g, best_s = -1, -np.inf
                for g in range(n_group):
                    if not group_selected[g] and group_scores[g] > best_s:
                        best_s = group_scores[g]
                        best_g = g
                if best_g >= 0:
                    group_selected[best_g] = True
            # Mask unselected groups
            for e in range(num_experts):
                if not group_selected[e // experts_per_group]:
                    b_row[e] = -np.inf
            # Top-K from masked biased scores
            renorm_sum = 0.0
            for ki in range(k):
                best_e, best_v = 0, -np.inf
                for e in range(num_experts):
                    if b_row[e] > best_v or (b_row[e] == best_v
                                             and e < best_e):
                        best_v = b_row[e]
                        best_e = e
                topi[t, ki] = np.int32(best_e)
                topw[t, ki] = s_row[best_e]
                renorm_sum += s_row[best_e]
                b_row[best_e] = -np.inf
            if norm_topk_prob and renorm_sum > 0:
                topw[t] = (topw[t] / renorm_sum).astype(np.float32)
            topw[t] = (topw[t] * routed_scaling_factor).astype(np.float32)
        return topw, topi

    @staticmethod
    def nemotron_h_plugin_router_logits(hidden_flat: torch.Tensor,
                                        gate: nn.Linear) -> torch.Tensor:
        """
        Compute router logits for the C++ ``moeTopkSoftmax`` kernel, matching HuggingFace
        ``NemotronHTopkRouter`` which uses sigmoid-based routing (not softmax).

        HF forward: ``scores = sigmoid(F.linear(x, w))``, top-k, renormalize, × routed_scaling_factor.
        The C++ plugin applies ``softmax(logits) → top-k → renorm``.

        To make ``softmax(z) / renorm == sigmoid(l) / renorm`` for the same top-k set, we pass
        ``z = log(sigmoid(l)) = F.logsigmoid(l)`` so that ``exp(z) = sigmoid(l)`` and
        ``softmax(log_sigmoid) == sigmoid(l) / Σ sigmoid`` — exactly the HF weight distribution.

        Expert selection is unchanged (log, sigmoid are both monotone → same top-k order).
        ``routed_scaling_factor`` is applied separately in ``forward()`` to match the HF ×scale step.
        """
        x = hidden_flat.float()
        w = gate.weight.float()
        if gate.bias is not None:
            logits = F.linear(x, w, gate.bias.float())
        else:
            logits = F.linear(x, w)
        # Transform raw logits → log(sigmoid(l)) so moeTopkSoftmax produces sigmoid-normalized weights.
        return F.logsigmoid(logits)

    @staticmethod
    def _nemotron_h_moe_gate_as_linear(gate: nn.Module, hidden_size: int,
                                       num_experts: int) -> nn.Linear:
        """
        Nemotron-H ``NemotronHMoE`` may use ``nn.Linear`` or ``NemotronHTopkRouter`` (DeepSeek-style
        ``weight`` matrix + ``F.linear``, no bias). Return an ``nn.Linear`` with the same mapping for
        plugin ONNX tracing.
        """
        if isinstance(gate, nn.Linear):
            return gate
        weight = getattr(gate, "weight", None)
        if isinstance(weight, nn.Parameter) and tuple(
                weight.shape) == (num_experts, hidden_size):
            lin = nn.Linear(
                hidden_size,
                num_experts,
                bias=False,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                lin.weight.copy_(weight)
            return lin
        raise TypeError(
            "NemotronHMoEW4A4Plugin expects moe_block.gate to be nn.Linear or a top-k router with "
            f"parameter weight of shape ({num_experts}, {hidden_size}); got {type(gate).__name__}"
        )

    def __init__(
        self,
        moe_block: "NemotronHMoE",
        *,
        activation_type: int = 0,
        quantization_group_size: int = 16,
    ):
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers.models.nemotron_h import failed)."
            )
        if not isinstance(moe_block, NemotronHMoE):
            raise TypeError(
                f"NemotronHMoEW4A4Plugin only supports NemotronHMoE, not {type(moe_block).__name__} "
                "(Qwen3 MoE uses fused gate/up and is incompatible with this plugin)."
            )
        super().__init__()
        cfg = moe_block.config
        hidden_size = int(cfg.hidden_size)
        latent = getattr(cfg, "moe_latent_size", None)
        expert_in = int(latent) if latent is not None else hidden_size
        if expert_in != hidden_size:
            raise ValueError(
                "NemotronHMoEW4A4Plugin needs expert input dim == hidden_size "
                f"(set moe_latent_size=None or {hidden_size}); got moe_latent_size={latent}"
            )
        # trust_remote_code NemotronHMOE: n_routed_experts/top_k live on .gate or
        # .experts.num_experts (NemotronHExperts has no __len__, use lazy eval).
        if hasattr(moe_block, "n_routed_experts"):
            num_experts = int(moe_block.n_routed_experts)
        elif hasattr(moe_block.gate, "n_routed_experts"):
            num_experts = int(moe_block.gate.n_routed_experts)
        else:
            num_experts = int(moe_block.experts.num_experts)
        gate_linear = NemotronHMoEW4A4Plugin._nemotron_h_moe_gate_as_linear(
            moe_block.gate, hidden_size, num_experts)
        top_k = int(
            getattr(moe_block, "top_k",
                    getattr(moe_block.gate, "top_k", cfg.num_experts_per_tok)))

        # Extract NemotronH routing parameters from config.
        n_group = int(getattr(cfg, "n_group", 1))
        topk_group = int(getattr(cfg, "topk_group", 1))
        norm_topk_prob = bool(getattr(cfg, "norm_topk_prob", True))
        routed_scaling_factor = float(
            getattr(cfg, "routed_scaling_factor", 1.0))

        # Extract e_score_correction_bias from the gate/router.
        correction_bias = getattr(moe_block.gate, "e_score_correction_bias",
                                  None)
        if correction_bias is None:
            correction_bias = getattr(moe_block, "e_score_correction_bias",
                                      None)
        self._init_from_gate_and_dims(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            moe_inter_size=int(cfg.moe_intermediate_size),
            gate_layer=gate_linear,
            activation_type=int(activation_type),
            quantization_group_size=int(quantization_group_size),
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
        )
        # routed_scaling_factor: HF NemotronHTopkRouter multiplies topk_weights by this
        # value after sigmoid + renorm (see modeling_nemotron_h.py, line ~917).
        # The C++ moeTopkSoftmax uses softmax + renorm and does NOT apply this factor,
        # so we compensate by scaling the routed expert output here before returning.
        # Default 1.0 for models that don't use this (standard softmax routing).
        self.routed_scaling_factor: float = float(
            getattr(cfg, "routed_scaling_factor", 1.0))

        # Shared expert is intentionally NOT stored here.
        # It is handled by NemotronHMoEWithSharedExperts, which wraps this plugin
        # and registers shared_experts as its own named submodule.  This ensures
        # each layer's shared expert gets a unique path in the ONNX graph
        # (e.g. layers.1.mlp.shared_experts vs layers.4.mlp.shared_experts),
        # avoiding the "Output name is not unique" ONNX error caused by modelopt
        # quantizer nodes sharing identical names when embedded inside the plugin.

    @classmethod
    def from_nemotron_h_moe(
        cls,
        moe_block: "NemotronHMoE",
        *,
        activation_type: int = 0,
        quantization_group_size: int = 16,
    ) -> "NemotronHMoEW4A4Plugin":
        """Alias for ``NemotronHMoEW4A4Plugin(moe_block, ...)``."""
        return cls(
            moe_block,
            activation_type=activation_type,
            quantization_group_size=quantization_group_size,
        )

    def _init_from_gate_and_dims(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_inter_size: int,
        gate_layer: nn.Linear,
        activation_type: int = 0,
        quantization_group_size: int = 16,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: torch.Tensor | nn.Parameter | None = None,
    ) -> None:
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.moe_inter_size = int(moe_inter_size)
        self.activation_type = int(activation_type)
        self.quantization_group_size = int(quantization_group_size)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.norm_topk_prob = int(bool(norm_topk_prob))
        self.routed_scaling_factor = float(routed_scaling_factor)
        # NemotronHMoEW4A4Plugin always drives the sigmoid + grouped top-k routing kernel (routing_mode=1).
        self.routing_mode = 1

        if self.quantization_group_size != 16:
            raise ValueError(
                "Nvfp4MoePlugin only supports quantization_group_size == 16 "
                f"(Marlin NVFP4 tile scales); got {self.quantization_group_size}"
            )
        if self.hidden_size % self.quantization_group_size != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be a multiple of "
                f"quantization_group_size ({self.quantization_group_size})")

        if self.hidden_size % 64 != 0:
            raise ValueError(
                f"hidden_size must be a multiple of 64, got {self.hidden_size}"
            )
        if self.moe_inter_size % 64 != 0:
            raise ValueError(
                f"moe_inter_size must be a multiple of 64 for Nvfp4MoePlugin (decode GEMV strips), "
                f"got {self.moe_inter_size}")

        self.num_chunks = self.hidden_size // 64

        if gate_layer.out_features != self.num_experts:
            raise ValueError(
                f"gate.out_features ({gate_layer.out_features}) must equal num_experts ({self.num_experts})"
            )
        if gate_layer.in_features != self.hidden_size:
            raise ValueError(
                f"gate.in_features ({gate_layer.in_features}) must equal hidden_size ({self.hidden_size})"
            )

        # FP32 weights match ``NemotronHTopkRouter`` / ``DeepseekV3TopkRouter`` (FP32 matmul). Using FP16 here
        # would quantize router weights and diverge from HF logits; call ``module.half()`` before export if FP16
        # parameters are required for the rest of the graph.
        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=gate_layer.bias is not None,
            dtype=torch.float32,
        )
        self.gate.weight.data = gate_layer.weight.data.clone().to(
            torch.float32)
        if gate_layer.bias is not None:
            self.gate.bias.data = gate_layer.bias.data.clone().to(
                torch.float32)

        e, inter = self.num_experts, self.moe_inter_size
        k_half = self.hidden_size // 2
        k_per_group = self.hidden_size // self.quantization_group_size
        # Cutlass Atom layout requires SF M-dimension padded to multiple of 128.
        up_sf_m_padded = ((self.hidden_size + 127) //
                          128) * 8  # 8 SF rows per 128-row M-tile
        dn_sf_m_padded = ((inter + 127) // 128) * 128
        self.register_buffer("fc_up_qweights",
                             torch.zeros(e, k_half, inter, dtype=torch.int8))
        self.register_buffer(
            "fc_up_blocks_scale",
            torch.zeros(e, up_sf_m_padded, inter, dtype=torch.int8))
        self.register_buffer("fc_up_global_scale",
                             torch.ones(e, dtype=torch.float32))
        self.register_buffer("fc_down_qweights",
                             torch.zeros(e, inter, k_half, dtype=torch.int8))
        self.register_buffer(
            "fc_down_blocks_scale",
            torch.zeros(e, dn_sf_m_padded, k_per_group, dtype=torch.int8))
        self.register_buffer("fc_down_global_scale",
                             torch.ones(e, dtype=torch.float32))
        # W4A16 export: unused by the plugin; W4A4 graphs replace with real NVFP4 activation scales.
        # Keep dtype=int8: the C++ plugin's supportsFormatCombination() expects INT8 here.
        # Identity nodes that torch.onnx.export inserts between the shared initializer and
        # each plugin node are removed in _elide_int8_identity_nodes() (onnx_utils.py).
        self.register_buffer(
            "hidden_block_scale",
            torch.zeros(1, 1, 1, dtype=torch.int8),
        )
        self.register_buffer("hidden_global_scale",
                             torch.ones(1, dtype=torch.float32))
        # e_score_correction_bias [E] FP32: NemotronH expert load balancing.
        if correction_bias is not None:
            bias_data = correction_bias.data.clone().to(torch.float32)
        else:
            bias_data = torch.zeros(e, dtype=torch.float32)
        self.register_buffer("e_score_correction_bias", bias_data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(
                f"hidden_dim {hidden_dim} != hidden_size {self.hidden_size}")

        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = type(self).nemotron_h_plugin_router_logits(
            hidden_flat, self.gate)

        out = nvfp4_moe_plugin(
            router_logits,
            hidden_states,
            self.hidden_block_scale,
            self.hidden_global_scale,
            self.fc_up_qweights,
            self.fc_up_blocks_scale,
            self.fc_up_global_scale,
            self.fc_down_qweights,
            self.fc_down_blocks_scale,
            self.fc_down_global_scale,
            self.e_score_correction_bias,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_inter_size,
            self.activation_type,
            self.quantization_group_size,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
            self.routing_mode,
        )
        # Apply routed_scaling_factor: HF NemotronHTopkRouter scales topk_weights by
        # this value but moeTopkSoftmax (used inside the C++ plugin) does not.
        # This compensates so routed expert contributions match the HF reference.
        if self.routed_scaling_factor != 1.0:
            out = out * self.routed_scaling_factor
        return out

    def populate_marlin_plugin_buffers(
        self,
        w_up_ehi,
        w_down_eih,
    ) -> None:
        """Pack dense Marlin-layout weights into this plugin's NVFP4 buffers and block scales.

        ``w_up_ehi`` is ``[E, H, I]``, ``w_down_eih`` is ``[E, I, H]`` (``Nvfp4MoePlugin`` layout).
        Per-expert ``fc_*_global_scale`` is ``S_max / MarlinConverter.FP8_MAX`` with ``S_max`` from
        that expert's up/down tensor.

        Vectorized over all tiles within each expert (numpy ops) — replaces the previous
        triple-nested Python loop (~20M calls for 128 experts) with a handful of broadcast
        operations per expert.  Peak extra RAM: ~50 MB per expert (processed sequentially).
        """
        e, h, inter = w_up_ehi.shape
        assert w_down_eih.shape == (e, inter, h)
        if int(e) != int(self.num_experts):
            raise ValueError(
                f"w_up_ehi expert count {e} != self.num_experts {self.num_experts}"
            )
        if int(h) != int(self.hidden_size) or int(inter) != int(
                self.moe_inter_size):
            raise ValueError(
                f"w_up_ehi shape ({e},{h},{inter}) mismatches plugin "
                f"(hidden_size={self.hidden_size}, moe_inter_size={self.moe_inter_size})"
            )
        nIC = inter // 64  # I-dimension chunks (tiles for up_proj)
        nHC = h // 64  # H-dimension chunks (tiles for down_proj)
        assert nIC * 64 == inter
        assert nHC * 64 == h

        w_up_np = w_up_ehi.float().detach().cpu().numpy()  # [E, H, I]
        w_dn_np = w_down_eih.float().detach().cpu().numpy()  # [E, I, H]

        s_max_up = np.maximum(
            np.abs(w_up_np).reshape(e, -1).max(axis=1) / 6.0, 1e-12)  # [E]
        s_max_dn = np.maximum(
            np.abs(w_dn_np).reshape(e, -1).max(axis=1) / 6.0, 1e-12)  # [E]

        k_fp8 = MarlinConverter.FP8_MAX

        # Midpoint boundaries between consecutive FP4 E2M1 positive levels:
        # levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        # np.searchsorted(bounds, |x|) gives the correct level index without
        # materialising an [N, 8] distance tensor.
        _E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                                dtype=np.float32)

        def _pack_one_expert(
            w_ah: np.ndarray,
            s_max_ex: float,
            nC: int,
        ):
            """Vectorized quantization + packing for one expert weight matrix.

            Args:
                w_ah:     [A, B] float32, B = nC * 64 (tiles along B dimension).
                s_max_ex: expert-level scale max (scalar float).
                nC:       number of 64-element tile chunks along B.

            Returns:
                payload_i8: [A, nC, 32] int8  — 2 FP4 E2M1 nibbles packed per byte.
                bs_i32:     [A, nC]     int32 — Marlin f8x4 block-scale word per tile.

            Buffer reshape invariants (caller's responsibility):
                up:  [H, nIC, 32] int8  → [H//2,   I  ] (H*nIC*32 == (H//2)*I)
                up:  [H, nIC]     int32 → [H//16,  I  ] (H*nIC*4  == (H//16)*I)
                dn:  [I, nHC, 32] int8  → [I,      H//2 ] (nHC*32 == H//2)
                dn:  [I, nHC]     int32 → [I,      H//16] (nHC*4  == H//16)
            """
            A, B = w_ah.shape
            # Reshape into sub-group tiles: [A, nC, 4, 16]
            w_tiles = w_ah.reshape(A, nC, 4, 16)

            # Per-sub-group scale: max(|tile_group|) / 6.0, clamped
            group_max = np.abs(w_tiles).max(axis=-1)  # [A, nC, 4]
            group_scales = np.maximum(group_max / 6.0, 1e-12)  # [A, nC, 4]

            # Scale elements by their sub-group scale and clip to [-6, 6]
            w_scaled = (w_tiles / group_scales[..., np.newaxis]).clip(
                -6.0, 6.0)
            # [A, nC, 4, 16]

            # FP4 E2M1 nearest-neighbour (vectorized via searchsorted on 7 boundaries)
            x_abs = np.abs(w_scaled)
            best_idx = np.searchsorted(_E2M1_BOUNDS, x_abs).astype(
                np.uint8)  # [A, nC, 4, 16]
            sign_bits = (w_scaled < 0).astype(np.uint8) << np.uint8(3)
            nibbles = (best_idx | sign_bits).reshape(A, nC,
                                                     64)  # values 0x0..0xF

            # Pack pairs of nibbles into bytes: lo | (hi << 4) → [A, nC, 32] uint8
            lo = nibbles[..., ::2] & np.uint8(0xF)
            hi = nibbles[..., 1::2] & np.uint8(0xF)
            payload_i8 = (lo | (hi << np.uint8(4))).astype(np.uint8).view(
                np.int8)
            # [A, nC, 32] int8

            # Vectorized fp32x4_to_marlin_f8x4_block_scale:
            # Normalise to FP8 range, cast to FP16, then repack via Marlin bit shuffle.
            scales_norm = (group_scales / s_max_ex * k_fp8).astype(np.float16)
            # [A, nC, 4]

            # View each FP16 as uint16, then combine pairs into uint32 half2 words
            h16 = scales_norm.view(np.uint16).astype(np.uint32)  # [A, nC, 4]
            # out2 packs s[0],s[1]; out1 packs s[2],s[3]  (matches fp32x4_to_marlin order)
            out2_raw = h16[..., 0] | (h16[..., 1] << np.uint32(16))  # [A, nC]
            out1_raw = h16[..., 2] | (h16[..., 3] << np.uint32(16))  # [A, nC]

            # Vectorized fp32x4_to_marlin_f8x4_block_scale end-to-end:
            #
            # The scalar path is: project(out_raw) then f16x4_in_u32x2_...(proj1, proj2).
            # f16x4_in_u32x2 applies another << 1 to proj before extracting bytes, so the
            # >> 1 inside project and the << 1 inside f16x4 cancel.  Combined formula:
            #
            #   u = (out_raw << 1) & 0xFF00FF00          (shift, keep odd bytes)
            #   b_high = (u >> 24) & 0xFF                (top byte)
            #   b_low  = (u >>  8) & 0xFF                (second byte)
            #
            # out1_raw → b1 (low), b3 (high); out2_raw → b0 (low), b2 (high).
            u1 = (out1_raw.astype(np.uint64) <<
                  np.uint64(1)) & np.uint64(0xFF00FF00)
            u2 = (out2_raw.astype(np.uint64) <<
                  np.uint64(1)) & np.uint64(0xFF00FF00)
            b0 = (u2 >> np.uint64(8)) & np.uint64(0xFF)
            b1 = (u1 >> np.uint64(8)) & np.uint64(0xFF)
            b2 = (u2 >> np.uint64(24)) & np.uint64(0xFF)
            b3 = (u1 >> np.uint64(24)) & np.uint64(0xFF)
            bs_i32 = (b0 | (b1 << np.uint64(8)) | (b2 << np.uint64(16)) |
                      (b3 << np.uint64(24))).astype(np.int32)  # [A, nC]

            return payload_i8, bs_i32

        # --- up_proj: tile along I; buffer layout [E, H//2, I] / [E, H//16, I] ---
        # Tile (jj, c): w_up[ex, jj, c*64:(c+1)*64].  Tile index = jj*nIC + c.
        # Memory: [H, nIC, 32] int8 flattens to H*nIC*32 = H*I//2 = (H//2)*I  ✓
        for ex in range(e):
            pl, sc = _pack_one_expert(w_up_np[ex], float(s_max_up[ex]), nIC)
            assert pl.shape == (h, nIC, 32), (
                f"up pl shape {pl.shape} != ({h}, {nIC}, 32) for expert {ex}")
            assert sc.shape == (h, nIC), (
                f"up sc shape {sc.shape} != ({h}, {nIC}) for expert {ex}")
            # pl: [H, nIC, 32] → [H//2, I]  (H*nIC*32 == (H//2)*I since nIC==I//64)
            self.fc_up_qweights[ex].copy_(
                torch.from_numpy(pl.reshape(h // 2, inter)))
            # sc: [H, nIC] int32 → view [H, nIC, 4] int8 → [H//16, I]
            # Buffer may be padded (atom-layout); copy into the valid rows only.
            sf_rows_up = h // 16
            self.fc_up_blocks_scale[ex, :sf_rows_up, :].copy_(
                torch.from_numpy(sc.view(np.int8).reshape(sf_rows_up, inter)))

        self.fc_up_global_scale.copy_(
            torch.from_numpy(s_max_up / k_fp8).to(
                device=self.fc_up_global_scale.device,
                dtype=self.fc_up_global_scale.dtype))

        # --- down_proj: tile along H; buffer layout [E, I, H//2] / [E, I, H//16] ---
        # Tile (c, j): w_down[ex, j, c*64:(c+1)*64].  Stored at dn_pl[ex, j, c*32:].
        # Memory: [I, nHC, 32] int8 → [I, nHC*32] = [I, H//2]  ✓
        for ex in range(e):
            pl, sc = _pack_one_expert(w_dn_np[ex], float(s_max_dn[ex]), nHC)
            assert pl.shape == (inter, nHC, 32), (
                f"dn pl shape {pl.shape} != ({inter}, {nHC}, 32) for expert {ex}"
            )
            assert sc.shape == (inter, nHC), (
                f"dn sc shape {sc.shape} != ({inter}, {nHC}) for expert {ex}")
            # pl: [I, nHC, 32] → [I, H//2]  (nHC*32 == H//2 since nHC==H//64)
            self.fc_down_qweights[ex].copy_(
                torch.from_numpy(pl.reshape(inter, h // 2)))
            # sc: [I, nHC] int32 → view [I, nHC, 4] int8 → [I, H//16]
            # Buffer may be padded (atom-layout); copy into the valid rows only.
            self.fc_down_blocks_scale[ex, :inter, :].copy_(
                torch.from_numpy(sc.view(np.int8).reshape(inter, h // 16)))

        self.fc_down_global_scale.copy_(
            torch.from_numpy(s_max_dn / k_fp8).to(
                device=self.fc_down_global_scale.device,
                dtype=self.fc_down_global_scale.dtype))

    def pack_experts_weights_to_marlin(self, moe: "NemotronHMoE") -> None:
        """
        Pack **all** NemotronH routed experts from HuggingFace ``NemotronHExperts`` into ``self`` buffers.

        Reads ``moe.experts.up_proj`` / ``down_proj`` (HF ``[E,I,H]`` / ``[E,H,I]``) with ``transpose(1,2)``
        into Marlin layout. Only supports ``moe.config.moe_latent_size is None`` (expert input dim ==
        ``hidden_size``); latent MoE raises ``ValueError``.
        """
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers.models.nemotron_h import failed)."
            )
        if not isinstance(moe, NemotronHMoE):
            raise TypeError(
                f"moe must be NemotronHMoE, got {type(moe).__name__!r}")
        if not hasattr(moe, "experts"):
            raise TypeError(
                f"moe must have ``experts``; got {type(moe).__name__!r}")
        # trust_remote_code NemotronHMOE: n_routed_experts lives on .gate or
        # .experts.num_experts (NemotronHExperts has no __len__, use lazy eval).
        _gate = getattr(moe, "gate", None)
        if hasattr(moe, "n_routed_experts"):
            _n_experts = int(moe.n_routed_experts)
        elif _gate is not None and hasattr(_gate, "n_routed_experts"):
            _n_experts = int(_gate.n_routed_experts)
        else:
            _n_experts = int(moe.experts.num_experts)
        if _n_experts != int(self.num_experts):
            raise ValueError(
                f"moe n_routed_experts ({_n_experts}) != self.num_experts ({self.num_experts})"
            )

        cfg = getattr(moe, "config", None)
        if cfg is None:
            raise TypeError(
                "packing Marlin weights from HF needs ``moe.config`` (NemotronHConfig) "
                "to resolve expert weight shapes (moe_intermediate_size, moe_latent_size, hidden_size)"
            )
        if int(cfg.moe_intermediate_size) != int(self.moe_inter_size):
            raise ValueError(
                f"moe.config.moe_intermediate_size ({cfg.moe_intermediate_size}) != "
                f"self.moe_inter_size ({self.moe_inter_size})")
        latent = getattr(cfg, "moe_latent_size", None)
        expert_in = int(latent) if latent is not None else int(cfg.hidden_size)
        if expert_in != int(self.hidden_size):
            raise ValueError(
                "Nvfp4MoePlugin / NemotronHMoEW4A4Plugin expect expert input dim == model hidden_size "
                f"(use ``moe_latent_size=None``); got expert_input_dim={expert_in} from moe.config, "
                f"self.hidden_size={self.hidden_size}")

        experts = moe.experts
        e_ct = _n_experts
        if isinstance(experts, nn.ModuleList):
            # trust_remote_code NemotronHMOE: individual MLP modules in a ModuleList.
            # Each expert.up_proj.weight is [I, H], expert.down_proj.weight is [H, I].
            # Stack into [E, I, H] / [E, H, I] matching the fused NemotronHExperts layout.
            up_list = [e.up_proj.weight.data for e in experts]
            down_list = [e.down_proj.weight.data for e in experts]
            up_stacked = torch.stack(up_list, dim=0)  # [E, I, H]
            down_stacked = torch.stack(down_list, dim=0)  # [E, H, I]
        else:
            up_raw = experts.up_proj
            down_raw = experts.down_proj
            if not isinstance(up_raw, nn.Parameter) or not isinstance(
                    down_raw, nn.Parameter):
                raise TypeError(
                    "expected experts.up_proj and experts.down_proj to be nn.Parameter stacked tensors "
                    f"(got up_proj={type(up_raw).__name__!r}, down_proj={type(down_raw).__name__!r})"
                )
            if int(up_raw.shape[0]) != e_ct or int(down_raw.shape[0]) != e_ct:
                raise ValueError(
                    f"expert dim mismatch: n_routed_experts={e_ct}, up.shape={tuple(up_raw.shape)}, "
                    f"down.shape={tuple(down_raw.shape)}")
            up_stacked = up_raw.data  # [E, I, H]
            down_stacked = down_raw.data  # [E, H, I]
        # Marlin layout: w_up [E, H, I], w_down [E, I, H]
        w_up_ehi = up_stacked.transpose(1, 2).contiguous()
        w_down_eih = down_stacked.transpose(1, 2).contiguous()
        self.populate_marlin_plugin_buffers(w_up_ehi, w_down_eih)


def register_nvfp4_moe_plugin_onnx_symbolic_functions() -> None:
    register_custom_op_symbolic(
        "trt::nvfp4_moe_plugin",
        symbolic_nvfp4_moe_plugin,
        ONNX_OPSET_VERSION,
    )


class NemotronHMoEWithSharedExperts(nn.Module):
    """Thin wrapper that runs NemotronHMoEW4A4Plugin + shared expert in sequence.

    The shared expert is registered as a named submodule (``self.shared_experts``)
    so that, at each MoE layer position, it appears under a unique path in the model
    hierarchy (e.g. ``layers.1.mlp.shared_experts`` vs ``layers.4.mlp.shared_experts``).

    Each instance receives a *deepcopy* of the original shared_experts module.  This
    is required even when the caller passes distinct module objects, because in some
    Nemotron-H checkpoints the shared_experts attribute is weight-tied across several
    MoE layers (the same Python object appears in multiple ``NemotronHMoE`` blocks).
    When torch.onnx.export encounters the same module object called from multiple
    scopes it collapses the scope names, producing "Output name is not unique" for the
    shared-expert sub-graph.  The deepcopy gives each wrapper a genuinely independent
    nn.Module object with its own Python identity, so TorchScript assigns every wrapper
    a distinct scope path and the duplicate-name error disappears.

    Mirrors ``NemotronHMoE.forward()``:
        out = moe(hidden_states) + shared_experts(hidden_states)
    """

    def __init__(
        self,
        plugin: "NemotronHMoEW4A4Plugin",
        shared_experts: nn.Module,
    ) -> None:
        super().__init__()
        self.moe_plugin = plugin
        # deepcopy so every wrapper instance owns a distinct module object.
        # Without this, weight-tied shared_experts (same Python id across layers)
        # collapse into a single ONNX scope → "Output name is not unique".
        try:
            self.shared_experts = copy.deepcopy(shared_experts)
        except Exception as _e:
            print(
                f"[NemotronHMoEWithSharedExperts] WARNING: deepcopy failed ({type(_e).__name__}: {_e}), "
                "using original module — ONNX export may fail with duplicate output names"
            )
            self.shared_experts = shared_experts  # fallback: best-effort

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe_plugin(hidden_states) + self.shared_experts(
            hidden_states)


def replace_moe_blocks_with_nvfp4_plugin(model: nn.Module) -> nn.Module:
    """Replace ``NemotronHMoE`` blocks with ``NemotronHMoEW4A4Plugin`` (NVFP4 buffers must be filled).

    When the ``NemotronHMoE`` block has a ``shared_experts`` attribute, the replacement
    is wrapped in ``NemotronHMoEWithSharedExperts`` so that the shared expert is a
    registered submodule at the layer's position in the hierarchy.  This keeps ONNX
    node names unique across all MoE layers (each layer gets its own path prefix).
    """
    if NemotronHMoE is None:
        return model
    for name, module in list(model.named_modules()):
        new_module = None
        if isinstance(module, NemotronHMoE):
            try:
                plugin = NemotronHMoEW4A4Plugin(module)
                shared = getattr(module, "shared_experts", None)
                if shared is not None:
                    # Wrap: plugin handles routed MoE, wrapper owns shared_experts
                    # as a registered submodule so each layer gets a unique ONNX path.
                    new_module = NemotronHMoEWithSharedExperts(plugin, shared)
                else:
                    new_module = plugin
            except Exception as _e:
                print(
                    f"[nvfp4_moe_plugin] WARNING: failed to replace {name!r}: {type(_e).__name__}: {_e}"
                )
                new_module = None
        if new_module is None:
            continue
        parent = model
        if "." in name:
            parent_name, module_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
        else:
            module_name = name
        setattr(parent, module_name, new_module)
    return model


def is_moe_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", "")
    return "moe" in model_type.lower()
