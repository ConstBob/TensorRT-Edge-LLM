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
``kernel::moeTopkSoftmax`` (softmax over experts, top-k, renormalize); see
:meth:`NemotronHMoEW4A4Plugin.moe_topk_softmax_renormalize_torch`. ``hidden_size`` and ``moe_inter_size`` must be
multiples of 64 (decode GEMV / Marlin tile chunks). Expert up
quantized weights are INT8 ``[E, hidden_size/2, moe_inter_size]`` (two NVFP4 nibbles per byte) with
INT8 Marlin block scales ``[E, hidden_size/16, moe_inter_size]``. Down weights are INT8
``[E, moe_inter_size, hidden_size/2]`` with block scales ``[E, moe_inter_size, hidden_size/16]``.
Activations are dense FP16 (W4A16).
Plugin output tensors are FP16.

HuggingFace ``NemotronHMoE`` (``transformers.models.nemotron_h.modeling_nemotron_h``) uses
``NemotronHTopkRouter`` / ``DeepseekV3TopkRouter`` for **pre-routing logits** (FP32 matmul), then
``route_tokens_to_experts`` (sigmoid, grouped top-k, correction bias). That post-processing is **not** in
the TRT plugin; :class:`NemotronHMoEW4A4Plugin` matches the router **linear** only via
:meth:`NemotronHMoEW4A4Plugin.nemotron_h_plugin_router_logits`.

**Expert weight layout:** HuggingFace ``NemotronHExperts`` stores ``up_proj`` as ``[E, I, H_in]`` and
``down_proj`` as ``[E, H_in, I]`` (``I`` = ``moe_intermediate_size``, ``H_in`` = ``moe_latent_size`` or
``hidden_size``). The TensorRT ``Nvfp4MoePlugin`` / Marlin pack path uses ``w_up_ehi`` ``[E, H_in, I]`` and
``w_down_eih`` ``[E, I, H_in]``; use ``experts.up_proj.data.transpose(1, 2)`` /
``experts.down_proj.data.transpose(1, 2)`` into Marlin layout (see :meth:`NemotronHMoEW4A4Plugin.pack_experts_weights_to_marlin`).
"""

from __future__ import annotations

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
             "softmax + top-k + renormalize (moeTopkSoftmax) internally."),
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
    "i",
    "i",
    "i",
    "i",
    "i",
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
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
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
        num_experts_i=num_experts,
        top_k_i=top_k,
        hidden_size_i=hidden_size,
        moe_inter_size_i=moe_inter_size,
        activation_type_i=activation_type,
        quantization_group_size_i=quantization_group_size,
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
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
) -> torch.Tensor:
    """
    Placeholder for ONNX tracing; TensorRT ``Nvfp4MoePlugin`` executes the real path (including
    ``moeTopkSoftmax`` on ``router_logits``).
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
        num_experts,
        top_k,
        moe_inter_size,
        activation_type,
        quantization_group_size,
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
    :meth:`moe_topk_softmax_renormalize_torch`’s equivalent in CUDA—not HF ``route_tokens_to_experts``
    (sigmoid, ``e_score_correction_bias``, grouped top-k).

    Router logits are ``(num_tokens, num_experts)`` with ``num_tokens = batch * seq_len``; FP16
    ``hidden_states`` are ``(batch, seq_len, hidden_size)`` (same linear memory as a flattened
    ``(num_tokens, hidden_size)`` view). The custom op returns ``(batch, seq_len, hidden_size)`` in FP16,
    matching HF layout.

    Packed expert weights follow plugin layout ``w_up_ehi`` ``[E,H,I]`` / ``w_down_eih`` ``[E,I,H]``;
    HF ``NemotronHExperts`` stores ``up_proj`` / ``down_proj`` as ``[E,I,H]`` / ``[E,H,I]`` (transpose
    last two axes for Marlin packing).
    """

    @staticmethod
    def moe_topk_softmax_renormalize_torch(
        router_logits: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch equivalent of TensorRT ``kernel::moeTopkSoftmax`` with ``renormalize=True`` (see
        ``cpp/kernels/moe/moeTopkSoftmaxKernels.h`` / ``Nvfp4MoePlugin::enqueue``).

        Applies softmax on the expert dimension, takes the top-k probabilities, then renormalizes those
        weights so each row sums to 1. Returned indices are ``int32`` to match plugin workspace tensors.

        **Tie-breaking:** ``torch.topk`` ordering for equal probabilities is not guaranteed to match CUDA
        ``moeTopkSoftmax`` (lower expert index wins). For strict parity with the TRT plugin, use
        :meth:`moe_topk_softmax_renormalize_numpy` on logits in NumPy.
        """
        logits = router_logits.float()
        probs = F.softmax(logits, dim=-1)
        k = min(int(top_k), probs.shape[-1])
        topw, topi = torch.topk(probs, k, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
        return topw, topi.to(torch.int32)

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
    def moe_topk_softmax_renormalize_numpy(
        router_logits: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NumPy softmax + top-k + renormalize aligned with CUDA ``moeTopkSoftmax``: iterative argmax with
        masking, **lower expert id wins on equal probability** (same as the fused kernel's warp reduce).
        """
        logits = np.asarray(router_logits, dtype=np.float32)
        m = np.max(logits, axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        probs = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
        num_tokens, num_experts = probs.shape
        k = min(int(top_k), int(num_experts))
        topw = np.zeros((num_tokens, k), dtype=np.float32)
        topi = np.zeros((num_tokens, k), dtype=np.int32)
        for t in range(int(num_tokens)):
            row = probs[t]
            used = np.zeros(num_experts, dtype=bool)
            for ki in range(k):
                best_e = -1
                best_p = np.float32(-1.0)
                for e in range(int(num_experts)):
                    if used[e]:
                        continue
                    p = row[e]
                    if best_e < 0 or p > best_p or (p == best_p
                                                    and e < best_e):
                        best_p = p
                        best_e = e
                used[best_e] = True
                topi[t, ki] = np.int32(best_e)
                topw[t, ki] = best_p
            denom = float(np.sum(topw[t])) + 1e-20
            topw[t] = (topw[t] / denom).astype(np.float32)
        return topw, topi

    @staticmethod
    def nemotron_h_plugin_router_logits(hidden_flat: torch.Tensor,
                                        gate: nn.Linear) -> torch.Tensor:
        """
        Compute router logits the same way as HuggingFace ``NemotronHTopkRouter.forward`` (generated from
        ``DeepseekV3TopkRouter`` in ``modeling_nemotron_h.py`` / ``modeling_deepseek_v3.py``): flatten to
        ``[num_tokens, hidden_size]``, cast to FP32, then ``F.linear(x, weight.float()[, bias.float()])``.

        ``gate`` is the :class:`nn.Linear` clone built by :meth:`_nemotron_h_moe_gate_as_linear` (bias False
        for the usual TopkRouter).
        """
        x = hidden_flat.float()
        w = gate.weight.float()
        if gate.bias is not None:
            return F.linear(x, w, gate.bias.float())
        return F.linear(x, w)

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
        latent = cfg.moe_latent_size
        expert_in = int(latent) if latent is not None else hidden_size
        if expert_in != hidden_size:
            raise ValueError(
                "NemotronHMoEW4A4Plugin needs expert input dim == hidden_size "
                f"(set moe_latent_size=None or {hidden_size}); got moe_latent_size={latent}"
            )
        num_experts = int(moe_block.n_routed_experts)
        gate_linear = NemotronHMoEW4A4Plugin._nemotron_h_moe_gate_as_linear(
            moe_block.gate, hidden_size, num_experts)
        self._init_from_gate_and_dims(
            num_experts=num_experts,
            top_k=int(moe_block.top_k),
            hidden_size=hidden_size,
            moe_inter_size=int(cfg.moe_intermediate_size),
            gate_layer=gate_linear,
            activation_type=int(activation_type),
            quantization_group_size=int(quantization_group_size),
        )

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
    ) -> None:
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.moe_inter_size = int(moe_inter_size)
        self.activation_type = int(activation_type)
        self.quantization_group_size = int(quantization_group_size)

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
        self.register_buffer("fc_up_qweights",
                             torch.zeros(e, k_half, inter, dtype=torch.int8))
        self.register_buffer(
            "fc_up_blocks_scale",
            torch.zeros(e, k_per_group, inter, dtype=torch.int8))
        self.register_buffer("fc_up_global_scale",
                             torch.ones(e, dtype=torch.float32))
        self.register_buffer("fc_down_qweights",
                             torch.zeros(e, inter, k_half, dtype=torch.int8))
        self.register_buffer(
            "fc_down_blocks_scale",
            torch.zeros(e, inter, k_per_group, dtype=torch.int8))
        self.register_buffer("fc_down_global_scale",
                             torch.ones(e, dtype=torch.float32))
        # W4A16 export: unused by the plugin; W4A4 graphs replace with real NVFP4 activation scales.
        self.register_buffer(
            "hidden_block_scale",
            torch.zeros(1, 1, 1, dtype=torch.int8),
        )
        self.register_buffer("hidden_global_scale",
                             torch.ones(1, dtype=torch.float32))

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
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_inter_size,
            self.activation_type,
            self.quantization_group_size,
        )
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
        num_inter_chunks = inter // 64
        num_hidden_chunks = h // 64
        assert num_inter_chunks * 64 == inter
        assert num_hidden_chunks * 64 == h

        w_up_f = w_up_ehi.float()
        w_dn_f = w_down_eih.float()
        s_max_up = (w_up_f.abs().amax(dim=(1, 2)) / 6.0).clamp(min=1e-12)
        s_max_dn = (w_dn_f.abs().amax(dim=(1, 2)) / 6.0).clamp(min=1e-12)
        s_max_up_np = s_max_up.detach().cpu().numpy()
        s_max_dn_np = s_max_dn.detach().cpu().numpy()
        k_fp8 = MarlinConverter.FP8_MAX

        up_pl = self.fc_up_qweights
        up_bs = self.fc_up_blocks_scale
        dn_pl = self.fc_down_qweights
        dn_bs = self.fc_down_blocks_scale
        for ex in range(e):
            s_max_up_ex = float(s_max_up_np[ex])
            s_max_dn_ex = float(s_max_dn_np[ex])
            up_pl_flat = up_pl[ex].reshape(-1)
            up_bs_flat = up_bs[ex].reshape(-1)
            for jj in range(h):
                for c in range(num_inter_chunks):
                    up_seg = w_up_ehi[ex, jj, c * 64:(c + 1) *
                                      64].float().detach().cpu().numpy()
                    pl_u, sq_u = MarlinConverter.quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
                        up_seg, expert_block_scale_max_fp32=s_max_up_ex)
                    tile_u = jj * num_inter_chunks + c
                    up_pl_flat[tile_u * 32:(tile_u + 1) * 32].copy_(
                        torch.from_numpy(pl_u.view(np.int8)))
                    sq_u_t = torch.tensor(
                        [MarlinConverter.marlin_f8x4_block_scale_as_i32(sq_u)],
                        dtype=torch.int32,
                    )
                    up_bs_flat[tile_u * 4:(tile_u + 1) * 4].copy_(
                        sq_u_t.view(torch.int8))
            for c in range(num_hidden_chunks):
                for j in range(inter):
                    dn_seg = w_down_eih[ex, j, c * 64:(c + 1) *
                                        64].float().detach().cpu().numpy()
                    pl_d, sq_d = MarlinConverter.quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
                        dn_seg, expert_block_scale_max_fp32=s_max_dn_ex)
                    dn_pl[ex, j, c * 32:(c + 1) * 32].copy_(
                        torch.from_numpy(pl_d.view(np.int8)))
                    sq_d_t = torch.tensor(
                        [MarlinConverter.marlin_f8x4_block_scale_as_i32(sq_d)],
                        dtype=torch.int32,
                    )
                    dn_bs[ex, j,
                          c * 4:(c + 1) * 4].copy_(sq_d_t.view(torch.int8))

        self.fc_up_global_scale.copy_(
            (s_max_up / k_fp8).to(device=self.fc_up_global_scale.device,
                                  dtype=self.fc_up_global_scale.dtype))
        self.fc_down_global_scale.copy_(
            (s_max_dn / k_fp8).to(device=self.fc_down_global_scale.device,
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
        if not hasattr(moe, "experts") or not hasattr(moe, "n_routed_experts"):
            raise TypeError(
                f"moe must have ``experts`` and ``n_routed_experts``; got {type(moe).__name__!r}"
            )
        if int(moe.n_routed_experts) != int(self.num_experts):
            raise ValueError(
                f"moe.n_routed_experts ({moe.n_routed_experts}) != self.num_experts ({self.num_experts})"
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
        up = experts.up_proj
        down = experts.down_proj
        if not isinstance(up, nn.Parameter) or not isinstance(
                down, nn.Parameter):
            raise TypeError(
                "expected experts.up_proj and experts.down_proj to be nn.Parameter stacked tensors "
                f"(got up_proj={type(up).__name__!r}, down_proj={type(down).__name__!r})"
            )
        e_ct = int(moe.n_routed_experts)
        if int(up.shape[0]) != e_ct or int(down.shape[0]) != e_ct:
            raise ValueError(
                f"expert dim mismatch: n_routed_experts={e_ct}, up.shape={tuple(up.shape)}, "
                f"down.shape={tuple(down.shape)}")
        w_up_ehi = up.data.transpose(1, 2).contiguous()
        w_down_eih = down.data.transpose(1, 2).contiguous()
        self.populate_marlin_plugin_buffers(w_up_ehi, w_down_eih)


def register_nvfp4_moe_plugin_onnx_symbolic_functions() -> None:
    register_custom_op_symbolic(
        "trt::nvfp4_moe_plugin",
        symbolic_nvfp4_moe_plugin,
        ONNX_OPSET_VERSION,
    )


def replace_moe_blocks_with_nvfp4_plugin(model: nn.Module) -> nn.Module:
    """Replace ``NemotronHMoE`` blocks with ``NemotronHMoEW4A4Plugin`` (NVFP4 buffers must be filled)."""
    if NemotronHMoE is None:
        return model
    for name, module in list(model.named_modules()):
        new_module = None
        if isinstance(module, NemotronHMoE):
            try:
                new_module = NemotronHMoEW4A4Plugin(module)
            except (TypeError, ValueError):
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
