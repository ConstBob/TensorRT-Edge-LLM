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
"""High-level weight accessors over a safetensors checkpoint.

Linears retain their checkpoint precision. The returned descriptors feed
TensorRT FP8/NVFP4/MXFP8 Q/DQ patterns, INT4 plugins, INT8 Q/DQ, or native
FP16 MatMul. MoE expert weights remain packed for their dedicated plugins.
"""

import re
from dataclasses import dataclass, replace
from types import ModuleType
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..weight_packing import int4 as int4_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import quantization
from .safetensors_np import SafetensorsStore

__all__ = ["LinearWeights", "Weights"]


@dataclass(frozen=True)
class LinearWeights:
    """Tensor buffers and metadata for one checkpoint linear."""

    quant_type: str
    weight: np.ndarray
    bias: Optional[np.ndarray] = None
    weight_scale: Optional[np.ndarray] = None
    weight_scale_2: Optional[float] = None
    input_scale: Optional[float] = None
    pre_quant_scale: Optional[np.ndarray] = None
    activation_permutation: Optional[np.ndarray] = None
    group_size: int = 1

    @property
    def out_features(self) -> int:
        """Infer the output dimension for supported checkpoint layouts."""
        if self.quant_type in (quantization.QUANT_INT4_AWQ,
                               quantization.QUANT_INT4_AWQ_MODELOPT,
                               quantization.QUANT_INT4_GPTQ):
            return int(self.weight.shape[0]) * 2
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        """Infer the input dimension for supported checkpoint layouts."""
        if self.quant_type == quantization.QUANT_NVFP4:
            return int(self.weight.shape[1]) * 2
        return int(self.weight.shape[1])


class Weights:

    def __init__(self,
                 model_dir: str,
                 group_size: int = 16,
                 quant: Optional[quantization.QuantConfig] = None,
                 component: str = "llm",
                 spec_type: str = "none",
                 spec_role: str = "none",
                 vocab_map: Optional[np.ndarray] = None,
                 conversion: Optional[ModuleType] = None) -> None:
        self.conversion = conversion
        checkpoint_dir = getattr(conversion, "checkpoint_dir", None)
        source_dir = (checkpoint_dir(model_dir, component)
                      if checkpoint_dir is not None else model_dir)
        self.store = SafetensorsStore(source_dir)
        self.group_size = group_size
        self.quant = quant or quantization.QuantConfig(group_size=group_size)
        self.component = component
        self.spec_type = spec_type
        self.spec_role = spec_role
        self.vocab_map = (None if vocab_map is None else np.ascontiguousarray(
            vocab_map, dtype=np.int64))

    def close(self) -> None:
        self.store.close()

    # -- presence -----------------------------------------------------------

    def has(self, name: str) -> bool:
        return self._resolve(name, required=False) is not None

    def _resolve(self, name: str, required: bool = True) -> Optional[str]:
        candidates = [name]
        resolve_candidates = getattr(self.conversion, "resolve_candidates",
                                     None)
        if resolve_candidates is not None:
            candidates.extend(
                resolve_candidates(name,
                                   component=self.component,
                                   spec_type=self.spec_type,
                                   spec_role=self.spec_role,
                                   quant_type=self.quant.quant_type))
        for candidate in candidates:
            if self.store.has(candidate):
                return candidate
        if required:
            raise KeyError(f"checkpoint tensor not found: {name!r}")
        return None

    def is_nvfp4(self, prefix: str) -> bool:
        return self.has(prefix + ".weight_scale") and (
            self.has(prefix + ".weight_scale_2")
            or self.has(prefix + ".weight_global_scale"))

    def _nvfp4_names(self, prefix: str) -> Tuple[str, str, str, str, bool]:
        """Resolve ModelOpt or compressed-tensors NVFP4 tensor names."""
        if self.has(prefix + ".weight_global_scale"):
            return (prefix + ".weight_packed", prefix + ".weight_scale",
                    prefix + ".weight_global_scale",
                    prefix + ".input_global_scale", True)
        return (prefix + ".weight", prefix + ".weight_scale",
                prefix + ".weight_scale_2", prefix + ".input_scale", False)

    @staticmethod
    def _nvfp4_global_scale(value: float, reciprocal: bool) -> float:
        """Return a scale in the multiplier convention used by the builder."""
        if not reciprocal:
            return value
        if value == 0.0:
            raise ValueError("compressed-tensors NVFP4 global scale is zero")
        return 1.0 / value

    # -- plain tensors ------------------------------------------------------

    def f16(self, name: str) -> np.ndarray:
        return self.store.get_f16(self._resolve(name))

    def f32(self, name: str) -> np.ndarray:
        return self.store.get_f32(self._resolve(name))

    def array(self, name: str) -> np.ndarray:
        """Return one tensor while retaining its stored integer/float dtype."""
        return np.ascontiguousarray(self.store.get_numpy(self._resolve(name)))

    def opt_f16(self, name: str) -> Optional[np.ndarray]:
        return self.f16(name) if self.has(name) else None

    def find(self, *names: str) -> str:
        """Return the first checkpoint key that exists."""
        for name in names:
            resolved = self._resolve(name, required=False)
            if resolved is not None:
                return resolved
        raise KeyError("none of the checkpoint keys exist: " +
                       ", ".join(names))

    def keys(self) -> Tuple[str, ...]:
        """Return checkpoint tensor names for component-specific discovery."""
        return tuple(self.store.keys())

    def layer_prefixes(self, markers: Sequence[str]) -> List[str]:
        """Discover and numerically order model-specific layer prefixes."""
        found = set()
        patterns = [re.compile(marker) for marker in markers]
        for key in self.keys():
            for pattern in patterns:
                match = pattern.search(key)
                if match:
                    found.add(match.group(1))
                    break

        def order(prefix: str) -> Tuple[int, str]:
            numbers = re.findall(r"\.(\d+)(?:\.|$)", prefix)
            return (int(numbers[-1]) if numbers else -1, prefix)

        return sorted(found, key=order)

    def find_suffix(self, suffix: str, contains: str = "") -> str:
        """Find a unique component tensor by suffix and optional substring."""
        matches = [
            name for name in self.store.keys()
            if name.endswith(suffix) and (not contains or contains in name)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one tensor ending {suffix!r} containing {contains!r}; "
                f"found {len(matches)}")
        return matches[0]

    def _linear_unreduced(self, prefix: str, quant_type: str) -> LinearWeights:
        """Load one linear in the representation consumed by TensorRT."""
        bias = self.opt_f16(prefix + ".bias")
        if quant_type == quantization.QUANT_FP16:
            weight, bias = self.linear_fp16(prefix)
            return LinearWeights(quant_type, weight, bias)
        if quant_type == quantization.QUANT_NVFP4:
            raw = self.linear_nvfp4_raw(prefix)
            return LinearWeights(
                quant_type,
                raw["packed"],
                raw["bias"],
                weight_scale=raw["weight_scale"],
                weight_scale_2=raw["weight_scale_2"],
                input_scale=raw["input_scale"],
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_FP8:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.f16(prefix + ".weight_scale"),
                input_scale=self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")),
            )
        if quant_type == quantization.QUANT_MXFP8:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.array(prefix + ".weight_scale").astype(
                    np.uint8, copy=False),
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_INT4_AWQ:
            qweight = self.array(prefix + ".qweight")
            qzeros = self.array(prefix + ".qzeros")
            scales = self.f16(prefix + ".scales")
            if self._reduce_lm_head(prefix):
                order = (0, 2, 4, 6, 1, 3, 5, 7)
                qweight = int4_pack.select_column_packed(
                    qweight, self.vocab_map, order)
                qzeros = int4_pack.select_column_packed(
                    qzeros, self.vocab_map, order)
                scales = self._select_output_axis(scales, self.vocab_map)
            packed = int4_pack.repack_awq(qweight, qzeros)
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=scales,
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_INT4_AWQ_MODELOPT:
            weight = self.array(prefix + ".weight")
            scales = self.f16(prefix + ".weight_scale")
            if self._reduce_lm_head(prefix):
                weight = int4_pack.select_pair_packed_rows(
                    weight, self.vocab_map)
                scales = self._select_output_axis(scales, self.vocab_map)
            packed = int4_pack.repack_modelopt_awq(weight)
            scales = scales.T
            pre_quant_scale = (self.f16(prefix + ".pre_quant_scale")
                               if self.has(prefix + ".pre_quant_scale") else
                               np.ones(packed.shape[1], dtype=np.float16))
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=np.ascontiguousarray(scales),
                pre_quant_scale=np.ascontiguousarray(pre_quant_scale),
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_INT4_GPTQ:
            qweight = self.array(prefix + ".qweight")
            qzeros = (self.array(prefix + ".qzeros")
                      if self.has(prefix + ".qzeros") else np.empty(
                          (1, 0), dtype=np.int32))
            scales = self.f16(prefix + ".scales")
            if self._reduce_lm_head(prefix):
                qweight = np.ascontiguousarray(qweight[:, self.vocab_map])
                if qzeros.size:
                    qzeros = int4_pack.select_column_packed(
                        qzeros, self.vocab_map, tuple(range(8)))
                scales = self._select_output_axis(scales, self.vocab_map)
            g_idx = (self.array(prefix +
                                ".g_idx") if self.has(prefix +
                                                      ".g_idx") else None)
            packed, permutation = int4_pack.repack_gptq(
                qweight,
                qzeros,
                g_idx,
                self.quant.gptq_zero_point_offset,
            )
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=scales,
                activation_permutation=permutation,
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_INT8_SQ:
            pre_quant_scale = (self.f16(prefix + ".pre_quant_scale")
                               if self.has(prefix + ".pre_quant_scale") else
                               np.ones(self.store.shape(
                                   self._resolve(prefix + ".weight"))[1],
                                       dtype=np.float16))
            return LinearWeights(
                quant_type,
                self.array(prefix + ".weight").astype(np.int8, copy=False),
                bias,
                weight_scale=self.f32(prefix + ".weight_scale"),
                input_scale=self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")),
                pre_quant_scale=np.ascontiguousarray(pre_quant_scale),
            )
        raise ValueError(f"unsupported linear quantization {quant_type!r}")

    def linear(self, prefix: str, quant_type: str) -> LinearWeights:
        """Load a linear and apply an optional output-vocabulary map."""
        if (quant_type == quantization.QUANT_NVFP4
                and not self.is_nvfp4(prefix)):
            quant_type = quantization.QUANT_FP16
        linear = self._linear_unreduced(prefix, quant_type)
        if not self._reduce_lm_head(prefix) or quant_type in (
                quantization.QUANT_INT4_AWQ,
                quantization.QUANT_INT4_AWQ_MODELOPT,
                quantization.QUANT_INT4_GPTQ,
        ):
            return linear
        indices = self.vocab_map
        weight = np.ascontiguousarray(linear.weight[indices])
        bias = (None if linear.bias is None else np.ascontiguousarray(
            linear.bias[indices]))
        weight_scale = linear.weight_scale
        if weight_scale is not None and np.ndim(weight_scale) > 0:
            weight_scale = self._select_output_axis(weight_scale, indices)
        return replace(linear,
                       weight=weight,
                       bias=bias,
                       weight_scale=weight_scale)

    def linear_adapter(self, prefix: str):
        """Return a model-owned static low-rank adapter, when present."""
        adapter = getattr(self.conversion, "linear_adapter", None)
        return adapter(self, prefix) if adapter is not None else None

    def _reduce_lm_head(self, prefix: str) -> bool:
        return (self.vocab_map is not None
                and (prefix == "lm_head" or prefix.endswith(".lm_head")
                     or prefix.endswith("embed_tokens")))

    @staticmethod
    def _select_output_axis(array: np.ndarray,
                            indices: np.ndarray) -> np.ndarray:
        max_index = int(indices.max())
        axes = [
            axis for axis, extent in enumerate(array.shape)
            if extent > max_index
        ]
        if not axes:
            return array
        return np.ascontiguousarray(np.take(array, indices, axis=axes[-1]))

    @staticmethod
    def shard_linear(linear: LinearWeights, mode: str, tp_size: int,
                     tp_rank: int) -> LinearWeights:
        """Return one contiguous tensor-parallel shard of a linear."""
        if tp_size == 1 or mode == "replicated":
            return linear
        full_out = linear.out_features
        full_in = linear.in_features
        split_size = full_out if mode == "column" else full_in
        if split_size % tp_size:
            raise ValueError(
                f"cannot {mode}-shard linear dimension {split_size} over {tp_size} ranks"
            )

        def split(array, axis):
            if array is None or np.ndim(array) == 0:
                return array
            return np.ascontiguousarray(
                np.split(array, tp_size, axis=axis)[tp_rank])

        weight = split(linear.weight, 0 if mode == "column" else 1)
        bias = split(linear.bias, 0) if mode == "column" else linear.bias
        if mode == "row" and bias is not None and tp_rank != 0:
            bias = np.zeros_like(bias)

        weight_scale = linear.weight_scale
        if weight_scale is not None and np.ndim(weight_scale) > 0:
            if mode == "column":
                matching = [
                    axis for axis, extent in enumerate(weight_scale.shape)
                    if extent in (full_out, full_out // 2)
                ]
            else:
                groups = full_in // max(1, linear.group_size)
                matching = [
                    axis for axis, extent in enumerate(weight_scale.shape)
                    if extent in (full_in, groups)
                ]
            if matching:
                weight_scale = split(weight_scale, matching[0])

        pre_quant_scale = linear.pre_quant_scale
        if mode == "row" and pre_quant_scale is not None:
            matching = [
                axis for axis, extent in enumerate(pre_quant_scale.shape)
                if extent == full_in
            ]
            if matching:
                pre_quant_scale = split(pre_quant_scale, matching[0])

        permutation = linear.activation_permutation
        if mode == "row" and permutation is not None:
            start = tp_rank * (full_in // tp_size)
            stop = start + full_in // tp_size
            local = permutation[(permutation >= start) & (permutation < stop)]
            permutation = np.ascontiguousarray(local - start, dtype=np.int64)

        return replace(linear,
                       weight=weight,
                       bias=bias,
                       weight_scale=weight_scale,
                       pre_quant_scale=pre_quant_scale,
                       activation_permutation=permutation)

    # -- dense linear (NVFP4 decoded to FP16, or plain) ---------------------

    def linear_fp16(self,
                    prefix: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(W [out, in] fp16, bias [out] fp16 | None)`` for a linear.

        NVFP4 weights are dequantized to FP16; plain weights are cast to FP16.
        """
        if self.is_nvfp4(prefix):
            weight_name, scale_name, global_scale_name, _, reciprocal = \
                self._nvfp4_names(prefix)
            packed = self.store.get_packed_fp4(self._resolve(weight_name))
            sf = self.store.get_fp8_bytes(self._resolve(scale_name))
            ws2 = self._nvfp4_global_scale(
                self.store.get_scalar_f32(self._resolve(global_scale_name)),
                reciprocal)
            dense = nvfp4_pack.decode_modelopt_nvfp4(packed, sf, ws2,
                                                     self.group_size)
            w = dense.astype(np.float16)
        elif self.has(prefix + ".weight"):
            w = self.f16(prefix + ".weight")
        else:
            convert = getattr(self.conversion, "convert_linear_fp16", None)
            converted = convert(self, prefix) if convert is not None else None
            if converted is None:
                raise KeyError(f"no weight for linear {prefix!r}")
            return converted
        bias = self.opt_f16(prefix + ".bias")
        return np.ascontiguousarray(w), (np.ascontiguousarray(bias)
                                         if bias is not None else None)

    def qkv_scales(self, attn_prefix: str) -> Tuple[float, float, float]:
        """Per-layer ``[q, k, v]`` scales for the FP8 KV cache.

        The k/v scales come from checkpoint metadata when present and default
        to 1.0 otherwise.
        """

        def scalar(name: str) -> float:
            return (float(self.store.get_scalar_f32(self._resolve(name)))
                    if self.has(name) else 1.0)

        return (1.0, scalar(f"{attn_prefix}.k_proj.k_scale"),
                scalar(f"{attn_prefix}.v_proj.v_scale"))

    def linear_nvfp4_raw(self, prefix: str) -> dict:
        """Raw NVFP4 pieces of a dense linear for the in-graph Q/DQ path."""
        (weight_name, scale_name, global_scale_name, input_scale_name,
         reciprocal) = \
            self._nvfp4_names(prefix)
        weight_scale_2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        input_scale = (self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(input_scale_name)),
            reciprocal) if self.has(input_scale_name) else 1.0)
        return {
            "packed": self.store.get_packed_fp4(self._resolve(weight_name)),
            "weight_scale":
            self.store.get_fp8_bytes(self._resolve(scale_name)),
            "weight_scale_2": weight_scale_2,
            "input_scale": input_scale,
            "bias": self.opt_f16(prefix + ".bias"),
        }

    # -- NVFP4 MoE expert accessors -----------------------------------------

    def expert_dense_f32(self, prefix: str) -> np.ndarray:
        """Load one NVFP4 or plain expert projection as dense fp32."""
        if not self.is_nvfp4(prefix):
            return np.ascontiguousarray(self.f32(prefix + ".weight"))
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        packed = self.store.get_packed_fp4(self._resolve(weight_name))
        sf = self.store.get_fp8_bytes(self._resolve(scale_name))
        ws2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        return nvfp4_pack.decode_modelopt_nvfp4(packed, sf, ws2,
                                                self.group_size)

    def expert_raw_nvfp4(self, prefix: str) -> dict:
        """Return raw NVFP4 bytes for one expert projection (byte-reuse path)."""
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        alpha = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        return {
            "packed": self.store.get_packed_fp4(self._resolve(weight_name)),
            "sf": self.store.get_fp8_bytes(self._resolve(scale_name)),
            "alpha": alpha,
        }
