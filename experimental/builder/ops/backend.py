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
"""Internal TensorRT lowering backend for symbolic operations.

A thin, stateful wrapper over ``trt.INetworkDefinition`` providing the
lowerings used by the public ``Tensor`` and ``functional`` APIs. Public model
definitions do not know whether a lowering is a TensorRT-native layer or an
Edge-LLM extension operation, and never receive this backend object.
"""

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import tensorrt as trt

from ..core import quantization

__all__ = ["Net"]

_NP_TO_TRT = {
    np.dtype(np.float16): trt.float16,
    np.dtype(np.float32): trt.float32,
    np.dtype(np.int8): trt.int8,
    np.dtype(np.int32): trt.int32,
    np.dtype(np.int64): trt.int64,
    np.dtype(np.uint8): trt.uint8,
    np.dtype(np.bool_): trt.bool,
}

_OPERATION_CREATORS = {
    "all_reduce": "AllReducePlugin",
    "attention": "AttentionPlugin",
    "dflash_target_cache_update": "DFlashTargetKVCacheUpdate",
    "gemma4_attention": "Gemma4AudioAttentionPlugin",
    "int4_groupwise_gemm": "Int4GroupwiseGemmPlugin",
    "int4_moe": "Int4MoePlugin",
    "nvfp4_moe": "Nvfp4MoePlugin",
    "nvfp4_moe_sm12x": "NvFP4MoEPluginGeforce",
    "vit_attention": "ViTAttentionPlugin",
}


class Net:
    """Lower symbolic operations into one ``INetworkDefinition``."""

    def __init__(self, builder: "trt.Builder",
                 network: "trt.INetworkDefinition") -> None:
        self.builder = builder
        self.network = network
        self._weight_refs: List[np.ndarray] = []  # keep numpy alive for TRT
        self._inputs = {}
        self._external_weights: Dict[str, Dict[str, np.ndarray]] = {}
        self._n = 0

    # -- low level ----------------------------------------------------------

    def _name(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    @staticmethod
    def _unwrap(value):
        unwrap = getattr(value, "_as_trt", None)
        return unwrap() if unwrap is not None else value

    def const(self,
              arr: np.ndarray,
              name: Optional[str] = None) -> "trt.ITensor":
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _NP_TO_TRT:
            raise TypeError(f"unsupported constant dtype {arr.dtype}")
        self._weight_refs.append(arr)  # TRT references the buffer; keep alive
        layer = self.network.add_constant(tuple(int(d) for d in arr.shape),
                                          trt.Weights(arr))
        if name:
            layer.name = self._name(name)
        return layer.get_output(0)

    def add_input(self, name: str, dtype: "trt.DataType",
                  shape: Sequence[int]) -> "trt.ITensor":
        if name in self._inputs:
            return self._inputs[name]
        tensor = self.network.add_input(name, dtype, tuple(shape))
        self._inputs[name] = tensor
        return tensor

    def weight_input(self, name: str, value: np.ndarray,
                     kind: str) -> "trt.ITensor":
        """Declare one static engine input backed by a runtime weight file."""
        value = np.ascontiguousarray(value)
        dtype = _NP_TO_TRT.get(value.dtype)
        if dtype is None:
            raise TypeError(f"unsupported external weight dtype {value.dtype}")
        tensors = self._external_weights.setdefault(kind, {})
        previous = tensors.get(name)
        if previous is not None and (previous.dtype != value.dtype
                                     or previous.shape != value.shape):
            raise ValueError(f"conflicting external weight input {name!r}")
        tensors.setdefault(name, value)
        return self.add_input(name, dtype, value.shape)

    def take_external_weights(self) -> Mapping[str, Mapping[str, np.ndarray]]:
        """Transfer registered external weights to the artifact writer."""
        weights = self._external_weights
        self._external_weights = {}
        return weights

    def mark_output(self,
                    tensor: "trt.ITensor",
                    name: str,
                    dtype: Optional["trt.DataType"] = None) -> None:
        tensor = self._unwrap(tensor)
        tensor.name = name
        self.network.mark_output(tensor)
        if dtype is not None:
            tensor.dtype = dtype

    # -- shape ops ----------------------------------------------------------

    def reshape(self, x: "trt.ITensor", shape: Sequence[int]) -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_shuffle(x)
        layer.reshape_dims = tuple(int(d) for d in shape)
        return layer.get_output(0)

    def dynamic_reshape(self, x: "trt.ITensor",
                        shape: "trt.ITensor") -> "trt.ITensor":
        """Reshape ``x`` using a runtime shape tensor."""
        layer = self.network.add_shuffle(self._unwrap(x))
        layer.set_input(1, self._unwrap(shape))
        return layer.get_output(0)

    def unsqueeze(self, x: "trt.ITensor", axis: int,
                  rank: int) -> "trt.ITensor":
        """Insert one unit dimension into a tensor with dynamic extents."""
        x = self._unwrap(x)
        normalized_axis = axis if axis >= 0 else axis + rank + 1
        if not 0 <= normalized_axis <= rank:
            raise ValueError(
                f"unsqueeze axis {axis} is invalid for rank {rank}")
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        pieces = []
        if normalized_axis:
            pieces.append(
                self.network.add_slice(shape, (0, ), (normalized_axis, ),
                                       (1, )).get_output(0))
        pieces.append(self.const(np.array([1], dtype=np.int32), "unit_dim"))
        if normalized_axis < rank:
            pieces.append(
                self.network.add_slice(shape, (normalized_axis, ),
                                       (rank - normalized_axis, ),
                                       (1, )).get_output(0))
        layer = self.network.add_shuffle(x)
        layer.set_input(1, self.concat(pieces, 0))
        return layer.get_output(0)

    def squeeze(self, x: "trt.ITensor", axis: int, rank: int) -> "trt.ITensor":
        """Remove one static unit dimension while preserving dynamic extents."""
        x = self._unwrap(x)
        normalized_axis = axis if axis >= 0 else axis + rank
        if not 0 <= normalized_axis < rank:
            raise ValueError(f"squeeze axis {axis} is invalid for rank {rank}")
        if int(x.shape[normalized_axis]) != 1:
            raise ValueError("squeeze requires a static unit dimension")
        if rank == 1:
            layer = self.network.add_shuffle(x)
            layer.reshape_dims = ()
            return layer.get_output(0)
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        pieces = []
        if normalized_axis:
            pieces.append(
                self.network.add_slice(shape, (0, ), (normalized_axis, ),
                                       (1, )).get_output(0))
        trailing = rank - normalized_axis - 1
        if trailing:
            pieces.append(
                self.network.add_slice(shape, (normalized_axis + 1, ),
                                       (trailing, ), (1, )).get_output(0))
        layer = self.network.add_shuffle(x)
        layer.set_input(1, self.concat(pieces, 0))
        return layer.get_output(0)

    def transpose(self, x: "trt.ITensor",
                  permutation: Sequence[int]) -> "trt.ITensor":
        """Permute tensor axes without changing element type."""
        x = self._unwrap(x)
        layer = self.network.add_shuffle(x)
        layer.second_transpose = tuple(int(axis) for axis in permutation)
        return layer.get_output(0)

    def cast(self, x: "trt.ITensor", dtype: "trt.DataType") -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_cast(x, dtype)
        return layer.get_output(0)

    def elementwise(self, a: "trt.ITensor", b: "trt.ITensor",
                    op: "trt.ElementWiseOperation") -> "trt.ITensor":
        a, b = self._unwrap(a), self._unwrap(b)
        return self.network.add_elementwise(a, b, op).get_output(0)

    def matmul(
        self,
        lhs: "trt.ITensor",
        rhs: "trt.ITensor",
        lhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE,
        rhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE
    ) -> "trt.ITensor":
        """Lower a symbolic matrix multiplication."""
        return self.network.add_matrix_multiply(self._unwrap(lhs), lhs_op,
                                                self._unwrap(rhs),
                                                rhs_op).get_output(0)

    def reduce(self,
               x: "trt.ITensor",
               op: "trt.ReduceOperation",
               axes: int,
               keep_dims: bool = False) -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_reduce(x, op, axes, keep_dims).get_output(0)

    def unary(self, x: "trt.ITensor",
              op: "trt.UnaryOperation") -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_unary(x, op).get_output(0)

    def activation(self, x: "trt.ITensor",
                   act: "trt.ActivationType") -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_activation(x, act).get_output(0)

    def softmax(self, x: "trt.ITensor", axis: int) -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_softmax(x)
        layer.axes = 1 << axis
        return layer.get_output(0)

    def log_softmax(self, x: "trt.ITensor", axis: int) -> "trt.ITensor":
        """Numerically stable softmax followed by logarithm."""
        return self.unary(self.softmax(x, axis), trt.UnaryOperation.LOG)

    def concat(self, tensors: Sequence["trt.ITensor"],
               axis: int) -> "trt.ITensor":
        layer = self.network.add_concatenation(
            [self._unwrap(tensor) for tensor in tensors])
        layer.axis = axis
        return layer.get_output(0)

    def silu(self, x: "trt.ITensor") -> "trt.ITensor":
        s = self.activation(x, trt.ActivationType.SIGMOID)
        return self.elementwise(x, s, trt.ElementWiseOperation.PROD)

    def relu(self, x: "trt.ITensor") -> "trt.ITensor":
        return self.activation(x, trt.ActivationType.RELU)

    def tanh(self, x: "trt.ITensor") -> "trt.ITensor":
        return self.activation(x, trt.ActivationType.TANH)

    def gelu(self, x: "trt.ITensor") -> "trt.ITensor":
        """Exact GELU represented with elementwise operations."""
        x = self._unwrap(x)
        output_dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        scalar_shape = (1, ) * len(x.shape)
        sqrt2 = self.const(
            np.array(np.sqrt(2.0), dtype=np.float32).reshape(scalar_shape),
            "sqrt2")
        half = self.const(
            np.array(0.5, dtype=np.float32).reshape(scalar_shape), "half")
        one = self.const(
            np.array(1.0, dtype=np.float32).reshape(scalar_shape), "one")
        scaled = self.elementwise(x32, sqrt2, trt.ElementWiseOperation.DIV)
        erf = self.unary(scaled, trt.UnaryOperation.ERF)
        factor = self.elementwise(one, erf, trt.ElementWiseOperation.SUM)
        factor = self.elementwise(factor, half, trt.ElementWiseOperation.PROD)
        return self.cast(
            self.elementwise(x32, factor, trt.ElementWiseOperation.PROD),
            output_dtype)

    def gelu_tanh(self, x: "trt.ITensor") -> "trt.ITensor":
        """Tanh-approximate GELU with rank-matched scalar constants."""
        x = self._unwrap(x)
        output_dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        scalar_shape = (1, ) * len(x.shape)

        def scalar(value: float, name: str) -> "trt.ITensor":
            data = np.array(value, dtype=np.float32).reshape(scalar_shape)
            return self.const(data, name)

        half = scalar(0.5, "gelu_half")
        one = scalar(1.0, "gelu_one")
        cubic = scalar(0.044715, "gelu_cubic")
        scale = scalar(np.sqrt(2.0 / np.pi), "gelu_scale")
        squared = self.elementwise(x32, x32, trt.ElementWiseOperation.PROD)
        cubed = self.elementwise(squared, x32, trt.ElementWiseOperation.PROD)
        inner = self.elementwise(
            x32, self.elementwise(cubic, cubed, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM)
        tanh = self.tanh(
            self.elementwise(scale, inner, trt.ElementWiseOperation.PROD))
        factor = self.elementwise(one, tanh, trt.ElementWiseOperation.SUM)
        result = self.elementwise(
            self.elementwise(half, x32, trt.ElementWiseOperation.PROD), factor,
            trt.ElementWiseOperation.PROD)
        return self.cast(result, output_dtype)

    def gather(self, x: "trt.ITensor", indices: np.ndarray,
               axis: int) -> "trt.ITensor":
        """Gather a constant index vector along one axis."""
        x = self._unwrap(x)
        index_tensor = self.const(np.asarray(indices, dtype=np.int64),
                                  "indices")
        return self.network.add_gather(x, index_tensor, axis).get_output(0)

    def gather_tensor(self, x: "trt.ITensor", indices: "trt.ITensor",
                      axis: int) -> "trt.ITensor":
        """Gather runtime indices along one axis."""
        x, indices = self._unwrap(x), self._unwrap(indices)
        return self.network.add_gather(x, indices, axis).get_output(0)

    def gather_nd(self,
                  x: "trt.ITensor",
                  indices: "trt.ITensor",
                  num_elementwise_dims: int = 0) -> "trt.ITensor":
        """Gather runtime indices with TensorRT's ND semantics."""
        layer = self.network.add_gather_v2(self._unwrap(x),
                                           self._unwrap(indices),
                                           trt.GatherMode.ND)
        layer.num_elementwise_dims = num_elementwise_dims
        return layer.get_output(0)

    def shape_of(self, x: "trt.ITensor") -> "trt.ITensor":
        """Return the runtime shape vector as INT32."""
        shape = self.network.add_shape(self._unwrap(x)).get_output(0)
        return self.cast(shape, trt.int32)

    def dynamic_slice(self, x: "trt.ITensor", start: "trt.ITensor",
                      size: "trt.ITensor",
                      stride: Sequence[int]) -> "trt.ITensor":
        """Slice ``x`` using runtime start and size vectors."""
        rank = len(stride)
        layer = self.network.add_slice(self._unwrap(x), (0, ) * rank,
                                       (0, ) * rank, tuple(stride))
        layer.set_input(1, self._unwrap(start))
        layer.set_input(2, self._unwrap(size))
        return layer.get_output(0)

    def slice_last_dim(self, x: "trt.ITensor", offset: int, size: int,
                       rank: int) -> "trt.ITensor":
        """Slice ``x[..., offset:offset+size]`` keeping dynamic leading dims."""
        x = self._unwrap(x)
        start = tuple([0] * (rank - 1) + [offset])
        stride = tuple([1] * rank)
        layer = self.network.add_slice(x, start, tuple([0] * rank), stride)
        shp = self.network.add_shape(x).get_output(0)
        shp32 = self.cast(shp, trt.int32)
        lead = self.network.add_slice(shp32, (0, ), (rank - 1, ),
                                      (1, )).get_output(0)
        size_c = self.const(np.array([size], dtype=np.int32))
        full = self.concat([lead, size_c], 0)
        layer.set_input(2, full)
        return layer.get_output(0)

    def slice_axis(self, x: "trt.ITensor", axis: int, offset: int, size: int,
                   rank: int) -> "trt.ITensor":
        """Slice one axis while copying all dynamic extents from the input."""
        x = self._unwrap(x)
        start = [0] * rank
        start[axis] = offset
        layer = self.network.add_slice(x, tuple(start), tuple([0] * rank),
                                       tuple([1] * rank))
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        before = (self.network.add_slice(
            shape, (0, ), (axis, ), (1, )).get_output(0) if axis else None)
        selected = self.const(np.array([size], dtype=np.int32), "slice_size")
        after_count = rank - axis - 1
        after = (self.network.add_slice(shape, (axis + 1, ), (after_count, ),
                                        (1, )).get_output(0)
                 if after_count else None)
        pieces = [
            piece for piece in (before, selected, after) if piece is not None
        ]
        layer.set_input(2, self.concat(pieces, 0))
        return layer.get_output(0)

    def slice_lead_dims(self, x: "trt.ITensor", size: int,
                        rank: int) -> "trt.ITensor":
        """Slice ``x[..., :size]`` along the last axis (alias of slice_last_dim)."""
        return self.slice_last_dim(x, 0, size, rank)

    def pad_last_dim(self, x: "trt.ITensor", padding: int,
                     rank: int) -> "trt.ITensor":
        """Append dynamic-shape zeros to the last dimension."""
        x = self._unwrap(x)
        if padding <= 0:
            return x
        source = self.slice_last_dim(x, 0, padding, rank)
        zero = self.const(np.zeros((1, ) * rank, dtype=np.float16), "zero")
        zeros = self.elementwise(source, zero, trt.ElementWiseOperation.PROD)
        return self.concat((x, zeros), rank - 1)

    def empty_sequence(self, x: "trt.ITensor", last_dim: int) -> "trt.ITensor":
        """Create a ``[B, 0, last_dim]`` view carrying x's dynamic batch."""
        x = self._unwrap(x)
        layer = self.network.add_slice(x, (0, 0, 0), (0, 0, last_dim),
                                       (1, 1, 1))
        source_shape = self.cast(
            self.network.add_shape(x).get_output(0), trt.int32)
        batch = self.network.add_slice(source_shape, (0, ), (1, ),
                                       (1, )).get_output(0)
        empty_shape = self.concat(
            (batch, self.const(np.array([0, last_dim], dtype=np.int32))), 0)
        layer.set_input(2, empty_shape)
        return layer.get_output(0)

    def topk(self, x: "trt.ITensor", k: int,
             axis: int) -> tuple["trt.ITensor", "trt.ITensor"]:
        x = self._unwrap(x)
        layer = self.network.add_topk(x, trt.TopKOperation.MAX, k, 1 << axis)
        return layer.get_output(0), layer.get_output(1)

    def select(self, condition: "trt.ITensor", when_true: "trt.ITensor",
               when_false: "trt.ITensor") -> "trt.ITensor":
        condition = self._unwrap(condition)
        when_true = self._unwrap(when_true)
        when_false = self._unwrap(when_false)
        return self.network.add_select(condition, when_true,
                                       when_false).get_output(0)

    # -- TensorRT attention -------------------------------------------------

    def rotary_embedding(self,
                         x: "trt.ITensor",
                         cos_cache: "trt.ITensor",
                         sin_cache: "trt.ITensor",
                         position_ids: "trt.ITensor",
                         rotary_dim: int,
                         interleaved: bool = False) -> "trt.ITensor":
        """Lower rotary embedding."""
        layer = self.network.add_rotary_embedding(self._unwrap(x),
                                                  self._unwrap(cos_cache),
                                                  self._unwrap(sin_cache),
                                                  interleaved, rotary_dim)
        layer.set_input(3, self._unwrap(position_ids))
        return layer.get_output(0)

    def kv_cache_update(self, cache: "trt.ITensor", update: "trt.ITensor",
                        write_indices: "trt.ITensor") -> "trt.ITensor":
        """Update a linear TensorRT KV cache at per-request positions."""
        layer = self.network.add_kv_cache_update(self._unwrap(cache),
                                                 self._unwrap(update),
                                                 self._unwrap(write_indices),
                                                 trt.KVCacheMode.LINEAR)
        return layer.get_output(0)

    def scaled_dot_product_attention(
        self,
        query: "trt.ITensor",
        key: "trt.ITensor",
        value: "trt.ITensor",
        mask: Optional["trt.ITensor"] = None,
        key_value_lengths: Optional["trt.ITensor"] = None,
        scale: Optional[float] = None,
    ) -> "trt.ITensor":
        """Apply non-causal TensorRT scaled dot-product attention."""
        query = self._unwrap(query)
        if scale is None:
            head_size = int(query.shape[-1])
            if head_size <= 0:
                raise ValueError(
                    "attention scale must be explicit for a dynamic head size")
            scale = head_size**-0.5
        if scale != 1.0:
            if query.dtype != trt.float16:
                raise TypeError(
                    "scaled attention currently requires FP16 inputs")
            scalar = self.const(
                np.array(scale, dtype=np.float16).reshape(
                    (1, ) * len(query.shape)), "attention_scale")
            query = self.elementwise(query, scalar,
                                     trt.ElementWiseOperation.PROD)
        layer = self.network.add_attention_v2(
            query, self._unwrap(key), self._unwrap(value),
            trt.AttentionNormalizationOp.SOFTMAX, trt.CausalMaskKind.NONE)
        layer.decomposable = True
        if mask is not None:
            layer.mask = self._unwrap(mask)
        if key_value_lengths is not None:
            layer.key_value_lengths = self._unwrap(key_value_lengths)
        return layer.get_output(0)

    # -- linear / matmul ----------------------------------------------------

    def linear(self,
               x: "trt.ITensor",
               weight: np.ndarray,
               bias: Optional[np.ndarray] = None,
               rank: int = 3) -> "trt.ITensor":
        """FP16 linear: ``y = x @ W^T (+ bias)`` with ``W`` of shape ``[out, in]``.

        TRT matmul requires both operands to have the same rank, so the weight
        is reshaped with leading 1s (``[1,...,out,in]``) to match ``x``.
        """
        x = self._unwrap(x)
        w16 = weight.astype(np.float16)
        wshape = [1] * (rank - 2) + list(w16.shape)
        w = self.const(np.ascontiguousarray(w16.reshape(wshape)), "w")
        out = self.network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is not None:
            bshape = [1] * (rank - 1) + [int(bias.shape[0])]
            b = self.const(bias.astype(np.float16).reshape(bshape), "b")
            out = self.elementwise(out, b, trt.ElementWiseOperation.SUM)
        return out

    def linear_f32(self,
                   x: "trt.ITensor",
                   weight: np.ndarray,
                   bias: Optional[np.ndarray] = None,
                   rank: int = 3) -> "trt.ITensor":
        """FP32 linear used where the pre-normalization accumulation is wide."""
        x = self._unwrap(x)
        x32 = self.cast(x, trt.float32)
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        shape = [1] * (rank - 2) + list(weight.shape)
        constant = self.const(weight.reshape(shape), "w32")
        output = self.network.add_matrix_multiply(
            x32, trt.MatrixOperation.NONE, constant,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is not None:
            bias_shape = [1] * (rank - 1) + [int(bias.shape[0])]
            bias_tensor = self.const(
                np.asarray(bias, dtype=np.float32).reshape(bias_shape), "b32")
            output = self.elementwise(output, bias_tensor,
                                      trt.ElementWiseOperation.SUM)
        return output

    def dynamic_lora(self, x: "trt.ITensor", base: "trt.ITensor", prefix: str,
                     in_features: int, out_features: int) -> "trt.ITensor":
        """Add runtime LoRA A/B matrices to a linear's base output."""
        x, base = self._unwrap(x), self._unwrap(base)
        lora_a = self.add_input(f"{prefix}.lora_A.weight", trt.float16,
                                (in_features, -1))
        lora_b = self.add_input(f"{prefix}.lora_B.weight", trt.float16,
                                (-1, out_features))
        intermediate = self.network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, lora_a,
            trt.MatrixOperation.NONE).get_output(0)
        update = self.network.add_matrix_multiply(
            intermediate, trt.MatrixOperation.NONE, lora_b,
            trt.MatrixOperation.NONE).get_output(0)
        return self.elementwise(base, update, trt.ElementWiseOperation.SUM)

    def convolution(
            self,
            x: "trt.ITensor",
            weight: np.ndarray,
            bias: Optional[np.ndarray] = None,
            stride: Sequence[int] = (1, ),
            padding: Sequence[int] = (0, ),
            dilation: Sequence[int] = (1, ),
            groups: int = 1,
            pre_padding: Optional[Sequence[int]] = None,
            post_padding: Optional[Sequence[int]] = None) -> "trt.ITensor":
        """Add an N-D convolution with checkpoint NumPy weights."""
        x = self._unwrap(x)
        weight = np.ascontiguousarray(weight.astype(np.float16))
        bias_array = (None if bias is None or bias.size == 0 else
                      np.ascontiguousarray(bias.astype(np.float16)))
        promoted_1d = weight.ndim == 3
        if promoted_1d:
            x = self.unsqueeze(x, -1, 3)
            weight = np.expand_dims(weight, -1)
        spatial_rank = weight.ndim - 2

        def spatial(values: Sequence[int], name: str,
                    promoted_value: int) -> tuple:
            result = tuple(int(value) for value in values)
            if promoted_1d and len(result) == 1:
                return result + (promoted_value, )
            if len(result) == 1 and spatial_rank > 1:
                result *= spatial_rank
            if len(result) != spatial_rank:
                raise ValueError(
                    f"{name} rank {len(result)} does not match convolution "
                    f"spatial rank {spatial_rank}")
            return result

        self._weight_refs.append(weight)
        bias_weights = trt.Weights()
        if bias_array is not None:
            self._weight_refs.append(bias_array)
            bias_weights = trt.Weights(bias_array)
        layer = self.network.add_convolution_nd(
            x,
            int(weight.shape[0]), tuple(int(dim) for dim in weight.shape[2:]),
            trt.Weights(weight), bias_weights)
        layer.stride_nd = spatial(stride, "stride", 1)
        layer.dilation_nd = spatial(dilation, "dilation", 1)
        if pre_padding is None and post_padding is None:
            layer.padding_nd = spatial(padding, "padding", 0)
        else:
            layer.pre_padding = spatial(pre_padding or padding, "pre_padding",
                                        0)
            layer.post_padding = spatial(post_padding or padding,
                                         "post_padding", 0)
        layer.num_groups = groups
        output = layer.get_output(0)
        return self.reshape(output, (0, 0, 0)) if promoted_1d else output

    def deconvolution(
            self,
            x: "trt.ITensor",
            weight: np.ndarray,
            bias: Optional[np.ndarray] = None,
            stride: Sequence[int] = (1, ),
            padding: Sequence[int] = (0, ),
            groups: int = 1,
            pre_padding: Optional[Sequence[int]] = None,
            post_padding: Optional[Sequence[int]] = None) -> "trt.ITensor":
        """Add an N-D transposed convolution."""
        x = self._unwrap(x)
        weight = np.ascontiguousarray(weight.astype(np.float16))
        bias_array = (None if bias is None or bias.size == 0 else
                      np.ascontiguousarray(bias.astype(np.float16)))
        promoted_1d = weight.ndim == 3
        if promoted_1d:
            x = self.unsqueeze(x, -1, 3)
            weight = np.expand_dims(weight, -1)
        spatial_rank = weight.ndim - 2

        def spatial(values: Sequence[int], name: str,
                    promoted_value: int) -> tuple:
            result = tuple(int(value) for value in values)
            if promoted_1d and len(result) == 1:
                return result + (promoted_value, )
            if len(result) == 1 and spatial_rank > 1:
                result *= spatial_rank
            if len(result) != spatial_rank:
                raise ValueError(
                    f"{name} rank {len(result)} does not match deconvolution "
                    f"spatial rank {spatial_rank}")
            return result

        self._weight_refs.append(weight)
        bias_weights = trt.Weights()
        if bias_array is not None:
            self._weight_refs.append(bias_array)
            bias_weights = trt.Weights(bias_array)
        layer = self.network.add_deconvolution_nd(
            x, int(weight.shape[1] * groups),
            tuple(int(dim) for dim in weight.shape[2:]), trt.Weights(weight),
            bias_weights)
        layer.stride_nd = spatial(stride, "stride", 1)
        if pre_padding is None and post_padding is None:
            layer.padding_nd = spatial(padding, "padding", 0)
        else:
            layer.pre_padding = spatial(pre_padding or padding, "pre_padding",
                                        0)
            layer.post_padding = spatial(post_padding or padding,
                                         "post_padding", 0)
        layer.num_groups = groups
        output = layer.get_output(0)
        return self.reshape(output, (0, 0, 0)) if promoted_1d else output

    def linear_from_weights(self,
                            x: "trt.ITensor",
                            linear_weights,
                            rank: int = 3,
                            name: str = "") -> "trt.ITensor":
        """Emit a linear using its checkpoint quantization."""
        quant_type = linear_weights.quant_type
        if quant_type == quantization.QUANT_FP16:
            return self.linear(x, linear_weights.weight, linear_weights.bias,
                               rank)
        if quant_type == quantization.QUANT_NVFP4:
            raw = {
                "packed": linear_weights.weight,
                "weight_scale": linear_weights.weight_scale,
                "weight_scale_2": linear_weights.weight_scale_2,
                "input_scale": linear_weights.input_scale,
                "bias": linear_weights.bias,
            }
            return self.nvfp4_linear(x, raw, rank)
        if quant_type == quantization.QUANT_FP8:
            return self.fp8_linear(x, linear_weights, rank)
        if quant_type == quantization.QUANT_MXFP8:
            return self.mxfp8_linear(x, linear_weights, rank)
        if quant_type in (quantization.QUANT_INT4_AWQ,
                          quantization.QUANT_INT4_AWQ_MODELOPT,
                          quantization.QUANT_INT4_GPTQ):
            return self.int4_linear(x, linear_weights, rank, name)
        if quant_type == quantization.QUANT_INT8_SQ:
            return self.int8_sq_linear(x, linear_weights, rank)
        raise ValueError(f"unsupported linear quantization {quant_type!r}")

    # -- normalization ------------------------------------------------------

    def rmsnorm(self,
                x: "trt.ITensor",
                weight: np.ndarray,
                eps: float,
                rank: int = 3,
                weight_before_cast: bool = False) -> "trt.ITensor":
        """Apply decomposed RMSNorm over the last axis."""
        last_axis = rank - 1
        x32 = self.cast(x, trt.float32)
        sq = self.elementwise(x32, x32, trt.ElementWiseOperation.PROD)
        var = self.network.add_reduce(sq, trt.ReduceOperation.AVG,
                                      1 << last_axis, True).get_output(0)
        eps_c = self.const(
            np.array(eps, dtype=np.float32).reshape([1] * rank), "eps")
        var = self.elementwise(var, eps_c, trt.ElementWiseOperation.SUM)
        std = self.unary(var, trt.UnaryOperation.SQRT)
        normed = self.elementwise(x32, std, trt.ElementWiseOperation.DIV)
        wshape = [1] * (rank - 1) + [int(weight.shape[0])]
        if weight_before_cast:
            w = self.const(weight.astype(np.float32).reshape(wshape), "rmsw")
            weighted = self.elementwise(normed, w,
                                        trt.ElementWiseOperation.PROD)
            return self.cast(weighted, trt.float16)
        normed16 = self.cast(normed, trt.float16)
        w = self.const(weight.astype(np.float16).reshape(wshape), "rmsw")
        return self.elementwise(normed16, w, trt.ElementWiseOperation.PROD)

    def layernorm(self, x: "trt.ITensor", weight: np.ndarray, bias: np.ndarray,
                  eps: float, rank: int) -> "trt.ITensor":
        """Apply LayerNorm over the last tensor axis."""
        axis = rank - 1
        x32 = self.cast(x, trt.float32)
        mean = self.reduce(x32, trt.ReduceOperation.AVG, 1 << axis, True)
        centered = self.elementwise(x32, mean, trt.ElementWiseOperation.SUB)
        squared = self.elementwise(centered, centered,
                                   trt.ElementWiseOperation.PROD)
        variance = self.reduce(squared, trt.ReduceOperation.AVG, 1 << axis,
                               True)
        epsilon = self.const(
            np.array(eps, dtype=np.float32).reshape([1] * rank), "eps")
        deviation = self.unary(
            self.elementwise(variance, epsilon, trt.ElementWiseOperation.SUM),
            trt.UnaryOperation.SQRT)
        normalized = self.cast(
            self.elementwise(centered, deviation,
                             trt.ElementWiseOperation.DIV), trt.float16)
        shape = [1] * (rank - 1) + [int(weight.shape[0])]
        scale = self.const(weight.astype(np.float16).reshape(shape), "ln_w")
        shift = self.const(bias.astype(np.float16).reshape(shape), "ln_b")
        normalized = self.elementwise(normalized, scale,
                                      trt.ElementWiseOperation.PROD)
        return self.elementwise(normalized, shift,
                                trt.ElementWiseOperation.SUM)

    # -- token selection ----------------------------------------------------

    def gather_last_tokens(self, hidden: "trt.ITensor",
                           last_token_ids: "trt.ITensor") -> "trt.ITensor":
        """Select token states with GatherND and batch dimensions enabled.

        ``hidden`` has shape ``[B,S,H]`` and ``last_token_ids`` has shape
        ``[B,T]``. The indices are expanded to ``[B,T,1]`` and the result has
        shape ``[B,T,H]``.
        """
        hidden = self._unwrap(hidden)
        idx32 = self.cast(last_token_ids, trt.int32)
        # [B,T] -> [B,T,1] via shuffle (0 copies the corresponding input dim).
        sh = self.network.add_shuffle(idx32)
        sh.reshape_dims = (0, 0, 1)
        gather = self.network.add_gather_v2(hidden, sh.get_output(0),
                                            trt.GatherMode.ND)
        gather.num_elementwise_dims = 1
        return gather.get_output(0)

    # -- operation implementations -----------------------------------------

    @staticmethod
    def _creator(name: str):
        creator_name = _OPERATION_CREATORS.get(name, name)
        registry = trt.get_plugin_registry()
        creator = registry.get_creator(creator_name, "1", "")
        if creator is None:
            raise RuntimeError(
                f"implementation for operation {name!r} ({creator_name!r}, v1) "
                "was not found; is libNvInfer_edgellm_plugin.so loaded?")
        return creator

    def _operation_field(self, name: str, value) -> "trt.PluginField":
        data = np.asarray(value)
        if data.dtype.kind in "biu":
            data = np.ascontiguousarray(data, dtype=np.int32).reshape(-1)
            field_type = trt.PluginFieldType.INT32
        elif data.dtype.kind == "f":
            data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
            field_type = trt.PluginFieldType.FLOAT32
        else:
            raise TypeError(
                f"operation attribute {name!r} has unsupported dtype "
                f"{data.dtype}")
        self._weight_refs.append(data)
        return trt.PluginField(name, data, field_type)

    def operation(self, name: str, attributes: Mapping[str, object],
                  inputs: Sequence["trt.ITensor"]):
        """Lower one semantic operation into the network."""
        fields = [
            self._operation_field(key, value)
            for key, value in attributes.items()
        ]
        instance_name = self._name(name)
        implementation = self._creator(name).create_plugin(
            instance_name, trt.PluginFieldCollection(fields),
            trt.TensorRTPhase.BUILD)
        if implementation is None:
            raise RuntimeError(
                f"operation {name!r} rejected the requested configuration; "
                "check the TensorRT log and rebuild "
                "libNvInfer_edgellm_plugin.so with the required optional "
                "kernels")
        layer = self.network.add_plugin_v3(
            [self._unwrap(value) for value in inputs], [], implementation)
        layer.name = instance_name
        return layer

    def operation_attributes(self, name: str) -> frozenset[str]:
        """Return attributes accepted by one operation implementation."""
        return frozenset(field.name
                         for field in self._creator(name).field_names)

    # -- NVFP4 dense Q/DQ (activation dynamic quantization) -----------------

    def const_fp4(self,
                  packed: np.ndarray,
                  shape,
                  name: str = "w_fp4") -> "trt.ITensor":
        """FP4 constant from packed nibbles (low nibble = even element)."""
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        self._weight_refs.append(packed)
        count = int(np.prod(shape))
        weights = trt.Weights(trt.DataType.FP4, packed.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(d) for d in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def const_fp8(self,
                  raw_bytes: np.ndarray,
                  shape,
                  name: str = "w_fp8") -> "trt.ITensor":
        """FP8 (E4M3) constant from raw bytes."""
        raw_bytes = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        self._weight_refs.append(raw_bytes)
        count = int(np.prod(shape))
        weights = trt.Weights(trt.DataType.FP8, raw_bytes.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(d) for d in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def const_ue8m0(self,
                    raw_bytes: np.ndarray,
                    shape,
                    name: str = "scale_ue8m0") -> "trt.ITensor":
        """UE8M0 constant used as an MXFP8 block scale."""
        if not hasattr(trt.DataType, "UE8M0"):
            raise RuntimeError("MXFP8 requires a TensorRT build with UE8M0")
        raw_bytes = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        self._weight_refs.append(raw_bytes)
        count = int(np.prod(shape))
        weights = trt.Weights(trt.DataType.UE8M0, raw_bytes.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(dim) for dim in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def _add_bias(self, x: "trt.ITensor", bias: Optional[np.ndarray],
                  rank: int) -> "trt.ITensor":
        x = self._unwrap(x)
        if bias is None:
            return x
        shape = [1] * (rank - 1) + [int(bias.shape[0])]
        constant = self.const(bias.astype(np.float16).reshape(shape), "b")
        return self.elementwise(x, constant, trt.ElementWiseOperation.SUM)

    def fp8_linear(self,
                   x: "trt.ITensor",
                   linear_weights,
                   rank: int = 3) -> "trt.ITensor":
        """FP8 Q/DQ MatMul with per-tensor activation and weight scales."""
        x = self._unwrap(x)
        input_scale = self.const(
            np.array(linear_weights.input_scale, dtype=np.float16), "x_scale")
        quantized_x = self.network.add_quantize(x, input_scale,
                                                trt.DataType.FP8)
        dequantized_x = self.network.add_dequantize(quantized_x.get_output(0),
                                                    input_scale, trt.float16)
        out_features, in_features = linear_weights.weight.shape
        weight = self.const_fp8(linear_weights.weight,
                                (out_features, in_features), "w_fp8")
        weight_scale = self.const(
            np.asarray(linear_weights.weight_scale, dtype=np.float16),
            "w_scale")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 0
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(w_tensor, [1] * (rank - 2) +
                                    [out_features, in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def mxfp8_linear(self,
                     x: "trt.ITensor",
                     linear_weights,
                     rank: int = 3) -> "trt.ITensor":
        """MXFP8 dynamic-activation Q/DQ and UE8M0 block-weight DQ."""
        x = self._unwrap(x)
        if not hasattr(trt.DataType, "UE8M0"):
            raise RuntimeError("MXFP8 requires TensorRT UE8M0 support")
        axis = rank - 1
        dynamic = self.network.add_dynamic_quantize(x, axis,
                                                    linear_weights.group_size,
                                                    trt.DataType.FP8,
                                                    trt.DataType.UE8M0)
        activation_scale = dynamic.get_output(1)
        dequantized_x = self.network.add_dequantize(dynamic.get_output(0),
                                                    activation_scale,
                                                    trt.float16)
        out_features, in_features = linear_weights.weight.shape
        weight = self.const_fp8(linear_weights.weight,
                                (out_features, in_features), "w_mxfp8")
        weight_scale = self.const_ue8m0(
            linear_weights.weight_scale,
            (out_features, in_features // linear_weights.group_size), "w_e8m0")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 1
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(w_tensor, [1] * (rank - 2) +
                                    [out_features, in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def int4_linear(self,
                    x: "trt.ITensor",
                    linear_weights,
                    rank: int = 3,
                    name: str = "") -> "trt.ITensor":
        """Lower groupwise INT4 GEMM."""
        if not name:
            raise ValueError("INT4 linear requires a stable module name")
        if linear_weights.pre_quant_scale is not None:
            shape = [1] * (rank - 1) + [linear_weights.in_features]
            smoother = self.const(
                linear_weights.pre_quant_scale.astype(
                    np.float16).reshape(shape), "pre_quant_scale")
            x = self.elementwise(x, smoother, trt.ElementWiseOperation.PROD)
        if linear_weights.activation_permutation is not None:
            x = self.gather(x, linear_weights.activation_permutation, rank - 1)
        weight = self.weight_input(
            name + ".qweight", linear_weights.weight.astype(np.int8,
                                                            copy=False),
            "int4_gemm")
        scales = self.weight_input(
            name + ".scales",
            np.asarray(linear_weights.weight_scale, dtype=np.float16),
            "int4_gemm")
        output = self.operation(
            "int4_groupwise_gemm", {
                "gemm_n": linear_weights.out_features,
                "gemm_k": linear_weights.in_features,
                "group_size": linear_weights.group_size,
            }, [x, weight, scales]).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def int8_sq_linear(self,
                       x: "trt.ITensor",
                       linear_weights,
                       rank: int = 3) -> "trt.ITensor":
        """SmoothQuant W8A8 Q/DQ MatMul."""
        smooth_shape = [1] * (rank - 1) + [linear_weights.in_features]
        smoother = self.const(
            linear_weights.pre_quant_scale.astype(
                np.float16).reshape(smooth_shape), "pre_quant_scale")
        smoothed = self.elementwise(x, smoother, trt.ElementWiseOperation.PROD)
        input_scale = self.const(
            np.array(linear_weights.input_scale, dtype=np.float32), "x_scale")
        quantized_x = self.network.add_quantize(smoothed, input_scale,
                                                trt.int8)
        dequantized_x = self.network.add_dequantize(quantized_x.get_output(0),
                                                    input_scale, trt.float16)
        weight = self.const(linear_weights.weight.astype(np.int8), "w_int8")
        weight_scale = self.const(
            linear_weights.weight_scale.astype(np.float32), "w_scale")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 0
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(
                w_tensor, [1] * (rank - 2) +
                [linear_weights.out_features, linear_weights.in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def nvfp4_act_qdq(self,
                      x: "trt.ITensor",
                      input_scale: float,
                      rank: int = 3) -> "trt.ITensor":
        """Quantize activations to FP4 with FP8 block scales, then
        dequantize the scales and activations to FP16."""
        x = self._unwrap(x)
        axis = rank - 1
        scale32 = self.const(np.array(input_scale, dtype=np.float32),
                             "act_scale32")
        dynq = self.network.add_dynamic_quantize(x, axis, 16, trt.DataType.FP4,
                                                 trt.DataType.FP8)
        dynq.set_input(1, scale32)
        x_f4 = dynq.get_output(0)
        block_scales_f8 = dynq.get_output(1)

        scale16 = self.const(np.array(input_scale, dtype=np.float16),
                             "act_scale16")
        dq_scales = self.network.add_dequantize(block_scales_f8, scale16,
                                                trt.float16)
        dq_x = self.network.add_dequantize(x_f4, dq_scales.get_output(0),
                                           trt.float16)
        dq_x.axis = axis
        return dq_x.get_output(0)

    def nvfp4_linear(self,
                     x: "trt.ITensor",
                     raw: dict,
                     rank: int = 3,
                     weight_dq_in_graph: bool = True) -> "trt.ITensor":
        """NVFP4 dense linear with activation and weight Q/DQ.

        ``raw`` comes from ``Weights.linear_nvfp4_raw``. With
        ``weight_dq_in_graph`` the FP4 weight and FP8 block scales become
        typed constants dequantized in-graph; otherwise the weight is
        pre-decoded to an FP16 constant (numerically equivalent).
        """
        x_dq = self.nvfp4_act_qdq(x, float(raw["input_scale"]), rank=rank)

        packed = raw["packed"]
        out_features = int(packed.shape[0])
        in_features = int(packed.shape[1]) * 2
        if weight_dq_in_graph:
            w_f4 = self.const_fp4(packed, (out_features, in_features), "w_fp4")
            ws_f8 = self.const_fp8(raw["weight_scale"],
                                   (out_features, in_features // 16),
                                   "w_scales")
            ws2 = self.const(np.array(raw["weight_scale_2"], dtype=np.float32),
                             "ws2")
            dq_ws = self.network.add_dequantize(ws_f8, ws2, trt.float32)
            dq_w = self.network.add_dequantize(w_f4, dq_ws.get_output(0),
                                               trt.float32)
            dq_w.axis = 1
            w16 = self.cast(dq_w.get_output(0), trt.float16)
            if rank != 2:
                w16 = self.reshape(w16, [1] * (rank - 2) +
                                   [out_features, in_features])
        else:
            from ..weight_packing import nvfp4 as nvfp4_pack
            dense = nvfp4_pack.decode_modelopt_nvfp4(packed,
                                                     raw["weight_scale"],
                                                     raw["weight_scale_2"], 16)
            wshape = [1] * (rank - 2) + [out_features, in_features]
            w16 = self.const(
                np.ascontiguousarray(dense.astype(np.float16).reshape(wshape)),
                "w")

        out = self.network.add_matrix_multiply(
            x_dq, trt.MatrixOperation.NONE, w16,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        bias = raw.get("bias")
        if bias is not None:
            bshape = [1] * (rank - 1) + [int(bias.shape[0])]
            b = self.const(bias.astype(np.float16).reshape(bshape), "b")
            out = self.elementwise(out, b, trt.ElementWiseOperation.SUM)
        return out
