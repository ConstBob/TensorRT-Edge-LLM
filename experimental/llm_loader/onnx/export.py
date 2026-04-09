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
ONNX export via ``torch.onnx.export(dynamo=True)``.


One graph covers prefill (``past_len=0``) and decode (``past_len>0``); custom
attention and Mamba ops expose state as I/O.

ONNX input / output layout - attention-only model
--------------------------------------------------
Inputs:
    inputs_embeds           [batch, seq_len, hidden_size]            float16
    past_key_values_0..N    [batch, 2, num_kv_heads, past, head_dim] float16
    rope_rotary_cos_sin     [batch, max_pos, rotary_dim]  float32
    context_lengths         [batch]                       int32
    kvcache_start_index     [batch]                       int32
    last_token_ids          [batch, 1]                    int64

Outputs:
    logits                  [batch, seq_len, vocab_size]             float32
    present_key_values_0..N [batch, 2, num_kv_heads, past+seq_len, head_dim] float16

Additional I/O for hybrid (Mamba) models
-----------------------------------------
Extra inputs:
    conv_state_0..M   [batch, conv_dim, conv_kernel-1]        float16
    ssm_state_0..M    [batch, num_heads, head_dim, ssm_state] float16

Extra outputs:
    present_conv_0..M   updated conv states
    present_ssm_0..M    updated ssm states
"""

import contextlib
import logging
import os

import onnx
import torch

from ..checkpoint.checkpoint_utils import write_runtime_artifacts
from ..models.default.modeling_default import CausalLM
from .dynamo_translations import build_custom_translation_table

logger = logging.getLogger(__name__)

__all__ = ["export_onnx"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_onnx(
    model: CausalLM,
    output_path: str,
    model_dir: str = "",
) -> None:
    """Export *model* to ONNX using the dynamo exporter.

    Writes ``model.onnx``, ``model.onnx.data``, ``config.json``,
    ``embedding.safetensors``, and any tokenizer files present in
    *model_dir* to the same output directory.

    Args:
        model:       A :class:`~modules.CausalLM` with weights loaded.
        output_path: Destination ``.onnx`` file path.
        model_dir:   Checkpoint directory (for tokenizer file copying).
                     If empty, tokenizer files are skipped.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    _export_model(model, output_path)
    write_runtime_artifacts(model, model_dir, out_dir)


# ---------------------------------------------------------------------------
# ONNX post-processing: TRT compatibility
# ---------------------------------------------------------------------------


def _fix_nvfp4_weight_dtype(onnx_path: str) -> None:
    """Reinterpret INT8 NVFP4 weight initializers as FLOAT4E2M1.

    Our model stores packed FP4 weights as int8 [out, in//2] (2 nibbles per byte).
    TRT's DequantizeLinear with block_size requires FLOAT4E2M1 (elem_type=23)
    type with logical shape [out, in] -- same bytes, different ONNX element type
    and shape declaration.

    This pass finds all int8 initialisers whose name ends with ``.weight``
    (NVFP4 linear weights) and rewrites them:
      elem_type  INT8  -> FLOAT4E2M1 (23)
      dims       [N, M] -> [N, M*2]   (double the last dim -- nibble unpacking)
    """
    _FLOAT4E2M1 = 23  # ONNX TensorProto.FLOAT4E2M1
    _INT8 = 3  # ONNX TensorProto.INT8

    model = onnx.load(onnx_path, load_external_data=False)
    changed = 0
    for init in model.graph.initializer:
        # Only INT8 tensors whose name ends with ".weight" (not scale / qweight)
        if (init.data_type != _INT8 or not init.name.endswith(".weight")
                or "scale" in init.name or "qweight" in init.name):
            continue
        if len(init.dims) < 1:
            continue
        # Reinterpret: same raw bytes, element type -> FLOAT4E2M1, last dim *2
        init.data_type = _FLOAT4E2M1
        old_dims = list(init.dims)
        init.dims[-1] = old_dims[-1] * 2
        changed += 1

    if not changed:
        return

    logger.info("TRT fix: reinterpreted %d NVFP4 weight(s) as FLOAT4E2M1",
                changed)
    data_file = os.path.basename(onnx_path) + ".data"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

_OPSET_VERSION = 24


@contextlib.contextmanager
def _permissive_inline_opset():
    """Patch onnx-ir InlinePass to resolve opset-version conflicts by taking max.

    torch TORCHLIB functions are compiled at opset 18; our custom onnxscript
    translation functions use opset 21 (required for FP8 ``QuantizeLinear``
    with ``output_dtype``).  ``InlinePass._instantiate_call`` raises
    ``ValueError: Opset mismatch: 18 != 21`` when it encounters both in the
    same model.

    The standard ONNX domain is strictly backwards-compatible, so taking the
    higher version is correct: opset 21 is a superset of opset 18.
    """
    try:
        from onnx_ir.passes.common.inliner import InlinePass
    except ImportError:
        yield
        return

    _orig = InlinePass._instantiate_call

    def _patched(self, node, call_site_id):
        # Pre-merge opset_imports taking max to avoid ValueError in original.
        # Also align function.opset_imports so _orig's equality check passes.
        op_id = node.op_identifier()
        function = self._functions.get(op_id)
        if function is not None:
            for key, value in list(function.opset_imports.items()):
                merged = max(self._opset_imports.get(key, value), value)
                self._opset_imports[key] = merged
                function.opset_imports[key] = merged
        return _orig(self, node, call_site_id)

    InlinePass._instantiate_call = _patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        InlinePass._instantiate_call = _orig  # type: ignore[method-assign]


def _setup_fp8kv_scales_for_export(model: "CausalLM") -> None:
    """Pre-cache FP8 KV scales as Python floats before torch.export tracing.

    During tracing, calling ``.item()`` on a tensor buffer creates a
    data-dependent symbolic expression that ``torch.export`` cannot guard on.
    By extracting the float values here (before the trace) and storing them
    as plain Python attributes on each attention module, they appear as
    compile-time constants during export.

    Stored attribute: ``module._qkv_scales_float = [q, k, v]``
      - q_scale : 1.0 (not stored in any current checkpoint)
      - k_scale : ``k_proj.k_scale`` buffer value if present, else 1.0
      - v_scale : ``v_proj.v_scale`` buffer value if present, else 1.0
    """
    for module in model.modules():
        if not getattr(module, "enable_fp8_kv_cache", False):
            continue
        k_buf = getattr(getattr(module, "k_proj", None), "k_scale", None)
        v_buf = getattr(getattr(module, "v_proj", None), "v_scale", None)
        module._qkv_scales_float = [
            1.0,
            float(k_buf.item()) if k_buf is not None else 1.0,
            float(v_buf.item()) if v_buf is not None else 1.0,
        ]


def _fix_initializer_dtypes(onnx_path: str) -> None:
    """Single-pass ONNX initializer dtype fixup for TRT compatibility.

    Performs two corrections in one ONNX load+save:

    1. **FP32 weights → FP16**: The dynamo exporter may emit FP32 constants
       for FP16 model weights (e.g. tied lm_head in BF16 checkpoints).  TRT
       requires uniform dtype in MatMul inputs.  Scalars and quantization
       scale tensors are left as FP32.

    2. **Mamba ssm_A → FP32**: ONNX constant folding may collapse the
       ``A_log.to(float32) → exp → neg`` chain into a single initializer.
       The ``update_ssm_state`` plugin requires its A input (position 1) to
       be FP32, so any such initializer is kept (or restored to) FP32.
    """
    import numpy as np

    _onnx = __import__("onnx")
    model = _onnx.load(onnx_path)

    # Collect Mamba A-input initializer names — these must stay FP32.
    mamba_a_names: set = set()
    for node in model.graph.node:
        if node.op_type == "update_ssm_state" and len(node.input) > 1:
            mamba_a_names.add(node.input[1])

    n_to_fp16 = 0
    n_to_fp32 = 0
    for init in model.graph.initializer:
        # --- Mamba A: ensure FP32 ---
        if init.name in mamba_a_names and init.data_type == 10:  # FP16
            dims = list(init.dims)
            data = np.frombuffer(init.raw_data, dtype=np.float16).reshape(dims)
            init.data_type = 1  # FLOAT (FP32)
            init.raw_data = data.astype(np.float32).tobytes()
            n_to_fp32 += 1
            logger.info("_fix_initializer_dtypes: %s %s FP16→FP32 (mamba A)",
                        init.name, dims)
            continue

        # --- FP32 weight → FP16 ---
        if init.data_type != 1:  # not FP32
            continue
        if init.name in mamba_a_names:  # already FP32, must stay
            continue
        dims = list(init.dims)
        if len(dims) == 0 or (len(dims) == 1 and dims[0] <= 1):
            continue  # keep scalars as FP32
        if (init.name.endswith(".weight_scale")
                or init.name.endswith(".input_scale")
                or init.name.endswith(".pre_quant_scale")
                or init.name.endswith("_scale")
                or init.name.endswith("_scale_2")):
            continue  # keep quantization scales as FP32
        data = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
        init.data_type = 10  # FLOAT16
        init.raw_data = data.astype(np.float16).tobytes()
        n_to_fp16 += 1
        logger.info("_fix_initializer_dtypes: %s %s FP32→FP16", init.name,
                    dims)

    if n_to_fp16 == 0 and n_to_fp32 == 0:
        return

    # Update matching value_info entries
    vi_map = {vi.name: vi for vi in model.graph.value_info}
    for init in model.graph.initializer:
        if init.name in vi_map:
            vi_map[init.name].type.tensor_type.elem_type = init.data_type

    logger.info("_fix_initializer_dtypes: %d→FP16, %d→FP32, saving...",
                n_to_fp16, n_to_fp32)
    _onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        convert_attribute=True,
    )


def _export_model(model: "CausalLM",
                  output_path: str,
                  optimize: bool = True) -> None:
    _setup_fp8kv_scales_for_export(model)
    spec = model.onnx_export_spec()
    translation_table = build_custom_translation_table()

    logger.info("Exporting ONNX to %s (opset %d, dynamo) ...", output_path,
                _OPSET_VERSION)
    with _permissive_inline_opset():
        prog = torch.onnx.export(
            spec.wrapped,
            spec.args,
            dynamo=True,
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_shapes=spec.dynamic_shapes,
            opset_version=_OPSET_VERSION,
            custom_translation_table=translation_table,
            external_data=True,
            optimize=optimize,
        )
    prog.save(output_path)
    with open(output_path, "rb") as _f:
        os.fsync(_f.fileno())
    if model.config.quant.uses_nvfp4_weights:
        _fix_nvfp4_weight_dtype(output_path)
    _fix_initializer_dtypes(output_path)
    logger.info("Export complete: %s", output_path)
