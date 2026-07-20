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
"""Qwen3-Omni Thinker + Talker quantization driver (NVFP4 + INT4 AWQ).

End-to-end quantization for both the Thinker text subgraph and the Talker
text subgraph of Qwen3-Omni.  Two model-variant conventions are handled
via a single ``quantize_qwen3_omni`` entry point:

  * ``model_type="qwen3_omni_moe"`` -- top-level class
    ``Qwen3OmniMoeForConditionalGeneration`` (released transformers).
  * ``model_type="qwen3_omni"``     -- top-level class
    ``Qwen3OmniForConditionalGeneration`` (requires a transformers build
    that carries the non-MoE class).

Two backbones are supported and selected via the ``quantization`` argument:

  * ``"nvfp4"`` (default) -- per-block FP4 weights + FP8 scale; consumed by
    the NVFP4 MoE plugin.
  * ``"int4_awq"``         -- W4A16-AWQ group=128; consumed by the Marlin
    ``int4MoePlugin`` GEMM.

Produces a single HF root under ``<output_dir>`` (same layout as the
source HF root) with quantized Thinker + Talker text weights and
FP16-kept visual / audio_tower / code2wav / code_predictor / talker
sidecars all in one consolidated safetensors set.  ``tensorrt-edgellm-export``
consumes the whole directory in one shot; no submodel split.

Calibration is **joint and multimodal**: a single ``mtq.quantize(model, ...)``
call observes both Thinker and Talker quantizers in the same forward loop.
Per sample:

  1. Thinker(audio / image / text) with ``output_hidden_states=True`` --
     Thinker quantizers observe true FP16 activations (modelopt's
     observe-only mode does not apply QDQ during calibration).
  2. ``thinker_hidden = hidden_states[accept_hidden_layer]`` and
     ``thinker_embed = hidden_states[0]`` (= inputs_embeds) feed
     ``talker.hidden_projection`` / ``talker.text_projection`` to produce
     realistic Talker ``inputs_embeds`` (same data flow as
     ``buildTalkerPrefillFromSegments`` in the C++ runtime).
  3. Talker(inputs_embeds=..., talker_input_ids=thinker_input_ids,
     attention_mask=...) -- Talker quantizers observe activations.

For ``quantization="int4_awq"`` two modelopt 0.44.0 quirks are patched
inline (see ``_int4_awq_modelopt_wars``); NVFP4 is unaffected.
"""

import copy
import os
import shutil
import time
import warnings
from contextlib import contextmanager, nullcontext
from typing import Optional

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from datasets import (Audio, concatenate_datasets, get_dataset_config_names,
                      load_dataset)
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .quantization_configs import build_quant_config

try:
    import librosa
except ImportError:
    librosa = None

# ---------------------------------------------------------------------------
# Quant config — Thinker + Talker text path
# ---------------------------------------------------------------------------


def _build_full_model_quant_cfg(quantization: str,
                                lm_head_quantization: Optional[str],
                                kv_cache_quantization: Optional[str],
                                is_moe: bool = True,
                                visual_quantization: Optional[str] = None,
                                audio_quantization: Optional[str] = None,
                                cp_quantization: Optional[str] = None) -> dict:
    """Build a quant_cfg for the full Qwen3-Omni model (both variants).

    ``quantization`` selects the backbone: ``"nvfp4"`` (per-block FP4 + FP8
    scale) or ``"int4_awq"`` (W4A16-AWQ group=128). Both share the same
    disable_globs (FP16 keep-set) and the same joint multimodal calib
    downstream; only the backbone precision differs.

    ``is_moe`` toggles two MoE-only disable_globs (``mlp.gate`` router and
    ``shared_expert_gate``); the non-MoE variant lacks these modules so
    the globs are dropped from its config.

    ``visual_quantization`` / ``audio_quantization`` / ``cp_quantization``:
    when ``None`` the visual encoder / audio_tower / code_predictor stay
    FP16. When set (only ``"fp8"`` exposed today), :func:`build_quant_config`
    layers an explicit override on the corresponding submodule patterns, and
    this driver drops its own disable glob for that submodule so the
    override wins. The joint multimodal calibration forward loop already
    exercises visual + audio_tower activations; the code_predictor needs a
    dedicated drive (see :func:`qwen3_cp_calibration_loop`) appended to the
    forward loop by the caller.

    Used by a single joint ``mtq.quantize(model, cfg, forward_loop=...)`` call
    that calibrates both Thinker and Talker subgraphs in one multimodal pass.
    Because the recipe is applied to the full model, all module paths in the
    ``disable_globs`` carry the ``thinker.`` / ``talker.`` prefix; this avoids
    cross-contamination between submodels that bare wildcards would risk.

    Disables: code2wav, talker sidecar projections/output heads, all MoE
    routers and shared-expert gates; plus visual / audio_tower /
    code_predictor unless opted in via their respective parameters.
    """
    cfg = build_quant_config(quantization,
                             lm_head_quantization,
                             kv_cache_quantization,
                             visual_quantization=visual_quantization,
                             audio_quantization=audio_quantization,
                             cp_quantization=cp_quantization)

    # AWQ-Lite (per-channel alpha search): default-on for INT4 AWQ (needed
    # for the Talker's large per-channel absmax spread at g=128), env-var
    # opt-in for NVFP4 (RTN suffices). alpha_step=0.25 keeps the search
    # tractable on this MoE (modelopt's 0.1 default is ~4x slower).
    use_awq_lite = (quantization == "int4_awq"
                    or os.environ.get("QWEN3_OMNI_USE_AWQ_LITE") == "1")
    if use_awq_lite:
        alpha_step = float(os.environ.get("QWEN3_OMNI_AWQ_ALPHA_STEP", "0.25"))
        cfg["algorithm"] = {"method": "awq_lite", "alpha_step": alpha_step}
        print(
            f"[Omni quant] AWQ-Lite enabled: method=awq_lite, "
            f"alpha_step={alpha_step}",
            flush=True)

    disable_globs = [
        # ---- Thinker non-LLM submodules (FP16) ----
        "*thinker.lm_head*",  # Final LM head — FP16 for logit fidelity
        # ---- Talker non-LLM submodules (FP16) ----
        "*talker.code2wav.*",  # Code2Wav vocoder (not actually attached
        #   to the talker subtree in HF but kept
        #   here defensively for any wrappers)
        "*talker.hidden_projection*",  # Thinker hidden -> talker-space MLP
        "*talker.text_projection*",  # Thinker embed  -> talker-space MLP
        "*talker.codec_head*",  # Codec output projection (FP16: feeds
        #   Code2Wav; FP4 noise here materially
        #   hurts audio quality)
    ]
    # Only pin the visual encoder / audio_tower to FP16 when the caller did
    # not opt them into quantization via ``visual_quantization`` /
    # ``audio_quantization``. When they are opted in, ``build_quant_config``
    # has already layered the corresponding per-submodule override; adding
    # our own disable glob here would clobber that.
    if visual_quantization is None:
        disable_globs.append("*thinker.visual.*")
    if audio_quantization is None:
        disable_globs.append("*thinker.audio_tower.*")
    if cp_quantization is None:
        disable_globs.append("*talker.code_predictor.*")
    if is_moe:
        # ---- MoE routers and shared-expert gates (FP16 for top-k stability) ----
        # The router is a tiny [H × num_experts] linear; its output drives
        # discrete top-k selection so even small quant noise can flip experts
        # and propagate large output errors.
        disable_globs.append("*mlp.gate.*")
        # ``shared_expert_gate`` is a 1-output Linear producing a sigmoid
        # mixer scalar; quantizing it directly biases per-token shared-expert
        # contribution.
        disable_globs.append("*shared_expert_gate.*")
        # ``shared_expert`` FFN runs at NVFP4: a layer-wise SNR audit showed
        # its contribution to downstream noise is negligible (Δcos < 2e-4
        # at the late layers), so the quantization-sensitivity budget is
        # better spent on the routed experts. Add the three globs below to
        # put it back at FP16 if a future model proves more sensitive.
        # disable_globs.append("*shared_expert.gate_proj.*")
        # disable_globs.append("*shared_expert.up_proj.*")
        # disable_globs.append("*shared_expert.down_proj.*")
    # modelopt's ``quant_cfg`` is an ordered list of rule dicts; append a
    # disable rule per glob at the end so it overrides the earlier
    # ``*weight_quantizer`` / ``*input_quantizer`` enables. A ``dict``
    # ``quant_cfg`` (``{pattern: cfg}``) is handled too.
    qc = cfg["quant_cfg"]
    for g in disable_globs:
        if isinstance(qc, list):
            qc.append({"quantizer_name": g, "enable": False})
        else:
            qc[g] = {"enable": False}
    return cfg


# ---------------------------------------------------------------------------
# modelopt 0.44.0 INT4 AWQ workarounds (active only when quantization ==
# "int4_awq"; gated via ``_maybe_int4_awq_wars``).  See the module docstring
# for the two upstream quirks these patches address.
# ---------------------------------------------------------------------------


def _block_amax_fallback(quantizer, weight_2d) -> None:
    """Derive per-block amax for an uncalibrated INT4-block quantizer.

    ``weight_2d``: ``[out, in]``; the block dim is the last dim, with
    ``block_sizes.get(-1)`` giving the group size (defaults to 128).
    Writes back to ``quantizer._amax`` in flat ``[out*num_blocks, 1]``
    storage with ``_amax_shape_for_export=(out, -1)``.
    """
    bs = quantizer.block_sizes.get(-1) or quantizer.block_sizes.get(
        weight_2d.dim() - 1, 128)
    out, K = weight_2d.shape
    num_blocks = K // bs
    per_block = weight_2d.abs().reshape(out, num_blocks, bs).amax(dim=-1)
    flat = per_block.reshape(-1, 1).contiguous().to(torch.float32)
    if hasattr(quantizer, "_amax"):
        delattr(quantizer, "_amax")
    quantizer.register_buffer("_amax", flat)
    quantizer._amax_shape_for_export = (out, -1)


def _slice_fused_amax(w_quantizer, fused_start: int, weight_slice_dim0: int,
                      fused_total: int) -> bool:
    """Slice the export-shape view of ``_amax`` for a gate/up projection.

    The amax storage layout for INT4 block quant is flat
    ``[out_full * num_blocks, 1]`` with a separate reshape hint
    ``_amax_shape_for_export=(out_full, -1)``. The upstream slicing code
    in ``_export_fused_experts`` keys off ``amax.shape[0]`` (flat) instead
    of the export-shape dim 0, so the divisibility check always fails for
    block-quant and slicing is silently skipped. We slice on the
    export-shape dim 0 and rewrite the flat storage to match.

    Returns ``True`` if the buffer was sliced, ``False`` if the layout is
    not the flat-block form (caller falls back to upstream behaviour).
    """
    amax = getattr(w_quantizer, "_amax", None)
    if amax is None:
        return False
    shape_for_export = getattr(w_quantizer, "_amax_shape_for_export", None)
    if shape_for_export is None:
        return False
    try:
        amax_view = amax.reshape(shape_for_export)
    except Exception:
        return False
    if amax_view.dim() < 1 or amax_view.shape[0] != fused_total:
        return False
    sliced = amax_view[fused_start:fused_start +
                       weight_slice_dim0].contiguous()
    new_dim0 = sliced.shape[0]
    new_flat = sliced.reshape(-1, 1).contiguous()
    delattr(w_quantizer, "_amax")
    w_quantizer.register_buffer("_amax", new_flat)
    w_quantizer._amax_shape_for_export = (new_dim0, -1)
    return True


def _export_fused_experts_int4_aware(module, dtype) -> None:
    """Replacement for ``modelopt.torch.export.moe_utils._export_fused_experts``
    that slices flat per-block amax on the export-shape dim and supplies a
    per-block weight-derived amax fallback for uncalibrated experts.

    Functionally equivalent to the upstream implementation for non-INT4
    paths (the flat-amax slice and the uncalibrated fallback are guarded
    by ``block_sizes.get("type") == "static"``); only the INT4 AWQ block
    path takes the new code path.
    """
    from modelopt.torch.export.unified_export_hf import \
        _export_quantized_weight
    from modelopt.torch.quantization.plugins.huggingface import \
        _get_fused_expert_intermediate_dim

    expert_dim = _get_fused_expert_intermediate_dim(module)
    fused_dim0 = 2 * expert_dim
    n = module.num_experts
    gate_up_input_q = module.gate_up_proj_input_quantizer
    down_input_q = module.down_proj_input_quantizer
    gate_up = module.gate_up_proj.data
    down = module.down_proj.data

    for idx in range(n):
        expert = nn.Module()

        # Pre-export uncalibrated fallback for the fused gate_up quantizer.
        gate_up_q = module.gate_up_proj_weight_quantizers[idx]
        is_block_quant = (getattr(gate_up_q, "block_sizes", None) is not None
                          and gate_up_q.block_sizes.get("type",
                                                        None) == "static")
        if getattr(gate_up_q, "is_enabled",
                   False) and (not hasattr(gate_up_q, "_amax")
                               or gate_up_q._amax is None
                               or torch.all(gate_up_q._amax == 0)):
            if is_block_quant:
                _block_amax_fallback(gate_up_q, gate_up[idx])
                warnings.warn(
                    f"Expert {idx} gate_up_proj: uncalibrated, using "
                    f"per-block weight-derived amax.",
                    stacklevel=2)
            else:
                gate_up_q.amax = gate_up[idx].abs().amax().to(torch.float32)
                warnings.warn(
                    f"Expert {idx} gate_up_proj: uncalibrated, using "
                    f"scalar amax fallback.",
                    stacklevel=2)

        projections = [
            ("gate_proj", gate_up[idx, :expert_dim, :], 0, fused_dim0, True),
            ("up_proj", gate_up[idx,
                                expert_dim:, :], expert_dim, fused_dim0, True),
            ("down_proj", down[idx], 0, down.shape[1], False),
        ]

        for proj_name, weight_slice, fused_start, fused_total, is_gate_up in projections:
            w_quantizer_src = (module.gate_up_proj_weight_quantizers[idx]
                               if is_gate_up else
                               module.down_proj_weight_quantizers[idx])
            i_quantizer = gate_up_input_q if is_gate_up else down_input_q

            # gate/up share a weight quantizer -- clone so each gets its own
            # sliced amax. down_proj uses the source directly.
            w_quantizer = (copy.deepcopy(w_quantizer_src)
                           if is_gate_up else w_quantizer_src)

            sliced_ok = _slice_fused_amax(w_quantizer, fused_start,
                                          weight_slice.shape[0], fused_total)
            if not sliced_ok and hasattr(w_quantizer, "_amax") and \
                    w_quantizer._amax is not None and w_quantizer._amax.dim() >= 1:
                # Upstream behaviour for non-flat-block amax layout.
                amax = w_quantizer._amax
                amax_dim0 = amax.shape[0]
                if fused_total % amax_dim0 == 0:
                    slice_start = fused_start * amax_dim0 // fused_total
                    slice_end = (fused_start + weight_slice.shape[0]
                                 ) * amax_dim0 // fused_total
                    delattr(w_quantizer, "_amax")
                    w_quantizer.register_buffer(
                        "_amax", amax[slice_start:slice_end].contiguous())

            if (hasattr(w_quantizer, "is_enabled") and w_quantizer.is_enabled
                    and
                (not hasattr(w_quantizer, "_amax") or w_quantizer._amax is None
                 or torch.all(w_quantizer._amax == 0))):
                is_bq = (getattr(w_quantizer, "block_sizes", None) is not None
                         and w_quantizer.block_sizes.get("type",
                                                         None) == "static")
                if is_bq:
                    _block_amax_fallback(w_quantizer, weight_slice)
                    warnings.warn(
                        f"Expert {idx} {proj_name}: uncalibrated, using "
                        f"per-block weight-derived amax.",
                        stacklevel=2)
                else:
                    w_quantizer.amax = weight_slice.abs().amax().to(
                        torch.float32)
                    warnings.warn(
                        f"Expert {idx} {proj_name}: uncalibrated, using "
                        f"scalar amax fallback.",
                        stacklevel=2)

            wrapper = nn.Module()
            wrapper.weight = nn.Parameter(weight_slice.contiguous(),
                                          requires_grad=False)
            wrapper.weight_quantizer = w_quantizer
            wrapper.input_quantizer = i_quantizer

            _export_quantized_weight(wrapper, dtype)

            proj = nn.Module()
            proj.weight = wrapper.weight
            for attr in ("weight_scale", "weight_scale_2", "input_scale"):
                if hasattr(wrapper, attr):
                    proj.register_buffer(attr, getattr(wrapper, attr))

            expert.add_module(proj_name, proj)

        module.add_module(str(idx), expert)

    for attr in ("gate_up_proj", "down_proj", "gate_up_proj_weight_quantizers",
                 "gate_up_proj_input_quantizer", "down_proj_weight_quantizers",
                 "down_proj_input_quantizer"):
        if hasattr(module, attr):
            delattr(module, attr)


@contextmanager
def _int4_awq_modelopt_wars():
    """Patch two modelopt 0.44.0 functions for the duration of an INT4 AWQ
    ``mtq.quantize`` + submodel export run, then restore them.
    """
    import modelopt.torch.export.layer_utils as _layer_utils
    import modelopt.torch.export.moe_utils as _moe_utils
    import modelopt.torch.export.unified_export_hf as _uehf

    _orig_get_experts_list = _layer_utils.get_experts_list

    def _patched_get_experts_list(module, model_type):
        try:
            return _orig_get_experts_list(module, model_type)
        except NotImplementedError:
            mt = (model_type or "").lower()
            if any(s in mt
                   for s in ("qwen3omnimoe", "qwen3_omni_moe", "qwen3omni")):
                # Fused-experts share an input quantizer across experts, so
                # cross-expert resmooth is structurally unnecessary.
                return []
            raise

    _orig_export_fused = _moe_utils._export_fused_experts
    _orig_export_fused_uehf = getattr(_uehf, "_export_fused_experts", None)

    _layer_utils.get_experts_list = _patched_get_experts_list
    _uehf.get_experts_list = _patched_get_experts_list
    _moe_utils._export_fused_experts = _export_fused_experts_int4_aware
    if _orig_export_fused_uehf is not None:
        _uehf._export_fused_experts = _export_fused_experts_int4_aware

    try:
        yield
    finally:
        _layer_utils.get_experts_list = _orig_get_experts_list
        _uehf.get_experts_list = _orig_get_experts_list
        _moe_utils._export_fused_experts = _orig_export_fused
        if _orig_export_fused_uehf is not None:
            _uehf._export_fused_experts = _orig_export_fused_uehf


def _maybe_int4_awq_wars(quantization: str):
    """Return the INT4 AWQ WAR context manager, or a nullcontext for NVFP4."""
    return _int4_awq_modelopt_wars(
    ) if quantization == "int4_awq" else nullcontext()


# ---------------------------------------------------------------------------
# Multimodal calibration for Talker
# ---------------------------------------------------------------------------


class _OmniMultimodalCalibDataset(Dataset):
    """Mixed-modality calibration dataset for Qwen3-Omni Talker.

    Pre-encodes each sample with the HF processor so calibration just
    iterates ready-to-forward dicts.
    """

    def __init__(self, processor, audio_data, image_data, text_data):
        self.processor = processor
        self.samples = []
        for item in audio_data:
            self.samples.append({"type": "audio", "raw": item})
        for item in image_data:
            self.samples.append({"type": "image", "raw": item})
        for text in text_data:
            self.samples.append({"type": "text", "raw": text})

    def __len__(self):
        return len(self.samples)

    def _process_audio(self, raw):
        import io

        import soundfile as sf
        audio_item = raw["audio"]
        audio_bytes = audio_item.get("bytes")
        if audio_bytes is not None:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio, sr = sf.read(audio_item["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "placeholder"
                },
                {
                    "type": "text",
                    "text": "Describe what you hear."
                },
            ],
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_image(self, raw):
        from PIL import Image
        images = [
            v.convert("RGB") for k, v in raw.items()
            if "image" in k and isinstance(v, Image.Image)
        ]
        if not images:
            return self._process_text("Describe the scene.")
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": "Describe what you see."
                },
            ],
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text,
                                images=images[:1],
                                return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_text(self, content):
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kind = sample["type"]
        if kind == "audio":
            return self._process_audio(sample["raw"])
        if kind == "image":
            return self._process_image(sample["raw"])
        return self._process_text(sample["raw"])


def _build_multimodal_calib_dataset(
    processor,
    num_audio: int,
    num_image: int,
    num_text: int,
    audio_dataset_dir: str = "openslr/librispeech_asr",
    visual_dataset_dir: str = "lmms-lab/MMMU",
    text_ds=None,
) -> "_OmniMultimodalCalibDataset":
    """Build the multimodal calibration dataset (audio + image + text).

    Setting any ``num_*`` to 0 skips that modality. ``audio_dataset_dir`` /
    ``visual_dataset_dir`` are optional HF dataset overrides
    (Qwen3.5-Omni orchestrator). ``text_ds`` is a registered dataset name,
    a generator, or ``None`` for the registry default — resolved via
    :func:`~tensorrt_edgellm.quantization.datasets.resolve_dataset`.
    """
    audio_data = []
    if num_audio > 0:
        print(
            f"[Omni calib] Loading audio from {audio_dataset_dir}, n={num_audio}"
        )
        audio_stream = load_dataset(audio_dataset_dir,
                                    "clean",
                                    split="test",
                                    streaming=True)
        audio_stream = audio_stream.cast_column("audio", Audio(decode=False))
        audio_data = list(audio_stream.take(num_audio))

    image_data = []
    if num_image > 0:
        print(
            f"[Omni calib] Loading images from {visual_dataset_dir}, n={num_image}"
        )
        if "lmms-lab/MMMU" in visual_dataset_dir:
            image_dataset = load_dataset(visual_dataset_dir, split="dev")
        elif "MMMU" in visual_dataset_dir:
            configs = get_dataset_config_names(visual_dataset_dir)
            image_dataset = concatenate_datasets([
                load_dataset(visual_dataset_dir, c, split="dev")
                for c in configs
            ])
        else:
            image_dataset = load_dataset(visual_dataset_dir, split="dev")
        image_data = list(
            image_dataset.select(range(min(num_image, len(image_dataset)))))

    text_data = []
    if num_text > 0:
        from itertools import islice

        from .datasets import dataset_name, resolve_dataset
        text_ds = resolve_dataset(text_ds, "text")
        print(f"[Omni calib] Loading text ({dataset_name(text_ds)}), "
              f"n={num_text}")
        text_data = list(islice(text_ds(), num_text))

    return _OmniMultimodalCalibDataset(processor, audio_data, image_data,
                                       text_data)


def _calib_full_multimodal(model, calib_dataset,
                           accept_hidden_layer: int) -> None:
    """Joint multimodal calibration forward loop for the full Qwen3-Omni model.

    Runs one Thinker forward and one Talker forward per sample so that
    BOTH submodels' quantizers observe activations in a single calibration
    pass.

    During calibration, modelopt's freshly-inserted quantizers are in
    observe-only mode (collecting amax statistics; no QDQ applied), so the
    Thinker forward returns true FP16 hidden states.  Those hidden states
    are projected through Talker's text/hidden projection MLPs (the actual
    runtime data flow handled by ``buildTalkerPrefillFromSegments`` in
    ``qwen3OmniTTSRuntime``), giving Talker quantizers the same input
    distribution they will see at inference.

    Args:
        model: Top-level ``Qwen3OmniMoeForConditionalGeneration``.
        calib_dataset: Mixed audio/image/text dataset built by
            :func:`_build_multimodal_calib_dataset`.
        accept_hidden_layer: Which Thinker layer feeds Talker's
            ``hidden_projection`` for multimodal token positions.  Must be
            the same value the C++ runtime reads from talker config
            (``mTalkerConfig.acceptHiddenLayer``) and the same value the
            Thinker ONNX export emits at ``last_pre_norm_hidden_states``.
    """
    device = next(model.parameters()).device
    thinker_config = model.config.thinker_config
    multimodal_token_ids = [
        getattr(thinker_config, field, -1)
        for field in ("audio_token_id", "image_token_id", "video_token_id")
    ]
    talker_dtype = next(model.talker.parameters()).dtype
    tc = model.talker.config.text_config

    skipped = 0
    for i in tqdm(range(len(calib_dataset)),
                  desc="Calibrating Omni (multimodal: thinker+talker)"):
        try:
            data = calib_dataset[i]
        except Exception as error:
            skipped += 1
            if skipped <= 3:
                print(f"[Omni calib] Skipping sample {i}: {error}")
            continue
        data = {
            k:
            v.unsqueeze(0).to(
                device,
                dtype=model.thinker.dtype if v.is_floating_point() else None)
            for k, v in data.items()
        }
        with torch.no_grad():
            thinker_out = model.thinker(**data, output_hidden_states=True)
            all_hidden = thinker_out.hidden_states
            if all_hidden is None or len(all_hidden) <= accept_hidden_layer:
                raise RuntimeError(
                    f"Thinker returned "
                    f"{0 if all_hidden is None else len(all_hidden)} "
                    f"hidden-state tensors; "
                    f"accept_hidden_layer={accept_hidden_layer}")
            thinker_hidden = all_hidden[accept_hidden_layer]
            thinker_embed = all_hidden[0]

            input_ids = data["input_ids"]
            mm_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in multimodal_token_ids:
                if token_id >= 0:
                    mm_mask |= (input_ids == token_id)

            seq_len = thinker_hidden.shape[1]
            inputs_embeds = torch.empty(1,
                                        seq_len,
                                        tc.hidden_size,
                                        dtype=talker_dtype,
                                        device=device)
            if mm_mask.any():
                inputs_embeds[mm_mask] = model.talker.hidden_projection(
                    thinker_hidden[mm_mask].to(talker_dtype))
            text_mask = ~mm_mask
            if text_mask.any():
                inputs_embeds[text_mask] = model.talker.text_projection(
                    thinker_embed[text_mask].to(talker_dtype))

            # ``talker_input_ids`` reuses the Thinker's input_ids verbatim,
            # matching HF's reference flow (modeling_qwen3_omni_moe.py:
            # ``_get_talker_user_parts``).  Talker only consumes this tensor
            # for 3D RoPE position bookkeeping (``get_rope_index`` checks
            # equality against ``audio_token_id`` / ``image_token_id`` etc.)
            # and never does an embedding lookup on it, so the IDs can stay
            # in Thinker's full vocab range even though it exceeds Talker's
            # codec vocab size.
            talker_ids = input_ids
            attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
            model.talker(inputs_embeds=inputs_embeds,
                         attention_mask=attn_mask,
                         talker_input_ids=talker_ids)

    if skipped:
        print(f"[Omni calib] Skipped {skipped}/{len(calib_dataset)} samples "
              f"due to processor errors")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def quantize_qwen3_omni(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = "nvfp4",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    audio_quantization: Optional[str] = None,
    cp_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    text_dataset=None,
    num_samples: int = 64,
    max_length: int = 64,
    talker_num_audio: int = 150,
    talker_num_image: int = 150,
    talker_num_text: int = 200,
    talker_accept_hidden_layer: Optional[int] = None,
) -> str:
    """Quantize Thinker + Talker text and save a single HF root.

    ``quantization`` selects the backbone: ``"nvfp4"`` (default) or
    ``"int4_awq"``. Calibration is **joint and multimodal**: a single
    ``mtq.quantize(model, ...)`` call is made on the full
    ``Qwen3Omni(Moe)?ForConditionalGeneration``, with a per-sample forward
    loop that runs

        Thinker(multimodal input)  -- quantizers observe Thinker activations
          → hidden_projection / text_projection
          → Talker(projected inputs_embeds)  -- quantizers observe Talker
                                                activations

    Both submodels' amax statistics come from the SAME realistic multimodal
    distribution.  See the module docstring for the INT4 AWQ modelopt
    workarounds.

    ``cp_quantization="fp8"`` additionally quantizes the Talker
    CodePredictor: the quant config opts CP in (see
    :func:`_build_full_model_quant_cfg`) and the forward loop appends a
    dedicated Thinker → Talker → CP generation drive
    (:func:`qwen3_cp_calibration_loop`) so CP quantizers see real
    activations — the joint pass alone never reaches CP.

    Output is a single HF root under ``output_dir`` (same layout as the
    source ``model_dir``), consumable by ``tensorrt-edgellm-export`` in
    one shot.  ``num_samples`` / ``max_length`` are accepted for CLI parity
    but unused (joint multimodal calibration supersedes text-only);
    ``text_dataset`` (registered name, generator, or ``None`` for the
    default) feeds the joint calibration's text portion and the CP drive.
    Audio / image calibration data use a fixed recipe (LibriSpeech + MMMU).
    Returns ``output_dir``.
    """
    del num_samples, max_length  # Joint multimodal calib supersedes
    if (quantization is None and visual_quantization is None
            and audio_quantization is None and cp_quantization is None):
        raise ValueError(
            "Nothing to quantize: pass --quantization and/or one of "
            "--visual_quantization / --audio_quantization / "
            "--cp_quantization.")
    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    # 1. Detect model variant + load. Two Qwen3-Omni variants share this
    #    driver:
    #      - model_type "qwen3_omni_moe" -> Qwen3OmniMoeForConditionalGeneration
    #        (in released transformers).
    #      - model_type "qwen3_omni"     -> Qwen3OmniForConditionalGeneration
    #        (requires a transformers build that carries the non-MoE class;
    #        not present in released transformers tags).
    hf_cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(hf_cfg, "model_type", None)
    is_moe = (model_type == "qwen3_omni_moe")
    print(f"[load] {model_dir} (model_type={model_type}, is_moe={is_moe})")
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_moe:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        _model_cls = Qwen3OmniMoeForConditionalGeneration
    else:
        try:
            from transformers import Qwen3OmniForConditionalGeneration
        except ImportError as error:
            raise ImportError(
                f"model_type={model_type!r} requires a transformers build "
                "that exposes Qwen3OmniForConditionalGeneration; the "
                "released transformers tags only carry the MoE variant."
            ) from error
        _model_cls = Qwen3OmniForConditionalGeneration
    model = _model_cls.from_pretrained(
        model_dir, dtype=torch_dtype,
        trust_remote_code=True).to(device).eval()
    if talker_accept_hidden_layer is None or talker_accept_hidden_layer < 0:
        talker_accept_hidden_layer = int(
            getattr(model.config.talker_config, "accept_hidden_layer"))
    print(f"[calib] accept_hidden_layer={talker_accept_hidden_layer}")

    # 2. Fused-Parameter MoE experts need no pre-quantization patching.
    #    On transformers >= 5.0 the Thinker/Talker experts are stored fused
    #    (``gate_up_proj`` / ``down_proj`` 3-D ``nn.Parameter``); ModelOpt 0.44
    #    quantizes and exports them natively via its on-the-fly fused-expert
    #    plugins, emitting standard per-expert NVFP4 keys
    #    (``experts.{j}.{gate,up,down}_proj.{weight,weight_scale,...}``).

    # 3. Joint multimodal calibration in a single ``mtq.quantize`` call on
    #    the full model.  The forward loop runs Thinker → projection →
    #    Talker per sample; during calibration all quantizers are in
    #    observe-only mode, so the Thinker output is genuine FP16 and the
    #    Talker sees the same input distribution it will encounter at
    #    inference time.
    if not is_quantized(model.thinker) or not is_quantized(model.talker):
        # The joint multimodal pass only pays off when quantizers outside
        # CP are enabled; a CP-only run needs just the dedicated CP drive.
        needs_multimodal_calib = (quantization is not None
                                  or visual_quantization is not None
                                  or audio_quantization is not None)
        calib_ds = None
        if needs_multimodal_calib:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_dir,
                                                      trust_remote_code=True)
            calib_ds = _build_multimodal_calib_dataset(
                processor,
                num_audio=talker_num_audio,
                num_image=talker_num_image,
                num_text=talker_num_text,
                text_ds=text_dataset)
            print(f"[calib] multimodal dataset: {len(calib_ds)} samples "
                  f"(joint thinker + talker calibration)")

        cfg = _build_full_model_quant_cfg(
            quantization,
            lm_head_quantization,
            kv_cache_quantization,
            is_moe=is_moe,
            visual_quantization=visual_quantization,
            audio_quantization=audio_quantization,
            cp_quantization=cp_quantization)

        def _forward_loop(m):
            if calib_ds is not None:
                _calib_full_multimodal(model, calib_ds,
                                       talker_accept_hidden_layer)
            if cp_quantization is not None:
                from .datasets import resolve_dataset
                from .quantize import _text_calib_dataloader
                from .qwen3_cp_loader import qwen3_cp_calibration_loop
                cp_loader = _text_calib_dataloader(tokenizer,
                                                   resolve_dataset(
                                                       text_dataset, "text"),
                                                   batch_size=1,
                                                   num_samples=64)
                qwen3_cp_calibration_loop(model, cp_loader, num_cp_samples=64)

        with _maybe_int4_awq_wars(quantization):
            mtq.quantize(model, cfg, forward_loop=_forward_loop)
            mtq.print_quant_summary(model.thinker)
            mtq.print_quant_summary(model.talker)
    print(f"[quant] {time.time() - t0:.1f}s")

    # 3c. Persist modelopt state for PyTorch QDQ verification.
    os.makedirs(output_dir, exist_ok=True)
    mto_state_path = os.path.join(output_dir, "modelopt_state.pt")
    mto.save(model, mto_state_path)
    print(f"[mto] saved modelopt state to {mto_state_path}")

    # 4. Full-model HF-root export.
    with _maybe_int4_awq_wars(quantization), _omni_export_wars(model):
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=output_dir)

    # Copy tokenizer + preprocessor + chat template to make ``output_dir``
    # a self-contained HF root (same convention as `quantize_and_export`).
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as error:  # non-fatal: raw files are copied below
        warnings.warn(f"AutoProcessor save failed ({error}); relying on raw "
                      "processor-file copies.")
    for fname in ("preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json", "chat_template.json",
                  "chat_template.jinja"):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"[done] {output_dir}  (total {time.time() - t0:.1f}s)")
    return output_dir


# ---------------------------------------------------------------------------
# Full-model export workarounds
# ---------------------------------------------------------------------------


@contextmanager
def _omni_export_wars(model):
    """Scoped patches for ``export_hf_checkpoint`` on a full Qwen3-Omni model.

    - Strip ``codec_head`` from Talker ``_tied_weights_keys`` (declared but
      not actually tied at runtime; leaving it in drops ``codec_head.weight``).
    - Skip ``requantize_resmooth_fused_llm_layers`` (its dummy forward calls
      ``model(input_ids)`` but Qwen3-Omni's top-level class ships no forward).
    - Teach ``get_expert_linear_names`` about Qwen3-Omni MoE expert MLPs
      (substring match misses ``Qwen3OmniMoe(Thinker|Talker)TextSparseMoeBlock``).
    """
    saved_tied_keys = None
    _tied_was_patched = False
    if hasattr(model, "talker"):
        orig_tied = getattr(model.talker, "_tied_weights_keys", None)
        if isinstance(orig_tied, dict):
            saved_tied_keys = orig_tied
            model.talker._tied_weights_keys = {
                k: v
                for k, v in orig_tied.items() if "codec_head" not in k
            }
            _tied_was_patched = True
        elif isinstance(orig_tied, (list, tuple)):
            saved_tied_keys = orig_tied
            model.talker._tied_weights_keys = type(orig_tied)(
                k for k in orig_tied if "codec_head" not in str(k))
            _tied_was_patched = True

    import modelopt.torch.export.unified_export_hf as _ueh
    _orig_resmooth = _ueh.requantize_resmooth_fused_llm_layers
    _ueh.requantize_resmooth_fused_llm_layers = lambda m: None

    _orig_gel = _ueh.get_expert_linear_names

    def _patched_gel(module):
        cls_name = type(module).__name__
        if "Qwen3OmniMoe" in cls_name and ("Thinker" in cls_name
                                           or "Talker" in cls_name):
            return ["gate_proj", "down_proj", "up_proj"]
        return _orig_gel(module)

    _ueh.get_expert_linear_names = _patched_gel

    try:
        yield
    finally:
        _ueh.get_expert_linear_names = _orig_gel
        _ueh.requantize_resmooth_fused_llm_layers = _orig_resmooth
        if _tied_was_patched and hasattr(model, "talker"):
            model.talker._tied_weights_keys = saved_tied_keys


# ===========================================================================
#     Qwen3.5-Omni (qwen3_omni_next) orchestrator — auto-dispatched from
#     ``tensorrt-edgellm-quantize llm``. Separate from the Qwen3-Omni
#     joint driver above (``quantize_qwen3_omni``).
# ===========================================================================

# ---------------------------------------------------------------------------
# transformers 4.57.0.dev0 workarounds for Qwen3-Next Omni
# ---------------------------------------------------------------------------


def _patch_qwen3_omni_next_transformers() -> None:
    """Patch transformers in-place so the Qwen3-Next Omni HF checkpoint loads.

    Three latent bugs in transformers 4.57.0.dev0's ``qwen3_omni_next``
    module surface only on the Qwen3-Next Omni checkpoint:

    1. ``Qwen3OmniNextCode2WavConfig`` never sets ``rope_scaling`` but the
       shared ``Qwen3OmniNextRotaryEmbedding.__init__`` reads it unguarded,
       so Code2Wav init raises ``AttributeError``.
    2. ``Qwen3OmniNextTalkerDecoderLayer`` wires ``self_attn`` to the
       non-gated ``Qwen3OmniNextThinkerTextAttention`` but the checkpoint
       stores a gated ``q_proj`` (2× output).  Swap to the gated
       ``Qwen3OmniNextAttention`` whose forward splits q into query+gate.
    3. ``Qwen3OmniNextTalkerCodePredictorAttention.q_proj`` is non-gated
       in source but gated in checkpoint.  CodePredictor is excluded from
       quantization and never forward-called during calibration, so
       resizing q_proj to the 2× shape is enough for ``load_state_dict``.

    The patches are guarded with sentinel attributes so multiple calls
    are safe.
    """
    import torch.nn as _nn
    from transformers.models.qwen3_omni_next import \
        modeling_qwen3_omni_next as _mod
    from transformers.models.qwen3_omni_next.configuration_qwen3_omni_next import \
        Qwen3OmniNextCode2WavConfig

    if not hasattr(Qwen3OmniNextCode2WavConfig, "rope_scaling"):
        Qwen3OmniNextCode2WavConfig.rope_scaling = {}

    if not getattr(_mod.Qwen3OmniNextTalkerDecoderLayer,
                   "_edgellm_gated_patched", False):
        _orig_tdl_init = _mod.Qwen3OmniNextTalkerDecoderLayer.__init__

        def _tdl_init_gated(self, config, layer_idx):
            _orig_tdl_init(self, config, layer_idx)
            self.self_attn = _mod.Qwen3OmniNextAttention(config, layer_idx)

        _mod.Qwen3OmniNextTalkerDecoderLayer.__init__ = _tdl_init_gated
        _mod.Qwen3OmniNextTalkerDecoderLayer._edgellm_gated_patched = True

    if not getattr(_mod.Qwen3OmniNextTalkerCodePredictorAttention,
                   "_edgellm_gated_q_patched", False):
        _orig_cp_init = \
            _mod.Qwen3OmniNextTalkerCodePredictorAttention.__init__

        def _cp_init_gated(self, config, layer_idx):
            _orig_cp_init(self, config, layer_idx)
            self.q_proj = _nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim * 2,
                bias=config.attention_bias,
            )

        _mod.Qwen3OmniNextTalkerCodePredictorAttention.__init__ = \
            _cp_init_gated
        _mod.Qwen3OmniNextTalkerCodePredictorAttention._edgellm_gated_q_patched = True


# ---------------------------------------------------------------------------
# Model detection + loader
# ---------------------------------------------------------------------------

OMNI_MODEL_TYPES = frozenset(["qwen3_omni", "qwen3_omni_next"])


def is_omni_model_dir(model_dir: str) -> bool:
    """Return True iff ``config.json`` declares a Qwen3-Omni variant."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return cfg.get("model_type") in OMNI_MODEL_TYPES


def _read_model_type(model_dir: str) -> str:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f).get("model_type", "")


def _load_omni_model(model_dir: str, dtype: str, device: str):
    """Instantiate the right ``ForConditionalGeneration`` class + processor."""
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    model_type = _read_model_type(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
    except Exception as error:
        print(f"Warning: AutoProcessor failed ({error}); "
              "multimodal calibration will fall back to text-only.")
        processor = None

    if model_type == "qwen3_omni_next":
        _patch_qwen3_omni_next_transformers()
        from transformers import Qwen3OmniNextForConditionalGeneration
        model = Qwen3OmniNextForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)
    else:
        from transformers import Qwen3OmniForConditionalGeneration
        model = Qwen3OmniForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)

    # Qwen3-Omni / Qwen3-Next Omni ForConditionalGeneration classes are
    # generation-only — they don't define forward(). ModelOpt's
    # ``export_hf_checkpoint`` calls ``model(fake_input)`` though, so add
    # a forward that delegates to the Thinker (which covers every layer
    # we actually quantize).
    type(model).forward = lambda self, *args, **kwargs: self.thinker(
        *args, **kwargs)

    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor


# ---------------------------------------------------------------------------
# Calibration loop — works for both Qwen3-Omni and Qwen3-Next Omni. Respects
# the ``has_talker`` toggle and Qwen3-Next Omni's standalone text-embedding
# path (``get_input_text_embeddings()`` vs. ``text_projection``).
# ---------------------------------------------------------------------------


def _omni_calib_loop(model,
                     calib_dataset: "_OmniMultimodalCalibDataset",
                     accept_hidden_layer: int = 14) -> None:
    """Forward each sample through Thinker (and Talker when present).

    For each sample the Thinker is run with ``output_hidden_states=True``.
    The hidden states at ``accept_hidden_layer`` are projected into the
    Talker space (Qwen3-Omni uses ``text_projection``; Qwen3-Next Omni uses
    a standalone ``get_input_text_embeddings()``) so the Talker's first
    decoder layer sees realistic ``inputs_embeds``.
    """
    device = next(model.parameters()).device
    has_talker = bool(getattr(model, "has_talker", False))

    skipped = 0
    for i in tqdm(range(len(calib_dataset)),
                  desc="Calibrating Omni (multimodal)"):
        try:
            data = calib_dataset[i]
        except Exception as error:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"Skipping sample {i}: {error}")
            continue
        data = {
            k:
            v.unsqueeze(0).to(
                device,
                dtype=model.thinker.dtype if v.is_floating_point() else None)
            for k, v in data.items()
        }

        thinker_out = model.thinker(**data, output_hidden_states=True)
        if not has_talker:
            continue

        all_hidden = thinker_out.hidden_states
        if all_hidden is None or len(all_hidden) <= accept_hidden_layer:
            continue
        thinker_hidden = all_hidden[accept_hidden_layer]
        thinker_embed = all_hidden[0]

        input_ids = data.get("input_ids")
        audio_tok = getattr(model.config.thinker_config, "audio_token_id", -1)
        image_tok = getattr(model.config.thinker_config, "image_token_id", -1)
        mm_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if audio_tok >= 0:
            mm_mask |= (input_ids == audio_tok)
        if image_tok >= 0:
            mm_mask |= (input_ids == image_tok)

        seq_len = thinker_hidden.shape[1]
        talker_dim = model.talker.config.text_config.hidden_size
        inputs_embeds = torch.empty(1,
                                    seq_len,
                                    talker_dim,
                                    dtype=model.talker.dtype,
                                    device=device)

        if mm_mask.any():
            inputs_embeds[mm_mask] = model.talker.hidden_projection(
                thinker_hidden[mm_mask])
        text_mask = ~mm_mask
        if text_mask.any():
            # Qwen3-Omni: ``talker.text_projection(thinker_embed)``.
            # Qwen3-Next Omni: standalone ``get_input_text_embeddings()``
            # (text_projection does not exist on the Qwen3.5 talker).
            if hasattr(model.talker, "text_projection"):
                inputs_embeds[text_mask] = model.talker.text_projection(
                    thinker_embed[text_mask])
            else:
                inputs_embeds[
                    text_mask] = model.talker.get_input_text_embeddings()(
                        input_ids[text_mask]).to(inputs_embeds.dtype)

        tc = model.talker.config.text_config
        talker_ids = torch.randint(0,
                                   tc.vocab_size, (1, seq_len),
                                   device=device)
        attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

        model.talker(inputs_embeds=inputs_embeds,
                     attention_mask=attn_mask,
                     talker_input_ids=talker_ids)

    if skipped:
        print(
            f"Omni calibration: skipped {skipped}/{len(calib_dataset)} samples"
        )


# ---------------------------------------------------------------------------
# Post-calibration ``_amax`` backfill
# ---------------------------------------------------------------------------


def _backfill_missing_amax(model) -> int:
    """Set ``_amax`` for any quantizer that calibration did not populate.

    Required because ``modelopt`` exports NVFP4's per-tensor ``weight_scale_2``
    from ``weight_quantizer._amax``; if a Linear's forward never updated its
    quantizer (observed for Talker layers post the gated-attention swap),
    ``export_hf_checkpoint`` aborts with::

        AssertionError: Weight quantizer does not have attribute amax

    For weight quantizers we use ``weight.abs().max()`` (the static fallback
    a ``MaxCalibrator`` would have produced).  For input quantizers without
    observed activations we use the same value: it makes the per-tensor
    scale conservative but quantization still works at runtime.
    """
    n_filled = 0
    for module in model.modules():
        wq = getattr(module, "weight_quantizer", None)
        iq = getattr(module, "input_quantizer", None)
        weight = getattr(module, "weight", None)
        # We're looking specifically for quantized linears, which are the
        # only modules that have all three (weight + weight_quantizer +
        # input_quantizer).  Skip everything else.
        if wq is None or iq is None or weight is None:
            continue
        for q in (wq, iq):
            if getattr(q, "_disabled", False):
                continue
            if not hasattr(q, "_amax"):
                fallback = weight.detach().abs().max().to(weight.dtype)
                q.register_buffer("_amax", fallback.clone())
                n_filled += 1
    return n_filled


# ---------------------------------------------------------------------------
# Top-level entry point (called from ``tensorrt-edgellm-quantize llm``)
# ---------------------------------------------------------------------------


def quantize_and_export_omni(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    audio_quantization: Optional[str] = None,
    cp_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    audio_dataset: str = "openslr/librispeech_asr",
    visual_dataset: str = "lmms-lab/MMMU",
    text_dataset: str = "cnn_dailymail",
    num_samples: int = 500,
) -> str:
    """Load a Qwen3-Next Omni model, quantize it, and export.

    Shares the ``tensorrt-edgellm-quantize llm`` flag surface with the
    Qwen3-Omni driver; sub-encoder quantization flags are rejected until
    validated on this family. ``num_samples`` is split roughly evenly
    across the three calibration modalities.
    """
    for flag, val in (("visual_quantization", visual_quantization),
                      ("audio_quantization", audio_quantization),
                      ("cp_quantization", cp_quantization)):
        if val is not None:
            raise ValueError(
                f"--{flag} is not supported for qwen3_omni_next yet.")

    t0 = time.time()
    model, tokenizer, processor = _load_omni_model(model_dir, dtype, device)

    if is_quantized(model):
        print("Model already quantized — skipping.")
    else:
        if processor is None:
            raise RuntimeError(
                "Omni multimodal calibration requires a processor; "
                "AutoProcessor.from_pretrained failed.")
        accept_layer = getattr(getattr(model.config, "talker_config", None),
                               "accept_hidden_layer", 14)
        third = max(1, num_samples // 3)
        calib_dataset = _build_multimodal_calib_dataset(
            processor,
            num_audio=third,
            num_image=third,
            num_text=num_samples - 2 * third,
            audio_dataset_dir=audio_dataset,
            visual_dataset_dir=visual_dataset,
            text_ds=text_dataset,
        )
        has_talker = bool(getattr(model, "has_talker", False))
        print(f"Omni multimodal calibration: {len(calib_dataset)} samples "
              f"(accept_hidden_layer={accept_layer}, "
              f"talker={'yes' if has_talker else 'no'})")

        quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                       kv_cache_quantization)
        mtq.quantize(
            model,
            quant_cfg,
            forward_loop=lambda m: _omni_calib_loop(m, calib_dataset,
                                                    accept_layer),
        )
        mtq.print_quant_summary(model)

    print(f"Quantization: {time.time() - t0:.1f}s")

    n_filled = _backfill_missing_amax(model)
    if n_filled:
        print(f"Backfilled {n_filled} missing _amax buffers from weights "
              "(Talker layers calibration did not populate)")

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    print(f"Saved to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
