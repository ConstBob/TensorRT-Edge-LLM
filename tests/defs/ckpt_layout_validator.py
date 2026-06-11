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
"""Post-quantization layout validator.

After ``tensorrt_edgellm`` writes a quantized checkpoint to
``<onnx_dir>/<ckpt-name>``, this validator inspects the safetensors files
and asserts that every Linear-weight module's on-disk dtype/layout matches
the precision implied by the ``TestConfig`` (which mirrors the checkpoint
directory name, e.g. ``Qwen3.5-0.8B-NVFP4-LMNVFP4``).

This catches a producer-side class of bug where modelopt's
``exclude_modules`` glob is too broad (e.g. ``["model*"]``) and silently
leaves the LLM body unquantized — the checkpoint then "lies" about its
contents and breaks every downstream test that expects e.g. NVFP4 weights.

Failing here keeps such checkpoints from being uploaded to the hub.
"""
from __future__ import annotations

import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tensorrt_edgellm.config import _VL_LLM_PREFIXES
from tensorrt_edgellm.quantization.quantization_configs import (
    _AUDIO_PREFIXES, _VISUAL_PREFIXES)

# ---------------------------------------------------------------------------
# Safetensors header reader (no torch dependency — just parse the JSON header)
# ---------------------------------------------------------------------------


def _read_safetensors_header(fpath: Path) -> Dict[str, str]:
    """Return ``{tensor_name: dtype_string}`` from a single safetensors file."""
    out: Dict[str, str] = {}
    with open(fpath, "rb") as h:
        hdr_len = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(hdr_len))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        out[k] = v.get("dtype", "?")
    return out


def _safetensors_index(ckpt_dir: Path) -> Dict[str, str]:
    """Combine headers across all safetensors files (or sharded set)."""
    out: Dict[str, str] = {}
    idx = ckpt_dir / "model.safetensors.index.json"
    if idx.exists():
        try:
            with open(idx) as fh:
                shards = set(json.load(fh).get("weight_map", {}).values())
            for s in shards:
                fp = ckpt_dir / s
                if fp.exists():
                    out.update(_read_safetensors_header(fp))
            if out:
                return out
        except (OSError, json.JSONDecodeError):
            pass
    for f in sorted(ckpt_dir.glob("*.safetensors")):
        try:
            out.update(_read_safetensors_header(f))
        except (OSError, json.JSONDecodeError, struct.error):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-module precision classifier
# ---------------------------------------------------------------------------


def _classify_module(suffixes: Dict[str, str]) -> Optional[str]:
    """Infer the precision of a single Linear-style module from its tensor
    suffixes (``weight``, ``weight_scale``, ``weight_scale_2``, ``qweight``)."""
    if "qweight" in suffixes:
        return "int4-awq"
    if "weight" not in suffixes:
        return None
    wdt = suffixes["weight"]
    ws = suffixes.get("weight_scale")
    ws2 = "weight_scale_2" in suffixes
    if ws2 and wdt in ("U8", "I8") and ws and ws.startswith("F8"):
        return "nvfp4"
    if not ws2 and ws == "U8" and wdt.startswith("F8"):
        return "mxfp8"
    if not ws2 and ws in ("F32", "F16") and wdt.startswith("F8"):
        return "fp8"
    if not ws2 and wdt == "I8" and ws in ("F16", "F32"):
        return "int8-sq"
    if not ws2 and wdt == "U8" and ws in ("F16", "F32"):
        return "int4-awq"
    if wdt in ("F16", "BF16"):
        return "fp16"
    return f"unknown(w={wdt},ws={ws},ws2={ws2})"


# ---------------------------------------------------------------------------
# Bucket assignment (which sub-network a module belongs to)
# ---------------------------------------------------------------------------

# Reuse the producer's prefix sets so "what counts as visual / audio / body"
# is defined in exactly one place per concept across the whole codebase.
# A new vision/audio model added to ``quantization_configs`` automatically
# flows through here.
_VL_LLM_ROOTS = tuple(p.rstrip(".") for p in _VL_LLM_PREFIXES)
# Body markers — anchored substrings of a sentinel-padded module path.
# Covers standard HF LLM bodies (``model.layers``), Eagle3 / MTP drafts that
# use bare top-level namespaces (``layers``, ``mtp``, ``eagle``), and the
# Eagle3 feature combiner (``fc``). Visual / audio markers are checked
# first, so a visual encoder's own ``layers.N`` won't be mis-classified.
_BODY_MARKERS = (
    "model.layers",
    "layers",
    "model.mtp",
    "mtp",
    "eagle",
    "fc",
)


def _bucket(mod: str) -> str:
    """Sort a module path into a sub-network bucket.

    Matching uses sentinel-padded membership (``f".{prefix}."`` in
    ``f".{mod}."``) so prefixes match either at the root or under any
    wrapper (``visual.X`` / ``model.visual.X`` / ``model.language_model.visual.X``)
    without false positives like ``visualizer.X`` matching ``visual``.
    """
    if mod == "lm_head" or mod.endswith(".lm_head"):
        return "lm_head"
    padded = f".{mod}."
    # Visual must come before embed: Phi-4-MM stores its image embedder under
    # ``model.embed_tokens_extend.image_embed.*``; a loose ``embed_tokens in
    # mod`` check would steal those modules into the "embed" bucket.
    if any(f".{p}." in padded for p in _VISUAL_PREFIXES):
        return "visual"
    if any(f".{p}." in padded for p in _AUDIO_PREFIXES):
        return "audio"
    if ".embed_tokens." in padded:
        return "embed"
    # Norm: HF naming (``input_layernorm`` / ``post_attention_layernorm`` /
    # ``rms_norm``) plus the bare top-level ``norm`` that Eagle3 / some
    # drafts use for the final RMSNorm.
    if (any(tag in mod for tag in ("layernorm", "rms_norm")) or mod == "norm"
            or mod.endswith(".norm")):
        return "norm"
    if any(f".{p}." in padded for p in _VL_LLM_ROOTS):
        return "body"
    if any(f".{p}." in padded for p in _BODY_MARKERS):
        return "body"
    return "other"


# ---------------------------------------------------------------------------
# Precision-token normalisation
# ---------------------------------------------------------------------------

# Map the TestConfig precision strings to the classifier's labels.
_PRECISION_ALIAS = {
    "fp16": "fp16",
    "bf16": "fp16",  # BF16 and F16 both treated as unquantized here
    "fp8": "fp8",
    "nvfp4": "nvfp4",
    "mxfp8": "mxfp8",
    "int4_awq": "int4-awq",
    "int8_sq": "int8-sq",
    "int4_gptq": "int4-awq",  # GPTQ uses qweight too — same classifier output
}


def _normalize(p: Optional[str]) -> str:
    if not p:
        return "fp16"
    return _PRECISION_ALIAS.get(p.lower(), p.lower())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_ckpt_layout(
    ckpt_dir: str,
    *,
    body_precision: str,
    lm_head_precision: Optional[str] = None,
    visual_precision: Optional[str] = None,
    audio_precision: Optional[str] = None,
    label: str = "Quantized model",
) -> None:
    """Inspect ``ckpt_dir`` and raise ``AssertionError`` on layout mismatch.

    Args:
        ckpt_dir: Path to the just-written checkpoint directory.
        body_precision: Expected precision of the LLM body (decoder layers).
        lm_head_precision: Expected precision of ``lm_head`` (defaults to fp16).
        visual_precision: Expected precision of the visual tower (default fp16).
        audio_precision: Expected precision of the audio encoder (default fp16).
        label: Prefix for error messages.
    """
    p = Path(ckpt_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {ckpt_dir}")

    tensors = _safetensors_index(p)
    if not tensors:
        raise FileNotFoundError(
            f"{label} has no safetensors files: {ckpt_dir}")

    # Group tensors by module.
    by_mod: Dict[str, Dict[str, str]] = defaultdict(dict)
    for full, dt in tensors.items():
        if "." not in full:
            continue
        mod, suf = full.rsplit(".", 1)
        by_mod[mod][suf] = dt

    # Classify each module and bucket.
    actual: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[Tuple[str, str], str] = {}
    for mod, suffixes in by_mod.items():
        cls = _classify_module(suffixes)
        if cls is None:
            continue
        b = _bucket(mod)
        actual[b][cls] += 1
        examples.setdefault((b, cls), mod)

    # Dominant precision per bucket: catches a wholesale unquantized bucket,
    # not a partial miss (a few fp16 layers among many fp8).
    def _main_kind(b: str) -> str:
        c = actual.get(b, Counter())
        return c.most_common(1)[0][0] if c else "absent"

    expectations = {
        "body": _normalize(body_precision),
        "lm_head": _normalize(lm_head_precision),
        "visual": _normalize(visual_precision),
        "audio": _normalize(audio_precision),
    }

    mismatches: List[str] = []
    for bucket, expected in expectations.items():
        got = _main_kind(bucket)
        if got == "absent":
            # An "fp16" expectation is satisfied by absence (nothing to check);
            # any other expectation requires that bucket to actually exist.
            if expected != "fp16":
                mismatches.append(
                    f"{bucket}: expected {expected} but no Linear weights "
                    f"were found in that bucket")
            continue
        if got != expected:
            sample_mod = examples.get((bucket, got), "?")
            distribution = ", ".join(f"{k}={n}"
                                     for k, n in actual[bucket].most_common())
            mismatches.append(
                f"{bucket}: expected {expected} but checkpoint actually "
                f"contains {{ {distribution} }} (example module: {sample_mod})"
            )

    if mismatches:
        # Surface the hf_quant_config.json that produced this mess — that's
        # where the wrong ``exclude_modules`` glob lives.
        hq_path = p / "hf_quant_config.json"
        hq_blob = ""
        if hq_path.exists():
            try:
                with open(hq_path) as fh:
                    hq_data = json.load(fh)
                hq_blob = "  hf_quant_config.json: " + json.dumps(
                    hq_data.get("quantization", {}), sort_keys=True)
            except (OSError, json.JSONDecodeError):
                pass
        raise AssertionError(
            f"{label} layout does not match the checkpoint name "
            f"({p.name}):\n  - " + "\n  - ".join(mismatches) +
            (f"\n{hq_blob}" if hq_blob else "") +
            "\nThe producer (modelopt PTQ) likely used an exclude_modules "
            "glob that swept the layers you expected to be quantized. "
            "Re-run the quantization recipe with a narrower exclude.")
