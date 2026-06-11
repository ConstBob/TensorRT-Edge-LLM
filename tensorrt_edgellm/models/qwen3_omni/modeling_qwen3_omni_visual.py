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
"""Qwen3-Omni / Qwen3-Omni-MoE visual encoder adapter.

The Qwen3-Omni visual encoder is computationally identical to Qwen3-VL
(same ViT blocks, same patch merger structure, same deepstack mergers).
We reuse :class:`Qwen3VLVisualModel` for the computation graph, but the
HF Qwen3-Omni checkpoint uses different parameter naming than HF
Qwen3-VL, so this adapter translates the checkpoint keys before
delegating to :func:`build_qwen3_vl_visual`.

Naming differences (HF Qwen3-Omni -> HF Qwen3-VL):

* Top-level prefix ``thinker.visual.*`` -> ``model.visual.*``  (Qwen3-Omni
  nests the LLM under ``thinker``; Qwen3-VL keeps it flat under ``model``).
* ``merger.ln_q`` / ``merger_list.{i}.ln_q`` -> ``merger.norm`` /
  ``deepstack_merger_list.{i}.norm``.
* ``merger.mlp.0`` / ``merger_list.{i}.mlp.0`` -> ``merger.linear_fc1`` /
  ``deepstack_merger_list.{i}.linear_fc1`` (the first Linear of the MLP).
* ``merger.mlp.2`` / ``merger_list.{i}.mlp.2`` -> ``merger.linear_fc2`` /
  ``deepstack_merger_list.{i}.linear_fc2`` (the second Linear; index 1
  is a parameter-less ``GELU`` so no checkpoint entry).
* ``merger_list`` -> ``deepstack_merger_list`` (HF Qwen3-Omni names the
  ``ModuleList`` of deepstack patch mergers without the ``deepstack_``
  prefix; HF Qwen3-VL adds the prefix).

Without this translation, ``load_submodule_weights`` would silently skip
every checkpoint key (the ``_remap`` callable in ``build_qwen3_vl_visual``
returns ``None`` for keys not matching ``model.visual.*``, which does
*not* populate the missing-keys list).  The visual encoder would then
export to ONNX with random-initialised weights; image features would
poison the LLM hidden states and downstream OmniBench output would be
empty / nonsensical.
"""
import re
from typing import TYPE_CHECKING

from ..qwen3_vl.modeling_qwen3_vl_visual import \
    Qwen3VLVisualModel as Qwen3OmniVisualModel
from ..qwen3_vl.modeling_qwen3_vl_visual import build_qwen3_vl_visual

if TYPE_CHECKING:
    import torch

    from ...config import ModelConfig

__all__ = ["Qwen3OmniVisualModel", "build_qwen3_omni_visual"]

# Prefix translation: HF Qwen3-Omni -> canonical Qwen3-VL.
_HF_OMNI_VISUAL_PREFIX = "thinker.visual."
_TARGET_VISUAL_PREFIX = "model.visual."

# Sub-module suffix translations.  Replacements are disjoint, but we keep
# ln_q first for readability.
_SUBMODULE_RENAMES = (
    (".ln_q.", ".norm."),
    (".mlp.0.", ".linear_fc1."),
    (".mlp.2.", ".linear_fc2."),
)

# Compiled once.  Matches ``model.visual.merger_list.`` exactly so we
# don't accidentally rewrite an unrelated future submodule named
# ``merger_list``.
_MERGER_LIST_RE = re.compile(r"\b" + re.escape(_TARGET_VISUAL_PREFIX) +
                             r"merger_list\.")
_DEEPSTACK_MERGER_LIST = _TARGET_VISUAL_PREFIX + "deepstack_merger_list."


def _translate_qwen3_omni_visual_key(key: str) -> str:
    """Return *key* translated from HF Qwen3-Omni to Qwen3-VL naming.

    Non-visual keys (``thinker.audio_tower.*``, ``thinker.model.*``,
    ``talker.*``, ...) are returned unchanged so downstream filters
    (e.g. ``build_qwen3_vl_visual``'s ``model.visual.`` prefix filter)
    drop them naturally.
    """
    if not key.startswith(_HF_OMNI_VISUAL_PREFIX):
        return key
    out = _TARGET_VISUAL_PREFIX + key[len(_HF_OMNI_VISUAL_PREFIX):]
    out = _MERGER_LIST_RE.sub(_DEEPSTACK_MERGER_LIST, out)
    for old, new in _SUBMODULE_RENAMES:
        out = out.replace(old, new)
    return out


def _validate_visual_coverage(model,
                              remapped_weights: dict,
                              prefix: str = "model.visual") -> None:
    """Assert that every model parameter is populated by *remapped_weights*
    and that every visual-prefixed key in *remapped_weights* matches a
    model parameter.

    Without this check, ``load_submodule_weights`` warns on extra-source
    keys but silently leaves *model-side* parameters uninitialised
    (zero-init from ``nn.Parameter(torch.empty(...))``).  An uninitialised
    visual encoder exports to ONNX without raising — the bug only surfaces
    at inference time as garbled image features.
    """
    expected = set(model.state_dict().keys())
    strip = prefix + "." if prefix else ""
    provided = {
        k[len(strip):] if strip else k
        for k in remapped_weights.keys() if not strip or k.startswith(strip)
    }
    missing = sorted(expected - provided)
    extra_visual = sorted(provided - expected)
    if missing or extra_visual:

        def _fmt(items):
            head = items[:10]
            return f"{head}{' ...' if len(items) > 10 else ''}"

        raise RuntimeError(
            "Qwen3-Omni visual encoder weight coverage mismatch.\n"
            f"  Missing in checkpoint ({len(missing)}): {_fmt(missing)}\n"
            f"  Extra under '{prefix}.' ({len(extra_visual)}): "
            f"{_fmt(extra_visual)}\n"
            "Check the HF-Omni -> Qwen3-VL key translation in "
            "_translate_qwen3_omni_visual_key().")


def build_qwen3_omni_visual(config: dict, weights: dict,
                            model_config: "ModelConfig", dtype: "torch.dtype"):
    """Build a Qwen3-Omni visual encoder.

    Translates HF Qwen3-Omni checkpoint keys to the Qwen3-VL naming
    expected by :func:`build_qwen3_vl_visual`, then delegates.  After
    construction, validates that every model parameter was covered by the
    remapped checkpoint (see :func:`_validate_visual_coverage`).

    Used for both ``qwen3_omni`` (dense) and ``qwen3_omni_moe`` model
    types; the visual encoder is byte-for-byte identical in HF between
    the two variants (one-line diff outside the forward pass).
    """
    remapped = {
        _translate_qwen3_omni_visual_key(k): v
        for k, v in weights.items()
    }
    # HF Qwen3-Omni-MoE 30B-A3B-Instruct's vision_config omits the explicit
    # ``num_position_embeddings`` field that the (later) Qwen3-VL builder
    # demands, but it does ship ``image_size`` + ``patch_size`` from which
    # the field is unambiguously derivable.  Inject the computed value so
    # build_qwen3_vl_visual can run without forking its config schema.
    if "num_position_embeddings" not in config:
        image_size = config.get("image_size")
        patch_size = config.get("patch_size")
        if image_size is not None and patch_size:
            grid = int(image_size) // int(patch_size)
            config = {**config, "num_position_embeddings": grid * grid}
    model = build_qwen3_vl_visual(config,
                                  remapped,
                                  model_config=model_config,
                                  dtype=dtype)
    _validate_visual_coverage(model, remapped)
    return model
