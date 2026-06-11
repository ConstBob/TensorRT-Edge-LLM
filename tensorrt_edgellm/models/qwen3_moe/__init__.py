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
"""Qwen3-MoE modeling and checkpoint-key remap utilities.

:data:`MODELOPT_KEY_REMAP` adapts ModelOpt-flat per-expert checkpoint keys
to the internal ``experts._experts.{j}.`` layout used by
:class:`Qwen3MoEExperts`:

* Bare Qwen3-MoE / Qwen3-30B-A3B (``mlp.experts.{j}.{gate,up,down}_proj.*``)
  -- single ``experts.`` level; insert one ``._experts``.
* Qwen3-Omni-30B-A3B Thinker / Talker quantized via
  ``tensorrt_edgellm.quantization.qwen3_omni`` -- the
  ``_PerExpertLinears`` patch nests per-expert ``nn.Linear`` modules under
  ``experts.experts.{j}.``; collapse the double-``experts.experts.`` to
  ``experts._experts.``.

Pass into :meth:`tensorrt_edgellm.AutoModel.from_pretrained` as ``key_remap``
when loading ModelOpt NVFP4 (or any HF-flat per-expert) Qwen3-MoE
checkpoints, including Qwen3-Omni MoE Thinker / Talker text-MoE backbones.

Without this remap the loader's ``_navigate`` calls ``getattr(experts,
"experts")`` which raises ``AttributeError`` (the ``nn.ModuleList`` is
registered as ``_experts``), so every per-expert weight is silently
``skipped`` -- producing a thinker engine ~3 GB (attention + norms only,
expert weights missing) instead of the expected ~17 GB.  See
``PERF_REPORT_OMNI_A3B.md`` section "OMNI v1" for the historical incident.
"""
import re
from typing import Optional

# yapf: disable
from .modeling_qwen3_moe import (Qwen3MoeCausalLM, Qwen3MoeDecoderLayer,
                                 Qwen3MoEExperts, Qwen3MoERouter,
                                 Qwen3MoeTransformer, Qwen3SparseMoeBlock)

# yapf: enable

# Insert ``_experts.`` between ``experts.`` and the integer expert index, e.g.
#   ``model.layers.5.mlp.experts.42.gate_proj.weight_scale``           (BARE)
# becomes
#   ``model.layers.5.mlp.experts._experts.42.gate_proj.weight_scale``  (model)
# Qwen3-Omni Thinker / Talker quantization wraps per-expert linears in an
# extra ``experts.`` level (``_PerExpertLinears`` holds a child ``self.experts``
# ModuleList), so the source key for OMNI is ``mlp.experts.experts.42.gate_proj.weight``
# and must collapse the double ``experts.experts.`` to ``experts._experts.``.
# Anchored with ``\b`` and digits-only to avoid touching unrelated keys.
_EXPERTS_DOUBLE = re.compile(r"\bexperts\.experts\.(\d+)\.")
_EXPERTS_INDIRECTION = re.compile(r"(\bexperts)\.(\d+)\.")


def MODELOPT_KEY_REMAP(key: str) -> Optional[str]:
    """Transform a ModelOpt-flat checkpoint key for this modeling tree.

    Handles both single- and double- ``experts.`` checkpoint key conventions.
    Returns ``None`` to drop a key (currently never happens).
    """
    if "experts.experts." in key:
        return _EXPERTS_DOUBLE.sub(r"experts._experts.\1.", key)
    return _EXPERTS_INDIRECTION.sub(r"\1._experts.\2.", key)


__all__ = [
    "Qwen3MoeCausalLM",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoEExperts",
    "Qwen3MoERouter",
    "Qwen3MoeTransformer",
    "Qwen3SparseMoeBlock",
    "MODELOPT_KEY_REMAP",
]
