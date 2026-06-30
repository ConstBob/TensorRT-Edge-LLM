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

:data:`MODELOPT_KEY_REMAP` adapts ModelOpt's per-expert checkpoint keys to the
internal ``experts._experts.{j}.`` layout used by :class:`Qwen3MoEExperts`.
ModelOpt's fused-expert export emits a single
``mlp.experts.{j}.{gate,up,down}_proj.*`` level (bare Qwen3-MoE / Qwen3-30B-A3B
and Qwen3-Omni-30B-A3B Thinker / Talker quantized via
``tensorrt_edgellm.quantization.qwen3_omni``); this remap inserts one
``._experts`` segment before the expert index.

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
#   ``model.layers.5.mlp.experts.42.gate_proj.weight_scale``
# becomes
#   ``model.layers.5.mlp.experts._experts.42.gate_proj.weight_scale``
# Anchored with ``\b`` and digits-only to avoid touching unrelated keys.
_EXPERTS_INDIRECTION = re.compile(r"(\bexperts)\.(\d+)\.")


def MODELOPT_KEY_REMAP(key: str) -> Optional[str]:
    """Transform a ModelOpt per-expert checkpoint key for this modeling tree.

    Returns ``None`` to drop a key (currently never happens).
    """
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
