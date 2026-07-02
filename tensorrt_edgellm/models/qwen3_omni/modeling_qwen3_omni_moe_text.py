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
Qwen3-Omni-30B-A3B Thinker MoE backbone.

Mirrors :class:`Qwen3OmniLanguageModel` behavior on top of the MoE
:class:`Qwen3MoeCausalLM`.  The TTS pipeline feeds the
Thinker's full-sequence last-layer pre-norm hidden states into the
Talker's input projection, so the export enables the extra
``hidden_states`` ONNX output via ``emit_hidden_states``.
"""
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeCausalLM

__all__ = ["Qwen3OmniMoeThinkerCausalLM"]


class Qwen3OmniMoeThinkerCausalLM(Qwen3MoeCausalLM):
    """Qwen3-Omni MoE Thinker: Qwen3-MoE decoder with hidden_states output."""

    emit_hidden_states = True
