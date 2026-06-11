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
Qwen3-Omni-30B-A3B Talker MoE backbone.

The Talker text decoder is architecturally the same Qwen3 MoE stack as the
Thinker, but HF exposes two different names around the vocabulary:

* ``model.codec_embedding.weight`` is the input embedding table.
* ``codec_head.weight`` is the output projection.

These two tensors are not tied in the Qwen3-Omni-30B-A3B checkpoint.  This
subclass only aliases the HF names onto the generic Qwen3 MoE implementation
so the shared loader can populate both tensors without a model-wide key remap.
"""

from ..linear import make_linear
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeCausalLM

__all__ = ["Qwen3OmniMoeTalkerCausalLM"]


class Qwen3OmniMoeTalkerCausalLM(Qwen3MoeCausalLM):
    """Qwen3-Omni MoE Talker: codec-token LM with codec head output.

    Also emits ``hidden_states`` from the **post-final-norm** Talker output
    (= ``model.norm(last_layer_output)``) so the C++ TTS runtime can hand
    per-frame Talker hidden states off to the CodePredictor engine.

    This matches HF's ``Qwen3OmniMoeTalkerForConditionalGeneration``
    past_hidden semantics (``modeling_qwen3_omni_moe.py:3176``):
    ``past_hidden = hidden_states[0][-1][:, -1:]`` where ``hidden_states[0]``
    is the per-layer hidden-states tuple and ``[-1]`` is the post-norm output
    in the transformers version we target.

    Talker does not set ``config.accept_hidden_layer`` (mid-layer hook is
    Thinker-only), so the base ``Qwen3MoeCausalLM`` defaults to post-norm —
    the correct selection for the Talker → CodePredictor path.
    """

    emit_hidden_states = True

    def __init__(self, config) -> None:
        # ``accept_hidden_layer`` is a Thinker-side config (mid-layer hook for
        # the text→speech path). Talker checkpoints may inherit this field as
        # a remnant of the parent config dict (observed value 24 in some
        # checkpoints, but the Talker only has 20 decoder layers). Force -1
        # so :class:`Qwen3MoeTransformer` emits the post-final-norm output
        # used by the CodePredictor (matches HF semantics).
        try:
            config.accept_hidden_layer = -1
        except (AttributeError, TypeError):
            pass  # frozen config: ``Qwen3MoeTransformer.__init__`` getattr
            # will still default to -1 if the attribute is missing.
        super().__init__(config)
        # HF Talker names the input embedding ``codec_embedding``.
        self.model.codec_embedding = self.model.embed_tokens

        # HF Talker names the output projection ``codec_head``.  It is a
        # separate tensor from ``model.codec_embedding.weight``.  Recreate the
        # projection with the HF module name so quant exclusions for
        # ``codec_head`` select the correct linear class.
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="codec_head")
        self.codec_head = self.lm_head
