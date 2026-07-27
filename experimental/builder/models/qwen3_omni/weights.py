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
"""Qwen3-Omni checkpoint weight mapping."""

import os

import numpy as np

from ...weight_packing import nvfp4 as nvfp4_pack

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual": ("thinker.visual.", "visual.", "vision_tower.", "model.visual."),
    "audio":
    ("thinker.audio_tower.", "audio_tower.", "audio.", "model.audio."),
    "talker": ("talker.", ),
    "code-predictor": ("talker.code_predictor.", "code_predictor."),
    "code2wav": ("code2wav.", "speech_tokenizer.decoder.", "decoder."),
}
_WRAPPERS = (
    "model.language_model.",
    "thinker.model.",
    "language_model.",
    "text_model.",
    "llm.",
    "thinker.",
)


def checkpoint_dir(model_dir: str, component: str) -> str:
    """Select the checkpoint subtree that owns the requested component."""
    speech_tokenizer = os.path.join(model_dir, "speech_tokenizer")
    if (component == "code2wav" and os.path.isdir(speech_tokenizer) and any(
            name.endswith(".safetensors")
            for name in os.listdir(speech_tokenizer))):
        return speech_tokenizer
    return model_dir


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3-Omni component checkpoints."""
    del spec_type, spec_role
    prefixes = _PREFIXES.get(component, ())
    candidates = [prefix + name for prefix in prefixes]
    if component == "llm" and name.startswith("model."):
        nested_name = name[len("model."):]
        candidates.extend(prefix + nested_name for prefix in prefixes)
    if component == "talker" and name == "model.embed_tokens.weight":
        candidates.extend(
            ("talker.model.codec_embedding.weight",
             "talker.codec_embedding.weight", "codec_embedding.weight"))
    if component == "talker" and name.startswith("lm_head."):
        suffix = name[len("lm_head."):]
        candidates.extend(
            (f"talker.codec_head.{suffix}", f"codec_head.{suffix}"))
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.extend(("model.embed_tokens.weight",
                           "model.language_model.embed_tokens.weight"))
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by Qwen3-Omni checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def repack_nvfp4_experts(load_expert, num_experts: int, hidden_size: int,
                         intermediate_size: int, group_size: int,
                         fc1_layout: str):
    """Pack this family's SwiGLU experts for the Edge-LLM MoE operation."""
    if fc1_layout not in ("interleave", "concat"):
        raise ValueError(f"unsupported FC1 layout {fc1_layout!r}")

    def build_fc1(gate, up):
        if fc1_layout == "concat":
            return np.concatenate([up, gate],
                                  axis=0).reshape(2 * intermediate_size,
                                                  hidden_size)
        rows = 64
        if intermediate_size % rows:
            raise ValueError("moe_intermediate_size must be a multiple of 64")
        chunks = intermediate_size // rows
        up_chunks = up.reshape(chunks, rows, hidden_size)
        gate_chunks = gate.reshape(chunks, rows, hidden_size)
        return np.stack([up_chunks, gate_chunks],
                        axis=1).reshape(2 * intermediate_size, hidden_size)

    fc1_weights, fc1_scales = [], []
    fc2_weights, fc2_scales = [], []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        fc1 = build_fc1(expert["gate"], expert["up"])
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(fc1, group_size)
        fc1_weights.append(weight)
        fc1_scales.append(scale)
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(
            expert["down"], group_size)
        fc2_weights.append(weight)
        fc2_scales.append(scale)

    ones = np.ones(num_experts, dtype=np.float32)
    return (np.stack(fc1_weights), np.stack(fc1_scales), ones,
            np.stack(fc2_weights), np.stack(fc2_scales), ones.copy())
