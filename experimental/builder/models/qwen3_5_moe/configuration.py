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
"""Qwen3.5-MoE checkpoint configuration."""

from ...core import contracts


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    raise ValueError(f"Qwen3.5-MoE has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    config.setdefault("partial_rotary_factor", 0.25)
    if not (config.get("layers_block_type") or config.get("layer_types")):
        interval = int(config.get("full_attention_interval", 4) or 4)
        config["layer_types"] = [
            "full_attention" if
            (index + 1) % interval == 0 else "linear_attention"
            for index in range(int(config["num_hidden_layers"]))
        ]
    return config
