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
"""Qwen3-TTS checkpoint and component configuration."""

import json
import os

from ...core import contracts


def prepare_root(model_dir: str, root: dict) -> dict:
    root = dict(root)
    speech_config = os.path.join(model_dir, "speech_tokenizer", "config.json")
    if os.path.isfile(speech_config):
        with open(speech_config) as config_file:
            root["_speech_tokenizer_config"] = json.load(config_file)
    return root


def component_config(root: dict, component: contracts.Component) -> dict:
    root_type = str(root.get("model_type", ""))
    if component == contracts.Component.TALKER:
        talker = root.get("talker_config")
        if isinstance(talker, dict):
            return talker
        if root_type in ("qwen3_tts", "qwen3_tts_talker"):
            return root
        raise ValueError("Qwen3-TTS checkpoint has no talker_config")
    if component == contracts.Component.CODE_PREDICTOR:
        talker = root.get("talker_config") or root
        predictor = talker.get("code_predictor_config")
        if isinstance(predictor, dict):
            return predictor
        if root_type == "qwen3_tts_code_predictor":
            return root
        raise ValueError("Qwen3-TTS checkpoint has no code_predictor_config")
    if component == contracts.Component.CODE2WAV:
        talker = root.get("talker_config") or {}
        speech = root.get("_speech_tokenizer_config") or {}
        return (root.get("code2wav_config") or talker.get("code2wav_config")
                or speech.get("decoder_config") or root)
    raise ValueError(f"Qwen3-TTS has no {component.value} configuration")
