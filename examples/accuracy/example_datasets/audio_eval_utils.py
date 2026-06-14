# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared audio helpers for accuracy benchmarks."""

import os

OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech.")


def save_audio_array_to_wav(audio,
                            output_path: str,
                            sample_rate: int = 16000) -> str:
    """Write a mono waveform array to ``output_path`` as a 16-bit PCM WAV.

    The C++ runtime's ``audioLoader`` + ``MelExtractor`` consumes the WAV
    directly via the CLI's raw-audio path; mel extraction happens in C++.
    """
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return output_path
