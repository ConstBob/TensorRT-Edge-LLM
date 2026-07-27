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
"""Phi-4-MM visual embedding artifacts."""

import os

import numpy as np

from ...core import contracts
from ...core.artifacts.tensors import save_safetensors
from ...core.weights import Weights
from . import weights as weight_conversion


def write_component_embeddings(bundle, component: contracts.Component, args,
                               output_dir: str) -> None:
    """Write Phi-4-MM projected image separator embeddings."""
    if component != contracts.Component.VISUAL:
        return
    weights = Weights(args.model_dir,
                      component=contracts.Component.VISUAL.value,
                      conversion=weight_conversion)
    try:
        projection_key = weights.find_suffix("img_projection.0.weight")
        projection_root = projection_key[:-len("0.weight")]
        first_weight, first_bias = weights.linear_fp16(projection_root + "0")
        second_weight, second_bias = weights.linear_fp16(projection_root + "2")

        def project(key: str) -> np.ndarray:
            value = weights.f16(weights.find_suffix(key)).reshape(1, -1)
            value = value.astype(np.float32) @ first_weight.astype(
                np.float32).T
            if first_bias is not None:
                value += first_bias.astype(np.float32)
            value = value.astype(np.float16).astype(np.float32)
            value = (0.5 * value * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) *
                (value + 0.044715 * value * value * value)))).astype(
                    np.float16)
            value = value @ second_weight.astype(np.float32).T
            if second_bias is not None:
                value += second_bias.astype(np.float32)
            return np.ascontiguousarray(value.reshape(-1), dtype=np.float16)

        save_safetensors(
            os.path.join(output_dir, "phi4mm_gn_proj.safetensors"), {
                "glb_GN": project("image_embed.glb_GN"),
                "sub_GN": project("image_embed.sub_GN"),
            })
    finally:
        weights.close()
    _ = bundle
