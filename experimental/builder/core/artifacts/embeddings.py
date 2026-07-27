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
"""Embedding artifact writers."""

import os
import shutil

import numpy as np

from .. import contracts, numpy_dtypes
from ..config import DeviceConfig
from ..safetensors_np import load_safetensors_tensor
from ..weights import Weights
from .tensors import save_safetensors


def write_embedding(weights: Weights, cfg: DeviceConfig, args,
                    engine_dir: str) -> None:
    weight = weights.f16("model.embed_tokens.weight")
    weight = np.ascontiguousarray(weight * np.float16(cfg.embedding_scale))
    path = os.path.join(engine_dir, "embedding.safetensors")
    if args.fp8_embedding:
        if weight.shape[1] % 128:
            raise ValueError(
                "FP8 embedding hidden size must be divisible by 128")
        grouped = weight.astype(np.float32).reshape(weight.shape[0], -1, 128)
        scales = np.maximum(np.max(np.abs(grouped), axis=2), 1e-4) / 448.0
        normalized = grouped / scales[..., None]
        encoded = numpy_dtypes.f32_to_fp8_e4m3_bytes(normalized).reshape(
            weight.shape)
        save_safetensors(path, {
            "embedding": encoded,
            "embedding_scale": scales.astype(np.float32),
        }, {"embedding": "F8_E4M3"})
    else:
        save_safetensors(path, {"embedding": weight})


def write_ple_embedding(weights: Weights, cfg: DeviceConfig,
                        engine_dir: str) -> None:
    weight = weights.f16("model.embed_tokens_per_layer.weight")
    weight = np.ascontiguousarray(
        weight * np.float16(np.sqrt(cfg.hidden_size_per_layer_input)))
    save_safetensors(os.path.join(engine_dir, "ple_embedding.safetensors"),
                     {"weight": weight})


def copy_vocab_artifacts(args, engine_dir: str) -> None:
    source_dir = (args.draft_reduced_vocab_dir if args.resolved_spec_role
                  == contracts.SpecRole.DRAFT else args.reduced_vocab_dir)
    if not source_dir:
        return
    draft = args.resolved_spec_role == contracts.SpecRole.DRAFT
    destination_map = ("draft_vocab_map.safetensors"
                       if draft else "vocab_map.safetensors")
    source_map = os.path.join(source_dir, "vocab_map.safetensors")
    vocab_map = load_safetensors_tensor(source_map,
                                        "vocab_map").astype(np.int32)
    save_safetensors(os.path.join(engine_dir, destination_map),
                     {"vocab_map": vocab_map})
    source_info = os.path.join(source_dir, "reduced_vocab.json")
    if os.path.isfile(source_info):
        shutil.copy2(
            source_info,
            os.path.join(
                engine_dir,
                "draft_reduced_vocab.json" if draft else "reduced_vocab.json"))
