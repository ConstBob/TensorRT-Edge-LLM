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
"""Runtime files for static quantized weight inputs."""

import json
import logging
import os
from typing import Mapping, Sequence

import numpy as np

from .. import contracts
from .tensors import save_safetensors

logger = logging.getLogger(__name__)


def _filename(kind: str, role: contracts.SpecRole) -> str:
    prefix = "" if role == contracts.SpecRole.NONE else role.value + "_"
    return f"{prefix}external_{kind}_weights.safetensors"


def write_external_weight_files(
        args,
        weights_by_kind: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict]:
    """Write registered network inputs and return the runtime manifest."""
    output_dir = contracts.component_spec(args.resolved_component).output_dir(
        args.engine_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    for kind, tensors in weights_by_kind.items():
        if not tensors:
            continue
        filename = _filename(kind, args.resolved_spec_role)
        save_safetensors(os.path.join(output_dir, filename), dict(tensors))
        names = list(tensors)
        manifest.append({
            "file": filename,
            "kind": kind + "_weights",
            "tensors": names,
        })
        logger.info("Externalized %d %s tensor(s) to %s", len(names), kind,
                    filename)
    return manifest


def patch_external_weight_manifest(config_path: str,
                                   manifest: Sequence[dict]) -> None:
    """Publish external weight files in a model-owned runtime config."""
    if not manifest:
        return
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["external_weight_files"] = list(manifest)
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
