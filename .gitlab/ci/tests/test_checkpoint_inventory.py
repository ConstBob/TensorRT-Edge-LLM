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

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOLVER_PATH = REPO_ROOT / "tests" / "defs" / "checkpoint_resolver.py"


def _load_resolver():
    spec = importlib.util.spec_from_file_location("checkpoint_resolver",
                                                  RESOLVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").touch()


def test_public_variant_can_use_private_quantized_mirror(tmp_path):
    resolver = _load_resolver()
    variant = "Cosmos-Reason2-8B-NVFP4"
    checkpoint = tmp_path / "private" / variant
    _create_checkpoint(checkpoint)

    resolved = resolver.resolve_checkpoint(
        variant,
        torch_roots=(),
        quantized_root=str(tmp_path),
    )

    assert resolved == str(checkpoint)


def test_quantized_variant_can_use_declared_managed_path(tmp_path):
    resolver = _load_resolver()
    variant = "NVIDIA-Nemotron-3-Nano-4B-NVFP4"
    checkpoint = tmp_path / "models" / variant
    _create_checkpoint(checkpoint)

    resolved = resolver.resolve_checkpoint(
        variant,
        torch_roots=(),
        quantized_root=str(tmp_path / "quantized"),
        managed_roots=(str(tmp_path), ),
    )

    assert resolved == str(checkpoint)


def test_private_checkpoint_never_falls_back_to_hugging_face(
        tmp_path, monkeypatch):
    resolver = _load_resolver()
    downloaded = False

    def fail_download(*_args, **_kwargs):
        nonlocal downloaded
        downloaded = True
        pytest.fail("private checkpoint attempted a Hugging Face download")

    monkeypatch.setattr(resolver, "_download_from_hf", fail_download)
    with pytest.raises(ValueError, match="Private CI checkpoint"):
        resolver.resolve_checkpoint(
            "qwen3_5_omni_3b_final_multilingual_0324",
            torch_roots=(),
            allow_download=True,
            download_root=str(tmp_path / "downloads"),
        )

    assert not downloaded


def test_release_manifest_excludes_ci_checkpoint_data():
    manifest_rules = {
        line.strip()
        for line in (REPO_ROOT /
                     "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    }

    assert "prune .gitlab" in manifest_rules
    assert "prune tests" in manifest_rules
    assert (REPO_ROOT / ".gitlab" / "ci" / "checkpoint_inventory.py").is_file()
    assert RESOLVER_PATH.is_file()
