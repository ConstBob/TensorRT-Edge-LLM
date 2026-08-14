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
"""Apply the CI retention policy to one scratch engine directory."""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

RETAIN_MARKER_PATTERN = re.compile(r"\[retain engines\]", re.IGNORECASE)
STORAGE_NAMESPACES = frozenset(("L0", "L1"))


class EngineStorageError(RuntimeError):
    """Raised when engine cleanup would violate the storage contract."""


def _retention_requested(environ: Mapping[str, str]) -> bool:
    return RETAIN_MARKER_PATTERN.search(environ.get("CI_COMMIT_MESSAGE",
                                                    "")) is not None


def _engine_directory(environ: Mapping[str, str]) -> Optional[Path]:
    raw_engine_dir = environ.get("ENGINE_DIR", "")
    if not raw_engine_dir:
        return None

    raw_scratch_root = environ.get("EDGE_LLM_CACHE_DIR", "")
    if not raw_scratch_root:
        raise EngineStorageError(
            "EDGE_LLM_CACHE_DIR must be set when ENGINE_DIR is set")

    engine_dir = Path(raw_engine_dir)
    scratch_root = Path(raw_scratch_root)
    if not engine_dir.is_absolute() or not scratch_root.is_absolute():
        raise EngineStorageError(
            "ENGINE_DIR and EDGE_LLM_CACHE_DIR must be absolute paths")
    if engine_dir.is_symlink():
        raise EngineStorageError(
            f"ENGINE_DIR must not be a symbolic link: {engine_dir}")

    resolved_engine_dir = engine_dir.resolve(strict=False)
    resolved_scratch_root = scratch_root.resolve(strict=False)
    if resolved_scratch_root == Path(resolved_scratch_root.anchor):
        raise EngineStorageError(
            "EDGE_LLM_CACHE_DIR must not be the filesystem root")
    try:
        relative_engine_dir = resolved_engine_dir.relative_to(
            resolved_scratch_root)
    except ValueError as error:
        raise EngineStorageError(
            f"ENGINE_DIR is outside EDGE_LLM_CACHE_DIR: {engine_dir}"
        ) from error

    parts = relative_engine_dir.parts
    if (len(parts) < 3 or parts[0] not in STORAGE_NAMESPACES
            or parts[-1] != "engines"):
        raise EngineStorageError(
            "ENGINE_DIR must name a job engines directory beneath the L0 or "
            f"L1 scratch namespace: {engine_dir}")
    return resolved_engine_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Apply the engine retention policy and return its process status."""
    parser = argparse.ArgumentParser(
        description="Remove unretained TensorRT Edge-LLM CI engines")
    parser.parse_args(argv)

    try:
        engine_dir = _engine_directory(os.environ)
        if engine_dir is None:
            print("No ENGINE_DIR was assigned; engine cleanup is not needed")
            return 0
        if not engine_dir.exists():
            print(f"Engine directory is already absent: {engine_dir}")
            return 0
        if not engine_dir.is_dir():
            raise EngineStorageError(
                f"ENGINE_DIR is not a directory: {engine_dir}")
        if _retention_requested(os.environ):
            print(f"Retaining engine directory: {engine_dir}")
            return 0

        shutil.rmtree(engine_dir)
        print(f"Removed unretained engine directory: {engine_dir}")
        return 0
    except (EngineStorageError, OSError) as error:
        print(f"ERROR: Engine cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
