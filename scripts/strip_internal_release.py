#!/usr/bin/env python3
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
"""Strip internal-release regions from a clean OSS export tree."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

_DEFAULT_MANIFEST = (Path(__file__).resolve().parents[1] /
                     "oss_release_manifest.json")
_DEFAULT_DO_NOT_RELEASE = "DO_NOT_RELEASE"
_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
}


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_guarded_regions(path: Path, begin_marker: str,
                           end_marker: str) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    stripped: list[str] = []
    removed_regions = 0
    in_guarded_region = False

    for line_no, line in enumerate(lines, start=1):
        if begin_marker in line:
            if in_guarded_region:
                raise RuntimeError(
                    f"{path}:{line_no}: nested {begin_marker} marker")
            in_guarded_region = True
            removed_regions += 1
            continue
        if end_marker in line:
            if not in_guarded_region:
                raise RuntimeError(
                    f"{path}:{line_no}: unmatched {end_marker} marker")
            in_guarded_region = False
            continue
        if not in_guarded_region:
            stripped.append(line)

    if in_guarded_region:
        raise RuntimeError(f"{path}: missing {end_marker} marker")

    if removed_regions:
        path.write_text("".join(stripped), encoding="utf-8")
    return removed_regions


def _iter_scanned_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_file():
                yield path


def _read_text_if_possible(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _check_forbidden_patterns(root: Path, patterns: list[str]) -> None:
    leaks: list[str] = []
    for path in _iter_scanned_files(root):
        text = _read_text_if_possible(path)
        if text is None:
            continue
        rel = path.relative_to(root)
        for pattern in patterns:
            if pattern in text:
                leaks.append(f"{rel}: contains {pattern!r}")

    if leaks:
        joined = "\n".join(leaks)
        raise RuntimeError(f"Internal-release leak check failed:\n{joined}")


def _delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _delete_paths(root: Path, rel_paths: list[str]) -> None:
    for rel_path in rel_paths:
        _delete_path(root / rel_path)


def _load_do_not_release(root: Path, rel_path: str) -> list[str]:
    path = root / rel_path
    if not path.exists():
        return []

    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def _matching_paths(root: Path, pattern: str) -> Iterable[Path]:
    normalized = pattern.rstrip("/")
    has_glob = any(char in normalized for char in "*?[")
    if has_glob:
        if "/" in normalized:
            yield from root.glob(normalized)
        else:
            yield from root.rglob(normalized)
        return

    exact_path = root / normalized
    if exact_path.exists() or exact_path.is_symlink():
        yield exact_path

    if pattern.endswith("/") and "/" not in normalized:
        for path in root.rglob(normalized):
            if path.is_dir():
                yield path


def _delete_do_not_release_paths(root: Path, rel_path: str) -> None:
    seen: set[Path] = set()
    for pattern in _load_do_not_release(root, rel_path):
        for path in _matching_paths(root, pattern):
            if path in seen:
                continue
            seen.add(path)
            _delete_path(path)
    _delete_path(root / rel_path)


def _get_release_policy(manifest: dict) -> dict:
    return manifest.get("release_policy", manifest)


def _get_guarded_region_markers(policy: dict) -> tuple[str, str]:
    markers = policy.get("guarded_region_markers")
    if markers is not None:
        return markers["begin"], markers["end"]
    return policy["begin_marker"], policy["end_marker"]


def strip_internal_release(root: Path, manifest_path: Path,
                           check_only: bool) -> None:
    manifest = _load_manifest(manifest_path)
    policy = _get_release_policy(manifest)
    begin_marker, end_marker = _get_guarded_region_markers(policy)

    if not check_only:
        _delete_do_not_release_paths(
            root, policy.get("do_not_release_manifest",
                             _DEFAULT_DO_NOT_RELEASE))
        _delete_paths(root, policy.get("delete_paths", []))
        for rel_path in policy.get("strip_paths", []):
            path = root / rel_path
            if path.exists():
                _strip_guarded_regions(path, begin_marker, end_marker)

    _check_forbidden_patterns(root, policy.get("forbidden_patterns", []))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip internal-release code from an OSS export tree.")
    parser.add_argument("--root",
                        type=Path,
                        default=Path.cwd(),
                        help=("Clean export tree to sanitize. Defaults to the "
                              "current directory."))
    parser.add_argument("--manifest",
                        type=Path,
                        default=_DEFAULT_MANIFEST,
                        help=("Release policy manifest. Defaults to repo-root "
                              "oss_release_manifest.json."))
    parser.add_argument("--check-only",
                        action="store_true",
                        help="Only run the leak check; do not modify files.")
    args = parser.parse_args()

    strip_internal_release(args.root.resolve(), args.manifest.resolve(),
                           args.check_only)


if __name__ == "__main__":
    main()
