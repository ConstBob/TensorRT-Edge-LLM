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
"""Validate that the current tracked tree can be sanitized for OSS release."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(args: list[str],
         cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args,
                          cwd=cwd,
                          check=True,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def _repo_root() -> Path:
    result = _run(["git", "rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip()).resolve()


def _require_initialized_submodules(repo_root: Path) -> None:
    result = _run(["git", "submodule", "status", "--recursive"], cwd=repo_root)
    invalid = [
        line for line in result.stdout.splitlines()
        if line and not line.startswith(" ")
    ]
    if invalid:
        details = "\n".join(invalid)
        raise RuntimeError(
            "Submodules must be initialized at their pinned revisions before "
            f"staging the OSS tree:\n{details}")


def _git_ls_files(repo_root: Path, include_submodules: bool) -> list[str]:
    args = ["git", "ls-files"]
    if include_submodules:
        _require_initialized_submodules(repo_root)
        args.append("--recurse-submodules")
    args.append("-z")
    result = subprocess.run(args,
                            cwd=repo_root,
                            check=True,
                            stdout=subprocess.PIPE)
    return [path.decode() for path in result.stdout.split(b"\0") if path]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst, follow_symlinks=False)


def _copy_tracked_worktree(repo_root: Path, export_root: Path,
                           include_submodules: bool) -> None:
    for rel_path in _git_ls_files(repo_root, include_submodules):
        src = repo_root / rel_path
        if src.is_file() or src.is_symlink():
            _copy_file(src, export_root / rel_path)

    # Keep local validation useful while release-policy changes are being
    # developed before their first commit.
    for rel_path in [
            "oss_release_manifest.json",
            "scripts/check_oss_release_sanitizer.py",
            "scripts/strip_internal_release.py",
            "scripts/README.md",
    ]:
        src = repo_root / rel_path
        if src.exists():
            _copy_file(src, export_root / rel_path)


def check_oss_release_sanitizer(repo_root: Path, keep_temp: bool,
                                include_submodules: bool) -> Path | None:
    temp_dir = Path(tempfile.mkdtemp(prefix="edgellm-oss-release-check."))
    try:
        _copy_tracked_worktree(repo_root, temp_dir, include_submodules)
        subprocess.run([
            sys.executable,
            str(repo_root / "scripts/strip_internal_release.py"),
            "--root",
            str(temp_dir),
            "--manifest",
            str(repo_root / "oss_release_manifest.json"),
        ],
                       check=True)
    except Exception:
        if keep_temp:
            print(f"Temporary export tree preserved: {temp_dir}",
                  file=sys.stderr)
        else:
            shutil.rmtree(temp_dir)
        raise

    if keep_temp:
        return temp_dir
    shutil.rmtree(temp_dir)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the OSS release sanitizer on a tracked-tree copy."
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary sanitized tree for inspection.")
    parser.add_argument(
        "--include-submodules",
        action="store_true",
        help="Include initialized submodule contents in the sanitized tree.")
    args = parser.parse_args()

    temp_dir = check_oss_release_sanitizer(_repo_root(), args.keep_temp,
                                           args.include_submodules)
    if temp_dir is not None:
        print(f"OSS release sanitizer check passed: {temp_dir}")
    else:
        print("OSS release sanitizer check passed")


if __name__ == "__main__":
    main()
