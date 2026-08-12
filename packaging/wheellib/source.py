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
"""Enforce the packaging source boundary and emit source provenance."""

import argparse
import subprocess
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from .config import (REPO_ROOT, load_toml, require_clean_source,
                     source_snapshot, write_json)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref",
                        help="Diff base for the packaging change.")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "source_policy.toml")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def _changed_paths(root: Path, base_ref: str) -> List[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE)
    return [Path(value) for value in result.stdout.splitlines() if value]


def _under_roots(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _packaging_boundary_changes(changed: Sequence[Path]) -> List[Path]:
    boundary_roots = (Path("packaging"), Path("tensorrt_edgellm/_native"))
    boundary_files = {
        Path("tensorrt_edgellm/runtime.py"),
        Path("experimental/server/runtime/engine.py"),
    }
    if not any(path in boundary_files or _under_roots(path, boundary_roots)
               for path in changed):
        return []
    return list(changed)


def _validate_source_boundary(changed: Sequence[Path],
                              policy: Mapping[str, Any]) -> None:
    protected = tuple(Path(value) for value in policy["protected_roots"])
    violations = [path for path in changed if _under_roots(path, protected)]
    if violations:
        raise RuntimeError(
            "Version-1 packaging changes protected product paths: " +
            ", ".join(path.as_posix() for path in violations))

    approved_roots = tuple(Path(value) for value in policy["approved_roots"])
    approved_files = {Path(value) for value in policy["approved_files"]}
    unapproved = [
        path for path in changed if path not in approved_files
        and not _under_roots(path, approved_roots)
    ]
    if unapproved:
        raise RuntimeError(
            "Version-1 packaging changes paths outside its approved boundary: "
            + ", ".join(path.as_posix() for path in unapproved))


def _emit_snapshot(root: Path, output: Path | None) -> None:
    snapshot = source_snapshot(root)
    if output:
        write_json(output, snapshot)
    else:
        print(snapshot["sha256"])


def main(argv=None) -> None:
    """Check changed paths and optionally require a clean tracked checkout."""
    args = _arguments(argv)
    root = args.repo_root.resolve()
    policy = load_toml(args.policy.resolve())
    if args.base_ref:
        changed = _packaging_boundary_changes(
            _changed_paths(root, args.base_ref))
        _validate_source_boundary(changed, policy)
    if args.require_clean:
        require_clean_source(root)
    _emit_snapshot(root, args.output)


if __name__ == "__main__":
    main()
