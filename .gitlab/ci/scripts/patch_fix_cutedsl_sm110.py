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
"""Apply SM110 compatibility patches to an installed CuTeDSL package.

CI-only workaround invoked from `.gitlab/ci/cutedsl-jobs.yml` during the
`build_cutedsl_sm110_artifact` job. Delete this script (and its call site)
once CuTeDSL upstream supports SM110 natively.

Each patch checks for its anchor before applying. If an anchor is missing
(e.g. CuTeDSL bumped and renamed/restructured), the script exits non-zero
with a clear error rather than silently no-op'ing like `sed -i` does.
"""

import argparse
import importlib
import sys
from pathlib import Path


def _read(path: Path) -> str:
    if not path.is_file():
        sys.exit(f"FAIL: {path} not found")
    return path.read_text()


def replace_unique(path: Path, before: str, after: str) -> None:
    """Replace exactly one occurrence of `before` in `path` with `after`."""
    text = _read(path)
    n = text.count(before)
    if n != 1:
        sys.exit(
            f"FAIL: {path}: expected 1 occurrence of {before!r}, found {n}. "
            "CuTeDSL likely bumped — patch needs review.")
    path.write_text(text.replace(before, after, 1))
    print(f"  patched {path} (replaced 1 occurrence)")


def insert_after_unique(path: Path, anchor_line: str,
                        lines_to_insert: str) -> None:
    """Insert `lines_to_insert` immediately after the unique line equal to `anchor_line`.

    Anchor and inserted text must include their trailing newline.
    """
    text = _read(path)
    n = text.count(anchor_line)
    if n != 1:
        sys.exit(
            f"FAIL: {path}: expected 1 anchor {anchor_line.strip()!r}, found {n}. "
            "CuTeDSL likely bumped — patch needs review.")
    path.write_text(text.replace(anchor_line, anchor_line + lines_to_insert,
                                 1))
    print(f"  patched {path} (inserted after {anchor_line.strip()!r})")


def discover_cutedsl_pkg_root() -> Path:
    """Locate the installed nvidia_cutlass_dsl package root in the current Python env."""
    try:
        mod = importlib.import_module("nvidia_cutlass_dsl")
    except ImportError as e:
        sys.exit(
            f"FAIL: cannot import nvidia_cutlass_dsl from this Python: {e}")
    return Path(mod.__path__[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="Repo root.",
    )
    args = ap.parse_args()

    pkg = discover_cutedsl_pkg_root()
    repo = args.repo_root.resolve()
    print(f"CuTeDSL package root: {pkg}")
    print(f"Repo root:            {repo}")

    print("Patching installed CuTeDSL package…")
    insert_after_unique(
        pkg / "python_packages/cutlass/cute/nvgpu/tcgen05/mma.py",
        anchor_line="        Arch.sm_103a,\n",
        lines_to_insert="        Arch.sm_101a,\n        Arch.sm_110a,\n",
    )
    replace_unique(
        pkg / "python_packages/cutlass/cute/nvgpu/tcgen05/copy.py",
        before="if not arch.is_family_of(Arch.sm_100f):",
        after=
        "if not (arch.is_family_of(Arch.sm_100f) or arch.is_family_of(Arch.sm_110f)):",
    )

    print("Invalidating CuTeDSL .pyc cache…")
    for pyc in pkg.rglob("*.pyc"):
        pyc.unlink()

    print("All patches applied successfully.")


if __name__ == "__main__":
    main()
