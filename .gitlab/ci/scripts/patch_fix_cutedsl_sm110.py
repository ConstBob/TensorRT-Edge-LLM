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
"""Apply SM110 compatibility patches to an installed CuTeDSL package and
to in-tree nvfp4_moe_cutedsl kernel sources.

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


def replace_all(path: Path,
                before: str,
                after: str,
                *,
                min_count: int = 1) -> None:
    """Replace every occurrence of `before` with `after`. Require at least `min_count`."""
    text = _read(path)
    n = text.count(before)
    if n < min_count:
        sys.exit(
            f"FAIL: {path}: expected >= {min_count} occurrence(s) of {before!r}, found {n}. "
            "CuTeDSL likely bumped — patch needs review.")
    path.write_text(text.replace(before, after))
    print(f"  patched {path} (replaced {n} occurrence(s) of {before!r})")


def delete_lines_equal(path: Path,
                       stripped: str,
                       *,
                       min_count: int = 1) -> None:
    """Delete every line whose `.strip()` equals `stripped`. Require at least `min_count`."""
    lines = _read(path).splitlines(keepends=True)
    kept: list[str] = []
    deleted = 0
    for line in lines:
        if line.strip() == stripped:
            deleted += 1
        else:
            kept.append(line)
    if deleted < min_count:
        sys.exit(
            f"FAIL: {path}: expected >= {min_count} line(s) matching {stripped!r}, "
            f"deleted {deleted}. CuTeDSL likely bumped — patch needs review.")
    path.write_text("".join(kept))
    print(
        f"  patched {path} (deleted {deleted} line(s) matching {stripped!r})")


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
        help="Repo root (contains kernelSrcs/nvfp4_moe_cutedsl/).",
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

    print("Patching in-tree nvfp4_moe_cutedsl kernel sources…")
    moe_dir = repo / "kernelSrcs/nvfp4_moe_cutedsl"
    for fname in (
            "blockscaled_contiguous_grouped_gemm_n_major.py",
            "blockscaled_contiguous_grouped_gemm_finalize_n_major.py",
    ):
        f = moe_dir / fname
        replace_all(f,
                    "cute.arch.ProxyKind.async_shared",
                    '"async.shared"',
                    min_count=1)
        replace_all(f,
                    "cute.arch.SharedSpace.shared_cta",
                    '"cta"',
                    min_count=1)
    delete_lines_equal(moe_dir / "utils.py", "T.f32(),", min_count=1)

    print("Invalidating CuTeDSL .pyc cache…")
    for pyc in pkg.rglob("*.pyc"):
        pyc.unlink()

    print("All patches applied successfully.")


if __name__ == "__main__":
    main()
