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

"""Patch/check CuTeDSL 4.5.1 site-packages for SM110a AOT export.

This is a temporary local workaround for Thor SM110 NVFP4 MoE AOT
generation. Keep it idempotent so build instructions can run it every time.
Remove it once the upstream CuTeDSL wheel accepts SM110a in these tcgen05 paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

_MMA_REL = Path("python_packages/cutlass/cute/nvgpu/tcgen05/mma.py")
_COPY_REL = Path("python_packages/cutlass/cute/nvgpu/tcgen05/copy.py")

_MMA_OLD = """    admissible_archs = [
        Arch.sm_100a,
        Arch.sm_103a,
    ]"""
_MMA_NEW = """    admissible_archs = [
        Arch.sm_100a,
        Arch.sm_103a,
        Arch.sm_110a,
    ]"""

_COPY_OLD = """        if not arch.is_family_of(Arch.sm_100f):
            supported = Arch.filter(lambda a: a.is_family_of(Arch.sm_100f))"""
_COPY_NEW = """        if not (
            arch.is_family_of(Arch.sm_100f)
            or arch.is_family_of(Arch.sm_110a)
            or arch.is_family_of(Arch.sm_110f)
        ):
            supported = Arch.filter(
                lambda a: a.is_family_of(Arch.sm_100f)
                or a.is_family_of(Arch.sm_110a)
                or a.is_family_of(Arch.sm_110f)
            )"""
_COPY_OLD_WITH_SM110F = """        if not (arch.is_family_of(Arch.sm_100f) or arch.is_family_of(Arch.sm_110f)):
            supported = Arch.filter(lambda a: a.is_family_of(Arch.sm_100f) or a.is_family_of(Arch.sm_110f))"""
_COPY_NEW_WITH_SM110F = """        if not (
            arch.is_family_of(Arch.sm_100f)
            or arch.is_family_of(Arch.sm_110a)
            or arch.is_family_of(Arch.sm_110f)
        ):
            supported = Arch.filter(
                lambda a: a.is_family_of(Arch.sm_100f)
                or a.is_family_of(Arch.sm_110a)
                or a.is_family_of(Arch.sm_110f)
            )"""
_COPY_OLD_WITH_SM110F_PARTIAL = """        if not (arch.is_family_of(Arch.sm_100f) or arch.is_family_of(Arch.sm_110f)):
            supported = Arch.filter(lambda a: a.is_family_of(Arch.sm_100f))"""


def _package_dir() -> Path:
    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Could not find package nvidia_cutlass_dsl in this Python environment.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _is_mma_patched(text: str) -> bool:
    return "Arch.sm_110a" in text


def _is_copy_patched(text: str) -> bool:
    s2t_start = text.find("class _S2TCopyBase")
    if s2t_start < 0:
        return False
    s2t_end = text.find("@dataclass", s2t_start)
    s2t_text = text[s2t_start : s2t_end if s2t_end >= 0 else len(text)]
    return "Arch.sm_110a" in s2t_text and "Arch.sm_110f" in s2t_text


def _patch_file(path: Path, old: str, new: str, is_patched) -> bool:
    text = path.read_text(encoding="utf-8")
    if is_patched(text):
        return False
    if old not in text:
        raise RuntimeError(
            f"Could not find expected CuTeDSL 4.5.1 snippet in {path}. "
            "The installed wheel may have changed; inspect this file before patching manually."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _patch_file_any(path: Path, replacements: list[tuple[str, str]], is_patched) -> bool:
    text = path.read_text(encoding="utf-8")
    if is_patched(text):
        return False
    for old, new in replacements:
        if old in text:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return True
    raise RuntimeError(
        f"Could not find expected CuTeDSL 4.5.1 snippet in {path}. "
        "The installed wheel may have changed; inspect this file before patching manually."
    )


def _clean_pycache(tcgen05_dir: Path) -> None:
    cache_dir = tcgen05_dir / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _check(mma_path: Path, copy_path: Path) -> list[str]:
    errors = []
    if not mma_path.exists():
        errors.append(f"Missing {mma_path}")
    elif not _is_mma_patched(mma_path.read_text(encoding="utf-8")):
        errors.append(f"{mma_path} does not include Arch.sm_110a in BlockScaledMmaOp.admissible_archs")

    if not copy_path.exists():
        errors.append(f"Missing {copy_path}")
    elif not _is_copy_patched(copy_path.read_text(encoding="utf-8")):
        errors.append(f"{copy_path} does not allow Arch.sm_110a in _S2TCopyBase")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify that the SM110a patch is present.")
    args = parser.parse_args()

    pkg_dir = _package_dir()
    mma_path = pkg_dir / _MMA_REL
    copy_path = pkg_dir / _COPY_REL

    if args.check:
        errors = _check(mma_path, copy_path)
        if errors:
            print("CuTeDSL SM110a patch check failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print(f"CuTeDSL SM110a patch check passed under {pkg_dir}")
        return 0

    changed = False
    changed |= _patch_file(mma_path, _MMA_OLD, _MMA_NEW, _is_mma_patched)
    changed |= _patch_file_any(
        copy_path,
        [
            (_COPY_OLD, _COPY_NEW),
            (_COPY_OLD_WITH_SM110F, _COPY_NEW_WITH_SM110F),
            (_COPY_OLD_WITH_SM110F_PARTIAL, _COPY_NEW_WITH_SM110F),
        ],
        _is_copy_patched,
    )
    if changed:
        _clean_pycache(mma_path.parent)
        print(f"Patched CuTeDSL SM110a support under {pkg_dir}")
    else:
        print(f"CuTeDSL SM110a support was already patched under {pkg_dir}")

    errors = _check(mma_path, copy_path)
    if errors:
        print("CuTeDSL SM110a patch verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
