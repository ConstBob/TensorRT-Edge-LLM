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
"""Precheck, base-wheel, payload, and assembly CI job implementations."""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import shutil
import sys
import typing

from wheel_ci_lib import common, matrix, python_setup
from wheellib import assemble, base, config, source


def precheck() -> None:
    """Validate source, matrix, generated CI, and exact submodules."""
    with common.phase("precheck: submodules"):
        common.update_submodules()
    with common.phase("precheck: source and matrix"):
        source.main([
            "--output",
            str(config.REPO_ROOT / "artifacts" / "provenance" / "source.json")
        ])
        config.load_matrix(config.REPO_ROOT / "packaging" / "variants.toml")
        matrix.load_qualification()
        matrix.generate_ci(check=True)


def build_base_ci() -> None:
    """Build one scheduled Python base wheel."""
    python_abi = config.required_environment("PYTHON_ABI")
    actual = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if actual != python_abi:
        raise RuntimeError(
            f"Scheduled ABI {python_abi} does not match interpreter {actual}.")
    with common.phase(f"base {python_abi}: submodules"):
        common.update_submodules()
    with common.phase(f"base {python_abi}: toolchain"):
        common.install_toolchain("base")
    with common.phase(f"base {python_abi}: build"):
        base.main([
            "--output-dir",
            str(config.REPO_ROOT / "artifacts" / "base" / python_abi),
        ])


def _cross_arguments(
        row: typing.Mapping[str,
                            object], python_abi: str, trt_dir: pathlib.Path
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    build_mode = row["ci_build_mode"]
    if build_mode == "native":
        if row["cpu_arch"] != common.host_arch():
            raise RuntimeError(
                f"Native build for {row['variant_id']} requires a "
                f"{row['cpu_arch']} runner, found {common.host_arch()}.")
        return [], []
    if row["cpu_arch"] == common.host_arch():
        raise RuntimeError(
            f"Cross build for {row['variant_id']} requires a non-"
            f"{row['cpu_arch']} runner.")
    toolchain = pathlib.Path(
        config.required_environment("WHEEL_TOOLCHAIN_FILE"))
    sysroot = pathlib.Path(config.required_environment("WHEEL_TARGET_SYSROOT"))
    header_template = config.required_environment(
        "WHEEL_TARGET_PYTHON_INCLUDE_ROOT")
    python_version = common.ABI_INTERPRETERS[python_abi].removeprefix("python")
    headers = pathlib.Path(
        header_template.format(python_abi=python_abi,
                               python_version=python_version))
    if not toolchain.is_file() or not sysroot.is_dir():
        raise RuntimeError(
            "The scheduled cross toolchain or sysroot is missing.")
    if not (headers / "Python.h").is_file():
        raise RuntimeError(f"Target Python headers are missing: {headers}.")
    return (
        [
            "--toolchain-file",
            str(toolchain),
            "--target-sysroot",
            str(sysroot),
            "--target-python-include-dir",
            str(headers),
        ],
        ["--dependency-root",
         str(sysroot), "--dependency-root",
         str(trt_dir)],
    )


def _prepare_cutedsl(python_bin: str, variant: str,
                     trt_dir: pathlib.Path) -> None:
    config.run_checked([
        python_bin,
        str(config.REPO_ROOT / "packaging" / "wheel_cli.py"),
        "prepare-cutedsl",
        "--variant",
        variant,
        "--artifact-dir",
        str(config.REPO_ROOT / "kernelSrcs" / "cuteDSLPrebuilt"),
    ],
                       cwd=config.REPO_ROOT,
                       env=common.build_environment(trt_dir))


def _build_payload(row: typing.Mapping[str, object], python_abi: str,
                   python_bin: str,
                   staged_headers: typing.Optional[pathlib.Path],
                   trt_dir: pathlib.Path, build_dir: pathlib.Path,
                   compiler_launcher: typing.Optional[str]) -> None:
    variant = str(row["variant_id"])
    common.apply_variant_environment(row, common.BUILD_ENVIRONMENT)
    if staged_headers is not None:
        os.environ["WHEEL_TARGET_PYTHON_INCLUDE_ROOT"] = str(staged_headers)
    output = config.REPO_ROOT / "artifacts" / "payloads" / f"{variant}-{python_abi}"
    build_extra, verify_extra = _cross_arguments(row, python_abi, trt_dir)
    environment = common.build_environment(trt_dir)
    config.run_checked([
        python_bin,
        str(config.REPO_ROOT / "packaging" / "wheel_cli.py"),
        "build-payload",
        "--variant",
        variant,
        "--python-abi",
        python_abi,
        "--trt-package-dir",
        str(trt_dir),
        "--build-dir",
        str(build_dir),
        *build_extra,
        "--output-dir",
        str(output),
        *([] if compiler_launcher is None else
          ["--compiler-launcher", compiler_launcher]),
    ],
                       cwd=config.REPO_ROOT,
                       env=environment)
    config.run_checked([
        python_bin,
        str(config.REPO_ROOT / "packaging" / "wheel_cli.py"),
        "verify-payload",
        "--stage",
        str(output),
        *verify_extra,
    ],
                       cwd=config.REPO_ROOT,
                       env=environment)


def build_payload_ci() -> None:
    """Build and verify one compatible group of native payloads."""
    group_name = config.required_environment("BUILD_GROUP")
    rows = matrix.scheduled_build_group(group_name)
    first = rows[0]
    common.apply_variant_environment(first, common.BUILD_ENVIRONMENT)
    trt_dir = common.normalized_trt_package(
        pathlib.Path(config.required_environment("TRT_PACKAGE_DIR")).resolve())
    with common.phase(f"payload {group_name}: submodules"):
        common.update_submodules()
    with common.phase(f"payload {group_name}: Python toolchains"):
        build_pythons = python_setup.bootstrap_build_pythons(
            first, tuple(common.ABI_INTERPRETERS))
    compiler_launcher = common.compiler_cache()
    prepare_python = build_pythons["cp312"][0]
    for row in rows:
        variant = str(row["variant_id"])
        # Native targets do not depend on CPython. Reconfiguring one build tree
        # per ABI preserves them while relinking only the extension module.
        build_dir = config.REPO_ROOT / "build" / "wheel-ci" / variant
        shutil.rmtree(build_dir, ignore_errors=True)
        with common.phase(f"payload {variant}: CuTeDSL"):
            _prepare_cutedsl(prepare_python, variant, trt_dir)
        for python_abi, (python_bin, staged_headers) in build_pythons.items():
            with common.phase(
                    f"payload {variant}/{python_abi}: build and verify"):
                _build_payload(row, python_abi, python_bin, staged_headers,
                               trt_dir, build_dir, compiler_launcher)
    common.report_compiler_cache()


def _assemble_wheel(cpu_arch: str, python_abi: str) -> None:
    bases = sorted(
        (config.REPO_ROOT / "artifacts" / "base" / python_abi).glob("*.whl"))
    if len(bases) != 1:
        raise RuntimeError(f"Expected one {python_abi} base wheel.")
    output = config.REPO_ROOT / "dist" / cpu_arch / python_abi
    assemble.main([
        "--base-wheel",
        str(bases[0]),
        "--payload-root",
        str(config.REPO_ROOT / "artifacts" / "payloads"),
        "--cpu-arch",
        cpu_arch,
        "--python-abi",
        python_abi,
        "--output-dir",
        str(output),
    ])
    wheel = common.single_wheel(cpu_arch, python_abi)
    (output / "wheel.sha256").write_text(
        f"{config.sha256(wheel)}  {wheel.name}\n", encoding="utf-8")


def assemble_ci() -> None:
    """Assemble and checksum one architecture's ABI wheels concurrently."""
    cpu_arch = config.required_environment("WHEEL_ARCH")
    if cpu_arch not in {"x86_64", "aarch64"}:
        raise RuntimeError(f"Unsupported assembly architecture {cpu_arch!r}.")
    try:
        workers = int(os.environ.get("WHEEL_ASSEMBLY_JOBS", "2"))
    except ValueError as error:
        raise RuntimeError(
            "WHEEL_ASSEMBLY_JOBS must be an integer.") from error
    if workers < 1:
        raise RuntimeError("WHEEL_ASSEMBLY_JOBS must be positive.")
    with common.phase(f"assembly {cpu_arch}: submodules"):
        common.update_submodules()
    with common.phase(f"assembly {cpu_arch}: toolchain"):
        common.install_toolchain("assembly")

    def assemble_one(python_abi: str) -> None:
        with common.phase(f"assembly {cpu_arch}/{python_abi}"):
            _assemble_wheel(cpu_arch, python_abi)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(
            workers, len(common.ABI_INTERPRETERS))) as executor:
        futures = [
            executor.submit(assemble_one, python_abi)
            for python_abi in common.ABI_INTERPRETERS
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
