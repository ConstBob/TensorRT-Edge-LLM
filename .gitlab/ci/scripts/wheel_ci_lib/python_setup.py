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
"""Host and target Python provisioning for native and cross wheel builds."""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import typing

from wheel_ci_lib import common
from wheellib import config

_UV_REQUIREMENT = "uv==0.8.14"
_TARGET_PYTHON_UBUNTU_RELEASE = {
    "3.10": "jammy",
    "3.11": "jammy",
    "3.12": "noble",
}
_PROVISIONED_PYTHONS: typing.Dict[str, str] = {}


def python_for_abi(python_abi: str) -> str:
    try:
        interpreter = common.ABI_INTERPRETERS[python_abi]
    except KeyError as error:
        raise RuntimeError(f"Unsupported Python ABI {python_abi}.") from error
    executable = shutil.which(interpreter)
    if executable is None:
        version = interpreter.removeprefix("python")
        candidates = sorted(
            pathlib.Path("/usr/local/pyenv/versions").glob(
                f"{version}.*/bin/python"))
        if len(candidates) == 1:
            executable = os.fspath(candidates[0])
    if executable is None:
        raise RuntimeError(f"Required interpreter {interpreter} is missing.")
    return executable


def managed_python(version: str) -> str:
    bootstrap = config.REPO_ROOT / "venv" / "python-bootstrap"
    bootstrap_python = bootstrap / "bin" / "python"
    uv = bootstrap / "bin" / "uv"
    if not uv.is_file():
        shutil.rmtree(bootstrap, ignore_errors=True)
        config.run_checked([sys.executable, "-m", "venv", str(bootstrap)])
        config.run_checked(
            [str(bootstrap_python), "-m", "pip", "install", _UV_REQUIREMENT])
    install_root = config.REPO_ROOT / "venv" / "managed-python"
    environment = dict(os.environ)
    environment["UV_PYTHON_INSTALL_DIR"] = str(install_root)
    config.run_checked([str(uv), "python", "install", version],
                       env=environment)
    candidates = sorted(install_root.rglob(f"python{version}"))
    executables = [path for path in candidates if path.parent.name == "bin"]
    if len(executables) != 1:
        raise RuntimeError(
            f"Expected one managed Python {version}, found {executables}.")
    return str(executables[0])


def install_host_python(version: str) -> str:
    return prepare_host_pythons((version, ))[version]


def _install_apt_host_tools(versions: typing.Sequence[str],
                            install_ccache: bool) -> None:
    packages = []
    if install_ccache and shutil.which("ccache") is None:
        packages.append("ccache")
    if versions:
        packages.append("software-properties-common")
    if not packages:
        return
    config.run_checked(["apt-get", "update"])
    config.run_checked(
        ["apt-get", "install", "-y", "--no-install-recommends", *packages])
    if not versions:
        return
    config.run_checked(["add-apt-repository", "-y", "ppa:deadsnakes/ppa"])
    config.run_checked(["apt-get", "update"])
    python_packages = []
    for version in versions:
        python_packages.extend((f"python{version}", f"python{version}-dev",
                                f"python{version}-venv"))
    config.run_checked([
        "apt-get", "install", "-y", "--no-install-recommends", *python_packages
    ])


def prepare_host_pythons(
        versions: typing.Sequence[str],
        *,
        install_ccache: bool = False) -> typing.Dict[str, str]:
    """Provision requested host interpreters in one package-manager pass."""
    ordered = tuple(dict.fromkeys(versions))
    executables = {
        version:
        _PROVISIONED_PYTHONS.get(version) or shutil.which(f"python{version}")
        for version in ordered
    }
    missing = tuple(version for version, executable in executables.items()
                    if executable is None)
    if os.geteuid() == 0:
        _install_apt_host_tools(missing, install_ccache)
    else:
        if install_ccache and shutil.which("ccache") is None:
            print(
                "[wheel-ci] Cannot install ccache without root; continuing without it.",
                flush=True)
        for version in missing:
            executables[version] = managed_python(version)
    for version in ordered:
        if executables[version] is None:
            executables[version] = python_for_abi(
                f"cp{version.replace('.', '')}")
    resolved = typing.cast(typing.Dict[str, str], executables)
    _PROVISIONED_PYTHONS.update(resolved)
    return resolved


def _scope_deb_source_to_host(path: pathlib.Path) -> None:
    if not path.is_file():
        return
    changed = False
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.lstrip()
        prefix = line[:len(line) - len(stripped)]
        if stripped.startswith("deb ") and not stripped.startswith("deb ["):
            line = prefix + "deb [arch=amd64] " + stripped.removeprefix("deb ")
            changed = True
        lines.append(line)
    if changed:
        path.write_text("".join(lines), encoding="utf-8")


def _scope_deb822_source_to_host(path: pathlib.Path) -> None:
    if not path.is_file():
        return
    changed = False
    stanzas = []
    for stanza in path.read_text(encoding="utf-8").split("\n\n"):
        lines = stanza.splitlines()
        types_index = next((index for index, line in enumerate(lines)
                            if line.lower().startswith("types:")), None)
        has_architectures = any(line.lower().startswith("architectures:")
                                for line in lines)
        if types_index is not None and not has_architectures:
            lines.insert(types_index + 1, "Architectures: amd64")
            changed = True
        stanzas.append("\n".join(lines))
    if changed:
        path.write_text("\n\n".join(stanzas).rstrip() + "\n", encoding="utf-8")


def _configure_target_python_sources(
        values: typing.Sequence[typing.Tuple[str, pathlib.Path]]) -> None:
    missing_versions = []
    for version, headers in values:
        target_config = (headers.parent / "aarch64-linux-gnu" / headers.name /
                         "pyconfig.h")
        if not target_config.is_file():
            missing_versions.append(version)
    if not missing_versions:
        return
    for source in [
            pathlib.Path("/etc/apt/sources.list"),
            *pathlib.Path("/etc/apt/sources.list.d").glob("*.list")
    ]:
        _scope_deb_source_to_host(source)
    for source in pathlib.Path("/etc/apt/sources.list.d").glob("*.sources"):
        _scope_deb822_source_to_host(source)
    releases = []
    for version in missing_versions:
        try:
            releases.append(_TARGET_PYTHON_UBUNTU_RELEASE[version])
        except KeyError as error:
            raise RuntimeError(
                f"No arm64 CPython header source is configured for {version}."
            ) from error
    ports = pathlib.Path("/etc/apt/sources.list.d/ubuntu-ports-arm64.list")
    ports.write_text(
        "\n".join(f"deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports "
                  f"{release} main restricted universe multiverse"
                  for target_release in sorted(set(releases))
                  for release in (target_release, f"{target_release}-updates",
                                  f"{target_release}-security")) + "\n",
        encoding="utf-8")
    config.run_checked(["dpkg", "--add-architecture", "arm64"])
    config.run_checked(["apt-get", "update"])


def _install_target_python_headers(version: str,
                                   headers: pathlib.Path) -> pathlib.Path:
    target_config = (headers.parent / "aarch64-linux-gnu" / headers.name /
                     "pyconfig.h")
    if target_config.is_file():
        return headers
    target_root = pathlib.Path(
        tempfile.gettempdir()) / f"edgellm-python{version}-arm64"
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edgellm-python-deb-") as value:
        download_dir = pathlib.Path(value)
        download_dir.chmod(0o777)
        config.run_checked(
            ["apt-get", "download", f"libpython{version}-dev:arm64"],
            cwd=download_dir)
        packages = list(download_dir.glob("*.deb"))
        if len(packages) != 1:
            raise RuntimeError(
                f"Expected one target CPython development package: {packages}."
            )
        config.run_checked(
            ["dpkg-deb", "--extract",
             str(packages[0]),
             str(target_root)])
    include_root = target_root / "usr" / "include"
    staged_headers = include_root / headers.name
    target_link = staged_headers / "aarch64-linux-gnu"
    target_link.unlink(missing_ok=True)
    target_link.symlink_to(include_root / "aarch64-linux-gnu",
                           target_is_directory=True)
    staged_config = target_link / headers.name / "pyconfig.h"
    if not (staged_headers /
            "Python.h").is_file() or not staged_config.is_file():
        raise RuntimeError(
            f"Target CPython headers are incomplete under {staged_headers}.")
    return staged_headers


def bootstrap_build_pythons(
    row: typing.Mapping[str, object], python_abis: typing.Sequence[str]
) -> typing.Dict[str, typing.Tuple[str, typing.Optional[pathlib.Path]]]:
    """Create lean payload-build environments for all requested ABIs."""
    versions = {
        python_abi: common.ABI_INTERPRETERS[python_abi].removeprefix("python")
        for python_abi in python_abis
    }
    host_pythons = prepare_host_pythons(tuple(versions.values()),
                                        install_ccache=True)
    target_headers: typing.Dict[str, pathlib.Path] = {}
    if row["ci_build_mode"] == "cross":
        header_template = str(row["ci_python_headers"])
        for python_abi, version in versions.items():
            target_headers[python_abi] = pathlib.Path(
                header_template.format(python_abi=python_abi,
                                       python_version=version))
        _configure_target_python_sources(
            tuple((versions[python_abi], headers)
                  for python_abi, headers in target_headers.items()))

    environments = {}
    for python_abi, version in versions.items():
        staged_headers = None
        if python_abi in target_headers:
            staged_headers = _install_target_python_headers(
                version, target_headers[python_abi])
        environments[python_abi] = (common.toolchain_python(
            host_pythons[version], python_abi, "payload"), staged_headers)
    return environments
