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
"""Shared environment, toolchain, and artifact helpers for wheel CI."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import typing
import uuid

from wheellib import config

ABI_INTERPRETERS = {
    "cp310": "python3.10",
    "cp311": "python3.11",
    "cp312": "python3.12",
}
INTEGRATION_GATE_SCHEMA_VERSION = 2
BUILD_ENVIRONMENT = {
    "TRT_PACKAGE_DIR": "ci_trt_package",
    "WHEEL_TOOLCHAIN_FILE": "ci_toolchain",
    "WHEEL_TARGET_SYSROOT": "ci_sysroot",
    "WHEEL_TARGET_PYTHON_INCLUDE_ROOT": "ci_python_headers",
}
REMOTE_INTEGRATION_ENVIRONMENT = {
    "REMOTE_TARGET": "ci_remote",
    "BOARD_IP": "ci_board_ip",
    "BOARD_USER": "ci_board_user",
    "WHEEL_TARGET_TRT_WHEEL": "ci_target_trt_wheel",
    "WHEEL_TARGET_WORK_DIR": "ci_target_work_dir",
}
LOCAL_INTEGRATION_ENVIRONMENT = {
    "REMOTE_TARGET": "ci_remote",
    "TRT_PACKAGE_DIR": "ci_trt_package",
}

_TOOLCHAIN_REQUIREMENTS = {
    "assembly": "wheel-assembly-requirements.txt",
    "base": "wheel-base-requirements.txt",
    "payload": "wheel-payload-requirements.txt",
}


def _record_phase(name: str, status: str, exit_code: int, started_epoch: float,
                  finished_epoch: float) -> None:
    job_id = os.environ.get("CI_JOB_ID", "")
    if os.environ.get("CI_TELEMETRY_ENABLED", "1") == "0" or not job_id:
        return
    project_dir = pathlib.Path(os.environ.get("CI_PROJECT_DIR", os.getcwd()))
    telemetry_root = pathlib.Path(
        os.environ.get("CI_TELEMETRY_DIR", str(project_dir / ".ci-telemetry")))
    record_dir = telemetry_root / "phases" / job_id
    sanitized_name = "".join(character if " " <= character <= "~" else " "
                             for character in name)
    payload = {
        "schema_version":
        1,
        "label":
        sanitized_name,
        "status":
        status,
        "exit_code":
        exit_code,
        "started_at":
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
        "started_epoch":
        started_epoch,
        "finished_at":
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_epoch)),
        "finished_epoch":
        finished_epoch,
        "elapsed_seconds":
        max(0.0, finished_epoch - started_epoch),
    }
    temporary: typing.Optional[pathlib.Path] = None
    try:
        record_dir.mkdir(parents=True, exist_ok=True)
        record = record_dir / f"phase-wheel-{uuid.uuid4().hex}.json"
        temporary = record.with_name(f".{record.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, record)
    except OSError as error:
        print(f"[wheel-ci] telemetry warning: {error}",
              file=sys.stderr,
              flush=True)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@contextlib.contextmanager
def phase(name: str) -> typing.Iterator[None]:
    """Print an unbuffered duration for one CI phase."""
    started = time.monotonic()
    started_epoch = time.time()
    print(f"[wheel-ci] START {name}", flush=True)
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - started
        print(f"[wheel-ci] FAIL {name} ({elapsed:.1f}s)", flush=True)
        _record_phase(name, "failure", 1, started_epoch, time.time())
        raise
    elapsed = time.monotonic() - started
    print(f"[wheel-ci] DONE {name} ({elapsed:.1f}s)", flush=True)
    _record_phase(name, "success", 0, started_epoch, time.time())


def install_toolchain(profile: str, python: str = sys.executable) -> None:
    try:
        requirements = _TOOLCHAIN_REQUIREMENTS[profile]
    except KeyError as error:
        raise RuntimeError(
            f"Unknown wheel toolchain profile {profile!r}.") from error
    pip = subprocess.run([python, "-m", "pip", "--version"],
                         text=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    if pip.returncode != 0:
        config.run_checked([python, "-m", "ensurepip", "--upgrade"])
    config.run_checked([
        python, "-m", "pip", "install", "--requirement",
        str(config.REPO_ROOT / "packaging" / requirements)
    ],
                       cwd=config.REPO_ROOT)


def toolchain_python(python: str, python_abi: str, profile: str) -> str:
    environment = config.REPO_ROOT / "venv" / f"wheel-{python_abi}"
    shutil.rmtree(environment, ignore_errors=True)
    config.run_checked([python, "-m", "venv", str(environment)])
    isolated_python = str(environment / "bin" / "python")
    install_toolchain(profile, isolated_python)
    return isolated_python


def compiler_cache() -> typing.Optional[str]:
    """Configure a CI-restored ccache when the build host provides it."""
    executable = shutil.which("ccache")
    if executable is None:
        print(
            "[wheel-ci] ccache is unavailable; compiler caching is disabled.",
            flush=True)
        return None
    cache_dir = config.REPO_ROOT / "venv" / "wheel-compiler-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CCACHE_DIR"] = str(cache_dir)
    os.environ["CCACHE_BASEDIR"] = str(config.REPO_ROOT)
    os.environ["CCACHE_COMPILERCHECK"] = "content"
    os.environ["CCACHE_COMPRESS"] = "true"
    config.run_checked([executable, "--max-size", "2G"])
    config.run_checked([executable, "--zero-stats"])
    compilers = tuple(path for name in ("cc", "c++")
                      if (path := shutil.which(name)) is not None)
    if any(
            pathlib.Path(path).resolve() == pathlib.Path(executable).resolve()
            for path in compilers):
        return None
    return executable


def report_compiler_cache() -> None:
    executable = shutil.which("ccache")
    if executable is not None and "CCACHE_DIR" in os.environ:
        config.run_checked([executable, "--show-stats"])


def update_submodules() -> None:
    error = None
    for attempt in range(1, 4):
        try:
            config.run_checked(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=config.REPO_ROOT)
            return
        except RuntimeError as current:
            error = current
            if attempt < 3:
                delay = 15 * attempt
                print(f"Submodule update failed; retrying in {delay}s.",
                      file=sys.stderr)
                time.sleep(delay)
    assert error is not None
    raise error


def normalized_trt_package(trt_dir: pathlib.Path) -> pathlib.Path:
    if (trt_dir / "include" / "NvInfer.h").is_file():
        return trt_dir
    include_candidates = sorted((trt_dir / "include").glob("*/NvInfer.h"))
    library_candidates = sorted((trt_dir / "lib").glob("*/libnvinfer.so"))
    if len(include_candidates) != 1 or len(library_candidates) != 1:
        return trt_dir
    staged = config.REPO_ROOT / "artifacts" / "trt-packages" / host_arch()
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    (staged / "include").symlink_to(include_candidates[0].parent,
                                    target_is_directory=True)
    (staged / "lib").symlink_to(library_candidates[0].parent,
                                target_is_directory=True)
    return staged


def apply_variant_environment(row: typing.Mapping[str, object],
                              fields: typing.Mapping[str, str]) -> None:
    for variable, field in fields.items():
        value = row.get(field)
        if value is None or value == "":
            os.environ.pop(variable, None)
            continue
        if isinstance(value, bool):
            os.environ[variable] = "1" if value else "0"
            continue
        text = str(value)
        os.environ[variable] = (config.required_environment(text[1:])
                                if text.startswith("$") else text)


def host_arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def build_environment(trt_dir: pathlib.Path) -> typing.Dict[str, str]:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(trt_dir / "lib"),
                            environment.get("LD_LIBRARY_PATH", "")) if value)
    return environment


def single_wheel(cpu_arch: str, python_abi: str) -> pathlib.Path:
    directory = config.REPO_ROOT / "dist" / cpu_arch / python_abi
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one wheel under {directory}, found {len(wheels)}.")
    return wheels[0].resolve()
