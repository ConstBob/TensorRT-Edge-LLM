#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Build Edge-LLM against a PRE_BUILT TensorRT and run its Python E2E tests."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = _SOURCE_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from developer_toolkit_extensions import edgellm_code_manager, gpu
from trt_dev_toolkit.code_manager import (TRT_COMPONENT, ArtifactSource,
                                          ArtifactTarget, BuildMode,
                                          CodeManager, DeploymentMode)
from trt_dev_toolkit.code_manager.models import (ArtifactResult, BuildConfig,
                                                 PlanStep, PlatformConfig)
from trt_dev_toolkit.command_manager.command_manager import CommandManager
from trt_dev_toolkit.command_manager.data_structures import (CommandSpec,
                                                             OutputMode,
                                                             ShellType)
from trt_dev_toolkit.command_manager.targets import LocalTarget
from trt_dev_toolkit.constants import Arch, TargetType, parse_arch
from trt_dev_toolkit.container_manager import (ContainerKind, ContainerManager,
                                               ContainerPattern, MountSpec,
                                               format_mounts_csv)
from trt_dev_toolkit.container_manager.backends.git_trt_runc_backend import (
    GitTrtRuncBackend, GitTrtRuncConfig)
from trt_dev_toolkit.log_manager import configure_logging, get_logger
from trt_dev_toolkit.remote_connection_manager.config.config import (
    RemoteConfig, RemotePaths)
from trt_dev_toolkit.remote_connection_manager.config.jump_host import \
    JumpHostConfig
from trt_dev_toolkit.remote_connection_manager.config.ssh import SSHConfig
from trt_dev_toolkit.remote_connection_manager.remote_connection_manager import \
    RemoteConnectionManager
from trt_dev_toolkit.remote_connection_manager.types.target_types import \
    SSHTargetInfo

_PYTHON = Path(sys.executable)
_PYTHON_ROOT = Path(sys.prefix)
_RUN_WORKSPACE = PurePosixPath("/tmp/edgellm-trt-ci")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_CUDA_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+")
_NATIVE_BUILD_ENV_KEYS = (
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
)
_CONNECTION_TIMEOUT_S = 30
_TRANSFER_TIMEOUT_S = 1800
_TEST_TIMEOUT_S = 3600
_JOBS = 16
_DEFAULT_D7L_CUDA_DIR = PurePosixPath("/usr/local/cuda/targets/aarch64-linux")
_DEFAULT_D7L_CUDA_TARGET_DIR = PurePosixPath(
    "/usr/local/cuda/thor/targets/aarch64-linux")


@dataclasses.dataclass(frozen=True)
class ModelCase:
    name: str
    repository: str
    checkpoint_dir: str
    pipeline_param: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], context: str) -> ModelCase:
        allowed = {"name", "repository", "checkpoint_dir", "pipeline_param"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                f"{context} has unknown fields: {', '.join(unknown)}")
        values: dict[str, str] = {}
        for field in sorted(allowed):
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{context}.{field} must be a non-empty string")
            if any(char in value for char in "\r\n\0"):
                raise ValueError(
                    f"{context}.{field} contains an invalid character")
            values[field] = value.strip()
        return cls(**values)


# One entry enables optional checkpoint download, ONNX export, engine build, and inference.
_DEFAULT_MODEL_CASES = (
    ModelCase(
        name="Qwen3-0.6B",
        repository="Qwen/Qwen3-0.6B",
        checkpoint_dir="Qwen3/Qwen3-0.6B",
        pipeline_param="Qwen3-0.6B-fp16-mxsl4096-mxbs1-mxil2048",
    ),
    ModelCase(
        name="Qwen3.5-0.8B",
        repository="Qwen/Qwen3.5-0.8B",
        checkpoint_dir="Qwen3.5-0.8B",
        pipeline_param="Qwen3.5-0.8B-fp16-mxsl2048-mxbs1-mxil1024",
    ),
)


def _component_id(component: Any) -> str:
    return str(getattr(component, "value", component))


def _gpu_selection_for_arch(architecture: Arch,
                            compute_capability: str) -> gpu.GPUSelection:
    name = "D7L" if architecture is Arch.D7L else "x86 build GPU"
    return gpu.GPUSelection(
        index=0,
        name=name,
        uuid="0",
        free_memory_mib=0,
        total_memory_mib=0,
        utilization_percent=0,
        compute_capability=compute_capability,
    )


class FlowError(RuntimeError):

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclasses.dataclass(frozen=True)
class HostSSHConfig:
    host: str
    port: int = 22
    user: str | None = None
    password: str | None = None
    jump_host: HostSSHConfig | None = None

    @classmethod
    def from_mapping(cls,
                     payload: dict[str, Any],
                     context: str,
                     *,
                     allow_jump: bool = True) -> HostSSHConfig:
        allowed = {"host", "port", "user", "password", "jump_host"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                f"{context} has unknown fields: {', '.join(unknown)}")
        jump_payload = payload.get("jump_host")
        if jump_payload is not None and not allow_jump:
            raise ValueError(f"{context} cannot contain another jump_host")
        if jump_payload is not None and not isinstance(jump_payload, dict):
            raise ValueError(f"{context}.jump_host must be a JSON object")
        try:
            port = int(payload.get("port", 22))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}.port must be an integer") from error
        password = payload.get("password")
        config = cls(
            host=str(payload.get("host", "")).strip(),
            port=port,
            user=(str(payload["user"]).strip()
                  if payload.get("user") is not None else None),
            password=(str(password) if password is not None else None),
            jump_host=(cls.from_mapping(
                jump_payload, f"{context}.jump_host", allow_jump=False)
                       if jump_payload is not None else None),
        )
        config.validate(context)
        return config

    def validate(self, context: str) -> None:
        if not self.host or any(char in self.host for char in "\r\n\0"):
            raise ValueError(f"{context}.host must be set")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"{context}.port must be between 1 and 65535")
        if self.is_local:
            if self.jump_host is not None:
                raise ValueError(f"{context} localhost cannot use a jump_host")
            return
        if self.jump_host is not None:
            if self.jump_host.jump_host is not None:
                raise ValueError(
                    f"{context}.jump_host cannot contain another jump_host")
            self.jump_host.validate(f"{context}.jump_host")
        if not self.user or any(char in self.user for char in "\r\n\0"):
            raise ValueError(f"{context}.user must be set for a remote host")
        if self.password is None:
            raise ValueError(
                f"{context}.password must be set for a remote host")

    @property
    def is_local(self) -> bool:
        return self.host.lower() == "localhost"


def _read_json_object(value: str, context: str) -> dict[str, Any]:
    candidate = value.strip()
    if not candidate.startswith("{"):
        try:
            candidate = Path(candidate).expanduser().read_text()
        except OSError as error:
            raise ValueError(
                f"{context} must be a JSON object or readable JSON file"
            ) from error
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{context} contains invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return payload


def read_ssh_config(value: str, context: str) -> HostSSHConfig:
    return HostSSHConfig.from_mapping(_read_json_object(value, context),
                                      context)


def read_additional_model_cases(value: str) -> tuple[ModelCase, ...]:
    candidate = value.strip()
    if not candidate.startswith("["):
        try:
            candidate = Path(candidate).expanduser().read_text()
        except OSError as error:
            raise ValueError(
                "additional model cases must be a JSON array or readable JSON file"
            ) from error
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"additional model cases contain invalid JSON: {error.msg}"
        ) from error
    if not isinstance(payload, list):
        raise ValueError("additional model cases must contain a JSON array")
    cases = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(
                f"additional model cases[{index}] must be a JSON object")
        cases.append(
            ModelCase.from_mapping(entry, f"additional model cases[{index}]"))
    return tuple(cases)


@dataclasses.dataclass(frozen=True)
class Host:
    target: Any
    remote: Any | None = None

    @property
    def is_local(self) -> bool:
        return self.remote is None


@dataclasses.dataclass(frozen=True)
class Runtime:
    target: Any
    workspace: PurePosixPath
    edge: PurePosixPath
    trt: PurePosixPath
    onnx_root: PurePosixPath
    env_script: PurePosixPath
    local_workspace: Path | None = None
    deployment_mode: DeploymentMode | None = None
    nfs_export_path: PurePosixPath | None = None
    edge_llm_cache_root: PurePosixPath | None = None


@dataclasses.dataclass(frozen=True)
class Config:
    architecture: Arch
    trt_location: PurePosixPath
    build_host: HostSSHConfig
    run_host: HostSSHConfig
    run_id: str
    branch: str
    compute_capability: str
    onnx_root: PurePosixPath
    hf_checkpoint_root: PurePosixPath | None
    jobs: int
    source_root: Path = _SOURCE_ROOT
    download_hf_checkpoint: bool = False
    export_onnx: bool = False
    no_trt_containers: bool = False
    edge_llm_cache_root: Path | None = None
    run_workspace_root: PurePosixPath | None = None
    model_cases: tuple[ModelCase, ...] = _DEFAULT_MODEL_CASES
    cuda_root: PurePosixPath | None = None
    cuda_version: str | None = None

    cuda_dir: PurePosixPath | None = None
    cuda_target_dir: PurePosixPath | None = None
    run_python: PurePosixPath | None = None

    def validate(self) -> None:
        if self.architecture not in {Arch.X86_64, Arch.D7L}:
            raise ValueError("architecture must be x86/x86_64 or d7l")
        if not _valid_compute_capability(self.compute_capability):
            raise ValueError("compute capability must use major.minor format")
        native_x86 = (self.no_trt_containers
                      and self.architecture is Arch.X86_64)
        if native_x86 and self.cuda_root is None:
            raise ValueError(
                "--cuda-root is required for x86 --no-trt-containers")
        if native_x86 and self.cuda_version is None:
            raise ValueError(
                "--cuda-version is required for x86 --no-trt-containers")
        if self.cuda_root is not None and not _safe_path(self.cuda_root):
            raise ValueError("cuda_root must be an absolute, non-root path")

        if self.cuda_dir is not None and not _safe_path(self.cuda_dir):
            raise ValueError("cuda_dir must be an absolute, non-root path")
        if (self.cuda_target_dir is not None
                and not _safe_path(self.cuda_target_dir)):
            raise ValueError(
                "cuda_target_dir must be an absolute, non-root path")
        if (self.cuda_version is not None
                and not _CUDA_VERSION_RE.fullmatch(self.cuda_version)):
            raise ValueError("cuda_version must use major.minor format")
        if self.run_python is not None and not _safe_path(self.run_python):
            raise ValueError("run_python must be an absolute, non-root path")
        if not _safe_path(self.trt_location):
            raise ValueError(
                "trt_location must be an absolute, non-root PRE_BUILT directory path"
            )
        if not _safe_path(self.onnx_root):
            raise ValueError(
                "TRT_CI_ONNX_DIR must be an absolute, non-root path")
        if self.download_hf_checkpoint or self.export_onnx:
            if self.hf_checkpoint_root is None or not _safe_path(
                    self.hf_checkpoint_root):
                raise ValueError(
                    "TRT_CI_HF_CHECKPOINT_DIR must be an absolute, non-root path"
                )
        if self.edge_llm_cache_root is not None and not _safe_path(
                PurePosixPath(self.edge_llm_cache_root)):
            raise ValueError(
                "TRT_CI_EDGE_LLM_CACHE_DIR must be an absolute, non-root path")
        if self.run_workspace_root is not None and not _safe_path(
                self.run_workspace_root):
            raise ValueError(
                "TRT_CI_RUN_WORKSPACE_ROOT must be an absolute, non-root path")
        if self.jobs <= 0:
            raise ValueError("TRT_CI_JOBS must be a positive integer")
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("invalid internal run ID")
        if not self.source_root.is_dir():
            raise ValueError(
                f"Edge-LLM source root does not exist: {self.source_root}")
        if not self.model_cases:
            raise ValueError("at least one model case must be configured")
        duplicate_names = sorted({
            model.name
            for model in self.model_cases
            if sum(case.name == model.name for case in self.model_cases) > 1
        })
        if duplicate_names:
            raise ValueError(
                f"duplicate model case names: {', '.join(duplicate_names)}")
        self.build_host.validate("build_host")
        self.run_host.validate("run_host")

    @property
    def run_root(self) -> PurePosixPath:
        if self.run_workspace_root is not None:
            return self.run_workspace_root / f"run-{self.run_id}"
        if self.architecture is Arch.D7L and self.run_host.user:
            return (PurePosixPath("/home") / self.run_host.user /
                    "edgellm-trt-ci" / f"run-{self.run_id}")
        return _RUN_WORKSPACE / f"run-{self.run_id}"

    @property
    def runtime_root(self) -> PurePosixPath:
        return self.run_root / "runtime"

    @property
    def local_root(self) -> Path:
        root = Path(os.environ.get("TRT_CI_ARTIFACTS_DIR", "artifacts/trt-ci"))
        return root.expanduser().resolve() / f"run-{self.run_id}"

    @property
    def edgellm_root(self) -> Path:
        return self.source_root


def _valid_compute_capability(value: str) -> bool:
    major, separator, minor = value.partition(".")
    return bool(separator and major.isdigit() and minor.isdigit())


def _safe_path(path: PurePosixPath) -> bool:
    return (path.anchor == "/" and ".." not in path.parts and str(path) != "/"
            and not any(char in str(path) for char in "\r\n\0"))


def _architecture(value: str) -> Arch:
    normalized = value.strip().lower()
    try:
        arch = Arch.X86_64 if normalized == "x86" else parse_arch(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if arch not in {Arch.X86_64, Arch.D7L}:
        raise argparse.ArgumentTypeError(
            "architecture must be x86/x86_64 or d7l")
    return arch


def remote_config(connection: HostSSHConfig,
                  *,
                  local_path: Path,
                  remote_path: PurePosixPath,
                  arch: Arch = Arch.X86_64) -> RemoteConfig:
    jump = connection.jump_host
    return RemoteConfig(
        target=SSHTargetInfo(
            target_type=(TargetType.THOR_LINUX
                         if arch is Arch.D7L else TargetType.LINUX),
            arch=arch,
            address=connection.host,
            port=connection.port,
            username=connection.user,
            password=connection.password,
        ),
        jump_host=(JumpHostConfig(host=jump.host,
                                  port=jump.port,
                                  username=jump.user or "",
                                  password=jump.password,
                                  ssh=SSHConfig(
                                      timeout_s=_TRANSFER_TIMEOUT_S,
                                      connect_timeout_s=_CONNECTION_TIMEOUT_S,
                                      batch_mode=jump.password in {None, ""}))
                   if jump is not None else None),
        ssh=SSHConfig(
            timeout_s=_TRANSFER_TIMEOUT_S,
            connect_timeout_s=_CONNECTION_TIMEOUT_S,
            batch_mode=connection.password in {None, ""},
        ),
        paths=RemotePaths(local_path=str(local_path),
                          remote_path=str(remote_path)),
    )


def resolve_host(connection: HostSSHConfig,
                 commands: Any,
                 *,
                 local_path: Path,
                 remote_path: PurePosixPath,
                 arch: Arch = Arch.X86_64) -> Host:
    if connection.is_local:
        return Host(LocalTarget())
    config = remote_config(connection,
                           local_path=local_path,
                           remote_path=remote_path,
                           arch=arch)
    remote = RemoteConnectionManager(config, command_manager=commands)
    return Host(remote.target, remote)


def _cute_dsl_cmake_args(architecture: Arch) -> list[str]:
    if architecture is Arch.D7L:
        return [
            "-DENABLE_CUTE_DSL=fmha;fmha_v2;ffpa;gdn;gemm;ssd",
            "-DCUTE_DSL_ARTIFACT_TAG=sm_110",
        ]
    return ["-DENABLE_CUTE_DSL=ffpa;fmha_v2;gdn;gemm;int4_fp16_gemm;ssd"]


def build_targets(config: Config) -> list[ArtifactTarget]:
    native_x86 = (config.no_trt_containers
                  and config.architecture is Arch.X86_64)

    d7l_cuda_dir = config.cuda_dir or _DEFAULT_D7L_CUDA_DIR
    d7l_cuda_target_dir = (config.cuda_target_dir
                           or _DEFAULT_D7L_CUDA_TARGET_DIR)
    platform = PlatformConfig(
        arch=config.architecture,
        cuda_version=config.cuda_version if native_x86 else None,
    )
    platform_cmake_args = ([
        f"-DCMAKE_TOOLCHAIN_FILE={config.source_root}/cmake/aarch64_linux_toolchain.cmake",
        "-DEMBEDDED_TARGET=auto-thor",

        f"-DCUDA_DIR={d7l_cuda_dir}",
        f"-DCUDA_TARGET_DIR={d7l_cuda_target_dir}",
    ] if config.architecture is Arch.D7L else (
        [
            f"-DCMAKE_CUDA_COMPILER={config.cuda_root}/bin/nvcc",
            f"-DCUDAToolkit_ROOT={config.cuda_root}",
            f"-DCUDA_DIR={config.cuda_root}",
            f"-DCUDA_TARGET_DIR={config.cuda_root}",
        ] if native_x86 else
        ["-DCUDA_TARGET_DIR=/usr/local/cuda/targets/x86_64-linux"]))
    trt = ArtifactTarget(
        component=TRT_COMPONENT,
        mode=BuildMode.RELEASE,
        branch=config.branch,
        source=ArtifactSource.PRE_BUILT,
        platform=platform,
        build=BuildConfig(build_dir=str(config.trt_location)),
    )
    return [
        trt,
        ArtifactTarget(
            component=edgellm_code_manager.EDGELLM_COMPONENT,
            mode=BuildMode.RELEASE,
            branch=config.branch,
            source=ArtifactSource.BUILD,
            platform=platform,
            build=BuildConfig(
                repo_path=str(config.edgellm_root),
                build_dir=str(config.edgellm_root),
                log_file=str(config.local_root / "build-edgellm.log"),
                parallel_jobs=config.jobs,
                no_nvidia_runtime=True,
                trt_package_dir=str(config.trt_location),
                cmake_args=[
                    "--fresh", "-DBUILD_UNIT_TESTS=OFF",
                    *_cute_dsl_cmake_args(config.architecture),
                    *platform_cmake_args
                ],
            ),
        ),
    ]


def _test_container_pattern(run_result: Any) -> ContainerPattern:
    if run_result.plan is None:
        raise FlowError("CodeManager result is missing its execution plan")
    target = next(
        (step.target for step in run_result.plan.steps if _component_id(
            step.component) == edgellm_code_manager.EDGELLM_COMPONENT), None)
    if target is None:
        raise FlowError("CodeManager result has no Edge-LLM build step")
    platform = target.platform
    return ContainerPattern(
        kind=ContainerKind.TRT,
        arch=platform.arch,
        branch=target.branch,
        cuda_version=platform.cuda_version,
        ubuntu_version=platform.ubuntu_version,
        trt_type=platform.trt_type,
    )


def _status(result: Any) -> int:
    if result.success:
        return 0
    if getattr(result, "timed_out", False):
        return 124
    return result.exit_code or 1


def _cleanup(remote: Any, path: PurePosixPath, logger: Any) -> None:
    try:
        if not remote.filesystem.remove_dir(str(path),
                                            timeout_s=_TRANSFER_TIMEOUT_S):
            logger.warning("Could not clean run-host workspace")
    except Exception as error:
        logger.warning("Could not clean run-host workspace: %s", error)


# TODO(devtoolkit): add a CodeManager no_container option that builds source
# artifacts through CommandManager, exposes each component build command, runs
# E2E commands on the host, and emits the same runtime setup environment.
def _native_build_command(config: Config, target: ArtifactTarget) -> str:
    build = target.build
    if not build.repo_path or not build.build_dir or not build.trt_package_dir:
        raise FlowError("Native EdgeLLM build target is incomplete")
    arch = parse_arch(target.platform.arch)
    if arch is Arch.X86_64:
        if config.cuda_root is None or config.cuda_version is None:
            raise FlowError("Native x86 CUDA configuration is incomplete")
        cuda_version = config.cuda_version
        cuda_root = shlex.quote(str(config.cuda_root))
        cuda_compiler = shlex.quote(str(config.cuda_root / "bin/nvcc"))
        cuda_setup = [
            f"cuda_root={cuda_root}",
            f"export CUDACXX={cuda_compiler}",
            'test -d "$cuda_root" || { echo "CUDA root does not exist: '
            '${cuda_root}" >&2; exit 2; }',
            'test -x "$CUDACXX" || { echo "CUDA compiler is not executable: '
            '${CUDACXX}" >&2; exit 2; }',
            'cuda_release="$("$CUDACXX" --version | sed -n '
            "'s/.*release \\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p')\"",
            f'test "$cuda_release" = {shlex.quote(cuda_version)} || {{ '
            f'echo "CUDA version mismatch: expected {cuda_version}, got '
            '${cuda_release:-unknown}" >&2; exit 2; }',
            'export CUDA_HOME="$cuda_root"',
            'export CUDA_PATH="$cuda_root"',
            f"export CUDA_CTK_VERSION={shlex.quote(cuda_version)}",
        ]
    else:
        cuda_version = target.platform.cuda_version or arch.default_cuda_version
        cuda_setup = []
    configure = [
        "cmake",
        build.repo_path,
        f"-DTRT_PACKAGE_DIR={build.trt_package_dir}",
        f"-DCUDA_CTK_VERSION={cuda_version}",
        "-DBUILD_UNIT_TESTS=ON",
        f"-DCMAKE_BUILD_TYPE={BuildMode(target.mode).capitalized}",
        *build.cmake_args,
    ]
    make = [
        "make", f"-j{build.parallel_jobs or 1}", *build.targets,
        *build.make_args
    ]
    return "\n".join([
        "set -euo pipefail", *cuda_setup,
        f"mkdir -p {shlex.quote(build.build_dir)}",
        f"cd {shlex.quote(build.build_dir)}",
        shlex.join(configure),
        shlex.join(make)
    ])



def _native_build_environment() -> dict[str, str]:
    """Return activated toolchain paths that must reach the build host."""
    return {
        key: os.environ[key]
        for key in _NATIVE_BUILD_ENV_KEYS if os.environ.get(key)
    }


def _build_on_host(config: Config, commands: Any, target: Any,
                   edge: ArtifactTarget, run_result: Any) -> Any:
    platform = edge.platform
    if config.architecture is Arch.D7L:
        platform = dataclasses.replace(
            platform, cuda_version=platform.arch.default_cuda_version)
    edge = dataclasses.replace(edge,
                               platform=dataclasses.replace(
                                   platform, ubuntu_version="24.04"))
    result = commands.run(
        target,
        CommandSpec(
            name="Build EdgeLLM on host",
            command=_native_build_command(config, edge),
            shell_type=ShellType.BASH,
            cwd=edge.build.repo_path,
            env=(_native_build_environment()
                 if config.architecture is Arch.X86_64 else {}),
            timeout_s=_TEST_TIMEOUT_S,
            output_mode=OutputMode.PROGRESS,
            artifact_log_file=edge.build.log_file,
            operation_name="native-edgellm-build",
        ))
    if not result.success:
        raise FlowError("Native EdgeLLM build failed", _status(result))
    if run_result.plan is None:
        raise FlowError("CodeManager result is missing its execution plan")
    step_id = "native-edgellm"
    artifacts = dict(run_result.step_artifacts)
    artifacts[step_id] = ArtifactResult(
        success=True,
        component=edgellm_code_manager.EDGELLM_COMPONENT,
        output_dir=edge.build.build_dir,
        log_file=edge.build.log_file,
    )
    plan = dataclasses.replace(
        run_result.plan,
        steps=[
            *run_result.plan.steps,
            PlanStep(step_id, edge, edgellm_code_manager.EDGELLM_COMPONENT),
        ],
    )
    return dataclasses.replace(run_result, step_artifacts=artifacts, plan=plan)

def _build(config: Config, code: Any, build_target: Any) -> Any:
    targets = build_targets(config)
    run_result = code.plan_and_execute(
        targets[:1] if config.no_trt_containers else targets)
    if not run_result.success:
        raise FlowError("CodeManager build failed: " +
                        "; ".join(run_result.error_messages))
    if config.no_trt_containers:
        return _build_on_host(config, code.command_manager, build_target,
                              targets[1], run_result)
    return run_result


def _runtime_artifacts(config: Config, commands: Any, run_result: Any) -> Any:
    """Exclude build-only TRT archives from D7L deployment."""
    if config.architecture is not Arch.D7L:
        return run_result
    artifacts = dict(run_result.step_artifacts)
    for step_id, artifact in artifacts.items():
        if _component_id(artifact.component) != TRT_COMPONENT:
            continue
        runtime_trt = config.local_root / "runtime-trt"
        result = commands.run(
            LocalTarget(),
            CommandSpec(
                name="Stage D7L TRT runtime",
                argv=[
                    "rsync", "-a", "--delete", "--exclude=*.a",
                    f"{artifact.output_dir}/", f"{runtime_trt}/"
                ],
                cwd=str(config.edgellm_root),
                timeout_s=_TRANSFER_TIMEOUT_S,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(config.local_root /
                                      "stage-trt-runtime.log"),
                operation_name="stage-d7l-trt-runtime",
            ))
        if not result.success:
            raise FlowError("Could not stage the D7L TRT runtime",
                            _status(result))
        artifacts[step_id] = dataclasses.replace(artifact,
                                                 output_dir=str(runtime_trt))
        break
    return dataclasses.replace(run_result, step_artifacts=artifacts)


def _deploy(config: Config, code: Any, run_host: Host,
            run_result: Any) -> Runtime:
    if run_host.remote is not None:
        deployment = code.deploy_runtime(
            _runtime_artifacts(config, code.command_manager, run_result),
            run_host.remote.config,
            preferred_mode=(DeploymentMode.RSYNC
                            if config.architecture is Arch.X86_64 else None),
        )
        workspace = PurePosixPath(deployment.remote_workspace)
        onnx_root = (workspace / "onnx"
                     if config.architecture is Arch.D7L else config.onnx_root)
        return Runtime(
            target=deployment.remote_target,
            workspace=workspace,
            edge=workspace / "edgellm",
            trt=workspace / "trt",
            onnx_root=onnx_root,
            env_script=PurePosixPath(deployment.env_script_path),
            local_workspace=Path(deployment.metadata["local_workspace"]),
            deployment_mode=deployment.deployment_mode,
            nfs_export_path=(
                PurePosixPath(deployment.metadata["jump_host_export_path"])
                if "jump_host_export_path" in deployment.metadata else None),
            edge_llm_cache_root=((workspace / "edge_llm_cache")
                                 if config.edge_llm_cache_root is not None else None),
        )

    edge = PurePosixPath(str(config.edgellm_root))
    setup = code.write_environment_setup_script(
        run_result, preferred_component=edgellm_code_manager.EDGELLM_COMPONENT)
    cache_root = (PurePosixPath(config.edge_llm_cache_root)
                  if config.edge_llm_cache_root is not None else None)
    return Runtime(run_host.target, edge, edge, config.trt_location,
                   config.onnx_root, PurePosixPath(setup),
                   edge_llm_cache_root=cache_root)


def _download_hf_checkpoints(config: Config, commands: Any,
                             target: Any) -> None:
    if not config.download_hf_checkpoint:
        return
    assert config.hf_checkpoint_root is not None
    for model in config.model_cases:
        result = commands.run(
            target,
            CommandSpec(
                name=f"Download {model.name} checkpoint",
                argv=[
                    str(_PYTHON.parent / "hf"), "download", model.repository,
                    "--local-dir",
                    str(config.hf_checkpoint_root / model.checkpoint_dir)
                ],
                cwd=str(config.edgellm_root),
                timeout_s=_TRANSFER_TIMEOUT_S,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(config.local_root /
                                      f"download-{model.name}.log"),
                operation_name=f"huggingface-download-{model.name}",
            ))
        if not result.success:
            raise FlowError(
                f"HuggingFace checkpoint download failed for {model.name}",
                _status(result))


def _export_onnx(config: Config, code: Any, target: Any,
                 run_result: Any) -> None:
    if not config.export_onnx:
        return
    assert config.hf_checkpoint_root is not None
    edge = PurePosixPath(str(config.edgellm_root))
    setup = code.write_environment_setup_script(
        run_result, preferred_component=edgellm_code_manager.EDGELLM_COMPONENT)
    result = code.command_manager.run(
        target,
        CommandSpec(
            name="Export Edge-LLM ONNX models",
            command=_export_onnx_command(config, edge, PurePosixPath(setup)),
            shell_type=ShellType.BASH,
            cwd=str(edge),
            timeout_s=_TEST_TIMEOUT_S,
            output_mode=OutputMode.PROGRESS,
            artifact_log_file=str(config.local_root / "export-onnx.log"),
            operation_name="edgellm-export-onnx",
        ))
    if not result.success:
        raise FlowError("ONNX export failed", _status(result))


def _copy_onnx_cases(commands: Any, source_root: Path, destination_root: Path,
                     log_root: Path, model_cases: tuple[ModelCase,
                                                        ...]) -> None:
    for model in model_cases:
        result = commands.run(
            LocalTarget(),
            CommandSpec(
                name=f"Stage {model.name} ONNX model",
                argv=[
                    "rsync", "-a", "--delete", f"{source_root / model.name}/",
                    f"{destination_root / model.name}/"
                ],
                cwd=str(_SOURCE_ROOT),
                timeout_s=_TRANSFER_TIMEOUT_S,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(log_root /
                                      f"stage-onnx-{model.name}.log"),
                operation_name=f"stage-onnx-{model.name}",
            ))
        if not result.success:
            raise FlowError(f"Could not stage {model.name} ONNX model",
                            _status(result))


def _copy_onnx_cases_to_remote(remote: Any, source_root: Path,
                               destination_root: PurePosixPath,
                               model_cases: tuple[ModelCase, ...]) -> None:
    for model in model_cases:
        copied = remote.copy_local_directory_to_remote(
            local_path=str(source_root / model.name),
            remote_path=str(destination_root / model.name),
            timeout_s=_TRANSFER_TIMEOUT_S,
        )
        if not copied:
            raise FlowError(
                f"Could not stage {model.name} ONNX model on the D7L run host")


def _jump_host_remote_manager(commands: Any, run_host: Host, local_path: Path,
                              remote_path: PurePosixPath) -> Any | None:
    if run_host.remote is None or run_host.remote.config.jump_host is None:
        return None
    jump = run_host.remote.config.jump_host
    config = RemoteConfig(
        target=SSHTargetInfo(
            target_type=TargetType.LINUX,
            arch=Arch.X86_64,
            address=jump.host,
            port=jump.port,
            username=jump.username,
            password=jump.password,
        ),
        ssh=jump.ssh,
        paths=RemotePaths(local_path=str(local_path),
                          remote_path=str(remote_path)),
    )
    return RemoteConnectionManager(config, command_manager=commands)


def _stage_edge_llm_cache(config: Config, run_host: Host,
                          runtime: Runtime) -> None:
    if (run_host.remote is None or config.edge_llm_cache_root is None
            or runtime.edge_llm_cache_root is None):
        return
    rouge_cache = config.edge_llm_cache_root / "rouge"
    if not (rouge_cache / "rouge.py").is_file():
        return
    copied = run_host.remote.copy_local_directory_to_remote(
        local_path=str(rouge_cache),
        remote_path=str(runtime.edge_llm_cache_root / "rouge"),
        timeout_s=_TRANSFER_TIMEOUT_S,
    )
    if not copied:
        raise FlowError("Could not stage the EdgeLLM ROUGE cache")


def _stage_onnx(config: Config, code: Any, run_host: Host,
                runtime: Runtime) -> None:
    if run_host.remote is None or config.architecture is not Arch.D7L:
        return
    source_root = Path(str(config.onnx_root))
    if runtime.deployment_mode is DeploymentMode.NFS:
        if runtime.nfs_export_path is not None:
            jump_host_workspace = _jump_host_remote_manager(
                code.command_manager, run_host, source_root,
                runtime.nfs_export_path / "onnx")
            if jump_host_workspace is not None:
                _copy_onnx_cases_to_remote(jump_host_workspace, source_root,
                                           runtime.nfs_export_path / "onnx",
                                           config.model_cases)
                return
        if runtime.local_workspace is not None:
            destination_root = runtime.local_workspace / "onnx"
            _copy_onnx_cases(code.command_manager, source_root,
                             destination_root, config.local_root,
                             config.model_cases)
            return
    run_host.remote.filesystem.remove_dir(str(runtime.onnx_root),
                                          timeout_s=_TRANSFER_TIMEOUT_S)
    if not run_host.remote.filesystem.ensure_dir(
            str(runtime.onnx_root), timeout_s=_TRANSFER_TIMEOUT_S):
        raise FlowError("Could not create the D7L ONNX staging directory")
    _copy_onnx_cases_to_remote(run_host.remote, source_root, runtime.onnx_root,
                               config.model_cases)


def _edge_llm_cache_export(runtime: Runtime) -> str:
    if runtime.edge_llm_cache_root is None:
        return ""
    return f"export EDGE_LLM_CACHE_DIR={shlex.quote(str(runtime.edge_llm_cache_root))}\n"


def _run_tests(config: Config, code: Any, run_host: Host, run_result: Any,
               runtime: Runtime, logger: Any) -> int:
    if config.no_trt_containers or config.architecture is Arch.D7L:
        result = code.command_manager.run(
            runtime.target,
            CommandSpec(
                name="Run Edge-LLM Python E2E tests",
                command=_test_command(config, runtime),
                shell_type=ShellType.BASH,
                cwd=str(runtime.edge),
                timeout_s=_TEST_TIMEOUT_S,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(config.local_root / "tests.log"),
                operation_name="edgellm-python-e2e",
            ))
        status = _status(result)
        return _collect_results(config, run_host, runtime, status, logger)

    containers = code.container_manager
    mounts = [MountSpec(str(runtime.workspace), str(runtime.workspace))]
    if run_host.is_local:
        mounts.append(
            MountSpec(str(config.trt_location),
                      str(config.trt_location),
                      read_only=True))
    mounts.extend((
        MountSpec(str(runtime.onnx_root),
                  str(runtime.onnx_root),
                  read_only=True),
        MountSpec(str(_PYTHON_ROOT), str(_PYTHON_ROOT), read_only=True),
    ))
    if runtime.edge_llm_cache_root is not None:
        mounts.append(
            MountSpec(str(runtime.edge_llm_cache_root),
                      str(runtime.edge_llm_cache_root),
                      read_only=True))
    extra_args = None
    if run_host.remote is not None:
        remote_mounts = [
            dataclasses.replace(mount, read_only=False) for mount in mounts
        ]
        extra_args = ["--mounts", format_mounts_csv(remote_mounts)]
        mounts = None
    handle = None
    status = 1
    try:
        descriptor = containers.resolve(_test_container_pattern(run_result))
        handle = containers.launch(
            descriptor=descriptor,
            name=f"trt-ci-edgellm-{config.run_id}",
            mounts=mounts,
            workdir=str(runtime.edge),
            extra_args=extra_args,
            exec_target=runtime.target,
        )
        status = 0 if containers.exec_progress(
            handle,
            command=_test_command(config, runtime),
            cwd=str(runtime.edge),
            timeout_s=_TEST_TIMEOUT_S,
            tee_file=str(config.local_root / "tests.log"),
            exec_target=runtime.target,
        ) else 1
    finally:
        if handle is not None:
            try:
                if not containers.remove(
                        handle, force=True, exec_target=runtime.target):
                    logger.warning("Could not remove Edge-LLM test container")
                    status = 1
            except Exception as error:
                logger.warning("Could not remove Edge-LLM test container: %s",
                               error)
                status = 1
    return _collect_results(config, run_host, runtime, status, logger)


def _collect_results(config: Config, run_host: Host, runtime: Runtime,
                     status: int, logger: Any) -> int:
    if run_host.remote is None:
        return status
    copied = run_host.remote.copy_remote_directory_to_local(
        remote_path=str(runtime.workspace / "results"),
        local_path=str(config.local_root / "results"),
        timeout_s=_TRANSFER_TIMEOUT_S,
        operation_name="collect-e2e-results",
    )
    if not copied:
        logger.warning("Could not collect remote E2E results")
        return status or 1
    return status


def _export_onnx_command(config: Config, edge: PurePosixPath,
                         env_script: PurePosixPath) -> str:
    results = config.local_root / "results"
    export_tests = edge / "tests/defs/test_checkpoint_export.py"
    q = shlex.quote
    assert config.hf_checkpoint_root is not None
    export_cases = " ".join(f"--test-param={q(model.name + '-fp16')}"
                            for model in config.model_cases)
    generated_dirs = " ".join(
        q(str(config.onnx_root / model.name / "llm-fp16-fp16"))
        for model in config.model_cases)
    return f"""set -euo pipefail
source {q(str(env_script))}
export LLM_SDK_DIR={q(str(edge))}
export LLM_MODELS_DIR={q(str(config.hf_checkpoint_root))}
export ONNX_DIR={q(str(config.onnx_root))}
export BUILD_DIR={q(str(edge))}
export TEST_LOG_DIR={q(str(results / 'logs'))}
export TRT_PACKAGE_DIR={q(str(config.trt_location))}
rm -rf {generated_dirs}
mkdir -p {q(str(results))} {q(str(config.onnx_root))}
{q(str(_PYTHON))} -m pytest -q {q(str(export_tests))}::test_checkpoint_export \
  {export_cases} --junitxml={q(str(results / 'e2e-export.xml'))} \
  2>&1 | tee {q(str(results / 'e2e-export.log'))}
"""


def _test_command(config: Config, runtime: Runtime) -> str:
    results = runtime.workspace / "results"
    tests = runtime.edge / "tests/defs/test_llm_pipeline.py"
    plugin = runtime.edge / "libNvInfer_edgellm_plugin.so"
    q = shlex.quote
    build_cases = " ".join(f"--test-param={q(model.pipeline_param)}"
                           for model in config.model_cases)
    inference_cases = " ".join(
        f"--test-param={q(model.pipeline_param + '-llm_basic')}"
        for model in config.model_cases)
    use_host_python = (config.no_trt_containers
                       or config.architecture is Arch.D7L)
    python = (str(config.run_python)
              if use_host_python and config.run_python is not None else
              ("python3" if use_host_python else str(_PYTHON)))
    python_path = "" if use_host_python else f"export PATH={q(str(_PYTHON.parent))}:$PATH\n"
    return f"""set -euo pipefail
source {q(str(runtime.env_script))}
export EDGELLM_PLUGIN_PATH={q(str(plugin))}
{_edge_llm_cache_export(runtime)}export HF_HOME={q(str(results / 'hf_home'))}
export HF_MODULES_CACHE={q(str(results / 'hf_modules'))}
{python_path}export LLM_SDK_DIR={q(str(runtime.edge))}
export ONNX_DIR={q(str(runtime.onnx_root))}
export ENGINE_DIR={q(str(runtime.workspace / 'engines'))}
export BUILD_DIR={q(str(runtime.edge))}
export TEST_LOG_DIR={q(str(results / 'logs'))}
export TRT_PACKAGE_DIR={q(str(runtime.trt))}
rm -rf {q(str(results))}
mkdir -p {q(str(results))}
{q(python)} -m pytest -q {q(str(tests))}::TestLLMPipeline::test_engine_build \
  {build_cases} --junitxml={q(str(results / 'e2e-build.xml'))} \
  2>&1 | tee {q(str(results / 'e2e-build.log'))}
{q(python)} -m pytest -q {q(str(tests))}::TestLLMPipeline::test_inference \
  {inference_cases} --junitxml={q(str(results / 'e2e-inference.xml'))} \
  2>&1 | tee {q(str(results / 'e2e-inference.log'))}
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture",
                        type=_architecture,
                        help="x86/x86_64 or d7l")
    parser.add_argument(
        "trt_location",
        type=PurePosixPath,
        help="absolute TRT PRE_BUILT directory on the build host")
    parser.add_argument(
        "build_host",
        help="inline JSON object or JSON file path for the build host")
    parser.add_argument(
        "run_host",
        help="inline JSON object or JSON file path for the run host")
    parser.add_argument(
        "--compute-capability",
        help=("GPU compute capability used for CuTeDSL artifact selection, "
              "for example 8.0, 8.9, or 11.0"),
    )
    parser.add_argument(
        "--no-trt-containers",
        action="store_true",
        help="run the EdgeLLM build and E2E tests without TRT containers",
    )
    parser.add_argument(
        "--cuda-root",
        type=PurePosixPath,
        help=("CUDA toolkit root on the build host; required for x86 "
              "--no-trt-containers (env: TRT_CI_CUDA_ROOT)"),
    )
    parser.add_argument(
        "--cuda-version",
        help=("CUDA toolkit major.minor version; required for x86 "
              "--no-trt-containers (env: TRT_CI_CUDA_VERSION)"),
    )

    parser.add_argument(
        "--cuda-dir",
        type=PurePosixPath,
        help=("CUDA toolkit directory for D7L cross-builds "
              "(env: TRT_CI_CUDA_DIR)"),
    )
    parser.add_argument(
        "--cuda-target-dir",
        type=PurePosixPath,
        help=("CUDA platform target directory for D7L cross-builds "
              "(env: TRT_CI_CUDA_TARGET_DIR)"),
    )
    parser.add_argument(
        "--download_hf_checkpoint",
        action="store_true",
        help="download configured HuggingFace checkpoints before ONNX export")
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="export ONNX models from HF checkpoints on the build host")
    parser.add_argument(
        "--additional-model-cases",
        default="[]",
        help=("JSON array, or JSON file path, of extra model cases with "
              "name, repository, checkpoint_dir, and pipeline_param"),
    )
    parser.add_argument(
        "--run-python",
        type=PurePosixPath,
        help="absolute Python executable on the run host for no-container runs",
    )
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser


def _config(args: argparse.Namespace) -> Config:
    try:
        jobs = int(os.environ.get("TRT_CI_JOBS", str(_JOBS)))
    except ValueError as error:
        raise ValueError("TRT_CI_JOBS must be a positive integer") from error
    branch = os.environ.get("TRT_CI_BRANCH") or os.environ.get(
        "CI_COMMIT_REF_NAME") or "main"
    if not branch or any(char in branch for char in "\r\n\0"):
        raise ValueError("invalid TRT_CI_BRANCH")
    compute_capability = (args.compute_capability
                          or ("11.0" if args.architecture is Arch.D7L else "8.0"))
    if not _valid_compute_capability(compute_capability):
        raise ValueError("compute capability must use major.minor format")
    cache_dir = (os.environ.get("TRT_CI_EDGE_LLM_CACHE_DIR")
                 or os.environ.get("EDGE_LLM_CACHE_DIR"))
    if cache_dir is None and Path("/home/edge_llm_cache/rouge/rouge.py").is_file():
        cache_dir = "/home/edge_llm_cache"
    cuda_root = args.cuda_root
    if cuda_root is None and os.environ.get("TRT_CI_CUDA_ROOT"):
        cuda_root = PurePosixPath(os.environ["TRT_CI_CUDA_ROOT"])

    cuda_dir = args.cuda_dir
    if cuda_dir is None and os.environ.get("TRT_CI_CUDA_DIR"):
        cuda_dir = PurePosixPath(os.environ["TRT_CI_CUDA_DIR"])
    cuda_target_dir = args.cuda_target_dir
    if cuda_target_dir is None and os.environ.get("TRT_CI_CUDA_TARGET_DIR"):
        cuda_target_dir = PurePosixPath(os.environ["TRT_CI_CUDA_TARGET_DIR"])
    onnx_dir = os.environ.get("TRT_CI_ONNX_DIR")
    if onnx_dir is None:
        raise ValueError("TRT_CI_ONNX_DIR must be set")
    if not onnx_dir or any(char in onnx_dir for char in "\r\n\0"):
        raise ValueError("invalid TRT_CI_ONNX_DIR")
    hf_checkpoint_dir = os.environ.get("TRT_CI_HF_CHECKPOINT_DIR")
    workspace_root = os.environ.get("TRT_CI_RUN_WORKSPACE_ROOT")
    if cache_dir is not None:
        if not cache_dir or any(char in cache_dir for char in "\r\n\0"):
            raise ValueError("invalid TRT_CI_EDGE_LLM_CACHE_DIR")
    if workspace_root is not None:
        if not workspace_root or any(char in workspace_root
                                     for char in "\r\n\0"):
            raise ValueError("invalid TRT_CI_RUN_WORKSPACE_ROOT")
    if args.download_hf_checkpoint or args.export_onnx:
        if hf_checkpoint_dir is None:
            raise ValueError("TRT_CI_HF_CHECKPOINT_DIR must be set")
        if not hf_checkpoint_dir or any(char in hf_checkpoint_dir
                                        for char in "\r\n\0"):
            raise ValueError("invalid TRT_CI_HF_CHECKPOINT_DIR")
    model_cases = _DEFAULT_MODEL_CASES + read_additional_model_cases(
        args.additional_model_cases)
    config = Config(
        architecture=args.architecture,
        trt_location=args.trt_location,
        build_host=read_ssh_config(args.build_host, "build_host"),
        run_host=read_ssh_config(args.run_host, "run_host"),
        run_id=args.run_id or uuid.uuid4().hex[:12],
        branch=branch,
        compute_capability=compute_capability,
        onnx_root=PurePosixPath(onnx_dir),
        hf_checkpoint_root=(PurePosixPath(hf_checkpoint_dir)
                            if hf_checkpoint_dir is not None else None),
        jobs=jobs,
        download_hf_checkpoint=args.download_hf_checkpoint,
        export_onnx=args.export_onnx,
        no_trt_containers=args.no_trt_containers,
        edge_llm_cache_root=(Path(cache_dir) if cache_dir else None),
        run_workspace_root=(PurePosixPath(workspace_root)
                            if workspace_root is not None else None),
        model_cases=model_cases,
        cuda_root=cuda_root,
        cuda_version=(args.cuda_version
                      or os.environ.get("TRT_CI_CUDA_VERSION")),

        cuda_dir=cuda_dir,
        cuda_target_dir=cuda_target_dir,
        run_python=args.run_python,
    )
    config.validate()
    return config


def _report_status(config: Config, logger: Any, status: int) -> int:
    outcome = "PASSED" if status == 0 else "FAILED"
    message = f"TRT CI validation {outcome}: run={config.run_id} artifacts={config.local_root}"
    (logger.info if status == 0 else logger.error)(message)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        config = _config(
            parser.parse_args(sys.argv[1:] if argv is None else argv))
    except ValueError as error:
        parser.error(str(error))

    config.local_root.mkdir(parents=True, exist_ok=True)
    configure_logging(str(config.local_root / "run.log"), log_level="INFO")
    logger = get_logger(__name__)
    commands = CommandManager()
    try:
        build_host = resolve_host(
            config.build_host,
            commands,
            local_path=config.edgellm_root,
            remote_path=PurePosixPath(str(config.edgellm_root)),
            arch=Arch.X86_64,
        )
        run_host = resolve_host(
            config.run_host,
            commands,
            local_path=config.local_root / "deploy-stage",
            remote_path=config.runtime_root,
            arch=config.architecture,
        )
    except ValueError as error:
        parser.error(str(error))

    git_trt = GitTrtRuncBackend(
        commands,
        config=GitTrtRuncConfig(extra_launch_args=["--skip-branch-check"]))
    containers = ContainerManager(command_manager=commands,
                                  git_trt_backend=git_trt)
    code = CodeManager(
        command_manager=commands,
        container_manager=containers,
        work_dir=str(config.local_root / "code-manager"),
        keep_containers_running=False,
        exec_target=build_host.target,
        remote_connection_manager=run_host.remote,
        enable_high_core_auto=False,
    )
    edgellm_code_manager.EdgeLlmArtifactGenerator.register_with_code_manager(
        code,
        container_manager=containers,
        default_exec_target=build_host.target,
        gpu_selection=_gpu_selection_for_arch(config.architecture,
                                               config.compute_capability),
    )
    try:
        run_result = _build(config, code, build_host.target)
        _download_hf_checkpoints(config, commands, build_host.target)
        _export_onnx(config, code, build_host.target, run_result)
        runtime = _deploy(config, code, run_host, run_result)
        _stage_onnx(config, code, run_host, runtime)
        _stage_edge_llm_cache(config, run_host, runtime)
        status = _run_tests(config, code, run_host, run_result, runtime,
                            logger)
    except FlowError as error:
        logger.error("%s", error)
        status = error.exit_code
    except Exception as error:
        logger.exception("TRT CI validation failed: %s", error)
        status = 1
    if run_host.remote is not None:
        _cleanup(run_host.remote, config.run_root, logger)
    return _report_status(config, logger, status)


if __name__ == "__main__":
    raise SystemExit(main())
