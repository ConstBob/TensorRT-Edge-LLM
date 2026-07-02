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
"""Build Edge-LLM against a TRT candidate and run its C++ tests on D7L."""

from __future__ import annotations

import argparse
import dataclasses
import re
import shlex
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from trt_dev_toolkit.code_manager import (ArtifactResult, ArtifactSource,
                                          ArtifactTarget, BuildComponent,
                                          BuildMode, CodeManager,
                                          DeploymentMode, RunResult)
from trt_dev_toolkit.code_manager.models import BuildConfig, PlatformConfig
from trt_dev_toolkit.command_manager.command_manager import CommandManager
from trt_dev_toolkit.command_manager.data_structures import (CommandSpec,
                                                             OutputMode,
                                                             ShellType)
from trt_dev_toolkit.command_manager.targets import LocalTarget
from trt_dev_toolkit.constants import Arch, TargetType
from trt_dev_toolkit.container_manager import ContainerManager
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

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_LOG = get_logger(__name__)


class FlowError(RuntimeError):

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclasses.dataclass(frozen=True)
class Endpoint:
    host: str
    user: str
    port: int = 22
    jump: str | None = None


@dataclasses.dataclass(frozen=True)
class CiConfig:
    trt_root: PurePosixPath
    trt_build_dir: PurePosixPath
    trt_branch: str
    build_endpoint: Endpoint
    test_endpoint: Endpoint
    workspace: PurePosixPath
    run_id: str
    artifacts_dir: Path
    cuda_version: str | None = None
    jobs: int = 16
    gtest_filter: str = "*"
    connection_timeout_s: int = 30
    transfer_timeout_s: float = 1800
    build_timeout_s: float = 7200
    test_timeout_s: float = 3600
    strict_host_keys: bool = True
    known_hosts_file: str | None = None
    keep_workspace: bool = False
    build_python: str = "python3.12"
    source_root: Path = _SOURCE_ROOT

    @property
    def run_root(self) -> PurePosixPath:
        return self.workspace / f"run-{self.run_id}"

    @property
    def local_root(self) -> Path:
        return self.artifacts_dir / f"run-{self.run_id}"

    def remote(self, name: str) -> PurePosixPath:
        return self.run_root / name

    def validate(self) -> None:
        remote_paths = (self.workspace, self.trt_root, self.trt_build_dir)
        if any(path.anchor != "/" or ".." in path.parts or str(path) == "/"
               or any(c in str(path) for c in "\r\n\0")
               for path in remote_paths):
            raise ValueError(
                "workspace and TRT paths must be canonical absolute, non-root paths"
            )
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(
                "run ID may contain only letters, digits, '.', '_' and '-'")
        for endpoint in (self.build_endpoint, self.test_endpoint):
            text = endpoint.host + endpoint.user
            if (not endpoint.host or not endpoint.user
                    or endpoint.host.startswith("-")
                    or endpoint.user.startswith("-") or any(c in text
                                                            for c in "\r\n\0")
                    or not 1 <= endpoint.port <= 65535):
                raise ValueError("invalid SSH endpoint")
        values = (self.jobs, self.connection_timeout_s,
                  self.transfer_timeout_s, self.build_timeout_s,
                  self.test_timeout_s)
        if any(value <= 0 for value in values):
            raise ValueError("jobs and timeouts must be positive")
        if not self.source_root.is_dir():
            raise ValueError(
                f"Edge-LLM source root does not exist: {self.source_root}")


@dataclasses.dataclass
class ControllerServices:
    commands: Any
    build_remote: Any
    test_remote: Any
    code: Any


@dataclasses.dataclass
class WorkerServices:
    commands: Any
    code: Any


def _status(result: Any) -> int:
    if result.success:
        return 0
    if getattr(result, "timed_out", False):
        return 124
    return result.exit_code if result.exit_code is not None else 1


def _jump(value: str | None, ssh: SSHConfig) -> JumpHostConfig | None:
    if value is None:
        return None
    if value.startswith("-") or any(c in value for c in "\r\n\0"):
        raise ValueError("invalid jump host")
    user, has_user, host_port = value.rpartition("@")
    if not has_user:
        user, host_port = "", value
    host, has_port, port_text = host_port.rpartition(":")
    if not has_port or not port_text.isdigit():
        host, port = host_port, 22
    else:
        port = int(port_text)
    if not host or host.startswith("-") or not 1 <= port <= 65535:
        raise ValueError("jump host must use [user@]host[:port]")
    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "password": None,
        "ssh": ssh
    }
    if user:
        kwargs["username"] = user
    return JumpHostConfig(**kwargs)


def remote_config(config: CiConfig, endpoint: Endpoint,
                  target_type: TargetType, arch: Arch,
                  local_path: Path) -> RemoteConfig:
    ssh = SSHConfig(
        timeout_s=config.transfer_timeout_s,
        connect_timeout_s=config.connection_timeout_s,
        strict_host_key_checking=config.strict_host_keys,
        known_hosts_file=config.known_hosts_file,
        batch_mode=True,
    )
    return RemoteConfig(
        target=SSHTargetInfo(
            target_type=target_type,
            arch=arch,
            address=endpoint.host,
            port=endpoint.port,
            username=endpoint.user,
            password="",
        ),
        ssh=ssh,
        jump_host=_jump(endpoint.jump, ssh),
        paths=RemotePaths(local_path=str(local_path),
                          remote_path=str(config.run_root)),
    )


def build_targets(run_root: PurePosixPath, trt_root: PurePosixPath,
                  branch: str, cuda_version: str | None,
                  jobs: int) -> list[ArtifactTarget]:
    source, build, package = (run_root / name
                              for name in ("source", "build", "trt-package"))
    platform = PlatformConfig(arch=Arch.D7L, cuda_version=cuda_version)
    return [
        ArtifactTarget(
            component=BuildComponent.TRT,
            mode=BuildMode.RELEASE,
            branch=branch,
            source=ArtifactSource.PRE_BUILT,
            platform=platform,
            build=BuildConfig(build_dir=str(package), repo_path=str(trt_root)),
        ),
        ArtifactTarget(
            component=BuildComponent.EDGELLM,
            mode=BuildMode.RELEASE,
            branch=branch,
            source=ArtifactSource.BUILD,
            platform=platform,
            build=BuildConfig(
                repo_path=str(source),
                build_dir=str(build),
                log_file=str(run_root / "artifacts/build-edgellm.log"),
                parallel_jobs=jobs,
                cmake_args=[
                    "-DEMBEDDED_TARGET=auto-thor",
                    f"-DCMAKE_TOOLCHAIN_FILE={source}/cmake/aarch64_linux_toolchain.cmake",
                    "-DENABLE_CUTE_DSL=OFF",
                ],
            ),
        ),
    ]


def deployment_result(manager: Any, trt_dir: Path, edgellm_dir: Path,
                      branch: str, cuda_version: str | None) -> RunResult:
    platform = PlatformConfig(arch=Arch.D7L, cuda_version=cuda_version)
    outputs = {
        BuildComponent.TRT: str(trt_dir.resolve()),
        BuildComponent.EDGELLM: str(edgellm_dir.resolve()),
    }
    plan = manager.plan_artifact_generation([
        ArtifactTarget(
            component=component,
            mode=BuildMode.RELEASE,
            branch=branch,
            source=ArtifactSource.PRE_BUILT,
            platform=platform,
            build=BuildConfig(build_dir=path),
        ) for component, path in outputs.items()
    ])
    return RunResult(
        success=True,
        plan=plan,
        step_artifacts={
            step.step_id:
            ArtifactResult(
                success=True,
                component=step.component.value,
                output_dir=outputs[step.component],
            )
            for step in plan.steps
        },
    )


class BuildWorker:

    def __init__(self, args: argparse.Namespace,
                 services: WorkerServices) -> None:
        self.args = args
        self.services = services

    def run(self) -> int:
        artifacts = Path(str(self.args.run_root / "artifacts"))
        artifacts.mkdir(parents=True, exist_ok=True)
        result = self.services.commands.run(LocalTarget(),
                                            self._package_spec(artifacts))
        if not result.success:
            return _status(result)
        build = self.services.code.plan_and_execute(
            build_targets(self.args.run_root, self.args.trt_root,
                          self.args.trt_branch, self.args.cuda_version,
                          self.args.jobs))
        if not build.success:
            _LOG.error("CodeManager Edge-LLM build failed: %s",
                       "; ".join(build.error_messages))
            return 1
        return 0

    def _package_spec(self, artifacts: Path) -> CommandSpec:
        q = shlex.quote
        command = f"""set -euo pipefail
trt_root={q(str(self.args.trt_root))}
trt_build={q(str(self.args.trt_build_dir))}
package={q(str(self.args.run_root / "trt-package"))}
rm -rf -- "$package"
mkdir -p "$package/include" "$package/lib"
test -d "$trt_root/include" && test -d "$trt_root/parsers/onnx"
test -f "$trt_build/include/NvInferVersion.h" && test -d "$trt_build/Release/lib"
cp -a "$trt_root/include/." "$package/include/"
cp -a "$trt_root/parsers/onnx/." "$package/include/"
cp -a "$trt_build/include/NvInferVersion.h" "$package/include/"
cp -a "$trt_build/Release/lib/." "$package/lib/"
test -f "$package/include/NvInfer.h"
test -f "$package/include/NvInferVersion.h"
test -f "$package/include/NvOnnxParser.h"
find "$package/lib" -maxdepth 1 -name 'libnvinfer.so*' -print -quit | grep -q .
find "$package/lib" -maxdepth 1 -name 'libnvonnxparser.so*' -print -quit | grep -q .
while IFS= read -r -d '' link; do
  resolved="$(readlink -f "$link")"
  test -e "$resolved"
  case "$resolved" in "$package"/lib/*) ;; *) exit 1 ;; esac
done < <(find "$package/lib" -maxdepth 1 -type l -name 'lib*.so*' -print0)
"""
        return CommandSpec(
            name="Prepare TensorRT PRE_BUILT package",
            command=command,
            shell_type=ShellType.BASH,
            timeout_s=self.args.package_timeout,
            output_mode=OutputMode.PROGRESS,
            artifact_log_file=str(artifacts / "prepare-trt-package.log"),
            operation_name="prepare-trt-package",
        )


class ControllerFlow:

    def __init__(self, config: CiConfig, services: ControllerServices) -> None:
        self.config, self.services = config, services
        self.deployment: Any | None = None
        self.worker_started = self.test_reached = False

    def run(self) -> int:
        status = 1
        self.config.local_root.mkdir(parents=True, exist_ok=True)
        try:
            self._prepare()
            self._stage_source()
            self._run_worker()
            trt_dir, edge_dir = self._retrieve()
            run = deployment_result(self.services.code, trt_dir, edge_dir,
                                    self.config.trt_branch,
                                    self.config.cuda_version)
            self.deployment = self.services.code.deploy_runtime(
                run,
                self.services.test_remote.config,
                preferred_mode=DeploymentMode.RSYNC)
            self._stage_resources()
            self.test_reached = True
            status = _status(
                self.services.commands.run(self.deployment.remote_target,
                                           self._test_spec()))
        except FlowError as error:
            _LOG.error("%s", error)
            status = error.exit_code
        except Exception as error:
            _LOG.exception("TRT CI D7L validation failed: %s", error)
        finally:
            self._collect()
            if status == 0 and not self.config.keep_workspace:
                self._cleanup()
        return status

    def _prepare(self) -> None:
        for name, remote in (("build", self.services.build_remote),
                             ("D7L", self.services.test_remote)):
            if not remote.test_connection():
                raise FlowError(f"cannot connect to {name} host")
            if not remote.filesystem.remove_dir(
                    str(self.config.run_root),
                    timeout_s=self.config.connection_timeout_s):
                raise FlowError(f"cannot reset {name} workspace")
        if not self.services.build_remote.filesystem.ensure_dir(
                str(self.config.remote("source")),
                timeout_s=self.config.connection_timeout_s):
            raise FlowError("cannot create build-host source workspace")
        probes = (
            (self.services.build_remote.target, "build-host-prerequisites",
             f"command -v timeout git cp find grep {shlex.quote(self.config.build_python)} >/dev/null"
             ),
            (self.services.test_remote.target, "d7l-prerequisites",
             "test \"$(uname -m)\" = aarch64 && "
             "test -f /etc/nvidia/version-ubuntu-rootfs.txt && "
             "tr '\\0' '\\n' </proc/device-tree/compatible | grep -qi 'nvidia,tegra264' && "
             "command -v bash ldd readlink awk tee grep tr >/dev/null"),
        )
        for target, name, command in probes:
            result = self.services.commands.run(
                target,
                CommandSpec(name=name,
                            command=command,
                            shell_type=ShellType.BASH,
                            timeout_s=self.config.connection_timeout_s,
                            output_mode=OutputMode.CAPTURE,
                            operation_name=name))
            if not result.success:
                raise FlowError(f"{name} probe failed", _status(result))

    def _stage_source(self) -> None:
        stage = self.config.local_root / "source-stage"
        stage.mkdir(parents=True, exist_ok=True)
        excludes = [
            "/.git", "/.worktrees", "/codex-logs", "/build", "/artifacts",
            "__pycache__", "*.pyc"
        ]
        try:
            relative = self.config.local_root.resolve().relative_to(
                self.config.source_root.resolve())
            if relative.parts:
                excludes.append(f"/{relative.parts[0]}")
        except ValueError:
            pass
        argv = ["rsync", "-a", "--delete", "--delete-excluded"]
        argv.extend(f"--exclude={item}" for item in excludes)
        argv.extend((f"{self.config.source_root}/", f"{stage}/"))
        result = self.services.commands.run(
            LocalTarget(),
            CommandSpec(
                name="Stage Edge-LLM source",
                argv=argv,
                timeout_s=self.config.transfer_timeout_s,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(self.config.local_root /
                                      "stage-source.log"),
                operation_name="stage-source",
            ))
        if not result.success:
            raise FlowError("local source staging failed", _status(result))
        if not self.services.build_remote.copy_local_directory_to_remote(
                local_path=str(stage),
                remote_path=str(self.config.remote("source")),
                timeout_s=self.config.transfer_timeout_s):
            raise FlowError("source upload failed")

    def _run_worker(self) -> None:
        argv = [
            "timeout",
            "--signal=TERM",
            "--kill-after=60",
            f"{self.config.build_timeout_s:g}s",
            self.config.build_python,
            str(self.config.remote("source/scripts/run_trt_ci_d7l.py")),
            "--build-worker",
            "--run-root",
            str(self.config.run_root),
            "--trt-root",
            str(self.config.trt_root),
            "--trt-build-dir",
            str(self.config.trt_build_dir),
            "--trt-branch",
            self.config.trt_branch,
            "--jobs",
            str(self.config.jobs),
            "--package-timeout",
            str(self.config.transfer_timeout_s),
        ]
        if self.config.cuda_version:
            argv.extend(("--cuda-version", self.config.cuda_version))
        self.worker_started = True
        result = self.services.commands.run(
            self.services.build_remote.target,
            CommandSpec(
                name="Build Edge-LLM on build host",
                argv=argv,
                env={
                    "PYTHONPATH":
                    f"{self.config.trt_root}/scripts/devToolkit/src"
                },
                timeout_s=self.config.build_timeout_s + 120,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(self.config.local_root /
                                      "build-worker.log"),
                operation_name="build-worker",
            ))
        if not result.success:
            raise FlowError("build-host worker failed", _status(result))

    def _retrieve(self) -> tuple[Path, Path]:
        root = self.config.local_root / "runtime"
        paths = (("trt-package", root / "trt"), ("build", root / "edgellm"))
        for remote_name, local_path in paths:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.services.build_remote.copy_remote_directory_to_local(
                    remote_path=str(self.config.remote(remote_name)),
                    local_path=str(local_path),
                    timeout_s=self.config.transfer_timeout_s):
                raise FlowError(f"failed to retrieve {remote_name}")
        return paths[0][1], paths[1][1]

    def _stage_resources(self) -> None:
        resources = [(self.config.source_root / "unittests/resources",
                      self.config.remote("source/unittests/resources"))]
        templates = self.config.source_root / "tests/chat_templates"
        if templates.is_dir():
            resources.append(
                (templates, self.config.remote("source/tests/chat_templates")))
        for local, remote in resources:
            if (not local.is_dir()
                    or not self.services.test_remote.filesystem.ensure_dir(
                        str(remote),
                        timeout_s=self.config.connection_timeout_s)
                    or not self.services.test_remote.
                    copy_local_directory_to_remote(
                        local_path=str(local),
                        remote_path=str(remote),
                        timeout_s=self.config.transfer_timeout_s)):
                raise FlowError(f"failed to stage test resource: {local}")

    def _test_spec(self) -> CommandSpec:
        assert self.deployment is not None
        workspace = PurePosixPath(self.deployment.remote_workspace)
        unit, builder = workspace / "edgellm/unitTest", workspace / "edgellm/examples/llm/llm_build"
        candidate, results = workspace / "trt", self.config.remote("results")
        q = shlex.quote
        command = f"""set -euo pipefail
source {q(self.deployment.env_script_path)}
mkdir -p {q(str(results))}
check_trt() {{
  binary="$1"; shift
  test -x "$binary"
  output="$(ldd "$binary")"; printf '%s\\n' "$output"
  ! grep -q 'not found' <<<"$output"
  for library in "$@"; do
    resolved="$(awk -v name="$library" '$1 ~ ("^" name "\\\\.so") {{
      sub(/^[^=]*=>[[:space:]]*/, "")
      sub(/[[:space:]]+[(]0x[[:xdigit:]]+[)]$/, "")
      print; exit
    }}' <<<"$output")"
    test -n "$resolved"
    case "$(readlink -f "$resolved")" in {q(str(candidate))}/*) ;; *) exit 1 ;; esac
  done
}}
check_trt {q(str(unit))} libnvinfer
check_trt {q(str(builder))} libnvinfer libnvonnxparser
set +e
{q(str(unit))} --gtest_filter={q(self.config.gtest_filter)} \
  --gtest_output=xml:{q(str(results / 'unit-tests.xml'))} 2>&1 | tee {q(str(results / 'unit-tests.log'))}
status=${{PIPESTATUS[0]}}; set -e; exit "$status"
"""
        return CommandSpec(
            name="Run Edge-LLM D7L unit tests",
            command=command,
            shell_type=ShellType.BASH,
            cwd=str(workspace / "edgellm"),
            timeout_s=self.config.test_timeout_s,
            output_mode=OutputMode.PROGRESS,
            artifact_log_file=str(self.config.local_root / "unit-tests.log"),
            operation_name="d7l-unit-tests",
        )

    def _collect(self) -> None:
        copies = []
        if self.worker_started:
            copies.append(
                (self.services.build_remote, self.config.remote("artifacts"),
                 self.config.local_root / "build-logs"))
        if self.test_reached:
            copies.append(
                (self.services.test_remote, self.config.remote("results"),
                 self.config.local_root / "test-results"))
        for remote, source, target in copies:
            try:
                if not remote.copy_remote_directory_to_local(
                        remote_path=str(source),
                        local_path=str(target),
                        timeout_s=self.config.transfer_timeout_s):
                    _LOG.warning("Could not collect %s", source)
            except Exception as error:
                _LOG.warning("Could not collect %s: %s", source, error)

    def _cleanup(self) -> None:
        for name, remote in (("build", self.services.build_remote),
                             ("D7L", self.services.test_remote)):
            try:
                if not remote.filesystem.remove_dir(
                        str(self.config.run_root),
                        timeout_s=self.config.connection_timeout_s):
                    _LOG.warning("Could not clean %s workspace", name)
            except Exception as error:
                _LOG.warning("Could not clean %s workspace: %s", name, error)


def _controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-root", required=True, type=PurePosixPath)
    parser.add_argument("--trt-build-dir", required=True, type=PurePosixPath)
    parser.add_argument("--trt-branch", default="main")
    for prefix in ("build", "test"):
        parser.add_argument(f"--{prefix}-host", required=True)
        parser.add_argument(f"--{prefix}-user", required=True)
        parser.add_argument(f"--{prefix}-port", type=int, default=22)
        parser.add_argument(f"--{prefix}-jump-host", help="[user@]host[:port]")
    parser.add_argument("--workspace",
                        type=PurePosixPath,
                        default=PurePosixPath("/tmp/edgellm-trt-ci"))
    parser.add_argument("--run-id")
    parser.add_argument("--artifacts-dir",
                        type=Path,
                        default=Path("artifacts/trt-ci-d7l"))
    parser.add_argument("--cuda-version")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--gtest-filter", default="*")
    parser.add_argument("--connection-timeout", type=int, default=30)
    parser.add_argument("--transfer-timeout", type=float, default=1800)
    parser.add_argument("--build-timeout", type=float, default=7200)
    parser.add_argument("--test-timeout", type=float, default=3600)
    parser.add_argument("--host-key-policy",
                        choices=("yes", "no"),
                        default="yes")
    parser.add_argument("--known-hosts-file")
    parser.add_argument("--build-python", default="python3.12")
    parser.add_argument("--keep-workspace", action="store_true")
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-worker", action="store_true")
    parser.add_argument("--run-root", required=True, type=PurePosixPath)
    parser.add_argument("--trt-root", required=True, type=PurePosixPath)
    parser.add_argument("--trt-build-dir", required=True, type=PurePosixPath)
    parser.add_argument("--trt-branch", required=True)
    parser.add_argument("--cuda-version")
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--package-timeout", required=True, type=float)
    return parser


def _config(args: argparse.Namespace) -> CiConfig:
    config = CiConfig(
        trt_root=args.trt_root,
        trt_build_dir=args.trt_build_dir,
        trt_branch=args.trt_branch,
        build_endpoint=Endpoint(args.build_host, args.build_user,
                                args.build_port, args.build_jump_host),
        test_endpoint=Endpoint(args.test_host, args.test_user, args.test_port,
                               args.test_jump_host),
        workspace=args.workspace,
        run_id=args.run_id or uuid.uuid4().hex[:12],
        artifacts_dir=args.artifacts_dir.expanduser().resolve(),
        cuda_version=args.cuda_version,
        jobs=args.jobs,
        gtest_filter=args.gtest_filter,
        connection_timeout_s=args.connection_timeout,
        transfer_timeout_s=args.transfer_timeout,
        build_timeout_s=args.build_timeout,
        test_timeout_s=args.test_timeout,
        strict_host_keys=args.host_key_policy == "yes",
        known_hosts_file=args.known_hosts_file,
        keep_workspace=args.keep_workspace,
        build_python=args.build_python,
    )
    config.validate()
    return config


def _worker_main(args: argparse.Namespace,
                 services: WorkerServices | None = None) -> int:
    paths = (args.run_root, args.trt_root, args.trt_build_dir)
    if (any(path.anchor != "/" or ".." in path.parts or str(path) == "/"
            for path in paths) or args.jobs <= 0 or args.package_timeout <= 0):
        raise ValueError(
            "worker paths must be canonical absolute non-root paths; values must be positive"
        )
    artifacts = Path(str(args.run_root / "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)
    configure_logging(str(artifacts / "worker.log"), log_level="INFO")
    if services is None:
        commands = CommandManager()
        code = CodeManager(
            command_manager=commands,
            container_manager=ContainerManager(command_manager=commands,
                                               trt_repo=str(args.trt_root)),
            work_dir=str(args.run_root),
            keep_containers_running=False,
            enable_high_core_auto=False,
        )
        services = WorkerServices(commands, code)
    return BuildWorker(args, services).run()


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if "--build-worker" in raw:
        return _worker_main(_worker_parser().parse_args(raw))
    parser = _controller_parser()
    try:
        config = _config(parser.parse_args(raw))
    except ValueError as error:
        parser.error(str(error))
    config.local_root.mkdir(parents=True, exist_ok=True)
    configure_logging(str(config.local_root / "orchestrator.log"),
                      log_level="INFO")
    commands = CommandManager()
    build_cfg = remote_config(config, config.build_endpoint, TargetType.LINUX,
                              Arch.X86_64,
                              config.local_root / "build-transfer")
    test_cfg = remote_config(config, config.test_endpoint,
                             TargetType.THOR_LINUX, Arch.D7L,
                             config.local_root / "deploy-stage")
    build_remote = RemoteConnectionManager(build_cfg, command_manager=commands)
    test_remote = RemoteConnectionManager(test_cfg, command_manager=commands)
    code = CodeManager(
        command_manager=commands,
        work_dir=str(config.local_root / "code-manager"),
        remote_connection_manager=test_remote,
        enable_high_core_auto=False,
    )
    return ControllerFlow(
        config, ControllerServices(commands, build_remote, test_remote,
                                   code)).run()


if __name__ == "__main__":
    raise SystemExit(main())
