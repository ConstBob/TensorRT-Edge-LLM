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
"""Build Edge-LLM against a PRE_BUILT TensorRT and test it on x86."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from trt_dev_toolkit.code_manager import (ArtifactSource, ArtifactTarget,
                                          BuildComponent, BuildMode,
                                          CodeManager, DeploymentMode)
from trt_dev_toolkit.code_manager.models import BuildConfig, PlatformConfig
from trt_dev_toolkit.command_manager.command_manager import CommandManager
from trt_dev_toolkit.command_manager.data_structures import (CommandSpec,
                                                             OutputMode,
                                                             ShellType)
from trt_dev_toolkit.command_manager.targets import LocalTarget
from trt_dev_toolkit.constants import Arch, TargetType, parse_arch
from trt_dev_toolkit.container_manager import ContainerManager
from trt_dev_toolkit.log_manager import configure_logging, get_logger
from trt_dev_toolkit.remote_connection_manager.config.config import (
    RemoteConfig, RemotePaths)
from trt_dev_toolkit.remote_connection_manager.config.ssh import SSHConfig
from trt_dev_toolkit.remote_connection_manager.remote_connection_manager import \
    RemoteConnectionManager
from trt_dev_toolkit.remote_connection_manager.types.target_types import \
    SSHTargetInfo

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = PurePosixPath("/tmp/edgellm-trt-ci")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_CONNECTION_TIMEOUT_S = 30
_TRANSFER_TIMEOUT_S = 1800
_BUILD_TIMEOUT_S = 7200
_TEST_TIMEOUT_S = 3600
_JOBS = 16
_PYTHON = "python3"
_DEFAULT_ONNX_ROOT = PurePosixPath("/home/edge_llm_cache/trt-ci/onnx")
_MODEL_RELATIVE = PurePosixPath("Qwen2.5-0.5B-Instruct/llm-fp16-fp16")


class FlowError(RuntimeError):

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclasses.dataclass(frozen=True)
class Endpoint:
    host: str
    user: str
    port: int


@dataclasses.dataclass(frozen=True)
class Config:
    architecture: Arch
    trt_location: PurePosixPath
    build_host: str
    run_host: str
    run_id: str
    branch: str
    onnx_root: PurePosixPath
    jobs: int
    source_root: Path = _SOURCE_ROOT

    def validate(self) -> None:
        if self.architecture is not Arch.X86_64:
            raise ValueError(
                "this first draft supports only x86/x86_64; D7L is not supported"
            )
        if not _safe_path(self.trt_location):
            raise ValueError(
                "trt_location must be an absolute, non-root package or TRT source path"
            )
        if not _safe_path(self.onnx_root):
            raise ValueError(
                "TRT_CI_ONNX_DIR must be an absolute, non-root run-host path")
        if self.jobs <= 0:
            raise ValueError("TRT_CI_JOBS must be a positive integer")
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("invalid internal run ID")
        if not self.source_root.is_dir():
            raise ValueError(
                f"Edge-LLM source root does not exist: {self.source_root}")
        for name, value in (("build_host", self.build_host), ("run_host",
                                                              self.run_host)):
            if not value or value.startswith("-") or any(char in value
                                                         for char in "\r\n\0"):
                raise ValueError(f"invalid {name} SSH target")

    @property
    def run_root(self) -> PurePosixPath:
        return _WORKSPACE / f"run-{self.run_id}"

    @property
    def runtime_root(self) -> PurePosixPath:
        return self.run_root / "runtime"

    @property
    def edgellm_root(self) -> PurePosixPath:
        return self.runtime_root / "edgellm"

    @property
    def local_root(self) -> Path:
        root = Path(os.environ.get("TRT_CI_ARTIFACTS_DIR", "artifacts/trt-ci"))
        return root.expanduser().resolve() / f"run-{self.run_id}"

    @property
    def model_dir(self) -> PurePosixPath:
        return self.onnx_root / _MODEL_RELATIVE

    @property
    def worker_timeout_s(self) -> float:
        return _BUILD_TIMEOUT_S + 2 * _TRANSFER_TIMEOUT_S + _TEST_TIMEOUT_S + 300


@dataclasses.dataclass
class ControllerServices:
    commands: Any
    build_remote: Any


@dataclasses.dataclass
class WorkerServices:
    commands: Any
    run_remote: Any
    run_config: RemoteConfig
    code: Any


def _safe_path(path: PurePosixPath) -> bool:
    return (path.anchor == "/" and ".." not in path.parts and str(path) != "/"
            and not any(char in str(path) for char in "\r\n\0"))


def _architecture(value: str) -> Arch:
    normalized = value.strip().lower()
    try:
        arch = Arch.X86_64 if normalized == "x86" else parse_arch(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if arch is not Arch.X86_64:
        raise argparse.ArgumentTypeError(
            "this first draft supports only x86/x86_64; D7L is not supported")
    return arch


def _split_ssh(value: str) -> tuple[str | None, str, int | None]:
    if not value or value.startswith("-") or any(char in value
                                                 for char in "\r\n\0"):
        raise ValueError("SSH target must use [user@]host[:port]")
    user, separator, host_port = value.rpartition("@")
    if not separator:
        user, host_port = "", value
    port: int | None = None
    if host_port.startswith("["):
        closing = host_port.find("]")
        if closing < 0:
            raise ValueError("invalid bracketed SSH host")
        host = host_port[1:closing]
        suffix = host_port[closing + 1:]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:].isdigit():
                raise ValueError("SSH target must use [user@]host[:port]")
            port = int(suffix[1:])
    else:
        host, separator, port_text = host_port.rpartition(":")
        if separator and port_text.isdigit():
            port = int(port_text)
        else:
            host = host_port
    if not host or host.startswith("-") or user.startswith("-"):
        raise ValueError("SSH target must use [user@]host[:port]")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("SSH port must be between 1 and 65535")
    return user or None, host, port


def resolve_endpoint(value: str, commands: Any) -> Endpoint:
    explicit_user, host, explicit_port = _split_ssh(value)
    query = f"{explicit_user}@{host}" if explicit_user else host
    result = commands.run(
        LocalTarget(),
        CommandSpec(
            name=f"Resolve SSH target {host}",
            argv=["ssh", "-G", query],
            timeout_s=_CONNECTION_TIMEOUT_S,
            output_mode=OutputMode.CAPTURE,
            diagnostic=True,
            operation_name="resolve-ssh-target",
        ))
    if not result.success:
        raise ValueError(f"OpenSSH could not resolve target {host!r}")
    resolved: dict[str, str] = {}
    for line in (result.stdout or result.combined_output or "").splitlines():
        key, separator, item = line.partition(" ")
        if separator and key in {"user", "port"}:
            resolved.setdefault(key, item.strip())
    user = explicit_user or resolved.get("user")
    try:
        port = explicit_port or int(resolved.get("port", "22"))
    except ValueError as error:
        raise ValueError(
            f"OpenSSH returned an invalid port for {host!r}") from error
    if not user or user.startswith("-") or not 1 <= port <= 65535:
        raise ValueError(
            f"OpenSSH did not resolve valid SSH information for {host!r}")
    # Keep the original host alias so its IdentityFile and ProxyJump settings apply.
    return Endpoint(host=host, user=user, port=port)


def remote_config(endpoint: Endpoint, *, local_path: Path | PurePosixPath,
                  remote_path: PurePosixPath) -> RemoteConfig:
    return RemoteConfig(
        target=SSHTargetInfo(
            target_type=TargetType.LINUX,
            arch=Arch.X86_64,
            address=endpoint.host,
            port=endpoint.port,
            username=endpoint.user,
            password="",
        ),
        ssh=SSHConfig(
            timeout_s=_TRANSFER_TIMEOUT_S,
            connect_timeout_s=_CONNECTION_TIMEOUT_S,
            batch_mode=True,
        ),
        paths=RemotePaths(local_path=str(local_path),
                          remote_path=str(remote_path)),
    )


def build_targets(config: Config, trt_layout: str) -> list[ArtifactTarget]:
    platform = PlatformConfig(arch=Arch.X86_64)
    cmake_args = ["-DENABLE_CUTE_DSL=OFF"]
    extra_mounts: list[str] = []
    if trt_layout == "package":
        trt_build = BuildConfig(build_dir=str(config.trt_location))
    elif trt_layout == "source":
        build = config.trt_location / "build"
        trt_build = BuildConfig(build_dir=str(build),
                                repo_path=str(config.trt_location))
        cmake_args[:0] = [
            f"-DTensorRT_INCLUDE_DIR={config.trt_location}/include;{build}/include",
            f"-DTensorRT_OnnxParser_INCLUDE_DIR={config.trt_location}/parsers/onnx",
            f"-DTensorRT_LIBRARY={build}/Release/lib/libnvinfer.so",
            f"-DTensorRT_OnnxParser_LIBRARY={build}/Release/lib/libnvonnxparser.so",
        ]
        extra_mounts = [str(config.trt_location)]
    else:
        raise ValueError(f"unsupported TRT layout: {trt_layout}")
    return [
        ArtifactTarget(
            component=BuildComponent.TRT,
            mode=BuildMode.RELEASE,
            branch=config.branch,
            source=ArtifactSource.PRE_BUILT,
            platform=platform,
            build=trt_build,
        ),
        ArtifactTarget(
            component=BuildComponent.EDGELLM,
            mode=BuildMode.RELEASE,
            branch=config.branch,
            source=ArtifactSource.BUILD,
            platform=platform,
            build=BuildConfig(
                repo_path=str(config.edgellm_root),
                build_dir=str(config.edgellm_root),
                log_file=str(config.run_root / "artifacts/build-edgellm.log"),
                parallel_jobs=config.jobs,
                cmake_args=cmake_args,
                extra_mounts=extra_mounts,
            ),
        ),
    ]


def _status(result: Any) -> int:
    if result.success:
        return 0
    if getattr(result, "timed_out", False):
        return 124
    return result.exit_code if result.exit_code is not None else 1


def _cleanup(remote: Any, path: PurePosixPath, logger: Any,
             label: str) -> None:
    try:
        if not remote.filesystem.remove_dir(str(path),
                                            timeout_s=_TRANSFER_TIMEOUT_S):
            logger.warning("Could not clean %s workspace", label)
    except Exception as error:
        logger.warning("Could not clean %s workspace: %s", label, error)


class ControllerFlow:

    def __init__(self, config: Config, services: ControllerServices,
                 logger: Any) -> None:
        self.config = config
        self.services = services
        self.logger = logger

    def run(self) -> int:
        status = 1
        try:
            self._prepare()
            self._stage_source()
            status = self._run_worker()
        except FlowError as error:
            self.logger.error("%s", error)
            status = error.exit_code
        except Exception as error:
            self.logger.exception("TRT CI validation failed: %s", error)
        if status == 0:
            _cleanup(self.services.build_remote, self.config.run_root,
                     self.logger, "build-host")
        return status

    def _prepare(self) -> None:
        local = self.services.commands.run(
            LocalTarget(),
            CommandSpec(
                name="Check controller prerequisites",
                command="command -v rsync >/dev/null",
                shell_type=ShellType.POSIX,
                timeout_s=_CONNECTION_TIMEOUT_S,
                output_mode=OutputMode.CAPTURE,
                operation_name="controller-prerequisites",
            ))
        if not local.success:
            raise FlowError("controller requires rsync", _status(local))
        remote = self.services.build_remote
        if not remote.test_connection():
            raise FlowError("cannot connect to build host")
        if not remote.filesystem.remove_dir(str(self.config.run_root),
                                            timeout_s=_CONNECTION_TIMEOUT_S):
            raise FlowError("cannot reset build-host workspace")
        if not remote.filesystem.ensure_dir(str(self.config.edgellm_root),
                                            timeout_s=_CONNECTION_TIMEOUT_S):
            raise FlowError("cannot create build-host source workspace")
        result = self.services.commands.run(
            remote.target,
            CommandSpec(
                name="Check build-host prerequisites",
                command=
                ('test "$(uname -m)" = x86_64 && '
                 f"command -v timeout git docker bash cmake make ssh rsync {_PYTHON} >/dev/null"
                 ),
                shell_type=ShellType.BASH,
                timeout_s=_CONNECTION_TIMEOUT_S,
                output_mode=OutputMode.CAPTURE,
                operation_name="build-host-prerequisites",
            ))
        if not result.success:
            raise FlowError("build-host prerequisite probe failed",
                            _status(result))

    def _stage_source(self) -> None:
        stage = self.config.local_root / "source-stage"
        stage.mkdir(parents=True, exist_ok=True)
        argv = ["rsync", "-a", "--delete", "--delete-excluded"]
        for item in ("/.git", "/.worktrees", "/codex-logs", "/build",
                     "/artifacts", "/CMakeCache.txt", "/CMakeFiles",
                     "__pycache__", "*.pyc"):
            argv.append(f"--exclude={item}")
        argv.extend((f"{self.config.source_root}/", f"{stage}/"))
        result = self.services.commands.run(
            LocalTarget(),
            CommandSpec(
                name="Stage Edge-LLM source",
                argv=argv,
                timeout_s=_TRANSFER_TIMEOUT_S,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(self.config.local_root /
                                      "stage-source.log"),
                operation_name="stage-source",
            ))
        if not result.success:
            raise FlowError("local source staging failed", _status(result))
        if not self.services.build_remote.copy_local_directory_to_remote(
                local_path=str(stage),
                remote_path=str(self.config.edgellm_root),
                timeout_s=_TRANSFER_TIMEOUT_S):
            raise FlowError("source upload failed")
        shutil.rmtree(stage, ignore_errors=True)

    def _run_worker(self) -> int:
        worker = self.config.edgellm_root / "scripts/run_trt_ci.py"
        argv = [
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            f"{self.config.worker_timeout_s:g}s",
            _PYTHON,
            str(worker),
            self.config.architecture.value,
            str(self.config.trt_location),
            self.config.build_host,
            self.config.run_host,
            "--build-worker",
            "--run-id",
            self.config.run_id,
        ]
        result = self.services.commands.run(
            self.services.build_remote.target,
            CommandSpec(
                name="Build, deploy, and test Edge-LLM",
                argv=argv,
                env={
                    "TRT_CI_BRANCH": self.config.branch,
                    "TRT_CI_ONNX_DIR": str(self.config.onnx_root),
                    "TRT_CI_JOBS": str(self.config.jobs),
                },
                timeout_s=self.config.worker_timeout_s + 120,
                output_mode=OutputMode.PROGRESS,
                artifact_log_file=str(self.config.local_root /
                                      "build-worker.log"),
                operation_name="build-worker",
            ))
        return _status(result)


class BuildWorker:

    def __init__(self, config: Config, services: WorkerServices,
                 logger: Any) -> None:
        self.config = config
        self.services = services
        self.logger = logger
        self.deployment: Any | None = None

    def run(self) -> int:
        try:
            self._probe_run_host()
            trt_layout = self._probe_trt()
            run_result = self.services.code.plan_and_execute(
                build_targets(self.config, trt_layout))
            if not run_result.success:
                self.logger.error("CodeManager build failed: %s",
                                  "; ".join(run_result.error_messages))
                return 1
            self.deployment = self.services.code.deploy_runtime(
                run_result,
                self.services.run_config,
                preferred_mode=DeploymentMode.RSYNC,
            )
            status = _status(
                self.services.commands.run(self.deployment.remote_target,
                                           self._test_spec()))
        except FlowError as error:
            self.logger.error("%s", error)
            return error.exit_code
        except Exception as error:
            self.logger.exception("Build-host worker failed: %s", error)
            return 1
        if status == 0:
            _cleanup(self.services.run_remote, self.config.run_root,
                     self.logger, "run-host")
        return status

    def _probe_run_host(self) -> None:
        if not self.services.run_remote.test_connection():
            raise FlowError("cannot connect from build host to run host")
        model = shlex.quote(str(self.config.model_dir))
        result = self.services.commands.run(
            self.services.run_remote.target,
            CommandSpec(
                name="Check run-host prerequisites",
                command=
                ('test "$(uname -m)" = x86_64 && '
                 f"test -d {model} && "
                 "command -v bash ldd readlink awk tee grep nvidia-smi >/dev/null"
                 ),
                shell_type=ShellType.BASH,
                timeout_s=_CONNECTION_TIMEOUT_S,
                output_mode=OutputMode.CAPTURE,
                operation_name="run-host-prerequisites",
            ))
        if not result.success:
            raise FlowError("run-host prerequisite/model probe failed",
                            _status(result))

    def _probe_trt(self) -> str:
        location = shlex.quote(str(self.config.trt_location))
        command = f"""set -e
root={location}
if test -f "$root/include/NvInfer.h" &&
   test -f "$root/include/NvInferVersion.h" &&
   test -f "$root/include/NvOnnxParser.h" &&
   test -e "$root/lib/libnvinfer.so" &&
   test -e "$root/lib/libnvonnxparser.so"; then
  printf 'package\n'
elif test -f "$root/include/NvInfer.h" &&
     test -f "$root/parsers/onnx/NvOnnxParser.h" &&
     test -f "$root/build/include/NvInferVersion.h" &&
     test -e "$root/build/Release/lib/libnvinfer.so" &&
     test -e "$root/build/Release/lib/libnvonnxparser.so"; then
  printf 'source\n'
else
  exit 1
fi
"""
        result = self.services.commands.run(
            LocalTarget(),
            CommandSpec(
                name="Detect TensorRT PRE_BUILT layout",
                command=command,
                shell_type=ShellType.BASH,
                timeout_s=_CONNECTION_TIMEOUT_S,
                output_mode=OutputMode.CAPTURE,
                operation_name="detect-trt-prebuilt",
            ))
        if not result.success:
            raise FlowError(
                "trt_location is neither a package root nor a built TRT source root",
                _status(result),
            )
        lines = (result.stdout or result.combined_output
                 or "").strip().splitlines()
        layout = lines[-1] if lines else ""
        if layout not in {"package", "source"}:
            raise FlowError(f"unexpected TensorRT layout result: {layout!r}")
        self.logger.info("TensorRT PRE_BUILT layout: %s", layout)
        return layout

    def _test_spec(self) -> CommandSpec:
        assert self.deployment is not None
        workspace = PurePosixPath(self.deployment.remote_workspace)
        edge, candidate = workspace / "edgellm", workspace / "trt"
        results = self.config.run_root / "results"
        unit = edge / "unitTest"
        builder = edge / "examples/llm/llm_build"
        inference = edge / "examples/llm/llm_inference"
        engine = results / "qwen2.5-fp16-engine"
        test_case = edge / "tests/test_cases/llm_basic.json"
        output = results / "llm-basic-output.json"
        q = shlex.quote
        command = f"""set -euo pipefail
source {q(self.deployment.env_script_path)}
mkdir -p {q(str(results))}
check_trt() {{
  binary="$1"; shift
  test -x "$binary"
  linked="$(ldd "$binary")"; printf '%s\\n' "$linked"
  ! grep -q 'not found' <<<"$linked"
  for library in "$@"; do
    resolved="$(awk -v name="$library" 'index($1, name ".so") == 1 {{print $3; exit}}' <<<"$linked")"
    test -n "$resolved"
    case "$(readlink -f "$resolved")" in {q(str(candidate))}/*) ;; *) exit 1 ;; esac
  done
}}
check_trt {q(str(unit))} libnvinfer
check_trt {q(str(builder))} libnvinfer libnvonnxparser
check_trt {q(str(inference))} libnvinfer
set +e
{q(str(unit))} --gtest_output=xml:{q(str(results / 'unit-tests.xml'))} \\
  2>&1 | tee {q(str(results / 'unit-tests.log'))}
status=${{PIPESTATUS[0]}}
set -e
test "$status" -eq 0 || exit "$status"
{q(str(builder))} \\
  --onnxDir={q(str(self.config.model_dir))} \\
  --engineDir={q(str(engine))} \\
  --maxInputLen=2048 --maxKVCacheCapacity=4096 --maxBatchSize=1 \\
  2>&1 | tee {q(str(results / 'llm-build.log'))}
{q(str(inference))} \\
  --engineDir={q(str(engine))} \\
  --inputFile={q(str(test_case))} \\
  --outputFile={q(str(output))} --dumpProfile \\
  2>&1 | tee {q(str(results / 'llm-inference.log'))}
test -s {q(str(output))}
"""
        return CommandSpec(
            name="Run Edge-LLM unit and fixed Qwen E2E tests",
            command=command,
            shell_type=ShellType.BASH,
            cwd=str(edge),
            timeout_s=_TEST_TIMEOUT_S,
            output_mode=OutputMode.PROGRESS,
            artifact_log_file=str(self.config.run_root /
                                  "artifacts/tests.log"),
            operation_name="run-compatibility-subset",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture",
                        type=_architecture,
                        help="x86 or x86_64")
    parser.add_argument(
        "trt_location",
        type=PurePosixPath,
        help="absolute TRT package or built source root on the build host")
    parser.add_argument("build_host",
                        help="[user@]host[:port] or OpenSSH alias")
    parser.add_argument("run_host", help="[user@]host[:port] or OpenSSH alias")
    parser.add_argument("--build-worker",
                        action="store_true",
                        help=argparse.SUPPRESS)
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
    config = Config(
        architecture=args.architecture,
        trt_location=args.trt_location,
        build_host=args.build_host,
        run_host=args.run_host,
        run_id=args.run_id or uuid.uuid4().hex[:12],
        branch=branch,
        onnx_root=PurePosixPath(
            os.environ.get("TRT_CI_ONNX_DIR", str(_DEFAULT_ONNX_ROOT))),
        jobs=jobs,
    )
    config.validate()
    return config


def _worker_main(config: Config) -> int:
    artifacts = Path(str(config.run_root / "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)
    configure_logging(str(artifacts / "worker.log"), log_level="INFO")
    logger = get_logger(__name__)
    commands = CommandManager()
    try:
        endpoint = resolve_endpoint(config.run_host, commands)
    except ValueError as error:
        logger.error("%s", error)
        return 1
    run_cfg = remote_config(
        endpoint,
        local_path=config.run_root / "deploy-stage",
        remote_path=config.runtime_root,
    )
    run_remote = RemoteConnectionManager(run_cfg, command_manager=commands)
    code = CodeManager(
        command_manager=commands,
        container_manager=ContainerManager(command_manager=commands),
        work_dir=str(config.run_root / "code-manager"),
        keep_containers_running=False,
        remote_connection_manager=run_remote,
        enable_high_core_auto=False,
    )
    return BuildWorker(config,
                       WorkerServices(commands, run_remote, run_cfg, code),
                       logger).run()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
        config = _config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.build_worker:
        return _worker_main(config)

    config.local_root.mkdir(parents=True, exist_ok=True)
    configure_logging(str(config.local_root / "orchestrator.log"),
                      log_level="INFO")
    logger = get_logger(__name__)
    commands = CommandManager()
    try:
        endpoint = resolve_endpoint(config.build_host, commands)
    except ValueError as error:
        parser.error(str(error))
    build_cfg = remote_config(
        endpoint,
        local_path=config.local_root / "source-stage",
        remote_path=config.edgellm_root,
    )
    build_remote = RemoteConnectionManager(build_cfg, command_manager=commands)
    return ControllerFlow(config, ControllerServices(commands, build_remote),
                          logger).run()


if __name__ == "__main__":
    raise SystemExit(main())
