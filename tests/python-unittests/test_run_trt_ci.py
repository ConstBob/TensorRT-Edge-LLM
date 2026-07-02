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
"""Behavior tests for the x86 TRT CI controller and direct build worker."""

import argparse
import ast
import dataclasses
import os
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("trt_dev_toolkit", reason="TRT-internal CI dependency")

from trt_dev_toolkit.code_manager import (ArtifactSource, BuildComponent,
                                          BuildMode, DeploymentMode,
                                          DeploymentResult, EnvironmentExports,
                                          RunResult)
from trt_dev_toolkit.command_manager.data_structures import (CommandResult,
                                                             OutputMode,
                                                             ShellType)
from trt_dev_toolkit.command_manager.targets import (LocalTarget,
                                                     RemoteSshTarget)
from trt_dev_toolkit.constants import Arch, TargetType

from scripts import run_trt_ci as ci


def _result(success=True, exit_code=0, timed_out=False, stdout=None):
    return CommandResult(
        success=success,
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
    )


class FakeCommands:

    def __init__(self, events=None, outcomes=None):
        self.events = events if events is not None else []
        self.outcomes = dict(outcomes or {})
        self.calls = []

    def run(self, target, spec):
        self.calls.append((target, spec))
        self.events.append(f"command:{spec.operation_name}")
        return self.outcomes.get(spec.operation_name, _result())


class FakeFilesystem:

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.remove_calls = []
        self.ensure_calls = []
        self.remove_ok = True
        self.ensure_ok = True

    def remove_dir(self, path, **kwargs):
        self.remove_calls.append((path, kwargs))
        self.events.append(f"{self.name}:remove")
        return self.remove_ok

    def ensure_dir(self, path, **kwargs):
        self.ensure_calls.append((path, kwargs))
        self.events.append(f"{self.name}:ensure")
        return self.ensure_ok


class FakeRemote:
    """RCM fake intentionally exposing only the one allowed source upload."""

    def __init__(self, name, config, events):
        self.name = name
        self.config = config
        self.events = events
        self.target = RemoteSshTarget(config)
        self.filesystem = FakeFilesystem(name, events)
        self.connected = True
        self.upload_ok = True
        self.uploads = []

    def test_connection(self):
        self.events.append(f"{self.name}:probe")
        return self.connected

    def copy_local_directory_to_remote(self, *, local_path, remote_path,
                                       timeout_s):
        self.uploads.append((local_path, remote_path, timeout_s))
        self.events.append(f"{self.name}:upload")
        return self.upload_ok


class FakeCode:

    def __init__(self, target, workspace, events):
        self.target = target
        self.workspace = workspace
        self.events = events
        self.build_result = RunResult(success=True)
        self.build_calls = []
        self.deploy_calls = []
        self.deploy_error = None

    def plan_and_execute(self, targets):
        self.build_calls.append(list(targets))
        self.events.append("code:build")
        return self.build_result

    def deploy_runtime(self,
                       run_result,
                       remote_config,
                       *,
                       preferred_mode=None):
        self.deploy_calls.append((run_result, remote_config, preferred_mode))
        self.events.append("code:deploy")
        if self.deploy_error:
            raise self.deploy_error
        return DeploymentResult(
            remote_target=self.target,
            remote_workspace=str(self.workspace),
            env_script_path=f"{self.workspace}/setup_environment.sh",
            environment_exports=EnvironmentExports(),
            deployment_mode=DeploymentMode.RSYNC,
        )


class FakeLogger:

    def __init__(self):
        self.messages = []

    def error(self, *args):
        self.messages.append(("error", args))

    def exception(self, *args):
        self.messages.append(("exception", args))

    def warning(self, *args):
        self.messages.append(("warning", args))

    def info(self, *args):
        self.messages.append(("info", args))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "Edge LLM source"
    source.mkdir()
    value = ci.Config(
        architecture=Arch.X86_64,
        trt_location=PurePosixPath("/candidate TRT/package"),
        build_host="build-alias",
        run_host="run-alias",
        run_id="unit-451",
        branch="rel-ci-test",
        onnx_root=PurePosixPath("/models/onnx"),
        jobs=7,
        source_root=source,
    )
    value.validate()
    return value


def _remote(endpoint, config, name, events):
    remote = ci.remote_config(
        endpoint,
        local_path=config.local_root / name,
        remote_path=(config.edgellm_root
                     if name == "build" else config.runtime_root),
    )
    return FakeRemote(name, remote, events)


def _worker_harness(config, trt_layout="package"):
    events = []
    run_remote = _remote(ci.Endpoint("run-alias", "runner", 2202), config,
                         "run", events)
    commands = FakeCommands(
        events,
        {"detect-trt-prebuilt": _result(stdout=f"{trt_layout}\n")},
    )
    code = FakeCode(run_remote.target, config.runtime_root, events)
    services = ci.WorkerServices(commands, run_remote, run_remote.config, code)
    return services, commands, run_remote, code, events


def _spec(commands, operation):
    return next((target, spec) for target, spec in commands.calls
                if spec.operation_name == operation)


def _assert_subsequence(values, expected):
    iterator = iter(values)
    for item in expected:
        assert any(value == item for value in iterator), (item, values)


@pytest.mark.parametrize("name", ["x86", "x86_64"])
def test_four_positional_cli_accepts_x86_aliases_and_rejects_d7l(
        name, monkeypatch):
    monkeypatch.setenv("TRT_CI_BRANCH", "rel-env")
    monkeypatch.setenv("TRT_CI_ONNX_DIR", "/models/env-onnx")
    monkeypatch.setenv("TRT_CI_JOBS", "9")
    args = ci._parser().parse_args([
        name,
        "/trt/package",
        "build-alias",
        "run-alias",
    ])
    config = ci._config(args)

    assert config.architecture is Arch.X86_64
    assert config.trt_location == PurePosixPath("/trt/package")
    assert (config.build_host, config.run_host) == ("build-alias", "run-alias")
    assert (config.branch, config.onnx_root,
            config.jobs) == ("rel-env", PurePosixPath("/models/env-onnx"), 9)
    with pytest.raises(argparse.ArgumentTypeError,
                       match="D7L is not supported"):
        ci._architecture("d7l")


def test_ssh_resolution_preserves_alias_and_derives_user_and_port():
    commands = FakeCommands(
        outcomes={
            "resolve-ssh-target": _result(
                stdout="user alias-user\nport 2222\n")
        })

    alias = ci.resolve_endpoint("lab-alias", commands)
    explicit = ci.resolve_endpoint("cli-user@build.example:2200", commands)

    assert alias == ci.Endpoint("lab-alias", "alias-user", 2222)
    assert explicit == ci.Endpoint("build.example", "cli-user", 2200)
    target, spec = commands.calls[0]
    assert isinstance(target, LocalTarget)
    assert spec.argv == ["ssh", "-G", "lab-alias"]
    assert spec.output_mode is OutputMode.CAPTURE and spec.diagnostic

    resolved = ci.remote_config(
        alias,
        local_path=Path("/tmp/local"),
        remote_path=PurePosixPath("/tmp/remote"),
    )
    assert (resolved.target.target_type, resolved.target.arch) == (
        TargetType.LINUX,
        Arch.X86_64,
    )
    assert resolved.target.address == "lab-alias"
    assert resolved.target.password == ""
    assert resolved.ssh.batch_mode


@pytest.mark.parametrize("layout", ["package", "source"])
def test_trt_probe_and_targets_support_package_or_built_source(config, layout):
    services, commands, _remote, _code, _events = _worker_harness(
        config, layout)

    assert ci.BuildWorker(config, services,
                          FakeLogger())._probe_trt() == layout

    target, probe = _spec(commands, "detect-trt-prebuilt")
    assert isinstance(target, LocalTarget)
    assert probe.output_mode is OutputMode.CAPTURE
    for path in (
            '"$root/include/NvInfer.h"',
            '"$root/include/NvInferVersion.h"',
            '"$root/include/NvOnnxParser.h"',
            '"$root/lib/libnvinfer.so"',
            '"$root/build/include/NvInferVersion.h"',
            '"$root/build/Release/lib/libnvinfer.so"',
    ):
        assert path in probe.command

    trt, edge = ci.build_targets(config, layout)

    assert [trt.component,
            edge.component] == [BuildComponent.TRT, BuildComponent.EDGELLM]
    assert [trt.source,
            edge.source] == [ArtifactSource.PRE_BUILT, ArtifactSource.BUILD]
    assert trt.mode is edge.mode is BuildMode.RELEASE
    assert trt.platform.arch is edge.platform.arch is Arch.X86_64
    assert trt.branch == edge.branch == config.branch
    expected_build = (config.trt_location if layout == "package" else
                      config.trt_location / "build")
    expected_repo = None if layout == "package" else str(config.trt_location)
    assert trt.build.build_dir == str(expected_build)
    assert trt.build.repo_path == expected_repo
    assert edge.build.repo_path == edge.build.build_dir == str(
        config.edgellm_root)
    assert edge.build.parallel_jobs == config.jobs
    if layout == "package":
        assert edge.build.cmake_args == ["-DENABLE_CUTE_DSL=OFF"]
        assert edge.build.extra_mounts == []
    else:
        build = config.trt_location / "build"
        assert edge.build.cmake_args == [
            f"-DTensorRT_INCLUDE_DIR={config.trt_location}/include;{build}/include",
            f"-DTensorRT_OnnxParser_INCLUDE_DIR={config.trt_location}/parsers/onnx",
            f"-DTensorRT_LIBRARY={build}/Release/lib/libnvinfer.so",
            f"-DTensorRT_OnnxParser_LIBRARY={build}/Release/lib/libnvonnxparser.so",
            "-DENABLE_CUTE_DSL=OFF",
        ]
        assert edge.build.extra_mounts == [str(config.trt_location)]
    assert edge.build.trt_package_dir is None


def test_trt_probe_rejects_empty_output(config):
    services, commands, _remote, _code, _events = _worker_harness(config)
    commands.outcomes["detect-trt-prebuilt"] = _result(stdout="")

    with pytest.raises(ci.FlowError, match="unexpected TensorRT layout"):
        ci.BuildWorker(config, services, FakeLogger())._probe_trt()


@pytest.mark.parametrize("toolkit_override", [None, "/ci/toolkit/src"])
def test_controller_stages_source_and_invokes_hidden_worker(
        config, monkeypatch, toolkit_override):
    inherited_pythonpath = os.pathsep.join(
        ("/existing/python", "/team/python"))
    monkeypatch.setenv("PYTHONPATH", inherited_pythonpath)
    if toolkit_override is None:
        monkeypatch.delenv("TRT_CI_TOOLKIT_PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("TRT_CI_TOOLKIT_PYTHONPATH", toolkit_override)
    events = []
    commands = FakeCommands(events)
    remote = _remote(ci.Endpoint("build-alias", "builder", 2201), config,
                     "build", events)
    flow = ci.ControllerFlow(
        config,
        ci.ControllerServices(commands, remote),
        FakeLogger(),
    )

    assert flow.run() == 0

    _assert_subsequence(events, [
        "build:probe",
        "build:remove",
        "build:ensure",
        "command:build-host-prerequisites",
        "command:stage-source",
        "build:upload",
        "command:build-worker",
        "build:remove",
    ])
    probe_target, probe = _spec(commands, "build-host-prerequisites")
    assert probe_target is remote.target
    assert probe.output_mode is OutputMode.CAPTURE
    assert "x86_64" in probe.command and "command -v timeout git docker" in probe.command

    stage_target, stage = _spec(commands, "stage-source")
    assert isinstance(stage_target, LocalTarget)
    assert stage.argv[:4] == ["rsync", "-a", "--delete", "--delete-excluded"]
    assert remote.uploads == [(
        str(config.local_root / "source-stage"),
        str(config.edgellm_root),
        1800,
    )]
    assert not (config.local_root / "source-stage").exists()

    worker_target, worker = _spec(commands, "build-worker")
    assert worker_target is remote.target
    assert worker.argv == [
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{config.worker_timeout_s:g}s",
        "python3",
        str(config.edgellm_root / "scripts/run_trt_ci.py"),
        "x86_64",
        str(config.trt_location),
        config.build_host,
        config.run_host,
        "--build-worker",
        "--run-id",
        config.run_id,
    ]
    toolkit = toolkit_override or ci._TOOLKIT_SRC
    assert worker.env == {
        "PYTHONPATH": os.pathsep.join((toolkit, inherited_pythonpath)),
        "TRT_CI_BRANCH": config.branch,
        "TRT_CI_ONNX_DIR": str(config.onnx_root),
        "TRT_CI_JOBS": str(config.jobs),
    }
    if toolkit_override is None:
        assert (Path(toolkit) / "trt_dev_toolkit" / "__init__.py").is_file()
    assert worker.timeout_s == config.worker_timeout_s + 120
    assert remote.filesystem.remove_calls[-1][1]["timeout_s"] == 1800


def test_worker_deploys_exact_build_result_then_tests_and_cleans(config):
    services, commands, run_remote, code, events = _worker_harness(config)
    original = code.build_result
    worker = ci.BuildWorker(config, services, FakeLogger())

    assert worker.run() == 0

    _assert_subsequence(events, [
        "run:probe",
        "command:run-host-prerequisites",
        "command:detect-trt-prebuilt",
        "code:build",
        "code:deploy",
        "command:run-compatibility-subset",
        "run:remove",
    ])
    assert len(code.build_calls) == len(code.deploy_calls) == 1
    deployed, remote_config, mode = code.deploy_calls[0]
    assert deployed is original
    assert remote_config is services.run_config
    assert services.run_config is run_remote.config
    assert mode is DeploymentMode.RSYNC
    assert not run_remote.uploads

    probe_target, probe = _spec(commands, "run-host-prerequisites")
    assert probe_target is run_remote.target
    assert probe.output_mode is OutputMode.CAPTURE
    assert "x86_64" in probe.command and "command -v bash ldd" in probe.command
    assert f"test -d {config.model_dir}" in probe.command

    test_target, test = _spec(commands, "run-compatibility-subset")
    assert test_target is run_remote.target
    assert test.shell_type is ShellType.BASH
    assert test.output_mode is OutputMode.PROGRESS
    assert test.cwd == str(config.runtime_root / "edgellm")
    for text in ("not found", "readlink -f", "libnvinfer", "libnvonnxparser",
                 "unitTest", "llm_build", "llm_inference", "PIPESTATUS[0]"):
        assert text in test.command
    results = config.run_root / "results"
    edge = config.runtime_root / "edgellm"
    for argument in (
            f"--onnxDir={config.model_dir}",
            f"--engineDir={results / 'qwen2.5-fp16-engine'}",
            "--maxInputLen=2048",
            "--maxKVCacheCapacity=4096",
            "--maxBatchSize=1",
            f"--inputFile={edge / 'tests/test_cases/llm_basic.json'}",
            f"--outputFile={results / 'llm-basic-output.json'}",
            "--dumpProfile",
            f"test -s {results / 'llm-basic-output.json'}",
    ):
        assert argument in test.command
    assert "--help" not in test.command
    assert test.artifact_log_file == str(config.run_root /
                                         "artifacts/tests.log")
    assert len(run_remote.filesystem.remove_calls) == 1
    assert run_remote.filesystem.remove_calls[0][1]["timeout_s"] == 1800


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("controller", 13),
        ("connect", 1),
        ("reset", 1),
        ("ensure", 1),
        ("prerequisite", 17),
        ("stage", 19),
        ("upload", 1),
        ("worker", 37),
        ("worker-timeout", 124),
    ],
)
def test_controller_preserves_workspace_and_failure_status(
        config, failure, expected):
    events = []
    commands = FakeCommands(events)
    remote = _remote(ci.Endpoint("build-alias", "builder", 2201), config,
                     "build", events)
    if failure == "controller":
        commands.outcomes["controller-prerequisites"] = _result(False, 13)
    elif failure == "connect":
        remote.connected = False
    elif failure == "reset":
        remote.filesystem.remove_ok = False
    elif failure == "ensure":
        remote.filesystem.ensure_ok = False
    elif failure == "prerequisite":
        commands.outcomes["build-host-prerequisites"] = _result(False, 17)
    elif failure == "stage":
        commands.outcomes["stage-source"] = _result(False, 19)
    elif failure == "upload":
        remote.upload_ok = False
    elif failure == "worker":
        commands.outcomes["build-worker"] = _result(False, 37)
    elif failure == "worker-timeout":
        commands.outcomes["build-worker"] = _result(False, None, True)

    status = ci.ControllerFlow(
        config,
        ci.ControllerServices(commands, remote),
        FakeLogger(),
    ).run()

    assert status == expected
    assert len(remote.filesystem.remove_calls) <= 1


@pytest.mark.parametrize(
    ("failure", "expected", "cleanup"),
    [
        ("connect", 1, False),
        ("prerequisite", 23, False),
        ("trt", 29, False),
        ("build", 1, False),
        ("deploy", 1, False),
        ("test", 41, False),
        ("test-timeout", 124, False),
        ("success", 0, True),
    ],
)
def test_worker_preserves_status_and_cleans_only_success(
        config, failure, expected, cleanup):
    services, commands, run_remote, code, _events = _worker_harness(config)
    if failure == "connect":
        run_remote.connected = False
    elif failure == "prerequisite":
        commands.outcomes["run-host-prerequisites"] = _result(False, 23)
    elif failure == "trt":
        commands.outcomes["detect-trt-prebuilt"] = _result(False, 29)
    elif failure == "build":
        code.build_result = RunResult(success=False,
                                      error_messages=["build failed"])
    elif failure == "deploy":
        code.deploy_error = RuntimeError("deploy failed")
    elif failure == "test":
        commands.outcomes["run-compatibility-subset"] = _result(False, 41)
    elif failure == "test-timeout":
        commands.outcomes["run-compatibility-subset"] = _result(
            False, None, True)

    status = ci.BuildWorker(config, services, FakeLogger()).run()

    assert status == expected
    assert len(run_remote.filesystem.remove_calls) == (1 if cleanup else 0)
    if failure in {"test", "test-timeout", "success"}:
        assert code.deploy_calls[0][0] is code.build_result


def test_worker_main_returns_one_when_ssh_resolution_fails(
        config, monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ci, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ci, "get_logger", lambda _name: logger)
    monkeypatch.setattr(ci, "CommandManager", object)

    def reject_endpoint(*_args):
        raise ValueError("bad SSH target")

    monkeypatch.setattr(ci, "resolve_endpoint", reject_endpoint)

    assert ci._worker_main(config) == 1
    assert logger.messages[0][0] == "error"


def test_structure_forbids_manual_runtime_resource_and_result_copies():
    source = Path(ci.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = source[source.index("class ControllerFlow"):source.
                        index("class BuildWorker")]

    assert [field.name for field in dataclasses.fields(ci.ControllerServices)
            ] == ["commands", "build_remote"]
    assert [field.name for field in dataclasses.fields(ci.WorkerServices)
            ] == ["commands", "run_remote", "run_config", "code"]
    for forbidden in ("deploy_runtime", "copy_remote_", "_retrieve",
                      "_stage_resources", "_collect", "_test_spec"):
        assert forbidden not in controller
    assert "copy_remote_directory_to_local" not in source
    assert "copy_remote_file_to_local" not in source
    assert source.count(".copy_local_directory_to_remote(") == 1
    assert not hasattr(FakeRemote, "copy_remote_directory_to_local")
    assert not any(
        isinstance(node, ast.Import) and any(alias.name == "subprocess"
                                             for alias in node.names)
        for node in ast.walk(tree))
