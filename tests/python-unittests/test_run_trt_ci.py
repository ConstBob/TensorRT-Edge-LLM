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
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

pytest.importorskip("trt_dev_toolkit", reason="TRT-internal CI dependency")

from trt_dev_toolkit.code_manager import (ArtifactSource, BuildComponent,
                                          BuildMode, DeploymentMode,
                                          DeploymentResult, EnvironmentExports,
                                          Plan, PlanStep, RunResult)
from trt_dev_toolkit.command_manager.data_structures import (CommandResult,
                                                             OutputMode)
from trt_dev_toolkit.command_manager.targets import (LocalTarget,
                                                     RemoteSshTarget)
from trt_dev_toolkit.constants import Arch, TargetType
from trt_dev_toolkit.container_manager import (ContainerBackendType,
                                               ContainerDescriptor,
                                               ContainerHandle, ContainerKind,
                                               MountSpec)

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
        self.upload_ok = True
        self.uploads = []

    def copy_local_directory_to_remote(self, *, local_path, remote_path,
                                       timeout_s):
        self.uploads.append((local_path, remote_path, timeout_s))
        self.events.append(f"{self.name}:upload")
        return self.upload_ok


class FakeContainers:

    def __init__(self, events):
        self.events = events
        self.descriptor = ContainerDescriptor(
            backend=ContainerBackendType.GIT_TRT_RUNC,
            image_or_profile="main-native-x86_64-ubuntu24.04-cuda13.2",
            kind=ContainerKind.TRT,
        )
        self.handle = ContainerHandle(
            name="trt-ci-edgellm-unit-451",
            backend=ContainerBackendType.GIT_TRT_RUNC,
            image=self.descriptor.image_or_profile,
            kind=ContainerKind.TRT,
        )
        self.resolve_calls = []
        self.launch_calls = []
        self.exec_calls = []
        self.remove_calls = []
        self.resolve_error = None
        self.launch_error = None
        self.exec_error = None
        self.exec_ok = True
        self.remove_error = None
        self.remove_ok = True

    def resolve(self, pattern):
        self.resolve_calls.append(pattern)
        self.events.append("container:resolve")
        if self.resolve_error:
            raise self.resolve_error
        return self.descriptor

    def launch(self, **kwargs):
        self.launch_calls.append(kwargs)
        self.events.append("container:launch")
        if self.launch_error:
            raise self.launch_error
        return self.handle

    def exec_progress(self, handle, **kwargs):
        self.exec_calls.append((handle, kwargs))
        self.events.append("container:exec")
        if self.exec_error:
            raise self.exec_error
        return self.exec_ok

    def remove(self, handle, **kwargs):
        self.remove_calls.append((handle, kwargs))
        self.events.append("container:remove")
        if self.remove_error:
            raise self.remove_error
        return self.remove_ok


class FakeCode:

    def __init__(self, target, workspace, events):
        self.target = target
        self.workspace = workspace
        self.events = events
        self.build_result = RunResult(success=True)
        self.container_manager = FakeContainers(events)
        self.build_calls = []
        self.deploy_calls = []
        self.deploy_error = None

    def plan_and_execute(self, targets):
        targets = list(targets)
        self.build_calls.append(targets)
        edge = next(target for target in targets
                    if target.component is BuildComponent.EDGELLM)
        edge = dataclasses.replace(
            edge,
            platform=dataclasses.replace(edge.platform,
                                         cuda_version="13.2",
                                         ubuntu_version="24.04"),
        )
        self.build_result.plan = Plan(
            run_id="fake-run",
            steps=[PlanStep("edgellm-build", edge, BuildComponent.EDGELLM)],
        )
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


def _worker_harness(config):
    events = []
    run_remote = _remote(ci.Endpoint("run-alias", "runner", 2202), config,
                         "run", events)
    code = FakeCode(run_remote.target, config.runtime_root, events)
    services = ci.WorkerServices(run_remote, run_remote.config, code)
    return services, run_remote, code, events


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


def test_build_targets_use_prebuilt_trt_and_edgellm_source(config):
    trt, edge = ci.build_targets(config)

    assert [trt.component,
            edge.component] == [BuildComponent.TRT, BuildComponent.EDGELLM]
    assert [trt.source,
            edge.source] == [ArtifactSource.PRE_BUILT, ArtifactSource.BUILD]
    assert trt.mode is edge.mode is BuildMode.RELEASE
    assert trt.platform.arch is edge.platform.arch is Arch.X86_64
    assert trt.branch == edge.branch == config.branch
    assert trt.build.build_dir == str(config.trt_location)
    assert trt.build.repo_path is None
    assert edge.build.repo_path == edge.build.build_dir == str(
        config.edgellm_root)
    assert edge.build.parallel_jobs == config.jobs
    assert edge.build.cmake_args == ["-DENABLE_CUTE_DSL=OFF"]
    assert edge.build.extra_mounts == []
    assert edge.build.trt_package_dir is None


def test_test_container_pattern_requires_edgellm_plan(config):
    with pytest.raises(ci.FlowError, match="missing its execution plan"):
        ci._test_container_pattern(RunResult(success=True))

    trt = ci.build_targets(config)[0]
    result = RunResult(
        success=True,
        plan=Plan("trt-only", [PlanStep("trt", trt, BuildComponent.TRT)]),
    )
    with pytest.raises(ci.FlowError, match="no Edge-LLM build step"):
        ci._test_container_pattern(result)


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
        "build:remove",
        "build:ensure",
        "command:stage-source",
        "build:upload",
        "command:build-worker",
        "build:remove",
    ])
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
    assert [spec.operation_name for _target, spec in commands.calls] == [
        "stage-source",
        "build-worker",
    ]
    assert remote.filesystem.remove_calls[-1][1]["timeout_s"] == 1800


def test_worker_deploys_then_tests_in_edgellm_container_and_cleans(config):
    services, run_remote, code, events = _worker_harness(config)
    original = code.build_result
    worker = ci.BuildWorker(config, services, FakeLogger())

    assert worker.run() == 0

    _assert_subsequence(events, [
        "code:build",
        "code:deploy",
        "container:resolve",
        "container:launch",
        "container:exec",
        "container:remove",
        "run:remove",
    ])
    assert len(code.build_calls) == len(code.deploy_calls) == 1
    deployed, remote_config, mode = code.deploy_calls[0]
    assert deployed is original
    assert remote_config is services.run_config
    assert services.run_config is run_remote.config
    assert mode is DeploymentMode.RSYNC
    assert not run_remote.uploads

    containers = code.container_manager
    pattern = containers.resolve_calls[0]
    assert (pattern.kind, pattern.arch, pattern.branch) == (
        ContainerKind.TRT,
        Arch.X86_64,
        config.branch,
    )
    assert (pattern.cuda_version, pattern.ubuntu_version) == ("13.2", "24.04")
    launch = containers.launch_calls[0]
    assert launch["descriptor"] is containers.descriptor
    assert launch["name"] == f"trt-ci-edgellm-{config.run_id}"
    assert launch["workdir"] == str(config.runtime_root / "edgellm")
    assert launch["exec_target"] is run_remote.target
    assert launch["mounts"] == [
        MountSpec(str(config.runtime_root), str(config.runtime_root)),
        MountSpec(str(config.onnx_root), str(config.onnx_root),
                  read_only=True),
    ]

    handle, execution = containers.exec_calls[0]
    assert handle is containers.handle
    assert execution["exec_target"] is run_remote.target
    assert execution["cwd"] == str(config.runtime_root / "edgellm")
    assert execution["timeout_s"] == 3600
    assert execution["tee_file"] == str(config.run_root /
                                        "artifacts/tests.log")
    command = execution["command"]
    setup = f"source {config.runtime_root / 'setup_environment.sh'}"
    plugin = config.runtime_root / "edgellm/libNvInfer_edgellm_plugin.so"
    assert setup in command
    assert f"export EDGELLM_PLUGIN_PATH={plugin}" in command
    expected_filter = (
        "SanityCheck.*:DeploymentConfigTest.*:EngineExecutorTest.*:"
        "LLMEngineConfigTest.*:LLMEngineConfigRecipesTest.*:"
        "RegistryBuilderTest.*:AllKVDtypes/RegistryBuilderKVDtypeTest.*")
    assert ci._TRT_UNIT_FILTER == expected_filter
    assert "InitializeMRopeCosSin" not in ci._TRT_UNIT_FILTER
    assert "Benchmark" not in ci._TRT_UNIT_FILTER
    assert ci._TRT_UNIT_FILTER in command
    assert "--gtest_fail_if_no_test_selected" in command
    assert command.index(setup) < command.index("unitTest")
    for text in ("unitTest", "llm_build", "llm_inference", "PIPESTATUS[0]",
                 "run_step", 'exit "$status"'):
        assert text in command
    results = config.runtime_root / "results"
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
        assert argument in command
    assert "--help" not in command
    assert containers.remove_calls == [(
        containers.handle,
        {
            "force": True,
            "exec_target": run_remote.target
        },
    )]
    assert len(run_remote.filesystem.remove_calls) == 1
    assert run_remote.filesystem.remove_calls[0][1]["timeout_s"] == 1800


@pytest.mark.parametrize(
    ("unit_rc", "build_rc", "inference_rc", "write_output", "expected",
     "steps"),
    [
        (7, 0, 0, True, 1, ["unit", "build", "inference"]),
        (0, 9, 0, True, 1, ["unit", "build"]),
        (0, 0, 11, True, 1, ["unit", "build", "inference"]),
        (0, 0, 0, False, 1, ["unit", "build", "inference"]),
        (0, 0, 0, True, 0, ["unit", "build", "inference"]),
    ],
)
def test_generated_test_command_aggregates_step_status(config, tmp_path,
                                                       unit_rc, build_rc,
                                                       inference_rc,
                                                       write_output, expected,
                                                       steps):
    workspace = tmp_path / "runtime"
    edge = workspace / "edgellm"
    model_root = tmp_path / "onnx"
    model = model_root / ci._MODEL_RELATIVE
    marker = tmp_path / "steps.log"

    def executable(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    model.mkdir(parents=True)
    (edge / "tests/test_cases").mkdir(parents=True)
    (edge / "tests/test_cases/llm_basic.json").write_text("{}")
    setup = workspace / "setup_environment.sh"
    setup.write_text(":\n")
    executable(edge / "unitTest", """printf 'unit\n' >> "$FLOW_MARKER"
exit "$UNIT_RC"
""")
    executable(edge / "libNvInfer_edgellm_plugin.so", "exit 0\n")
    executable(
        edge / "examples/llm/llm_build", """printf 'build\n' >> "$FLOW_MARKER"
for argument in "$@"; do
  case "$argument" in
    --engineDir=*) mkdir -p "$(printf '%s' "$argument" | cut -d= -f2-)" ;;
  esac
done
exit "$BUILD_RC"
""")
    executable(
        edge / "examples/llm/llm_inference",
        """printf 'inference\n' >> "$FLOW_MARKER"
output=
for argument in "$@"; do
  case "$argument" in
    --outputFile=*) output="$(printf '%s' "$argument" | cut -d= -f2-)" ;;
  esac
done
if test "$WRITE_OUTPUT" = 1; then printf '{}\n' > "$output"; fi
exit "$INFERENCE_RC"
""")
    test_config = dataclasses.replace(config,
                                      onnx_root=PurePosixPath(str(model_root)))
    worker = ci.BuildWorker(test_config, None, FakeLogger())
    worker.deployment = SimpleNamespace(
        remote_workspace=str(workspace),
        env_script_path=str(setup),
    )
    env = os.environ.copy()
    env.update({
        "BUILD_RC": str(build_rc),
        "FLOW_MARKER": str(marker),
        "INFERENCE_RC": str(inference_rc),
        "UNIT_RC": str(unit_rc),
        "WRITE_OUTPUT": "1" if write_output else "0",
    })

    result = subprocess.run(
        ["bash", "-c", worker._test_command()],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False)

    assert result.returncode == expected, result.stdout + result.stderr
    assert marker.read_text().splitlines() == steps


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("reset", 1),
        ("ensure", 1),
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
    if failure == "reset":
        remote.filesystem.remove_ok = False
    elif failure == "ensure":
        remote.filesystem.ensure_ok = False
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
        ("build", 1, False),
        ("deploy", 1, False),
        ("container-resolve", 1, False),
        ("container-launch", 1, False),
        ("test", 1, False),
        ("test-error", 1, False),
        ("remove", 1, False),
        ("remove-error", 1, False),
        ("success", 0, True),
    ],
)
def test_worker_preserves_status_and_cleans_only_success(
        config, failure, expected, cleanup):
    services, run_remote, code, _events = _worker_harness(config)
    if failure == "build":
        code.build_result = RunResult(success=False,
                                      error_messages=["build failed"])
    elif failure == "deploy":
        code.deploy_error = RuntimeError("deploy failed")
    elif failure == "container-resolve":
        code.container_manager.resolve_error = RuntimeError("resolve failed")
    elif failure == "container-launch":
        code.container_manager.launch_error = RuntimeError("launch failed")
    elif failure == "test":
        code.container_manager.exec_ok = False
    elif failure == "test-error":
        code.container_manager.exec_error = TimeoutError("test timed out")
    elif failure == "remove":
        code.container_manager.remove_ok = False
    elif failure == "remove-error":
        code.container_manager.remove_error = RuntimeError("remove failed")

    status = ci.BuildWorker(config, services, FakeLogger()).run()

    assert status == expected
    assert len(run_remote.filesystem.remove_calls) == (1 if cleanup else 0)
    if failure in {
            "container-resolve", "container-launch", "test", "test-error",
            "remove", "remove-error", "success"
    }:
        assert code.deploy_calls[0][0] is code.build_result
    assert len(code.container_manager.remove_calls) == (1 if failure in {
        "test", "test-error", "remove", "remove-error", "success"
    } else 0)


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
            ] == ["run_remote", "run_config", "code"]
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
