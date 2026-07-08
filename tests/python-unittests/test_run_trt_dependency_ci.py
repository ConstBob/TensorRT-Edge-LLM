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
"""Focused behavior tests for the x86 and D7L TRT CI validation entry point."""

import argparse
import ast
import dataclasses
import json
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
from trt_dev_toolkit.code_manager.models import ArtifactResult
from trt_dev_toolkit.command_manager.data_structures import (CommandResult,
                                                             OutputMode)
from trt_dev_toolkit.command_manager.targets import (LocalTarget,
                                                     RemoteSshTarget)
from trt_dev_toolkit.constants import Arch, TargetType
from trt_dev_toolkit.container_manager import (ContainerBackendType,
                                               ContainerDescriptor,
                                               ContainerHandle, ContainerKind,
                                               MountSpec)

from scripts import run_trt_dependency_ci as ci


def _result(success=True, exit_code=0, timed_out=False, stdout=None):
    return CommandResult(success=success,
                         exit_code=exit_code,
                         timed_out=timed_out,
                         stdout=stdout)


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

    def remove_dir(self, path, **kwargs):
        self.remove_calls.append((path, kwargs))
        self.events.append(f"{self.name}:remove")
        return True


class FakeRemote:

    def __init__(self, name, config, events):
        self.name = name
        self.config = config
        self.events = events
        self.target = RemoteSshTarget(config)
        self.filesystem = FakeFilesystem(name, events)
        self.copy_calls = []

    def copy_remote_directory_to_local(self, **kwargs):
        self.copy_calls.append(kwargs)
        self.events.append(f"{self.name}:copy-results")
        return True


class FakeContainers:

    def __init__(self, events):
        self.events = events
        self.descriptor = ContainerDescriptor(
            backend=ContainerBackendType.GIT_TRT_RUNC,
            image_or_profile="main-native-x86_64-ubuntu24.04-cuda13.2",
            kind=ContainerKind.TRT)
        self.handle = ContainerHandle(
            name="trt-ci-edgellm-unit-451",
            backend=ContainerBackendType.GIT_TRT_RUNC,
            image=self.descriptor.image_or_profile,
            kind=ContainerKind.TRT)
        self.resolve_calls = []
        self.launch_calls = []
        self.exec_calls = []
        self.remove_calls = []
        self.resolve_error = self.launch_error = self.exec_error = None
        self.remove_error = None
        self.exec_ok = self.remove_ok = True

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

    def __init__(self, target, workspace, local_edge, events):
        self.target = target
        self.workspace = workspace
        self.local_edge = local_edge
        self.events = events
        self.command_manager = None
        self.build_result = RunResult(success=True)
        self.container_manager = FakeContainers(events)
        self.build_calls = []
        self.deploy_calls = []
        self.setup_calls = []
        self.deploy_error = self.setup_error = None

    def plan_and_execute(self, targets):
        targets = list(targets)
        self.build_calls.append(targets)
        self.events.append("code:build")
        if self.build_result.success:
            edge = next(t for t in targets
                        if t.component is BuildComponent.EDGELLM)
            edge = dataclasses.replace(edge,
                                       platform=dataclasses.replace(
                                           edge.platform,
                                           cuda_version="13.2",
                                           ubuntu_version="24.04"))
            self.build_result.plan = Plan(
                run_id="fake-run",
                steps=[PlanStep("edge", edge, BuildComponent.EDGELLM)])
        return self.build_result

    def deploy_runtime(self, result, remote, *, preferred_mode=None):
        self.deploy_calls.append((result, remote, preferred_mode))
        self.events.append("code:deploy")
        if self.deploy_error:
            raise self.deploy_error
        return DeploymentResult(
            remote_target=self.target,
            remote_workspace=str(self.workspace),
            env_script_path=f"{self.workspace}/setup_environment.sh",
            environment_exports=EnvironmentExports(),
            deployment_mode=DeploymentMode.RSYNC)

    def write_environment_setup_script(self,
                                       result,
                                       *,
                                       preferred_component=None):
        self.setup_calls.append((result, preferred_component))
        self.events.append("code:setup")
        if self.setup_error:
            raise self.setup_error
        return self.local_edge / "setup_environment.sh"


class FakeLogger:

    def __init__(self):
        self.messages = []

    def __getattr__(self, level):
        return lambda *args: self.messages.append((level, args))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("TRT_CI_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    source = tmp_path / "Edge LLM source"
    source.mkdir()
    local = ci.HostSSHConfig("localhost")
    value = ci.Config(Arch.X86_64, PurePosixPath("/candidate TRT/package"),
                      local, local, "unit-451", "rel-ci-test",
                      PurePosixPath("/models/onnx"), 7, source)
    value.validate()
    return value


def _remote(connection, config, name, events, arch=Arch.X86_64):
    build = name == "build"
    remote = ci.remote_config(
        connection,
        local_path=(config.edgellm_root if build else config.local_root /
                    "deploy-stage"),
        remote_path=(PurePosixPath(str(config.edgellm_root))
                     if build else config.runtime_root),
        arch=arch)
    return FakeRemote(name, remote, events)


def _main_harness(monkeypatch,
                  tmp_path,
                  build_remote,
                  run_remote,
                  architecture=Arch.X86_64,
                  download_onnx=False):
    monkeypatch.setenv("TRT_CI_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("TRT_CI_BRANCH", "rel-ci-test")
    monkeypatch.setenv("TRT_CI_ONNX_DIR", "/models/onnx")
    monkeypatch.setenv("TRT_CI_JOBS", "7")
    local = ci.HostSSHConfig("localhost")
    build_connection = (ci.HostSSHConfig("build-host", 2201, "builder", "")
                        if build_remote else local)
    run_connection = (ci.HostSSHConfig("run-host", 2202, "runner", "")
                      if run_remote else local)
    config = ci.Config(architecture,
                       PurePosixPath("/candidate TRT/package"),
                       build_connection,
                       run_connection,
                       "unit-451",
                       "rel-ci-test",
                       PurePosixPath("/models/onnx"),
                       7,
                       download_onnx=download_onnx)
    events = []
    commands = FakeCommands(events)
    build_rcm = (_remote(build_connection, config, "build", events)
                 if build_remote else None)
    run_rcm = (_remote(run_connection, config, "run", events, architecture)
               if run_remote else None)
    build_host = ci.Host(build_rcm.target,
                         build_rcm) if build_rcm else ci.Host(LocalTarget())
    run_host = ci.Host(run_rcm.target, run_rcm) if run_rcm else ci.Host(
        LocalTarget())
    workspace = (config.runtime_root if run_remote else PurePosixPath(
        str(config.edgellm_root)))
    code = FakeCode(run_host.target, workspace, config.edgellm_root, events)
    code.command_manager = commands
    logger = FakeLogger()
    calls = SimpleNamespace(resolve=[], containers=[], code=[])
    hosts = iter((build_host, run_host))

    def resolve(value, command_manager, **paths):
        calls.resolve.append((value, command_manager, paths))
        return next(hosts)

    def containers(**kwargs):
        calls.containers.append(kwargs)
        return code.container_manager

    def code_manager(**kwargs):
        calls.code.append(kwargs)
        return code

    monkeypatch.setattr(ci, "CommandManager", lambda: commands)
    monkeypatch.setattr(ci, "resolve_host", resolve)
    monkeypatch.setattr(ci, "ContainerManager", containers)
    monkeypatch.setattr(ci, "CodeManager", code_manager)
    monkeypatch.setattr(ci, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ci, "get_logger", lambda *_args: logger)

    def payload(connection):
        return json.dumps({
            "host": connection.host,
            "port": connection.port,
            "user": connection.user,
            "password": connection.password,
        } if not connection.is_local else {"host": "localhost"})

    argv = [
        architecture.value, "/candidate TRT/package",
        payload(build_connection),
        payload(run_connection), "--run-id", "unit-451"
    ]
    if download_onnx:
        argv.append("--download_onnx")
    return SimpleNamespace(argv=argv,
                           config=config,
                           events=events,
                           commands=commands,
                           build_host=build_host,
                           run_host=run_host,
                           build_remote=build_rcm,
                           run_remote=run_rcm,
                           code=code,
                           logger=logger,
                           calls=calls)


def _assert_subsequence(values, expected):
    iterator = iter(values)
    for item in expected:
        assert any(value == item for value in iterator), (item, values)


@pytest.mark.parametrize("name,expected", [("x86", Arch.X86_64),
                                           ("x86_64", Arch.X86_64),
                                           ("d7l", Arch.D7L)])
def test_cli_and_localhost_resolution(name, expected, monkeypatch):
    monkeypatch.setenv("TRT_CI_BRANCH", "rel-env")
    monkeypatch.setenv("TRT_CI_ONNX_DIR", "/models/env-onnx")
    monkeypatch.setenv("TRT_CI_JOBS", "9")
    localhost = '{"host":"localhost"}'
    args = ci._parser().parse_args(
        [name, "/trt/package", localhost, localhost])
    config = ci._config(args)
    commands = FakeCommands()
    host = ci.resolve_host(config.build_host,
                           commands,
                           local_path=Path("/tmp/local"),
                           remote_path=PurePosixPath("/tmp/remote"))

    assert config.architecture is expected
    assert config.trt_location == PurePosixPath("/trt/package")
    assert config.build_host == config.run_host == ci.HostSSHConfig(
        "localhost")
    assert (config.branch, config.onnx_root,
            config.jobs) == ("rel-env", PurePosixPath("/models/env-onnx"), 9)
    workspace = (PurePosixPath("/dev/shm/edgellm-trt-ci") if expected
                 is Arch.D7L else PurePosixPath("/tmp/edgellm-trt-ci"))
    assert config.run_root == workspace / f"run-{config.run_id}"
    assert not config.download_onnx
    assert host.is_local and isinstance(host.target, LocalTarget)
    assert commands.calls == []
    assert ci._status(_result(False, 0)) == 1
    with pytest.raises(argparse.ArgumentTypeError, match="architecture must"):
        ci._architecture("d7q")


def test_download_onnx_cli_is_x86_only(monkeypatch):
    monkeypatch.setenv("TRT_CI_ONNX_DIR", "/models/onnx")
    parser = ci._parser()
    localhost = '{"host":"localhost"}'
    args = parser.parse_args(
        ["x86", "/trt/package", localhost, localhost, "--download_onnx"])

    assert ci._config(args).download_onnx
    args = parser.parse_args(
        ["d7l", "/trt/package", localhost, localhost, "--download_onnx"])
    with pytest.raises(ValueError, match="currently supports x86 only"):
        ci._config(args)


def test_json_host_config_reads_inline_and_file(tmp_path):
    payload = {
        "host": "192.168.1.3",
        "port": 2222,
        "user": "nvidia",
        "password": "board-secret",
        "jump_host": {
            "host": "jump.example.com",
            "port": 2200,
            "user": "jump-user",
            "password": "",
        },
    }
    path = tmp_path / "run-host.json"
    path.write_text(json.dumps(payload))

    inline = ci.read_ssh_config(json.dumps(payload), "run_host")
    assert ci.read_ssh_config(str(path), "run_host") == inline
    assert inline == ci.HostSSHConfig(
        "192.168.1.3", 2222, "nvidia", "board-secret",
        ci.HostSSHConfig("jump.example.com", 2200, "jump-user", ""))
    with pytest.raises(ValueError, match="unknown fields"):
        ci.read_ssh_config('{"host":"localhost","alias":"bad"}', "build_host")
    with pytest.raises(ValueError, match="password must be set"):
        ci.read_ssh_config('{"host":"remote","user":"runner"}', "run_host")


def test_direct_json_target_preserves_jump_host(monkeypatch):
    events = []
    commands = FakeCommands(events)
    created = []

    def manager(config, command_manager):
        created.append((config, command_manager))
        return FakeRemote("d7l", config, events)

    connection = ci.read_ssh_config(
        json.dumps({
            "host": "192.168.1.3",
            "port": 22,
            "user": "nvidia",
            "password": "board-secret",
            "jump_host": {
                "host": "dl20-0092.ipp4a1.colossus.nvidia.com",
                "port": 22,
                "user": "jump-user",
                "password": "",
            },
        }), "run_host")
    monkeypatch.setattr(ci, "RemoteConnectionManager", manager)
    host = ci.resolve_host(connection,
                           commands,
                           local_path=Path("/tmp/local"),
                           remote_path=PurePosixPath("/tmp/remote"),
                           arch=Arch.D7L)

    config, command_manager = created[0]
    assert command_manager is commands and host.remote.config is config
    assert config.target.target_type is TargetType.THOR_LINUX
    assert (config.target.address, config.target.port, config.target.username,
            config.target.password) == ("192.168.1.3", 22, "nvidia",
                                        "board-secret")
    assert config.jump_host.host == "dl20-0092.ipp4a1.colossus.nvidia.com"
    assert (config.jump_host.port, config.jump_host.username,
            config.jump_host.password) == (22, "jump-user", "")
    assert config.ssh.batch_mode is False
    assert config.jump_host.ssh.batch_mode is True
    assert (config.paths.local_path,
            config.paths.remote_path) == ("/tmp/local", "/tmp/remote")
    assert commands.calls == []


@pytest.mark.parametrize(
    "architecture,platform_args",
    [(Arch.X86_64, ["-DCUDA_TARGET_DIR=/usr/local/cuda/targets/x86_64-linux"]),
     (Arch.D7L, [
         "toolchain", "-DEMBEDDED_TARGET=auto-thor",
         "-DCUDA_DIR=/usr/local/cuda-13.2/targets/sbsa-linux",
         "-DCUDA_TARGET_DIR=/usr/local/cuda-13.2/targets/sbsa-linux"
     ])])
def test_build_targets_use_prebuilt_trt_and_edgellm_source(
        config, architecture, platform_args):
    config = dataclasses.replace(config, architecture=architecture)
    trt, edge = ci.build_targets(config)

    assert [trt.component,
            edge.component] == [BuildComponent.TRT, BuildComponent.EDGELLM]
    assert [trt.source,
            edge.source] == [ArtifactSource.PRE_BUILT, ArtifactSource.BUILD]
    assert trt.mode is edge.mode is BuildMode.RELEASE
    assert trt.platform.arch is edge.platform.arch is architecture
    assert trt.branch == edge.branch == config.branch
    assert trt.build.build_dir == str(config.trt_location)
    assert trt.build.repo_path is None
    assert edge.build.repo_path == edge.build.build_dir == str(
        config.edgellm_root)
    assert edge.build.log_file == str(config.local_root / "build-edgellm.log")
    assert edge.build.parallel_jobs == config.jobs
    assert edge.build.no_nvidia_runtime
    assert edge.build.cmake_args[:3] == [
        "--fresh", "-DBUILD_UNIT_TESTS=OFF", "-DENABLE_CUTE_DSL=OFF"
    ]
    if architecture is Arch.D7L:
        assert edge.build.cmake_args[3].endswith(
            "/cmake/aarch64_linux_toolchain.cmake")
        assert edge.build.cmake_args[4:] == platform_args[1:]
    else:
        assert edge.build.cmake_args[3:] == platform_args


@pytest.mark.parametrize("build_remote", [False, True])
@pytest.mark.parametrize("run_remote", [False, True])
def test_main_routes_local_and_remote_host_combinations(
        monkeypatch, tmp_path, build_remote, run_remote):
    harness = _main_harness(monkeypatch, tmp_path, build_remote, run_remote)

    assert ci.main(harness.argv) == 0
    assert harness.logger.messages[-1][0] == "info"
    assert "TRT CI validation PASSED" in harness.logger.messages[-1][1][0]

    container_args = harness.calls.containers[0]
    git_trt = container_args["git_trt_backend"]
    assert git_trt._config.extra_launch_args == ["--skip-branch-check"]
    code_args = harness.calls.code[0]
    assert code_args["command_manager"] is harness.commands
    assert code_args["container_manager"] is harness.code.container_manager
    assert code_args["exec_target"] is harness.build_host.target
    assert code_args["remote_connection_manager"] is harness.run_host.remote
    assert bool(harness.code.deploy_calls) is run_remote
    assert bool(harness.code.setup_calls) is (not run_remote)
    assert harness.code.container_manager.launch_calls[0][
        "exec_target"] is harness.run_host.target
    assert bool(harness.run_remote
                and harness.run_remote.filesystem.remove_calls) is run_remote
    assert bool(harness.run_remote
                and harness.run_remote.copy_calls) is run_remote


def test_remote_success_deploys_exact_build_and_tests_in_container(
        monkeypatch, tmp_path):
    harness = _main_harness(monkeypatch, tmp_path, True, True)
    original = harness.code.build_result

    assert ci.main(harness.argv) == 0

    _assert_subsequence(harness.events, [
        "code:build", "code:deploy", "container:resolve", "container:launch",
        "container:exec", "container:remove", "run:remove"
    ])
    deployed, remote, mode = harness.code.deploy_calls[0]
    assert deployed is original
    assert remote is harness.run_remote.config
    assert mode is DeploymentMode.RSYNC

    containers = harness.code.container_manager
    pattern = containers.resolve_calls[0]
    assert (pattern.kind, pattern.arch,
            pattern.branch) == (ContainerKind.TRT, Arch.X86_64,
                                harness.config.branch)
    assert (pattern.cuda_version, pattern.ubuntu_version) == ("13.2", "24.04")
    launch = containers.launch_calls[0]
    assert launch["workdir"] == str(harness.config.runtime_root / "edgellm")
    assert launch["exec_target"] is harness.run_remote.target
    assert launch["mounts"] is None
    assert launch["extra_args"] == [
        "--mounts",
        (f"{harness.config.runtime_root}:{harness.config.runtime_root},"
         f"{harness.config.onnx_root}:{harness.config.onnx_root},"
         f"{ci._PYTHON_ROOT}:{ci._PYTHON_ROOT}"),
    ]
    handle, execution = containers.exec_calls[0]
    assert handle is containers.handle
    assert execution["exec_target"] is harness.run_remote.target
    assert execution["cwd"] == str(harness.config.runtime_root / "edgellm")
    assert execution["tee_file"] == str(harness.config.local_root /
                                        "tests.log")
    command = execution["command"]
    assert f"source {harness.config.runtime_root / 'setup_environment.sh'}" in command
    assert "unitTest" not in command
    assert f"{ci._PYTHON} -m pytest" in command
    assert f"export PATH={ci._PYTHON.parent}:$PATH" in command
    assert "TestLLMPipeline::test_engine_build" in command
    assert "TestLLMPipeline::test_inference" in command
    for model in ci._E2E_MODEL_FAMILIES:
        assert f"--test-param={model}" in command
        assert f"--test-param={model}-llm_basic" in command
    assert f"export ONNX_DIR={harness.config.onnx_root}" in command
    assert f"export BUILD_DIR={harness.config.runtime_root / 'edgellm'}" in command
    assert f"export ENGINE_DIR={harness.config.runtime_root / 'engines'}" in command


def test_download_onnx_prepares_checkpoints_and_exports_in_container(
        monkeypatch, tmp_path):
    harness = _main_harness(monkeypatch,
                            tmp_path,
                            build_remote=False,
                            run_remote=False,
                            download_onnx=True)

    assert ci.main(harness.argv) == 0

    downloads = [(target, spec) for target, spec in harness.commands.calls
                 if spec.operation_name.startswith("huggingface-download-")]
    assert len(downloads) == len(ci._ONNX_MODELS)
    for (model_name, (repository, checkpoint_dir)), (target, spec) in zip(
            ci._ONNX_MODELS.items(), downloads):
        assert target is harness.run_host.target
        assert spec.argv == [
            str(ci._PYTHON.parent / "hf"), "download", repository,
            "--local-dir",
            str(harness.config.onnx_root / ".checkpoints" / checkpoint_dir)
        ]
        assert spec.output_mode is OutputMode.PROGRESS
        assert spec.artifact_log_file.endswith(f"download-{model_name}.log")

    onnx_mount = next(
        mount
        for mount in harness.code.container_manager.launch_calls[0]["mounts"]
        if mount.host_path == str(harness.config.onnx_root))
    assert not onnx_mount.read_only
    command = harness.code.container_manager.exec_calls[0][1]["command"]
    assert "test_checkpoint_export.py::test_checkpoint_export" in command
    assert f"export LLM_MODELS_DIR={harness.config.onnx_root}/.checkpoints" in command
    assert "e2e-export.xml" in command
    for model_name in ci._ONNX_MODELS:
        assert f"--test-param={model_name}-fp16" in command


def test_download_onnx_failure_stops_before_tests(monkeypatch, tmp_path):
    harness = _main_harness(monkeypatch,
                            tmp_path,
                            build_remote=False,
                            run_remote=False,
                            download_onnx=True)
    first_model = next(iter(ci._ONNX_MODELS))
    harness.commands.outcomes[f"huggingface-download-{first_model}"] = _result(
        False, 7)

    assert ci.main(harness.argv) == 7
    assert harness.code.container_manager.launch_calls == []
    assert "TRT CI validation FAILED" in harness.logger.messages[-1][1][0]


def test_local_runtime_uses_code_manager_setup_and_container_handoff(
        monkeypatch, tmp_path):
    harness = _main_harness(monkeypatch, tmp_path, False, False)

    assert ci.main(harness.argv) == 0

    assert harness.code.deploy_calls == []
    assert harness.code.setup_calls == [(harness.code.build_result,
                                         BuildComponent.EDGELLM)]
    edge = PurePosixPath(str(harness.config.edgellm_root))
    containers = harness.code.container_manager
    launch = containers.launch_calls[0]
    assert launch["exec_target"] is harness.run_host.target
    assert launch["workdir"] == str(edge)
    assert launch["mounts"] == [
        MountSpec(str(edge), str(edge)),
        MountSpec(str(harness.config.trt_location),
                  str(harness.config.trt_location),
                  read_only=True),
        MountSpec(str(harness.config.onnx_root),
                  str(harness.config.onnx_root),
                  read_only=True),
        MountSpec(str(ci._PYTHON_ROOT), str(ci._PYTHON_ROOT), read_only=True)
    ]
    assert launch["extra_args"] is None
    assert f"source {edge / 'setup_environment.sh'}" in containers.exec_calls[
        0][1]["command"]


def test_d7l_runtime_excludes_static_trt_archives(config):
    config = dataclasses.replace(config, architecture=Arch.D7L)
    original = ArtifactResult(success=True,
                              component=BuildComponent.TRT.value,
                              output_dir="/trt/full")
    run_result = RunResult(success=True, step_artifacts={"trt": original})
    commands = FakeCommands()

    staged = ci._runtime_artifacts(config, commands, run_result)

    assert run_result.step_artifacts["trt"] is original
    assert staged.step_artifacts["trt"].output_dir == str(config.local_root /
                                                          "runtime-trt")
    target, spec = commands.calls[0]
    assert isinstance(target, LocalTarget)
    assert spec.operation_name == "stage-d7l-trt-runtime"
    assert spec.argv == [
        "rsync", "-a", "--delete", "--exclude=*.a", "/trt/full/",
        f"{config.local_root / 'runtime-trt'}/"
    ]


def test_d7l_runs_python_e2e_directly_on_configured_board(
        monkeypatch, tmp_path):
    harness = _main_harness(monkeypatch,
                            tmp_path,
                            build_remote=False,
                            run_remote=True,
                            architecture=Arch.D7L)

    assert ci.main(harness.argv) == 0

    assert harness.code.deploy_calls[0][2] is None
    assert harness.code.container_manager.resolve_calls == []
    assert harness.code.container_manager.launch_calls == []
    target, spec = next((target, spec)
                        for target, spec in harness.commands.calls
                        if spec.operation_name == "edgellm-python-e2e")
    assert target is harness.run_host.target
    assert spec.shell_type.value == "bash"
    assert spec.cwd == str(harness.config.runtime_root / "edgellm")
    assert "python3 -m pytest" in spec.command
    assert str(ci._PYTHON) not in spec.command
    assert "export PATH=" not in spec.command
    assert harness.run_remote.copy_calls[0]["remote_path"] == str(
        harness.config.runtime_root / "results")


@pytest.mark.parametrize("build_status,infer_status,expected,trace",
                         [(9, 0, 9, ["build"]), (0, 8, 8, ["build", "infer"]),
                          (0, 0, 0, ["build", "infer"])])
def test_test_command_fails_fast(tmp_path, monkeypatch, config, build_status,
                                 infer_status, expected, trace):
    workspace = tmp_path / "runtime with spaces"
    edge = workspace / "edgellm"
    tests = edge / "tests/defs"
    tests.mkdir(parents=True)
    (tests / "test_llm_pipeline.py").write_text("# test seam\n")
    (edge / "setup_environment.sh").write_text("export SETUP_SOURCED=1\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(r'''#!/usr/bin/env bash
set -eu
test "$SETUP_SOURCED" = 1 || exit 99
case " $* " in
  *TestLLMPipeline::test_engine_build*)
    printf "build\n" >> "$TRACE"; exit "$BUILD_STATUS";;
  *TestLLMPipeline::test_inference*)
    printf "infer\n" >> "$TRACE"; exit "$INFER_STATUS";;
esac
exit 98
''')
    fake_python.chmod(0o755)
    monkeypatch.setattr(ci, "_PYTHON", fake_python)
    monkeypatch.setattr(ci, "_PYTHON_ROOT", bin_dir)
    trace_file = tmp_path / "trace"
    runtime = ci.Runtime(LocalTarget(), PurePosixPath(str(workspace)),
                         PurePosixPath(str(edge)),
                         PurePosixPath(str(edge / "setup_environment.sh")))
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRACE": str(trace_file),
        "BUILD_STATUS": str(build_status),
        "INFER_STATUS": str(infer_status),
    }

    result = subprocess.run(
        ["bash", "-c", ci._test_command(config, runtime)],
        cwd=edge,
        env=env,
        check=False)

    assert result.returncode == expected
    assert trace_file.read_text().splitlines() == trace


@pytest.mark.parametrize("case,run_remote,expected,removed",
                         [("build", True, 1, False),
                          ("deploy", True, 1, False),
                          ("setup", False, 1, False),
                          ("resolve", True, 1, False),
                          ("launch", True, 1, False),
                          ("exec-false", True, 1, True),
                          ("exec-error", True, 1, True),
                          ("remove-false", True, 1, True),
                          ("remove-error", True, 1, True)])
def test_main_unifies_failures(monkeypatch, tmp_path, case, run_remote,
                               expected, removed):
    harness = _main_harness(monkeypatch, tmp_path, True, run_remote)
    containers = harness.code.container_manager
    if case == "build":
        harness.code.build_result = RunResult(success=False,
                                              error_messages=["broken build"])
    elif case == "deploy":
        harness.code.deploy_error = RuntimeError("deploy failed")
    elif case == "setup":
        harness.code.setup_error = RuntimeError("setup failed")
    elif case == "resolve":
        containers.resolve_error = RuntimeError("resolve failed")
    elif case == "launch":
        containers.launch_error = RuntimeError("launch failed")
    elif case == "exec-false":
        containers.exec_ok = False
    elif case == "exec-error":
        containers.exec_error = RuntimeError("test failed")
    elif case == "remove-false":
        containers.remove_ok = False
    elif case == "remove-error":
        containers.remove_error = RuntimeError("remove failed")

    assert ci.main(harness.argv) == expected
    assert harness.logger.messages[-1][0] == "error"
    assert "TRT CI validation FAILED" in harness.logger.messages[-1][1][0]
    assert bool(containers.remove_calls) is removed
    if harness.run_remote:
        assert harness.run_remote.filesystem.remove_calls == [
            (str(harness.config.run_root), {
                "timeout_s": ci._TRANSFER_TIMEOUT_S
            })
        ]


def test_structure_has_no_worker_or_manual_runtime_copy():
    source = Path(ci.__file__).read_text()
    tree = ast.parse(source)
    classes = {
        node.name
        for node in tree.body if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name
        for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    imports = {
        alias.name
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not classes.intersection(
        {"ControllerFlow", "BuildWorker", "ValidationFlow", "WorkerServices"})
    assert {"_build", "_deploy", "_run_tests", "main"}.issubset(functions)
    assert "_stage_source" not in functions
    assert "--build-worker" not in source
    assert source.count(".copy_local_directory_to_remote(") == 0
    assert source.count(".copy_remote_directory_to_local(") == 1
    assert ".copy_remote_file_to_local(" not in source
    assert "deploy_runtime(" in source
    assert "write_environment_setup_script(" in source
    assert "load_registry" not in source
    assert "from_target_name" not in source
    assert "resolve_endpoint" not in source
    assert "subprocess" not in imports
