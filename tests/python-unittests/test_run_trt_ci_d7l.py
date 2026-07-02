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
"""Behavior tests for the TRT-internal D7L CI controller."""

import argparse
import ast
import dataclasses
import shlex
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("trt_dev_toolkit", reason="TRT-internal CI dependency")

from trt_dev_toolkit.code_manager import (ArtifactSource, BuildComponent,
                                          BuildMode, DeploymentMode,
                                          DeploymentResult, EnvironmentExports,
                                          Plan, PlanStep, RunResult)
from trt_dev_toolkit.command_manager.command_manager import CommandManager
from trt_dev_toolkit.command_manager.data_structures import (CommandResult,
                                                             OutputMode,
                                                             ShellType)
from trt_dev_toolkit.command_manager.targets import (LocalTarget,
                                                     RemoteSshTarget)
from trt_dev_toolkit.constants import Arch, TargetType

from scripts import run_trt_ci_d7l as ci


def _result(success=True, exit_code=0, timed_out=False):
    return CommandResult(success=success,
                         exit_code=exit_code,
                         timed_out=timed_out)


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

    def __init__(self, name, config, events):
        self.name = name
        self.config = config
        self.events = events
        self.target = RemoteSshTarget(config)
        self.filesystem = FakeFilesystem(name, events)
        self.connected = True
        self.upload_fail_names = set()
        self.download_fail_names = set()
        self.uploads = []
        self.downloads = []

    def test_connection(self):
        self.events.append(f"{self.name}:probe")
        return self.connected

    def copy_local_directory_to_remote(self, *, local_path, remote_path,
                                       timeout_s):
        name = PurePosixPath(remote_path).name
        self.uploads.append((local_path, remote_path, timeout_s))
        self.events.append(f"{self.name}:upload:{name}")
        return name not in self.upload_fail_names

    def copy_remote_directory_to_local(self, *, remote_path, local_path,
                                       timeout_s):
        name = PurePosixPath(remote_path).name
        self.downloads.append((remote_path, local_path, timeout_s))
        self.events.append(f"{self.name}:download:{name}")
        if name in self.download_fail_names:
            return False
        Path(local_path).mkdir(parents=True, exist_ok=True)
        return True


class FakeCode:

    def __init__(self, events=None, remote_target=None):
        self.events = events if events is not None else []
        self.remote_target = remote_target
        self.plan_calls = []
        self.build_calls = []
        self.deploy_calls = []
        self.build_result = RunResult(success=True)
        self.deploy_error = None

    def plan_artifact_generation(self, targets):
        self.plan_calls.append(list(targets))
        self.events.append("code:plan")
        return Plan(
            run_id="unit-plan",
            steps=[
                PlanStep(
                    step_id=f"s{index:03d}",
                    target=target,
                    component=BuildComponent(target.component),
                ) for index, target in enumerate(targets)
            ],
        )

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
        if self.deploy_error is not None:
            raise self.deploy_error
        workspace = remote_config.paths.remote_path
        return DeploymentResult(
            remote_target=self.remote_target,
            remote_workspace=workspace,
            env_script_path=f"{workspace}/setup_environment.sh",
            environment_exports=EnvironmentExports(),
            deployment_mode=DeploymentMode.RSYNC,
        )


@dataclasses.dataclass
class Harness:
    events: list
    commands: FakeCommands
    build_remote: FakeRemote
    test_remote: FakeRemote
    code: FakeCode

    @property
    def services(self):
        return ci.ControllerServices(self.commands, self.build_remote,
                                     self.test_remote, self.code)


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "Edge LLM source"
    (source / "unittests/resources").mkdir(parents=True)
    (source / "tests/chat_templates").mkdir(parents=True)
    value = ci.CiConfig(
        trt_root=PurePosixPath("/candidate TRT/source"),
        trt_build_dir=PurePosixPath("/candidate TRT/build"),
        trt_branch="main",
        build_endpoint=ci.Endpoint("builder.example", "builder", 2201,
                                   "jump-user@jump.example:2222"),
        test_endpoint=ci.Endpoint("d7l.example", "tester", 2202),
        workspace=PurePosixPath("/remote Edge workspace"),
        run_id="unit-451",
        artifacts_dir=source / "ci artifacts",
        cuda_version="13.2",
        jobs=7,
        gtest_filter="Suite.$case;*",
        connection_timeout_s=9,
        transfer_timeout_s=30,
        build_timeout_s=0.5,
        test_timeout_s=60,
        strict_host_keys=False,
        known_hosts_file="/tmp/known hosts",
        source_root=source,
    )
    value.validate()
    return value


def _harness(config):
    events = []
    commands = FakeCommands(events)
    build_cfg = ci.remote_config(
        config,
        config.build_endpoint,
        TargetType.LINUX,
        Arch.X86_64,
        config.local_root / "build-transfer",
    )
    test_cfg = ci.remote_config(
        config,
        config.test_endpoint,
        TargetType.THOR_LINUX,
        Arch.D7L,
        config.local_root / "deploy-stage",
    )
    build_remote = FakeRemote("build", build_cfg, events)
    test_remote = FakeRemote("test", test_cfg, events)
    code = FakeCode(events, test_remote.target)
    return Harness(events, commands, build_remote, test_remote, code)


def _worker_args(tmp_path):
    return argparse.Namespace(
        run_root=PurePosixPath(str(tmp_path / "worker run")),
        trt_root=PurePosixPath("/candidate TRT/source"),
        trt_build_dir=PurePosixPath("/candidate TRT/build"),
        trt_branch="main",
        cuda_version="13.2",
        jobs=7,
        package_timeout=30,
    )


def _assert_subsequence(values, expected):
    iterator = iter(values)
    for item in expected:
        assert any(value == item for value in iterator), (item, values)


def test_remote_configs_map_linux_and_d7l_endpoints(config):
    build = ci.remote_config(config, config.build_endpoint, TargetType.LINUX,
                             Arch.X86_64, config.local_root / "build")
    test = ci.remote_config(config, config.test_endpoint,
                            TargetType.THOR_LINUX, Arch.D7L,
                            config.local_root / "test")

    assert (build.target.target_type, build.target.arch) == (TargetType.LINUX,
                                                             Arch.X86_64)
    assert (test.target.target_type,
            test.target.arch) == (TargetType.THOR_LINUX, Arch.D7L)
    assert build.target.password == test.target.password == ""
    assert (build.target.port, test.target.port) == (2201, 2202)
    assert build.jump_host.host == "jump.example"
    assert (build.jump_host.username, build.jump_host.port) == ("jump-user",
                                                                2222)
    assert build.ssh.batch_mode and not build.ssh.strict_host_key_checking
    assert build.ssh.known_hosts_file == "/tmp/known hosts"
    assert build.jump_host.ssh is build.ssh
    assert build.jump_host.ssh.connect_timeout_s == 9
    assert build.jump_host.ssh.batch_mode
    assert not build.jump_host.ssh.strict_host_key_checking
    assert build.jump_host.ssh.known_hosts_file == "/tmp/known hosts"
    assert build.paths.remote_path == str(config.run_root)

    for unsafe in (PurePosixPath("//"), PurePosixPath("/tmp/../escape")):
        with pytest.raises(ValueError, match="canonical absolute"):
            dataclasses.replace(config, workspace=unsafe).validate()


def test_build_targets_use_prebuilt_trt_and_source_edgellm(config):
    targets = ci.build_targets(config.run_root, config.trt_root,
                               config.trt_branch, config.cuda_version,
                               config.jobs)
    trt, edge = targets

    assert [target.component for target in targets
            ] == [BuildComponent.TRT, BuildComponent.EDGELLM]
    assert [target.source for target in targets
            ] == [ArtifactSource.PRE_BUILT, ArtifactSource.BUILD]
    assert all(target.mode is BuildMode.RELEASE for target in targets)
    assert all(target.platform.arch is Arch.D7L for target in targets)
    assert trt.build.build_dir == str(config.remote("trt-package"))
    assert trt.build.repo_path == str(config.trt_root)
    assert edge.build.repo_path == str(config.remote("source"))
    assert edge.build.build_dir == str(config.remote("build"))
    assert edge.build.parallel_jobs == 7
    assert edge.build.trt_package_dir is None
    assert edge.build.cmake_args == [
        "-DEMBEDDED_TARGET=auto-thor",
        f"-DCMAKE_TOOLCHAIN_FILE={config.remote('source')}/cmake/aarch64_linux_toolchain.cmake",
        "-DENABLE_CUTE_DSL=OFF",
    ]


def test_worker_prepares_package_then_uses_code_manager(tmp_path):
    args = _worker_args(tmp_path)
    commands, code = FakeCommands(), FakeCode()
    assert ci.BuildWorker(args, ci.WorkerServices(commands, code)).run() == 0

    target, spec = commands.calls[0]
    assert isinstance(target, LocalTarget)
    assert spec.operation_name == "prepare-trt-package"
    assert spec.shell_type is ShellType.BASH
    for text in ("NvInferVersion.h", "parsers/onnx", "Release/lib",
                 "libnvinfer.so", "libnvonnxparser.so", "cp -a", "readlink -f",
                 'case "$resolved"', '"$package"/lib/*', "-type l"):
        assert text in spec.command
    assert len(code.build_calls) == 1

    failed_commands = FakeCommands(
        outcomes={"prepare-trt-package": _result(False, 23)})
    unused_code = FakeCode()
    assert ci.BuildWorker(args, ci.WorkerServices(failed_commands,
                                                  unused_code)).run() == 23
    assert not unused_code.build_calls

    failed_code = FakeCode()
    failed_code.build_result = RunResult(success=False,
                                         error_messages=["build failed"])
    assert ci.BuildWorker(args, ci.WorkerServices(FakeCommands(),
                                                  failed_code)).run() == 1


def test_deployment_result_describes_local_prebuilt_artifacts(tmp_path):
    trt, edge = tmp_path / "runtime/trt", tmp_path / "runtime/edgellm"
    trt.mkdir(parents=True)
    edge.mkdir(parents=True)
    code = FakeCode()

    result = ci.deployment_result(code, trt, edge, "main", "13.2")

    assert result.success and len(code.plan_calls) == 1
    assert not code.build_calls
    assert [step.component for step in result.plan.steps
            ] == [BuildComponent.TRT, BuildComponent.EDGELLM]
    assert all(step.target.source is ArtifactSource.PRE_BUILT
               for step in result.plan.steps)
    assert [
        result.step_artifacts[step.step_id].output_dir
        for step in result.plan.steps
    ] == [str(trt.resolve()), str(edge.resolve())]


def test_controller_uses_toolkit_services_in_order(config):
    harness = _harness(config)
    assert ci.ControllerFlow(config, harness.services).run() == 0

    _assert_subsequence(harness.events, [
        "build:probe",
        "build:remove",
        "test:probe",
        "test:remove",
        "build:ensure",
        "command:build-host-prerequisites",
        "command:d7l-prerequisites",
        "command:stage-source",
        "build:upload:source",
        "command:build-worker",
        "build:download:trt-package",
        "build:download:build",
        "code:plan",
        "code:deploy",
        "test:upload:resources",
        "test:upload:chat_templates",
        "command:d7l-unit-tests",
        "build:download:artifacts",
        "test:download:results",
        "build:remove",
        "test:remove",
    ])
    worker = next(spec for _target, spec in harness.commands.calls
                  if spec.operation_name == "build-worker")
    assert worker.argv[0] == "timeout" and "--build-worker" in worker.argv
    assert "0.5s" in worker.argv
    assert worker.env == {
        "PYTHONPATH": f"{config.trt_root}/scripts/devToolkit/src"
    }
    assert worker.timeout_s > config.build_timeout_s
    assert harness.code.deploy_calls[0][2] is DeploymentMode.RSYNC

    probes = {
        spec.operation_name: spec
        for _target, spec in harness.commands.calls if spec.operation_name in
        {"build-host-prerequisites", "d7l-prerequisites"}
    }
    assert all(spec.output_mode is OutputMode.CAPTURE
               for spec in probes.values())
    for text in ("command -v timeout", config.build_python):
        assert text in probes["build-host-prerequisites"].command
    for text in ("aarch64", "/etc/nvidia/version-ubuntu-rootfs.txt",
                 "/proc/device-tree/compatible", "grep -qi 'nvidia,tegra264'",
                 "command -v bash ldd readlink awk tee grep tr"):
        assert text in probes["d7l-prerequisites"].command
    stage = next(spec for _target, spec in harness.commands.calls
                 if spec.operation_name == "stage-source")
    assert "--exclude=/ci artifacts" in stage.argv

    tree = ast.parse(Path(ci.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "subprocess" not in imports
    assert not any(
        isinstance(node, ast.ClassDef) and "Runner" in node.name
        for node in ast.walk(tree))


def test_test_spec_proves_candidate_trt_and_stages_resources(config):
    harness = _harness(config)
    assert ci.ControllerFlow(config, harness.services).run() == 0
    target, spec = next(call for call in harness.commands.calls
                        if call[1].operation_name == "d7l-unit-tests")

    assert target is harness.test_remote.target
    assert spec.shell_type is ShellType.BASH
    assert spec.output_mode is OutputMode.PROGRESS
    assert spec.cwd == str(config.remote("edgellm"))
    for text in (
            f"source {shlex.quote(str(config.run_root / 'setup_environment.sh'))}",
            "not found",
            "readlink -f",
            f"{shlex.quote(str(config.run_root / 'trt'))}/*",
            "libnvinfer",
            "libnvonnxparser",
            "--gtest_filter='Suite.$case;*'",
            "--gtest_output=xml:",
            "PIPESTATUS[0]",
    ):
        assert text in spec.command
    assert [
        remote for _local, remote, _timeout in harness.test_remote.uploads
    ] == [
        str(config.remote("source/unittests/resources")),
        str(config.remote("source/tests/chat_templates")),
    ]


def test_test_spec_runs_locally_with_candidate_paths_containing_spaces(
        config, tmp_path):
    config = dataclasses.replace(
        config,
        workspace=PurePosixPath(str(tmp_path / "D7L workspace with spaces")),
        artifacts_dir=tmp_path / "controller artifacts",
    )
    config.validate()
    workspace = Path(str(config.run_root))
    edge = workspace / "edgellm"
    candidate = workspace / "trt"
    tools = tmp_path / "fake tools with spaces"
    (edge / "examples/llm").mkdir(parents=True)
    candidate.mkdir(parents=True)
    tools.mkdir()
    for library in ("libnvinfer.so.10", "libnvonnxparser.so.10"):
        (candidate / library).touch()

    def executable(path, script):
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)

    executable(
        edge / "unitTest",
        """#!/bin/bash
for arg in "$@"; do
  case "$arg" in
    --gtest_output=xml:*)
      output="${arg#--gtest_output=xml:}"
      mkdir -p "$(dirname "$output")"
      printf '<testsuites/>\n' >"$output"
      ;;
  esac
done
printf 'unit tests passed\n'
""",
    )
    executable(edge / "examples/llm/llm_build", "#!/bin/bash\nexit 0\n")
    executable(
        tools / "ldd",
        """#!/bin/bash
printf 'libnvinfer.so.10 => %s (0xabc)\n' "$FAKE_TRT/libnvinfer.so.10"
case "$1" in
  *llm_build)
    printf 'libnvonnxparser.so.10 => %s (0xdef)\n' "$FAKE_TRT/libnvonnxparser.so.10"
    ;;
esac
""",
    )
    setup = workspace / "setup_environment.sh"
    setup.write_text(
        f"export PATH={shlex.quote(str(tools))}:$PATH\n"
        f"export FAKE_TRT={shlex.quote(str(candidate))}\n",
        encoding="utf-8",
    )
    config.local_root.mkdir(parents=True)
    harness = _harness(config)
    flow = ci.ControllerFlow(config, harness.services)
    flow.deployment = DeploymentResult(
        remote_target=harness.test_remote.target,
        remote_workspace=str(workspace),
        env_script_path=str(setup),
        environment_exports=EnvironmentExports(),
        deployment_mode=DeploymentMode.RSYNC,
    )

    result = CommandManager().run(LocalTarget(), flow._test_spec())

    assert result.success and result.exit_code == 0
    assert (workspace / "results/unit-tests.xml").read_text(
        encoding="utf-8") == "<testsuites/>\n"
    assert "unit tests passed" in (workspace /
                                   "results/unit-tests.log").read_text(
                                       encoding="utf-8")


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("probe", 1),
        ("source-stage", 19),
        ("source-upload", 1),
        ("worker", 37),
        ("retrieve", 1),
        ("deploy", 1),
        ("resource", 1),
    ],
)
def test_primary_failures_stop_before_testing(config, failure,
                                              expected_status):
    harness = _harness(config)
    if failure == "probe":
        harness.build_remote.connected = False
    elif failure == "source-stage":
        harness.commands.outcomes["stage-source"] = _result(False, 19)
    elif failure == "source-upload":
        harness.build_remote.upload_fail_names.add("source")
    elif failure == "worker":
        harness.commands.outcomes["build-worker"] = _result(False, 37)
    elif failure == "retrieve":
        harness.build_remote.download_fail_names.add("trt-package")
    elif failure == "deploy":
        harness.code.deploy_error = RuntimeError("deploy failed")
    elif failure == "resource":
        harness.test_remote.upload_fail_names.add("resources")

    assert ci.ControllerFlow(config, harness.services).run() == expected_status
    assert "command:d7l-unit-tests" not in harness.events
    assert len(harness.build_remote.filesystem.remove_calls) <= 1
    assert len(harness.test_remote.filesystem.remove_calls) <= 1


@pytest.mark.parametrize(
    ("test_result", "collection_fails", "keep_workspace", "expected",
     "cleanup"),
    [
        (_result(False, 41), False, False, 41, False),
        (_result(False, None, True), False, False, 124, False),
        (_result(False, None), True, False, 1, False),
        (_result(), True, False, 0, True),
        (_result(), False, True, 0, False),
    ],
)
def test_test_status_collection_and_cleanup_precedence(config, test_result,
                                                       collection_fails,
                                                       keep_workspace,
                                                       expected, cleanup):
    config = dataclasses.replace(config, keep_workspace=keep_workspace)
    harness = _harness(config)
    harness.commands.outcomes["d7l-unit-tests"] = test_result
    if collection_fails:
        harness.test_remote.download_fail_names.add("results")

    assert ci.ControllerFlow(config, harness.services).run() == expected
    assert any(
        PurePosixPath(remote).name == "results"
        for remote, _local, _timeout in harness.test_remote.downloads)
    expected_removals = 2 if cleanup else 1
    assert len(
        harness.build_remote.filesystem.remove_calls) == expected_removals
    assert len(
        harness.test_remote.filesystem.remove_calls) == expected_removals
