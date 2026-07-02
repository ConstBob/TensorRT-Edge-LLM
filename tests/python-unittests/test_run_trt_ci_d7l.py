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

import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import shlex
import subprocess
import sys
import time

import pytest

from scripts import run_trt_ci_d7l


def _assert_option(argv, option, value):
    index = argv.index(option)
    assert argv[index + 1] == value


def test_ssh_endpoint_renders_noninteractive_ssh_scp_and_rsync(tmp_path):
    identity = tmp_path / "identity key"
    identity.touch()
    target = run_trt_ci_d7l.SshTarget(
        "build.example",
        "ci-user",
        port=2201,
        identity_file=identity,
        jump_host="jump-user@jump.example:2222",
        host_key_policy="no",
    )

    ssh_argv = target.ssh_argv("printf ok", 9)
    assert ssh_argv[0:2] == ["ssh", "-T"]
    for option in [
            "BatchMode=yes",
            "StrictHostKeyChecking=no",
            "UserKnownHostsFile=/dev/null",
    ]:
        assert option in ssh_argv
    _assert_option(ssh_argv, "-p", "2201")
    _assert_option(ssh_argv, "-i", str(identity))
    _assert_option(ssh_argv, "-J", "jump-user@jump.example:2222")
    assert ssh_argv[-4:] == [
        "ci-user@build.example",
        "bash",
        "-lc",
        shlex.quote("printf ok"),
    ]

    remote_path = pathlib.PurePosixPath("/tmp/run with spaces/$unsafe;name")
    remote_spec = "ci-user@build.example:" + shlex.quote(str(remote_path))
    assert target.remote_spec(remote_path) == remote_spec
    scp_argv = target.scp_argv("local archive", remote_spec, 9)
    assert scp_argv[0:2] == ["scp", "-O"]
    _assert_option(scp_argv, "-P", "2201")
    assert scp_argv[-2:] == ["local archive", remote_spec]

    rsync_argv = shlex.split(target.rsync_transport(9))
    assert rsync_argv[0] == "ssh"
    _assert_option(rsync_argv, "-p", "2201")
    _assert_option(rsync_argv, "-i", str(identity))
    assert not any("password" in token.lower()
                   for token in ssh_argv + scp_argv + rsync_argv)
    with pytest.raises(ValueError, match="control"):
        target.remote_spec(pathlib.PurePosixPath("/tmp/bad\npath"))


@pytest.mark.parametrize(("field", "value"), [
    ("host", "-oProxyCommand=touch-pwned"),
    ("user", "-Fattacker-config"),
    ("jump_host", "-oProxyCommand=touch-pwned"),
    ("host", "build.example\n-oProxyCommand=pwned"),
    ("jump_host", "jump.example\n-oProxyCommand=pwned"),
])
def test_ssh_target_rejects_option_and_control_injection(field, value):
    endpoint = {"host": "build.example", "user": "ci-user", field: value}
    with pytest.raises(ValueError, match="SSH"):
        run_trt_ci_d7l.SshTarget(**endpoint)


def _process_is_running(pid):
    stat_path = pathlib.Path("/proc") / str(pid) / "stat"
    if not stat_path.exists():
        return False
    return stat_path.read_text(encoding="utf-8").split()[2] != "Z"


def test_command_runner_times_out_silent_process_group(tmp_path):
    child_script = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)")
    pid_file = tmp_path / "child.pid"
    parent_script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
        "pathlib.Path(sys.argv[2]).write_text(str(child.pid)); time.sleep(60)")
    runner = run_trt_ci_d7l.CommandRunner(tmp_path)

    start = time.monotonic()
    with pytest.raises(run_trt_ci_d7l.FlowError, match="timed out"):
        runner.run(
            "silent-timeout",
            [sys.executable, "-c", parent_script, child_script,
             str(pid_file)],
            timeout_s=0.5,
        )
    assert time.monotonic() - start < 3.0

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def test_command_runner_bounds_reader_after_leader_exits_with_detached_child(
        tmp_path):
    pid_file = tmp_path / "detached-pipe-child.pid"
    child_script = ("import os,pathlib,signal,sys,time\n"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                    "time.sleep(60)\n")
    leader_script = (
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],"
        "start_new_session=True)\n"
        "marker=pathlib.Path(sys.argv[2])\n"
        "deadline=time.monotonic()+2\n"
        "while not marker.exists():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(2)\n"
        "    time.sleep(0.01)\n")
    runner = run_trt_ci_d7l.CommandRunner(tmp_path)

    start = time.monotonic()
    try:
        with pytest.raises(run_trt_ci_d7l.FlowError) as error:
            runner.run(
                "detached-child",
                [
                    sys.executable, "-c", leader_script, child_script,
                    str(pid_file)
                ],
                timeout_s=0.5,
            )
        assert error.value.exit_code == 124
        assert time.monotonic() - start < 3.0
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text(encoding="utf-8")), 9)
            except ProcessLookupError:
                pass


def test_command_runner_streams_complete_output_to_phase_log(tmp_path):
    runner = run_trt_ci_d7l.CommandRunner(tmp_path)
    command = "for i in range(205): print('line-{}'.format(i))"

    runner.run("streamed-output", [sys.executable, "-c", command], timeout_s=5)

    log = (tmp_path / "streamed-output.log").read_text(encoding="utf-8")
    assert "line-0\n" in log and "line-204\n" in log


def test_command_runner_preserves_exact_exit_code(tmp_path):
    runner = run_trt_ci_d7l.CommandRunner(tmp_path)
    with pytest.raises(run_trt_ci_d7l.FlowError) as error:
        runner.run(
            "known-exit",
            [sys.executable, "-c", "raise SystemExit(23)"],
            timeout_s=5,
        )
    assert error.value.exit_code == 23


@pytest.fixture
def flow_config(tmp_path):
    repo_root = pathlib.Path(run_trt_ci_d7l.__file__).resolve().parents[1]
    return run_trt_ci_d7l.TrtCiConfig(
        source_root=repo_root,
        trt_root=pathlib.PurePosixPath("/candidate TRT/source;$root"),
        trt_build_dir=pathlib.PurePosixPath("/candidate TRT/build $output"),
        build_target=run_trt_ci_d7l.SshTarget("build.example", "builder"),
        test_target=run_trt_ci_d7l.SshTarget("d7l.example", "tester"),
        workspace=pathlib.PurePosixPath("/tmp/edge llm validation"),
        run_id="unit-123",
        artifacts_dir=tmp_path / "artifacts",
        cuda_version="13.2",
        jobs=7,
        gtest_filter="LoggerTest.$case;*",
        connection_timeout_s=9,
        build_timeout_s=120,
        transfer_timeout_s=30,
        test_timeout_s=60,
        keep_workspace=True,
    )


def test_source_root_and_workspace_are_independent_of_cwd(
        flow_config, tmp_path, monkeypatch):
    script_path = pathlib.Path(run_trt_ci_d7l.__file__).resolve()
    expected_root = script_path.parents[1]
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("_trt_ci_cwd_test",
                                                  script_path)
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    parsed = loaded.parse_args([
        "--trt-root",
        "/remote/trt",
        "--trt-build-dir",
        "/remote/build",
        "--build-host",
        "build",
        "--build-user",
        "ci",
        "--test-host",
        "test",
        "--test-user",
        "ci",
        "--artifacts-dir",
        str(tmp_path / "results"),
    ])

    assert parsed.source_root == expected_root
    assert flow_config.run_workspace == pathlib.PurePosixPath(
        "/tmp/edge llm validation/run-unit-123")


def _make_checkout(root, *, include_nvtx):
    """Create a minimal checkout with selected submodule marker files."""
    files = [
        "CMakeLists.txt",
        "3rdParty/googletest/CMakeLists.txt",
        "3rdParty/nlohmannJson/CMakeLists.txt",
    ]
    if include_nvtx:
        files.append("3rdParty/NVTX/CMakeLists.txt")
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_main_reports_missing_nvtx_submodule(tmp_path, flow_config,
                                             monkeypatch, capsys):
    source_root = _make_checkout(tmp_path / "source", include_nvtx=False)
    config = dataclasses.replace(flow_config, source_root=source_root)
    monkeypatch.setattr(run_trt_ci_d7l, "parse_args", lambda argv: config)

    assert run_trt_ci_d7l.main([]) == 1
    assert "3rdParty/NVTX/CMakeLists.txt" in capsys.readouterr().err


@pytest.mark.parametrize("workspace", [
    pathlib.PurePosixPath("/"),
    pathlib.PurePosixPath("//"),
    pathlib.PurePosixPath("relative/workspace"),
    pathlib.PurePosixPath("/tmp/../escape"),
])
def test_config_rejects_unsafe_workspace(flow_config, workspace):
    with pytest.raises(ValueError, match="workspace"):
        dataclasses.replace(flow_config, workspace=workspace)


@pytest.mark.parametrize("run_id",
                         ["../escape", "nested/id", ".", "bad;command"])
def test_config_rejects_unsafe_run_id(flow_config, run_id):
    with pytest.raises(ValueError, match="run"):
        dataclasses.replace(flow_config, run_id=run_id)


def test_build_and_bundle_quote_paths_and_use_full_candidate_trt(flow_config):
    flow = run_trt_ci_d7l.TrtCiFlow(flow_config, runner=object())
    script = flow.render_build_script()
    workspace = flow_config.run_workspace

    for path in [
            flow_config.trt_root,
            flow_config.trt_build_dir,
            workspace / "source",
    ]:
        assert shlex.quote(str(path)) in script
    for required in [
            "-DBUILD_UNIT_TESTS=ON",
            "-DEMBEDDED_TARGET=auto-thor",
            "cmake/aarch64_linux_toolchain.cmake",
            "-DTRT_PACKAGE_DIR=\"$pkg\"",
            "-DCUDA_CTK_VERSION=\"$cuda_version\"",
            "-DENABLE_CUTE_DSL=OFF",
            "include/NvInferVersion.h",
            "parsers/onnx",
            "NvOnnxParser.h",
    ]:
        assert required in script
    assert "cmake --build" in script and "--parallel 7" in script
    assert "--target unitTest" not in script
    assert "/usr/local/cuda/bin/nvcc --version" in run_trt_ci_d7l.TrtCiFlow(
        dataclasses.replace(flow_config, cuda_version=None),
        runner=object()).render_build_script()

    bundle = flow.render_bundle_script()
    for required in [
            "trt-package/lib",
            "llm_build",
            "unittests/resources",
            "tests/chat_templates",
            "-name '*.so*'",
    ]:
        assert required in bundle


def test_runtime_script_proves_candidate_trt_and_preserves_gtest_status(
        flow_config):
    flow = run_trt_ci_d7l.TrtCiFlow(flow_config, runner=object())
    script = flow.render_test_script()
    for required in [
            "not found",
            "libnvonnxparser",
            str(flow_config.run_workspace / "trt-package/lib"),
            "LD_LIBRARY_PATH",
            "check_ldd \"$run/build/llm_build\" 1",
            "--gtest_output=xml:",
            "PIPESTATUS",
    ]:
        assert required in script
    assert "--gtest_filter=" + shlex.quote(flow_config.gtest_filter) in script


def _complete_trt_lib_dir(path, marker=None):
    path.mkdir(parents=True)
    for name in ["libnvinfer.so.10", "libnvonnxparser.so.10"]:
        (path / name).write_text(name, encoding="utf-8")
    if marker:
        (path / marker).touch()


def _run_rendered_library_stage(flow, trt_build, package):
    lines = flow.render_build_script().splitlines()
    start = lines.index('exact="$trt_build/Release/lib"; lib_dir=')
    end = next(index for index, line in enumerate(lines[start:], start)
               if line.startswith('test -f "$pkg/include/NvInfer.h"'))
    script = "\n".join([
        "set -euo pipefail",
        "trt_build={}".format(shlex.quote(str(trt_build))),
        "pkg={}".format(shlex.quote(str(package))),
        'mkdir -p "$pkg/lib"',
    ] + lines[start:end])
    return subprocess.run(["bash", "-c", script],
                          text=True,
                          capture_output=True,
                          check=False)


def test_rendered_library_selection_prefers_exact_and_rejects_ambiguity(
        tmp_path, flow_config):
    flow = run_trt_ci_d7l.TrtCiFlow(flow_config, runner=object())
    trt_build = tmp_path / "TRT build"
    exact = trt_build / "Release/lib"
    _complete_trt_lib_dir(exact, "selected-exact")
    _complete_trt_lib_dir(trt_build / "a/Release/lib", "fallback-a")
    _complete_trt_lib_dir(trt_build / "b/Release/lib", "fallback-b")

    package = tmp_path / "package exact"
    result = _run_rendered_library_stage(flow, trt_build, package)
    assert result.returncode == 0, result.stderr
    assert (package / "lib/selected-exact").is_file()
    assert not (package / "lib/fallback-a").exists()

    ambiguous = tmp_path / "ambiguous build"
    _complete_trt_lib_dir(ambiguous / "a/Release/lib")
    _complete_trt_lib_dir(ambiguous / "b/Release/lib")
    result = _run_rendered_library_stage(flow, ambiguous,
                                         tmp_path / "package ambiguous")
    assert result.returncode == 2
    assert "Expected one complete Release/lib, found 2" in result.stderr


def test_rendered_library_stage_rejects_unsafe_symlink(tmp_path, flow_config):
    flow = run_trt_ci_d7l.TrtCiFlow(flow_config, runner=object())
    trt_build = tmp_path / "unsafe build"
    exact = trt_build / "Release/lib"
    _complete_trt_lib_dir(exact)
    (exact / "libunsafe.so").symlink_to("/tmp/outside-candidate-trt.so")

    result = _run_rendered_library_stage(flow, trt_build,
                                         tmp_path / "unsafe package")

    assert result.returncode == 2
    assert "Absolute TRT symlink" in result.stderr


def test_rendered_ldd_accepts_candidate_path_with_spaces(
        tmp_path, flow_config):
    config = dataclasses.replace(
        flow_config,
        workspace=pathlib.PurePosixPath(str(tmp_path / "remote workspace")),
        run_id="ldd-spaces",
    )
    flow = run_trt_ci_d7l.TrtCiFlow(config, runner=object())
    run = pathlib.Path(str(config.run_workspace))
    package = run / "trt-package/lib"
    build = run / "build"
    package.mkdir(parents=True)
    build.mkdir(parents=True)
    for library in ["libnvinfer.so.10", "libnvonnxparser.so.10"]:
        (package / library).touch()
    for binary in ["unitTest", "llm_build"]:
        path = build / binary
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    ldd = stub_dir / "ldd"
    ldd.write_text(
        "#!/bin/bash\n"
        "printf 'libnvinfer.so.10 => %s (0xabc)\\n' "
        "\"$FAKE_PACKAGE/libnvinfer.so.10\"\n"
        "case $1 in *llm_build) "
        "printf 'libnvonnxparser.so.10 => %s (0xdef)\\n' "
        "\"$FAKE_PACKAGE/libnvonnxparser.so.10\";; esac\n",
        encoding="utf-8",
    )
    ldd.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "FAKE_PACKAGE":
        str(package),
        "PATH":
        str(stub_dir) + os.pathsep + environment["PATH"],
    })

    result = subprocess.run(
        ["bash", "-c", flow.render_test_script()],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(package) in (run /
                            "results/ldd.log").read_text(encoding="utf-8")


class _RecordingRunner:

    def __init__(self, failures=None):
        self.calls = []
        self.failures = failures or {}

    @property
    def phases(self):
        return [phase for phase, _argv in self.calls]

    def run(self, phase, argv, timeout_s):
        self.calls.append((phase, list(argv)))
        payload = b"candidate TRT bundle"
        if phase == "download-bundle":
            pathlib.Path(argv[-1]).write_bytes(payload)
        elif phase == "download-digest":
            pathlib.Path(argv[-1]).write_text(
                hashlib.sha256(payload).hexdigest() + "  bundle\n",
                encoding="utf-8",
            )
        if phase in self.failures:
            raise run_trt_ci_d7l.FlowError(phase + " failed",
                                           self.failures[phase])


def test_flow_order_without_network(flow_config):
    runner = _RecordingRunner()
    run_trt_ci_d7l.TrtCiFlow(flow_config, runner).run()

    calls = dict(runner.calls)
    probe_script = calls["probe-test"][-1]
    assert "aarch64" in probe_script
    assert "/etc/nvidia/version-ubuntu-rootfs.txt" in probe_script

    sync_argv = calls["sync-source"]
    excludes = [
        sync_argv[index + 1] for index, token in enumerate(sync_argv)
        if token == "--exclude"
    ]
    assert ["/.git", "/.worktrees/", "/build/", "/build-*/"] == excludes[:4]
    assert all(pattern.startswith("/") for pattern in excludes)

    collect_argv = calls["collect-results"]
    assert collect_argv[0:3] == ["scp", "-r", "-O"]
    assert "/results" in collect_argv[-2]

    deploy_script = calls["deploy"][-1]
    assert "sha256sum" in deploy_script
    assert hashlib.sha256(b"candidate TRT bundle").hexdigest() in deploy_script

    assert runner.phases == [
        "probe-build",
        "probe-test",
        "prepare-build",
        "sync-source",
        "build",
        "bundle",
        "download-bundle",
        "download-digest",
        "prepare-deploy",
        "upload-bundle",
        "deploy",
        "test",
        "collect-results",
    ]


def test_collection_is_best_effort_and_never_masks_primary(flow_config):
    best_effort = _RecordingRunner({"collect-results": 91})
    run_trt_ci_d7l.TrtCiFlow(flow_config, best_effort).run()

    runner = _RecordingRunner({"build": 37, "collect-results": 91})

    with pytest.raises(run_trt_ci_d7l.FlowError) as error:
        run_trt_ci_d7l.TrtCiFlow(flow_config, runner).run()

    assert error.value.exit_code == 37
    assert runner.phases[-2:] == ["build", "collect-results"]


def test_main_returns_exact_flow_exit(tmp_path, monkeypatch, flow_config):
    source_root = _make_checkout(tmp_path / "complete-source",
                                 include_nvtx=True)
    config = dataclasses.replace(flow_config, source_root=source_root)
    monkeypatch.setattr(run_trt_ci_d7l, "parse_args", lambda argv: config)

    def fail(_flow):
        raise run_trt_ci_d7l.FlowError("expected", exit_code=41)

    monkeypatch.setattr(run_trt_ci_d7l.TrtCiFlow, "run", fail)
    assert run_trt_ci_d7l.main([]) == 41
