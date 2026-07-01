#!/usr/bin/env python3
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
"""Build Edge-LLM against a candidate TensorRT and run its tests on D7L."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import hashlib
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import typing
import uuid

SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SSH_USER_PATTERN = r"[A-Za-z_][A-Za-z0-9_.-]*"
_SSH_HOST_PATTERN = r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:.%]+\])"
_SSH_JUMP_RE = re.compile(r"(?:{}@)?{}(?::([1-9][0-9]{{0,4}}))?".format(
    _SSH_USER_PATTERN, _SSH_HOST_PATTERN))


class FlowError(RuntimeError):
    """A CI flow failure whose exit code must be returned to CI.
    Attributes:
        exit_code: Nonzero process-compatible exit code for the failed phase.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        """Initialize a flow error.
        Args:
            message: Contextual failure description.
            exit_code: Nonzero process-compatible exit code.
        """
        super().__init__(message)
        self.exit_code = exit_code if exit_code != 0 else 1


@dataclasses.dataclass(frozen=True)
class SshTarget:
    """Immutable non-interactive OpenSSH endpoint configuration.
    Attributes:
        host: Endpoint hostname or address.
        user: Remote SSH username.
        port: Endpoint SSH port.
        identity_file: Optional local private-key path.
        jump_host: Optional OpenSSH jump endpoint.
        host_key_policy: StrictHostKeyChecking value.
    """

    host: str
    user: str
    port: int = 22
    identity_file: typing.Optional[pathlib.Path] = None
    jump_host: typing.Optional[str] = None
    host_key_policy: str = "yes"

    def __post_init__(self) -> None:
        """Validate endpoint fields.
        Raises:
            ValueError: If required fields, port, or host-key policy are invalid.
        """
        if re.fullmatch(_SSH_HOST_PATTERN, self.host) is None:
            raise ValueError(
                "SSH host must be a hostname, address, or bracketed IPv6 literal"
            )
        if re.fullmatch(_SSH_USER_PATTERN, self.user) is None:
            raise ValueError("SSH user contains unsupported syntax")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be in the range 1..65535")
        if self.jump_host is not None:
            jump = _SSH_JUMP_RE.fullmatch(self.jump_host)
            if jump is None or (jump.group(1) is not None
                                and int(jump.group(1)) > 65535):
                raise ValueError(
                    "SSH jump host must use [user@]host[:port] syntax")
        if self.host_key_policy not in ("yes", "accept-new", "no"):
            raise ValueError("host-key policy must be yes, accept-new, or no")

    # yapf: disable
    def _options(self, connect_timeout_s: int, port_flag: str) -> typing.List[str]:
        """Render shared SSH-family options."""
        argv = [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout={}".format(connect_timeout_s),
            "-o", "StrictHostKeyChecking={}".format(self.host_key_policy), port_flag, str(self.port)]
        if self.host_key_policy == "no":
            argv[0:0] = ["-o", "UserKnownHostsFile=/dev/null"]
        if self.identity_file is not None:
            argv.extend(["-i", str(self.identity_file)])
        if self.jump_host:
            argv.extend(["-J", self.jump_host])
        return argv
    def ssh_argv(self, script: str, connect_timeout_s: int) -> typing.List[str]:
        """Render an argv-only SSH command that runs a Bash script.
        Args:
            script: Bash source to execute remotely.
            connect_timeout_s: OpenSSH connection timeout in seconds.
        Returns:
            Complete SSH argv without a TTY or password.
        """
        return ["ssh", "-T"] + self._options(connect_timeout_s, "-p") + [
            "{}@{}".format(self.user, self.host), "bash", "-lc", shlex.quote(script)]
    def scp_argv(self, source: str, destination: str, connect_timeout_s: int) -> typing.List[str]:
        """Render an argv-only SCP command.
        Args:
            source: Local path or qualified remote source.
            destination: Local path or qualified remote destination.
            connect_timeout_s: OpenSSH connection timeout in seconds.
        Returns:
            Complete legacy-SCP argv preserving remote path quoting.
        """
        prefix = "{}@{}:".format(self.user, self.host)
        if source.startswith(prefix) == destination.startswith(prefix):
            raise ValueError("SCP requires exactly one path qualified for this SSH target")
        return ["scp", "-O"] + self._options(connect_timeout_s, "-P") + [source, destination]
    def rsync_transport(self, connect_timeout_s: int) -> str:
        """Render the OpenSSH transport required by rsync.
        Args:
            connect_timeout_s: OpenSSH connection timeout in seconds.
        Returns:
            Shell-escaped transport string for rsync's ``-e`` option.
        """
        return shlex.join(["ssh", "-T"] + self._options(connect_timeout_s, "-p"))
    def remote_spec(self, path: pathlib.PurePosixPath) -> str:
        """Render a qualified remote path.
        Args:
            path: Remote POSIX path to quote.
        Returns:
            SCP/rsync-compatible ``user@host:path`` value.
        """
        text = str(path)
        if not path.is_absolute() or any(character in text for character in "\x00\r\n"):
            raise ValueError("remote path must be absolute and contain no control characters")
        return "{}@{}:{}".format(self.user, self.host, shlex.quote(text))
    # yapf: enable


# yapf: disable
class CommandRunner:
    """Stream a subprocess and enforce a process-group timeout.
    Attributes:
        artifacts_dir: Local directory receiving phase logs.
    """
    def __init__(self, artifacts_dir: pathlib.Path) -> None:
        """Create a command runner.
        Args:
            artifacts_dir: Local directory receiving phase logs.
        """
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    @staticmethod
    def _terminate_group(process: subprocess.Popen) -> None:
        """Terminate every process in the subprocess group."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    def run(self, phase: str, argv: typing.Sequence[str], timeout_s: float,
            check: bool = True) -> subprocess.CompletedProcess:
        """Run an argv command with streamed output and a deadline.
        Args:
            phase: Stable phase name used for its log file.
            argv: Command and argument sequence.
            timeout_s: Wall-clock timeout in seconds.
            check: Whether a nonzero exit status raises an error.
        Returns:
            Completed-process metadata with a bounded output tail.
        Raises:
            FlowError: If execution, output streaming, or status checking fails.
            OSError: If the process cannot be started.
        """
        log_path = self.artifacts_dir / "{}.log".format(
            re.sub(r"[^A-Za-z0-9_.-]", "-", phase))
        command = [str(value) for value in argv]
        tail = collections.deque(maxlen=200)  # type: typing.Deque[str]
        reader_errors = []  # type: typing.List[BaseException]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            rendered = shlex.join(command)
            print("$ {}".format(rendered))
            log.write("$ {}\n".format(rendered))
            log.flush()
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, start_new_session=True)
            def copy_output() -> None:
                """Stream output and retain only a bounded tail."""
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        tail.append(line)
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        log.flush()
                except (OSError, UnicodeError, ValueError) as error:
                    reader_errors.append(error)

            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            deadline = time.monotonic() + timeout_s
            timed_out = False
            try:
                return_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
            if not timed_out:
                reader.join(max(0.0, deadline - time.monotonic()))
                timed_out = reader.is_alive()
            if timed_out:
                self._terminate_group(process)
                # A detached descendant may still own stdout; never wait forever.
                reader.join(1.0)
                log.write("\n[timeout_seconds] {}\n".format(timeout_s))
                raise FlowError("{} timed out after {} seconds; see {}".format(
                    phase, timeout_s, log_path), 124)
            if reader_errors:
                raise FlowError("{} output reader failed: {}; see {}".format(
                    phase, reader_errors[0], log_path))
            log.write("\n[exit_code] {}\n".format(return_code))
        completed = subprocess.CompletedProcess(command, return_code, "".join(tail), "")
        if check and return_code != 0:
            raise FlowError("{} failed with exit code {}; see {}".format(
                phase, return_code, log_path), return_code)
        return completed
# yapf: enable


# yapf: disable
@dataclasses.dataclass(frozen=True)
class TrtCiConfig:
    """Validated immutable inputs for one compatibility run.
    Attributes:
        source_root: Local Edge-LLM checkout root.
        trt_root: TensorRT source root on the build host.
        trt_build_dir: Candidate TensorRT build directory on the build host.
        build_target: Linux cross-build SSH endpoint.
        test_target: D7L test SSH endpoint.
        workspace: Safe absolute remote workspace base.
        run_id: Safe identifier for the run-specific child.
        artifacts_dir: Local artifact base directory.
        cuda_version: typing.Optional CUDA toolkit version override.
        jobs: Parallel build job count.
        gtest_filter: GoogleTest filter passed to ``unitTest``.
        connection_timeout_s: SSH connection/probe timeout.
        build_timeout_s: Full cross-build timeout.
        test_timeout_s: D7L unit-test timeout.
        transfer_timeout_s: Source/archive/result transfer timeout.
        keep_workspace: Whether to retain successful remote workspaces.
    """

    source_root: pathlib.Path
    trt_root: pathlib.PurePosixPath
    trt_build_dir: pathlib.PurePosixPath
    build_target: SshTarget
    test_target: SshTarget
    workspace: pathlib.PurePosixPath
    run_id: str
    artifacts_dir: pathlib.Path
    cuda_version: typing.Optional[str] = None
    jobs: int = 8
    gtest_filter: str = "*"
    connection_timeout_s: int = 30
    build_timeout_s: int = 7200
    test_timeout_s: int = 1800
    transfer_timeout_s: int = 1800
    keep_workspace: bool = False
    def __post_init__(self) -> None:
        """Validate local inputs, remote paths, and numeric limits.
        Raises:
            ValueError: If any path, identifier, filter, job, or timeout is unsafe.
        """
        if not self.source_root.is_dir() or not (self.source_root / "CMakeLists.txt").is_file():
            raise ValueError("Edge-LLM source root is invalid: {}".format(self.source_root))
        if not self.artifacts_dir.is_absolute():
            raise ValueError("--artifacts-dir must be an absolute local path")
        if (not self.workspace.is_absolute()
                or self.workspace == pathlib.PurePosixPath(self.workspace.anchor)
                or ".." in self.workspace.parts):
            raise ValueError("--workspace must be a safe absolute non-root POSIX path")
        for label, path in (("--trt-root", self.trt_root), ("--trt-build-dir", self.trt_build_dir)):
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("{} must be an absolute remote POSIX path".format(label))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.run_id) or self.run_id in (".", ".."):
            raise ValueError("--run-id must contain only safe letters, digits, '.', '_', or '-'")
        if self.jobs <= 0 or min(self.connection_timeout_s, self.build_timeout_s,
                                 self.test_timeout_s, self.transfer_timeout_s) <= 0:
            raise ValueError("jobs and all timeouts must be positive")
        if "\x00" in self.gtest_filter or "\n" in self.gtest_filter:
            raise ValueError("--gtest-filter cannot contain NUL or a newline")
    @property
    def run_workspace(self) -> pathlib.PurePosixPath:
        """Return the sole remote run-specific child.
        Returns:
            Identical absolute workspace path used on both remote hosts.
        Raises:
            ValueError: If derivation ever escapes the validated workspace base.
        """
        child = self.workspace / "run-{}".format(self.run_id)
        if child.parent != self.workspace:
            raise ValueError("unsafe derived run workspace")
        return child

# yapf: enable


class TrtCiFlow:
    """Cross-build, transfer, run, and collect the compatibility check.
    Attributes:
        config: Immutable validated run configuration.
        runner: Injectable argv command runner.
    """

    def __init__(self, config: TrtCiConfig, runner: CommandRunner) -> None:
        """Initialize the flow.
        Args:
            config: Validated immutable run configuration.
            runner: Command runner or recording test double.
        """
        self.config = config
        self.runner = runner
        self._temp_dir = None  # type: typing.Optional[pathlib.Path]

    @staticmethod
    def _q(value: object) -> str:
        """Quote one value for Bash."""
        return shlex.quote(str(value))

    # yapf: disable
    def render_build_script(self) -> str:
        """Render candidate package creation and the full build.
        Returns:
            Fail-fast Bash source for the remote build phase.
        """
        cfg, q = self.config, self._q
        run, source = cfg.run_workspace, cfg.run_workspace / "source"
        cuda = q(cfg.cuda_version) if cfg.cuda_version else "$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \\([0-9][0-9.]*\\).*/\\1/p' | head -n1)"
        return "\n".join([
            "set -euo pipefail", "run={}".format(q(run)), "source={}".format(q(source)),
            "trt_root={}".format(q(cfg.trt_root)), "trt_build={}".format(q(cfg.trt_build_dir)),
            "pkg=\"$run/trt-package\"", "build=\"$run/build\"",
            "test -d \"$trt_root/include\"", "test -d \"$trt_root/parsers/onnx\"",
            "test -f \"$trt_build/include/NvInferVersion.h\"", "rm -rf \"$pkg\" \"$build\"",
            "mkdir -p \"$pkg/include\" \"$pkg/lib\"",
            "tar -C \"$trt_root/include\" -cf - . | tar -C \"$pkg/include\" -xf -",
            "cp -a \"$trt_build/include/NvInferVersion.h\" \"$pkg/include/NvInferVersion.h\"",
            "tar -C \"$trt_root/parsers/onnx\" -cf - . | tar -C \"$pkg/include\" -xf -",
            "exact=\"$trt_build/Release/lib\"; lib_dir=",
            "complete() { find \"$1\" -maxdepth 1 \\( -type f -o -type l \\) -name 'libnvinfer.so*' -print -quit | grep -q . && find \"$1\" -maxdepth 1 \\( -type f -o -type l \\) -name 'libnvonnxparser.so*' -print -quit | grep -q .; }",
            "if [ -d \"$exact\" ] && complete \"$exact\"; then lib_dir=$exact; else",
            "  mapfile -t candidates < <(find \"$trt_build\" -mindepth 2 -maxdepth 8 -type d -path '*/Release/lib' -print | LC_ALL=C sort)",
            "  complete_dirs=(); for candidate in \"${candidates[@]}\"; do complete \"$candidate\" && complete_dirs+=(\"$candidate\"); done",
            "  [ \"${#complete_dirs[@]}\" -eq 1 ] || { printf 'Expected one complete Release/lib, found %s\\n' \"${#complete_dirs[@]}\" >&2; exit 2; }",
            "  lib_dir=${complete_dirs[0]}", "fi",
            "tar -C \"$lib_dir\" -cf - . | tar -C \"$pkg/lib\" -xf -", "pkg_lib_root=$(readlink -f \"$pkg/lib\")",
            "while IFS= read -r -d '' link; do target=$(readlink \"$link\"); case \"$target\" in /*) echo \"Absolute TRT symlink: $link\" >&2; exit 2;; esac; resolved=$(readlink -f \"$link\") || exit 2; case \"$resolved\" in \"$pkg_lib_root/\"*) [ -e \"$resolved\" ];; *) echo \"Out-of-tree TRT symlink: $link\" >&2; exit 2;; esac; done < <(find \"$pkg/lib\" -type l -print0)",
            "test -f \"$pkg/include/NvInfer.h\"; test -f \"$pkg/include/NvInferVersion.h\"; test -f \"$pkg/include/NvOnnxParser.h\"",
            "complete \"$pkg/lib\"", "test -d \"$source/unittests/resources\"", "cuda_version={}".format(cuda), "test -n \"$cuda_version\"",
            "cmake -S \"$source\" -B \"$build\" -DCMAKE_BUILD_TYPE=Release -DBUILD_UNIT_TESTS=ON -DEMBEDDED_TARGET=auto-thor -DCMAKE_TOOLCHAIN_FILE=\"$source/cmake/aarch64_linux_toolchain.cmake\" -DTRT_PACKAGE_DIR=\"$pkg\" -DCUDA_CTK_VERSION=\"$cuda_version\" -DENABLE_CUTE_DSL=OFF",
            "cmake --build \"$build\" --parallel {}".format(cfg.jobs),
            "for artifact in \"$build/unitTest\" \"$build/examples/llm/llm_build\" \"$build/examples/llm/llm_inference\" \"$build/examples/multimodal/visual_build\" \"$build/examples/multimodal/audio_build\"; do test -x \"$artifact\"; done",
        ])
    def render_bundle_script(self) -> str:
        """Render staging of executables, DSOs, and test resources.
        Returns:
            Fail-fast Bash source that creates and hashes the board archive.
        """
        q, run = self._q, self.config.run_workspace
        return "\n".join([
            "set -euo pipefail", "run={}".format(q(run)), "stage=\"$run/bundle-root\"", "archive=\"$run/edgellm-d7l.tar.gz\"",
            "rm -rf \"$stage\"", "mkdir -p \"$stage/build\" \"$stage/trt-package\" \"$stage/source/unittests\"",
            "cp -a \"$run/build/unitTest\" \"$run/build/examples/llm/llm_build\" \"$stage/build/\"",
            "find \"$run/build\" -maxdepth 1 \\( -type f -o -type l \\) -name '*.so*' -exec cp -a {} \"$stage/build/\" \\;",
            "cp -a \"$run/trt-package/lib\" \"$stage/trt-package/lib\"",
            "cp -a \"$run/source/unittests/resources\" \"$stage/source/unittests/resources\"",
            "if [ -d \"$run/source/tests/chat_templates\" ]; then mkdir -p \"$stage/source/tests\"; cp -a \"$run/source/tests/chat_templates\" \"$stage/source/tests/\"; fi",
            "tar -C \"$stage\" -czf \"$archive\" .", "sha256sum \"$archive\" > \"$archive.sha256\"",
        ])
    def render_test_script(self) -> str:
        """Render dependency provenance checks and the GTest run.
        Returns:
            Bash source that records ldd/GTest output and preserves test status.
        """
        q, run = self._q, self.config.run_workspace
        return "\n".join([
            "set -euo pipefail", "run={}".format(q(run)), "package_lib={}".format(q(run / "trt-package/lib")),
            "mkdir -p \"$run/results\"", "export LD_LIBRARY_PATH=\"$package_lib:$run/build:${LD_LIBRARY_PATH:-}\"",
            "package_root=$(readlink -f \"$package_lib\")",
            "ldd_re='^[[:space:]]*([^[:space:]]+)[[:space:]]+=>[[:space:]]+(.+)[[:space:]]+\\(0x[[:xdigit:]]+\\)$'",
            "check_ldd() { binary=$1; need_parser=$2; have_nvinfer=0; have_parser=0; output=$(ldd \"$binary\") || return 1; printf '%s\\n' \"$output\"; while IFS= read -r line; do [[ $line != *'not found'* ]] || return 1; if [[ $line =~ $ldd_re ]]; then name=${BASH_REMATCH[1]}; path=${BASH_REMATCH[2]}; while [[ $path == *[[:space:]] ]]; do path=${path%?}; done; case $name in libnvinfer*) have_nvinfer=1;; libnvonnxparser*) have_parser=1;; *) continue;; esac; resolved=$(readlink -f -- \"$path\") || return 1; case $resolved in \"$package_root\"/*) ;; *) echo \"Unexpected candidate library path: $name => $resolved\" >&2; return 1;; esac; fi; done <<< \"$output\"; [ $have_nvinfer -eq 1 ] || { echo \"Missing candidate libnvinfer for $binary\" >&2; return 1; }; [ $need_parser -eq 0 ] || [ $have_parser -eq 1 ] || { echo \"Missing candidate libnvonnxparser for $binary\" >&2; return 1; }; }",
            "{ check_ldd \"$run/build/unitTest\" 0; check_ldd \"$run/build/llm_build\" 1; } 2>&1 | tee \"$run/results/ldd.log\"",
            "set +e", "\"$run/build/unitTest\" --gtest_filter={} --gtest_output=xml:\"$run/results/gtest.xml\" 2>&1 | tee \"$run/results/gtest.log\"".format(q(self.config.gtest_filter)),
            "status=${PIPESTATUS[0]}", "set -e", "printf '%s\\n' \"$status\" > \"$run/results/exit_code\"", "exit \"$status\"",
        ])
    # yapf: enable

    # yapf: disable
    def _remote(self, phase: str, target: SshTarget, script: str, timeout_s: float,
                check: bool = True) -> subprocess.CompletedProcess:
        """Run generated Bash over SSH."""
        argv = target.ssh_argv(script, self.config.connection_timeout_s)
        return self.runner.run(phase, argv, timeout_s, check)
    def _probe(self) -> None:
        """Probe both remote endpoints."""
        cfg = self.config
        self._remote("probe-build", cfg.build_target,
                     "set -eu; uname -a; command -v bash cmake tar find sha256sum readlink rsync",
                     cfg.connection_timeout_s)
        self._remote("probe-test", cfg.test_target,
                     "set -eu; test \"$(uname -m)\" = aarch64; test -s /etc/nvidia/version-ubuntu-rootfs.txt; uname -a; command -v bash tar ldd readlink sha256sum tee awk",
                     cfg.connection_timeout_s)
    def _sync(self) -> None:
        """Prepare the build workspace and sync source."""
        cfg, run = self.config, self.config.run_workspace
        script = "set -euo pipefail; rm -rf {0}; mkdir -p {0}/source".format(self._q(run))
        self._remote("prepare-build", cfg.build_target, script, cfg.connection_timeout_s)
        excludes = [
            "/.git", "/.worktrees/", "/build/", "/build-*/", "/artifacts/", "/logs/",
            "/tmp/", "/codex-logs/", "/.cache/", "/**/.cache/", "/dist/", "/.venv/",
            "/venv/", "/**/*.egg-info/", "/**/__pycache__/",
        ]
        argv = ["rsync", "-a", "--delete"]
        for pattern in excludes:
            argv.extend(["--exclude", pattern])
        argv.extend(["-e", cfg.build_target.rsync_transport(cfg.connection_timeout_s),
                     str(cfg.source_root) + "/", cfg.build_target.remote_spec(run / "source") + "/"])
        self.runner.run("sync-source", argv, cfg.transfer_timeout_s)
    def _transfer(self, remote_archive: pathlib.PurePosixPath) -> str:
        """Verify and transfer the board archive."""
        if self._temp_dir is None:
            raise FlowError("transfer called without an active temporary directory")
        cfg, run = self.config, self.config.run_workspace
        archive = self._temp_dir / "edgellm-d7l.tar.gz"
        digest_file = self._temp_dir / "edgellm-d7l.tar.gz.sha256"
        self.runner.run("download-bundle", cfg.build_target.scp_argv(
            cfg.build_target.remote_spec(remote_archive), str(archive), cfg.connection_timeout_s),
            cfg.transfer_timeout_s)
        self.runner.run("download-digest", cfg.build_target.scp_argv(
            cfg.build_target.remote_spec(pathlib.PurePosixPath(str(remote_archive) + ".sha256")),
            str(digest_file), cfg.connection_timeout_s), cfg.transfer_timeout_s)
        fields = digest_file.read_text(encoding="utf-8").split()
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        expected, actual = fields[0] if fields else "", digest.hexdigest()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected) or expected.lower() != actual:
            raise FlowError("downloaded bundle SHA-256 does not match build host")
        script = "set -euo pipefail; rm -rf {0}; mkdir -p {0}".format(self._q(run))
        self._remote("prepare-deploy", cfg.test_target, script, cfg.connection_timeout_s)
        self.runner.run("upload-bundle", cfg.test_target.scp_argv(
            str(archive), cfg.test_target.remote_spec(run / archive.name), cfg.connection_timeout_s),
            cfg.transfer_timeout_s)
        return actual
    def _collect(self) -> None:
        """Fetch D7L results."""
        cfg = self.config
        run_artifacts = cfg.artifacts_dir / "run-{}".format(cfg.run_id)
        run_artifacts.mkdir(parents=True, exist_ok=True)
        result_dir = run_artifacts / "results"
        if result_dir.exists():
            shutil.rmtree(str(result_dir))
        argv = cfg.test_target.scp_argv(cfg.test_target.remote_spec(cfg.run_workspace / "results"),
                                       str(run_artifacts), cfg.connection_timeout_s)
        argv.insert(1, "-r")
        self.runner.run("collect-results", argv, cfg.transfer_timeout_s)
    def run(self) -> None:
        """Execute ordered phases and preserve the primary result.
        Returns:
            None.
        Raises:
            FlowError: If a required build, transfer, deploy, or test phase fails.
            OSError: If a local artifact operation fails.
        """
        primary = None  # type: typing.Optional[Exception]
        with tempfile.TemporaryDirectory(prefix="edgellm-trt-ci-") as temp:
            self._temp_dir = pathlib.Path(temp)
            try:
                self._probe()
                self._sync()
                self._remote("build", self.config.build_target, self.render_build_script(),
                             self.config.build_timeout_s)
                self._remote("bundle", self.config.build_target, self.render_bundle_script(),
                             self.config.transfer_timeout_s)
                digest = self._transfer(self.config.run_workspace / "edgellm-d7l.tar.gz")
                run, q = self.config.run_workspace, self._q
                extract = "\n".join([
                    "set -euo pipefail", "run={}".format(q(run)), "archive=\"$run/edgellm-d7l.tar.gz\"",
                    "actual=$(sha256sum \"$archive\" | awk '{print $1}')", "test \"$actual\" = {}".format(q(digest)),
                    "tar -C \"$run\" -xzf \"$archive\"", "test -x \"$run/build/unitTest\""])
                self._remote("deploy", self.config.test_target, extract, self.config.transfer_timeout_s)
                self._remote("test", self.config.test_target, self.render_test_script(),
                             self.config.test_timeout_s)
            except Exception as error:
                primary = error
            finally:
                try:
                    self._collect()
                except Exception as error:
                    print(
                        "warning: result collection failed: {}".format(error),
                        file=sys.stderr)
                self._temp_dir = None
        if primary is not None:
            raise primary
        if not self.config.keep_workspace:
            cfg, run = self.config, self.config.run_workspace
            for name, target in (("cleanup-build", cfg.build_target), ("cleanup-test", cfg.test_target)):
                self._remote(name, target, "set -euo pipefail; rm -rf {}".format(self._q(run)),
                             cfg.connection_timeout_s)

def _make_parser(source_root: pathlib.Path) -> argparse.ArgumentParser:
    """Build the documented command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    ci_job_id = re.sub(r"[^A-Za-z0-9_.-]", "-", os.environ.get("CI_JOB_ID", "")).strip(".-")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    default_run_id = "job-{}".format(ci_job_id[:59]) if ci_job_id else "{}-{}-{}".format(
        stamp, os.getpid(), uuid.uuid4().hex[:8])
    # yapf: disable
    add("--trt-root", required=True, help="Absolute TensorRT source root on the build host.")
    add("--trt-build-dir", required=True, help="Absolute candidate TensorRT build directory on the build host.")
    add("--build-host", required=True, help="SSH hostname of the Linux build host.")
    add("--build-user", required=True, help="SSH user for the Linux build host.")
    add("--test-host", required=True, help="SSH hostname of the D7L test host.")
    add("--test-user", required=True, help="SSH user for the D7L test host.")
    add("--build-port", type=int, default=22, help="Build-host SSH port (default: 22).")
    add("--test-port", type=int, default=22, help="D7L SSH port (default: 22).")
    add("--identity-file", type=pathlib.Path, help="Shared local SSH private key; ssh-agent is used when omitted.")
    add("--build-jump-host", help="Build jump endpoint in [user@]host[:port] syntax.")
    add("--test-jump-host", help="D7L jump endpoint in [user@]host[:port] syntax.")
    add("--host-key-policy", choices=("yes", "accept-new", "no"), default="yes", help="OpenSSH StrictHostKeyChecking policy (default: yes).")
    add("--workspace", default="/tmp/edgellm-trt-ci", help="Safe absolute remote workspace base used on both hosts.")
    add("--run-id", default=default_run_id, help="Safe run identifier; defaults to CI job ID or a unique local value.")
    add("--artifacts-dir", type=pathlib.Path, default=source_root / "artifacts" / "trt-ci-d7l", help="Absolute local base for phase logs and D7L results.")
    add("--cuda-version", help="CUDA toolkit version override; default is build-host nvcc detection.")
    add("--jobs", type=int, default=8, help="Parallel build jobs (default: 8).")
    add("--gtest-filter", default="*", help="GoogleTest filter passed to unitTest.")
    add("--connection-timeout", type=int, default=30, help="SSH connection and probe timeout in seconds.")
    add("--build-timeout", type=int, default=7200, help="Full remote build timeout in seconds.")
    add("--test-timeout", type=int, default=1800, help="D7L unit-test timeout in seconds.")
    add("--transfer-timeout", type=int, default=1800, help="Source, archive, and result transfer timeout in seconds.")
    add("--keep-workspace", action="store_true", help="Keep successful remote run directories for debugging.")
    # yapf: enable
    return parser


# yapf: disable
def parse_args(
        argv: typing.Optional[typing.Sequence[str]] = None) -> TrtCiConfig:
    """Parse the stable TensorRT CI command-line interface.
    Args:
        argv: Optional argument sequence; defaults to ``sys.argv``.
    Returns:
        Validated immutable run configuration.
    Raises:
        SystemExit: If argparse detects missing or invalid arguments.
    """
    parser = _make_parser(SOURCE_ROOT)
    args = parser.parse_args(argv)
    if args.identity_file is not None and not args.identity_file.expanduser().is_file():
        parser.error("--identity-file must name an existing local file")
    identity = args.identity_file.expanduser().resolve() if args.identity_file else None
    try:
        common = dict(identity_file=identity, host_key_policy=args.host_key_policy)
        build_target = SshTarget(args.build_host, args.build_user, args.build_port,
                                 jump_host=args.build_jump_host, **common)
        test_target = SshTarget(args.test_host, args.test_user, args.test_port,
                                jump_host=args.test_jump_host, **common)
        # yapf: disable
        return TrtCiConfig(
            SOURCE_ROOT, pathlib.PurePosixPath(args.trt_root), pathlib.PurePosixPath(args.trt_build_dir),
            build_target, test_target, pathlib.PurePosixPath(args.workspace), args.run_id,
            args.artifacts_dir.expanduser().resolve(), args.cuda_version, args.jobs, args.gtest_filter,
            args.connection_timeout, args.build_timeout, args.test_timeout, args.transfer_timeout,
            args.keep_workspace)
        # yapf: enable
    except ValueError as error:
        parser.error(str(error))


# yapf: enable


# yapf: disable
def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    """Run the compatibility flow.
    Args:
        argv: Optional argument sequence; defaults to ``sys.argv``.
    Returns:
        Zero on success or the exact expected phase failure code.
    """
    try:
        config = parse_args(argv)
        markers = ["3rdParty/googletest/CMakeLists.txt", "3rdParty/nlohmannJson/CMakeLists.txt",
                   "3rdParty/NVTX/CMakeLists.txt"]
        missing = [name for name in markers if not (config.source_root / name).is_file()]
        if missing:
            raise ValueError("Edge-LLM submodules are unavailable ({}); run git submodule update --init".format(
                ", ".join(missing)))
        runner = CommandRunner(config.artifacts_dir / "run-{}".format(config.run_id))
        TrtCiFlow(config, runner).run()
    except FlowError as error:
        print("error: {}".format(error), file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    return 0


# yapf: enable

if __name__ == "__main__":
    sys.exit(main())
