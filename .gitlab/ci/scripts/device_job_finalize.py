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
"""Run best-effort post-job operations for a remote test board."""

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import typing

SSH_OPTIONS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ConnectionAttempts=1",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=2",
)


def _parse_args(
        argv: typing.Optional[typing.Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect remote reports and remove the job workspace.")
    parser.add_argument("--fetch-reports", action="store_true")
    return parser.parse_args(argv)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _remote_workspace(board_user: str) -> str:
    workspace = _required_environment("REMOTE_WORKSPACE")
    path = pathlib.PurePosixPath(workspace)
    expected_home = pathlib.PurePosixPath("/home") / board_user
    if path.parent != expected_home or not path.name.startswith(
            "tensorrt-edge-llm"):
        raise ValueError(f"unsafe remote workspace: {workspace}")
    return workspace


def _run(command: typing.List[str],
         password: str,
         operation: str,
         attempts: int = 1) -> bool:
    command_environment = os.environ.copy()
    command_environment["SSHPASS"] = password
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(command,
                                    check=False,
                                    env=command_environment)
        except OSError as error:
            print(f"WARNING: {operation} could not start: {error}",
                  file=sys.stderr)
        else:
            if result.returncode == 0:
                return True
            print(f"WARNING: {operation} exited with code {result.returncode}",
                  file=sys.stderr)

        if attempt < attempts:
            print(f"Retrying {operation} ({attempt + 1}/{attempts})")
    return False


def _publish_logs(source: pathlib.Path, destination: pathlib.Path) -> None:
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
    except OSError as error:
        print(f"WARNING: Local test log publication failed: {error}",
              file=sys.stderr)


def _publish_junit_reports(source: pathlib.Path,
                           project_dir: pathlib.Path) -> None:
    junit_dir = project_dir / ".ci-reports"
    reports = []
    for report in sorted(source.glob("*.xml")):
        if report.is_symlink() or not report.is_file():
            print(f"WARNING: Skipping non-regular JUnit report: {report.name}",
                  file=sys.stderr)
            continue
        reports.append(report)

    if not reports:
        print("WARNING: No JUnit XML reports were collected from the board",
              file=sys.stderr)
        return

    try:
        junit_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"WARNING: Cannot create JUnit report directory: {error}",
              file=sys.stderr)
        return

    published = 0
    for report in reports:
        try:
            shutil.copy2(report, junit_dir / report.name)
        except OSError as error:
            print(f"WARNING: Cannot publish {report.name}: {error}",
                  file=sys.stderr)
            continue
        published += 1

    print(f"Published {published} JUnit report(s) to {junit_dir}")


def _collect_reports(target: str, workspace: str, password: str) -> None:
    try:
        test_log_dir = pathlib.Path(_required_environment("TEST_LOG_DIR"))
        project_dir = pathlib.Path(_required_environment("CI_PROJECT_DIR"))
    except ValueError as error:
        print(f"WARNING: Remote report collection skipped: {error}",
              file=sys.stderr)
        return

    with tempfile.TemporaryDirectory(prefix="device-reports-") as staging_dir:
        staging_path = pathlib.Path(staging_dir)
        print(f"Fetching pytest reports from {target}:{workspace}/logs")
        fetched = _run([
            "sshpass",
            "-e",
            "rsync",
            "-a",
            "--delete",
            "--safe-links",
            "--timeout=60",
            "-e",
            shlex.join(["ssh", *SSH_OPTIONS]),
            f"{target}:{workspace}/logs/",
            f"{staging_path}/",
        ],
                       password,
                       "remote report collection",
                       attempts=2)
        if not fetched:
            return

        _publish_logs(staging_path, test_log_dir)
        _publish_junit_reports(staging_path, project_dir)


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        board_user = _required_environment("BOARD_USER")
        password = _required_environment("BOARD_PASSWORD_NVKS")
        workspace = _remote_workspace(board_user)
    except ValueError as error:
        print(f"WARNING: Device post-job finalization skipped: {error}",
              file=sys.stderr)
        return 0

    board_ip = os.environ.get("BOARD_IP", "192.168.55.1")
    target = f"{board_user}@{board_ip}"

    if args.fetch_reports:
        _collect_reports(target, workspace, password)

    print(f"Cleaning up remote workspace {workspace}")
    _run([
        "sshpass",
        "-e",
        "ssh",
        *SSH_OPTIONS,
        target,
        f"rm -rf -- {shlex.quote(workspace)}",
    ], password, "remote workspace cleanup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
