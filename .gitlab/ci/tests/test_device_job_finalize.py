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
"""Tests for the remote device post-job finalizer entry point."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / ".gitlab" / "ci" / "scripts" / "device_job_finalize.py"
_SPEC = importlib.util.spec_from_file_location("device_job_finalize",
                                               _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
device_job_finalize = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = device_job_finalize
_SPEC.loader.exec_module(device_job_finalize)


@pytest.fixture
def finalizer_environment(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    test_log_dir = tmp_path / "logs"
    project_dir.mkdir()
    monkeypatch.setenv("BOARD_USER", "tester")
    monkeypatch.setenv("BOARD_PASSWORD_NVKS", "password")
    monkeypatch.setenv("BOARD_IP", "192.0.2.10")
    monkeypatch.setenv("CI_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("TEST_LOG_DIR", str(test_log_dir))
    monkeypatch.setenv("REMOTE_WORKSPACE",
                       "/home/tester/tensorrt-edge-llm-ut-123")
    return project_dir, test_log_dir


def test_main_publishes_fresh_board_reports_and_removes_workspace(
        finalizer_environment, monkeypatch):
    project_dir, test_log_dir = finalizer_environment
    test_log_dir.mkdir()
    (test_log_dir / "stale.xml").write_text("<stale/>", encoding="utf-8")
    operations = []

    def run(command, **_kwargs):
        if "rsync" in command:
            operations.append("fetch")
            destination = pathlib.Path(command[-1])
            (destination / "test_report_l0_python_ut.xml").write_text(
                "<testsuites/>", encoding="utf-8")
            (destination / "test_report_l0_python_ut.html").write_text(
                "<html/>", encoding="utf-8")
        else:
            operations.append("cleanup")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main(["--fetch-reports"]) == 0

    assert operations == ["fetch", "cleanup"]
    assert (test_log_dir / "test_report_l0_python_ut.html").exists()
    assert (test_log_dir / "test_report_l0_python_ut.xml").exists()
    junit_dir = project_dir / ".ci-reports"
    assert (junit_dir / "test_report_l0_python_ut.xml").read_text(
        encoding="utf-8") == "<testsuites/>"
    assert not (junit_dir / "stale.xml").exists()


def test_main_does_not_publish_partial_reports_after_transfer_failure(
        finalizer_environment, monkeypatch, capsys):
    project_dir, _ = finalizer_environment
    operations = []

    def run(command, **_kwargs):
        if "rsync" in command:
            operations.append("fetch")
            pathlib.Path(command[-1],
                         "partial.xml").write_text("<testsuites>",
                                                   encoding="utf-8")
            return subprocess.CompletedProcess(command, 23)
        operations.append("cleanup")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main(["--fetch-reports"]) == 0

    assert operations == ["fetch", "fetch", "cleanup"]
    assert not (project_dir / ".ci-reports" / "partial.xml").exists()
    assert "remote report collection exited with code 23" in capsys.readouterr(
    ).err


def test_main_retries_report_collection_before_cleanup(finalizer_environment,
                                                       monkeypatch):
    project_dir, _ = finalizer_environment
    fetch_attempts = 0
    operations = []

    def run(command, **_kwargs):
        nonlocal fetch_attempts
        if "rsync" not in command:
            operations.append("cleanup")
            return subprocess.CompletedProcess(command, 0)

        operations.append("fetch")
        fetch_attempts += 1
        if fetch_attempts == 1:
            return subprocess.CompletedProcess(command, 12)
        pathlib.Path(command[-1], "report.xml").write_text("<testsuites/>",
                                                           encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main(["--fetch-reports"]) == 0

    assert operations == ["fetch", "fetch", "cleanup"]
    assert (project_dir / ".ci-reports" / "report.xml").exists()


def test_main_warns_when_board_produces_no_junit_report(
        finalizer_environment, monkeypatch, capsys):

    def run(command, **_kwargs):
        if "rsync" in command:
            pathlib.Path(command[-1],
                         "pytest.log").write_text("setup failed",
                                                  encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main(["--fetch-reports"]) == 0
    assert "No JUnit XML reports were collected" in capsys.readouterr().err


def test_main_still_removes_workspace_when_report_destination_is_missing(
        finalizer_environment, monkeypatch, capsys):
    monkeypatch.delenv("CI_PROJECT_DIR")
    operations = []

    def run(command, **_kwargs):
        operations.append("fetch" if "rsync" in command else "cleanup")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main(["--fetch-reports"]) == 0

    assert operations == ["cleanup"]
    assert "CI_PROJECT_DIR must be set" in capsys.readouterr().err


def test_main_treats_cleanup_failure_as_diagnostic(finalizer_environment,
                                                   monkeypatch, capsys):

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 255)

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main([]) == 0
    assert "remote workspace cleanup exited with code 255" in capsys.readouterr(
    ).err


def test_main_rejects_workspace_outside_board_home(finalizer_environment,
                                                   monkeypatch, capsys):
    monkeypatch.setenv("REMOTE_WORKSPACE", "/tmp/tensorrt-edge-llm-123")

    def run(*_args, **_kwargs):
        raise AssertionError("unsafe workspace must not be removed")

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main([]) == 0
    assert "unsafe remote workspace" in capsys.readouterr().err


def test_main_requires_workspace_from_environment(finalizer_environment,
                                                  monkeypatch, capsys):
    monkeypatch.delenv("REMOTE_WORKSPACE")

    def run(*_args, **_kwargs):
        raise AssertionError("cleanup must not guess the workspace")

    monkeypatch.setattr(device_job_finalize.subprocess, "run", run)

    assert device_job_finalize.main([]) == 0
    assert "REMOTE_WORKSPACE must be set" in capsys.readouterr().err
