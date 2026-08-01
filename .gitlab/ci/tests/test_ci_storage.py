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

import importlib.util
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = (Path(__file__).parents[1] / "scripts" / "ci_storage.py")
MODULE_SPEC = importlib.util.spec_from_file_location("ci_storage", SCRIPT_PATH)
ci_storage = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(ci_storage)
EXTERNAL_ROOT_MODE = 0o750


@pytest.fixture
def storage_environment(tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch.edge_llm_cache"
    l0_root = scratch_root / "L0"
    l0_root.mkdir(parents=True)
    l0_root.chmod(EXTERNAL_ROOT_MODE)

    values = {
        "EDGE_LLM_CACHE_DIR": str(scratch_root),
        "L0_STORAGE_KIND": "mr",
        "CI_MERGE_REQUEST_IID": "42",
        "CI_PIPELINE_ID": "100",
        "CI_PIPELINE_CREATED_AT": "2026-07-30T10:39:11-07:00",
        "CI_JOB_ID": "200",
        "CI_JOB_NAME_SLUG": "l0-drive-thor-1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return scratch_root


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def _source_environment(path, names):
    commands = ['. "$1"']
    commands.extend(f'printf "%s\\0" "${{{name}}}"' for name in names)
    result = subprocess.run(["sh", "-c", "\n".join(commands), "sh",
                             str(path)],
                            check=True,
                            capture_output=True)
    values = result.stdout.split(b"\0")[:-1]
    return dict(zip(names, (value.decode() for value in values)))


def test_prepare_storage_preserves_external_root_and_creates_protected_layout(
        storage_environment):
    assert ci_storage.main(["prepare-storage"]) == 0
    assert _mode(storage_environment / "L0") == EXTERNAL_ROOT_MODE
    mr_root = storage_environment / "L0" / "MR_42"
    for path in (mr_root, mr_root / "onnx_cache", mr_root / "workspace"):
        assert _mode(path) == 0o755


def test_prepare_job_creates_cross_identity_boundary(storage_environment):
    assert ci_storage.main(["prepare-job"]) == 0

    pipeline_root = (storage_environment / "L0" / "MR_42" / "workspace" /
                     "pipeline_100")
    job_root = pipeline_root / "job_200_l0-drive-thor-1"

    assert _mode(pipeline_root) == 0o755
    for path in (job_root, job_root / "engines", job_root / "logs",
                 job_root / "tmp"):
        assert _mode(path) == 0o777


def test_prepare_job_publishes_resolved_paths(storage_environment,
                                              monkeypatch):
    environment_file = storage_environment.parent / "job.env"
    monkeypatch.setenv("L0_ONNX_CACHE_RELATIVE_PATH", "checkpoint_export/onnx")

    assert ci_storage.main(
        ["prepare-job", "--environment-file",
         str(environment_file)]) == 0

    mr_root = storage_environment / "L0" / "MR_42"
    pipeline_root = mr_root / "workspace" / "pipeline_100"
    job_root = pipeline_root / "job_200_l0-drive-thor-1"
    expected = {
        "L0_ONNX_CACHE_ROOT": str(mr_root / "onnx_cache"),
        "PIPELINE_WORKSPACE_ROOT": str(pipeline_root),
        "JOB_WORKSPACE_ROOT": str(job_root),
        "TEST_LOG_DIR": str(job_root / "logs"),
        "ENGINE_DIR": str(job_root / "engines"),
        "ONNX_DIR": str(mr_root / "onnx_cache" / "checkpoint_export" / "onnx"),
    }
    assert _source_environment(environment_file, expected) == expected


def test_prepare_job_can_publish_job_local_onnx_path(storage_environment,
                                                     monkeypatch):
    environment_file = storage_environment.parent / "job.env"
    monkeypatch.setenv("L0_JOB_ONNX_RELATIVE_PATH", "tmp/onnx")

    assert ci_storage.main(
        ["prepare-job", "--environment-file",
         str(environment_file)]) == 0

    job_root = (storage_environment / "L0" / "MR_42" / "workspace" /
                "pipeline_100" / "job_200_l0-drive-thor-1")
    assert _source_environment(environment_file,
                               ["ONNX_DIR"])["ONNX_DIR"] == str(job_root /
                                                                "tmp" / "onnx")


def test_prepare_job_rejects_conflicting_onnx_locations(
        storage_environment, monkeypatch, capsys):
    monkeypatch.setenv("L0_ONNX_CACHE_RELATIVE_PATH", "checkpoint_export/onnx")
    monkeypatch.setenv("L0_JOB_ONNX_RELATIVE_PATH", "tmp/onnx")

    assert ci_storage.main(["prepare-job"]) == 1
    assert not (storage_environment / "L0" / "MR_42").exists()
    assert "only one L0 ONNX relative path" in capsys.readouterr().err


def test_prepare_job_rejects_onnx_path_traversal(storage_environment,
                                                 monkeypatch, capsys):
    monkeypatch.setenv("L0_ONNX_CACHE_RELATIVE_PATH", "../other-storage")

    assert ci_storage.main(["prepare-job"]) == 1
    assert not (storage_environment / "L0" / "MR_42").exists()
    assert "relative path without traversal" in capsys.readouterr().err


@pytest.mark.parametrize(
    "variable",
    ["CI_MERGE_REQUEST_IID", "CI_PIPELINE_ID", "CI_JOB_ID"],
)
def test_prepare_job_rejects_missing_identifiers_before_writing(
        storage_environment, monkeypatch, capsys, variable):
    monkeypatch.delenv(variable)

    assert ci_storage.main(["prepare-job"]) == 1
    assert not (storage_environment / "L0" / "MR_42").exists()
    assert f"{variable} must be set" in capsys.readouterr().err


def test_prepare_job_rejects_malformed_slug(storage_environment, monkeypatch,
                                            capsys):
    monkeypatch.setenv("CI_JOB_NAME_SLUG", "../other-job")

    assert ci_storage.main(["prepare-job"]) == 1
    assert not (storage_environment / "L0" / "MR_42").exists()
    assert "CI_JOB_NAME_SLUG" in capsys.readouterr().err


def test_delete_rejects_malformed_mr_id_without_writing(
        storage_environment, monkeypatch, capsys):
    monkeypatch.setenv("CI_MERGE_REQUEST_IID", "../43")

    assert ci_storage.main(["delete-storage"]) == 1
    assert list((storage_environment / "L0").iterdir()) == []
    assert "CI_MERGE_REQUEST_IID" in capsys.readouterr().err


def test_delete_removes_only_selected_mr(storage_environment, monkeypatch):
    assert ci_storage.main(["prepare-job"]) == 0
    board_output = (storage_environment / "L0" / "MR_42" / "workspace" /
                    "pipeline_100" / "job_200_l0-drive-thor-1" / "logs" /
                    "board-output")
    board_output.mkdir()
    board_output.chmod(0o777)
    read_only_file = board_output / "audio.wav"
    read_only_file.touch()
    read_only_file.chmod(0o444)

    monkeypatch.setenv("CI_MERGE_REQUEST_IID", "43")
    monkeypatch.setenv("CI_PIPELINE_ID", "102")
    monkeypatch.setenv("CI_JOB_ID", "202")
    assert ci_storage.main(["prepare-job"]) == 0

    monkeypatch.setenv("CI_MERGE_REQUEST_IID", "42")
    assert ci_storage.main(["delete-storage"]) == 0

    assert not (storage_environment / "L0" / "MR_42").exists()
    assert (storage_environment / "L0" / "MR_43").is_dir()


def test_delete_is_idempotent_when_storage_is_absent(storage_environment):
    assert ci_storage.main(["delete-storage"]) == 0
    assert not (storage_environment / "L0" / "MR_42").exists()


def test_delete_rejects_symlink_target(storage_environment, tmp_path, capsys):
    outside = tmp_path / "outside"
    outside.mkdir()
    mr_root = storage_environment / "L0" / "MR_42"
    mr_root.symlink_to(outside, target_is_directory=True)

    assert ci_storage.main(["delete-storage"]) == 1
    assert outside.is_dir()
    assert "must not be a symlink" in capsys.readouterr().err


def test_stability_storage_is_isolated_by_pipeline(storage_environment,
                                                   monkeypatch):
    monkeypatch.setenv("L0_STORAGE_KIND", "stability")
    environment_file = storage_environment.parent / "stability.env"
    assert ci_storage.main(
        ["prepare-job", "--environment-file",
         str(environment_file)]) == 0

    stability_root = storage_environment / "L0" / "STABILITY"
    first_name = "2026-07-30T17-39-11Z_pipeline_100"
    first_root = stability_root / first_name
    assert _source_environment(
        environment_file,
        ["PIPELINE_WORKSPACE_ROOT"])["PIPELINE_WORKSPACE_ROOT"] == str(
            first_root / "workspace")
    assert (first_root / "workspace" / "job_200_l0-drive-thor-1").is_dir()
    assert not (first_root / "workspace" / "pipeline_100").exists()

    monkeypatch.setenv("CI_PIPELINE_ID", "101")
    monkeypatch.setenv("CI_PIPELINE_CREATED_AT", "2026-07-30T17:00:05-07:00")
    monkeypatch.setenv("CI_JOB_ID", "201")
    assert ci_storage.main(["prepare-job"]) == 0
    second_root = stability_root / "2026-07-31T00-00-05Z_pipeline_101"

    monkeypatch.setenv("CI_PIPELINE_ID", "100")
    monkeypatch.setenv("CI_PIPELINE_CREATED_AT", "2026-07-30T10:39:11-07:00")
    assert ci_storage.main(["delete-storage"]) == 0
    assert not first_root.exists()
    assert second_root.is_dir()


@pytest.mark.parametrize("created_at", [
    None,
    "not-a-timestamp",
    "2026-07-30T10:39:11",
])
def test_stability_rejects_ambiguous_created_at_before_writing(
        storage_environment, monkeypatch, capsys, created_at):
    monkeypatch.setenv("L0_STORAGE_KIND", "stability")
    if created_at is None:
        monkeypatch.delenv("CI_PIPELINE_CREATED_AT")
    else:
        monkeypatch.setenv("CI_PIPELINE_CREATED_AT", created_at)

    assert ci_storage.main(["prepare-job"]) == 1
    assert not (storage_environment / "L0" / "STABILITY").exists()
    assert "CI_PIPELINE_CREATED_AT" in capsys.readouterr().err
