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

import datetime
import json
import os
import pathlib
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).parents[3]
SCRIPT_PATH = PROJECT_ROOT / ".gitlab/ci/scripts/record_runner_usage.sh"


def _write_runner_record(tmp_path, pipeline_environment):
    environment = {
        "PATH": os.environ["PATH"],
        "RUNNER_USAGE_DIR": str(tmp_path),
        "CI_PROJECT_DIR": str(tmp_path / "project"),
        "CI_PROJECT_PATH": "TensorRT/tensorrt-edge-llm/tensorrt-edge-llm",
        "CI_PROJECT_PATH_SLUG": "tensorrt-edge-llm",
        "CI_PIPELINE_ID": "123",
        "CI_PIPELINE_IID": "45",
        "CI_PIPELINE_CREATED_AT": "2020-01-01T12:00:00Z",
        "CI_PIPELINE_URL": "https://gitlab.example/pipelines/123",
        "CI_JOB_ID": "456",
        "CI_JOB_NAME": "l0_test",
        "CI_JOB_STAGE": "l0_test",
        "CI_JOB_STATUS": "success",
        "CI_JOB_STARTED_AT": "2020-01-01T12:01:00Z",
        "CI_JOB_QUEUED_DURATION": "5",
        "CI_JOB_URL": "https://gitlab.example/jobs/456",
        "NODE_NAME": "runner-node",
        **pipeline_environment,
    }
    result = subprocess.run(
        ["sh", str(SCRIPT_PATH)],
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    record_path = tmp_path / "tensorrt-edge-llm" / "123" / "456.json"
    assert record_path.is_file()
    return json.loads(record_path.read_text(encoding="utf-8"))


def test_record_attributes_usage_to_job_and_runner(tmp_path):
    record = _write_runner_record(
        tmp_path,
        {
            "CI_PIPELINE_SOURCE": "push",
            "CI_COMMIT_REF_NAME": "main",
            "CI_COMMIT_SHA": "tested-sha",
            "CI_RUNNER_TAGS": '["gpu", "a30"]',
            "EMBEDDED_TARGET": "orin",
            "IMAGE": "edge-image",
            "CUDA_VERSION": "13.0",
        },
    )

    assert record["project"] == "TensorRT/tensorrt-edge-llm/tensorrt-edge-llm"
    assert record["ref"] == "main"
    assert record["pipeline_id"] == "123"
    assert record["pipeline_source"] == "push"
    assert record["pipeline_url"] == "https://gitlab.example/pipelines/123"
    assert record["job_id"] == "456"
    assert record["job_name"] == "l0_test"
    assert record["job_stage"] == "l0_test"
    assert record["job_url"] == "https://gitlab.example/jobs/456"
    assert record["job_status"] == "success"
    assert record["commit_sha"] == "tested-sha"
    assert record["runner_node"] == "runner-node"
    assert json.loads(record["runner_tags"]) == ["gpu", "a30"]
    assert record["embedded_target"] == "orin"
    assert record["image"] == "edge-image"
    assert record["cuda_version"] == "13.0"
    assert record["queue_seconds"] == 5
    assert record["pipeline_to_job_start_seconds"] == 60

    started_at = datetime.datetime.fromisoformat(record["job_started_at"])
    finished_at = datetime.datetime.fromisoformat(record["job_finished_at"])
    assert record["run_seconds"] == int(
        (finished_at - started_at).total_seconds())


def test_merged_result_preserves_tested_source_and_target_revisions(tmp_path):
    record = _write_runner_record(
        tmp_path,
        {
            "CI_PIPELINE_SOURCE": "merge_request_event",
            "CI_COMMIT_REF_NAME": "refs/merge-requests/1288/merge",
            "CI_COMMIT_SHA": "merged-result-sha",
            "CI_MERGE_REQUEST_IID": "1288",
            "CI_MERGE_REQUEST_EVENT_TYPE": "merged_result",
            "CI_MERGE_REQUEST_SOURCE_BRANCH_SHA": "source-sha",
            "CI_MERGE_REQUEST_TARGET_BRANCH_SHA": "target-sha",
        },
    )

    assert record["merge_request_iid"] == "1288"
    assert record["merge_request_event_type"] == "merged_result"
    assert record["commit_sha"] == "merged-result-sha"
    assert record["source_commit_sha"] == "source-sha"
    assert record["target_commit_sha"] == "target-sha"


def test_detached_merge_request_uses_tested_revision_as_source(tmp_path):
    record = _write_runner_record(
        tmp_path,
        {
            "CI_PIPELINE_SOURCE": "merge_request_event",
            "CI_COMMIT_SHA": "source-sha",
            "CI_MERGE_REQUEST_IID": "1288",
            "CI_MERGE_REQUEST_EVENT_TYPE": "detached",
        },
    )

    assert record["merge_request_event_type"] == "detached"
    assert record["commit_sha"] == "source-sha"
    assert record["source_commit_sha"] == record["commit_sha"]
    assert record["target_commit_sha"] is None
