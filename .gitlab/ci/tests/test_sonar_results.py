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

import http.server
import json
import os
import pathlib
import subprocess
import threading
import urllib.parse

import pytest
import yaml

G_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
G_SCRIPT = G_REPO_ROOT / ".gitlab" / "ci" / "scripts" / "sonar_results.py"
G_SAST_REPORT = {
    "version": "15.0.0",
    "vulnerabilities": [],
    "scan": {
        "type": "sast"
    },
}


@pytest.fixture
def sonar_server():
    state = {
        "requests": [],
        "task_responses": [],
        "gate_status": "OK",
    }

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            state["requests"].append({
                "path":
                self.path,
                "authorization":
                self.headers.get("Authorization"),
            })
            request_path = urllib.parse.urlsplit(self.path).path
            if request_path == "/api/ce/task":
                if not state["task_responses"]:
                    self.send_error(500, "No task response configured")
                    return
                payload = {"task": state["task_responses"].pop(0)}
            elif request_path == "/api/issues/gitlab_sast_export":
                payload = G_SAST_REPORT
            elif request_path == "/api/qualitygates/project_status":
                payload = {"projectStatus": {"status": state["gate_status"]}}
            else:
                self.send_error(404)
                return

            response = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format_string, *args):
            del format_string, args

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def run_sonar_results(tmp_path, server_url):
    metadata_file = tmp_path / ".scannerwork" / "report-task.txt"
    metadata_file.parent.mkdir()
    metadata_file.write_text(
        "projectKey=edge-llm\n"
        f"serverUrl={server_url}\n"
        f"ceTaskUrl={server_url}/api/ce/task?id=task-1\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "gl-sast-sonar-report.json"
    environment = os.environ.copy()
    environment.update({
        "SONAR_TOKEN": "test-token",
        "CI_COMMIT_BRANCH": "main",
        "CI_MERGE_REQUEST_IID": "",
    })
    result = subprocess.run(
        [
            "python3",
            str(G_SCRIPT),
            "--output-file",
            str(output_file),
            "--timeout-seconds",
            "2",
            "--poll-interval-seconds",
            "0",
            "--request-timeout-seconds",
            "2",
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=environment,
        text=True,
        timeout=10,
    )
    return result, output_file


def test_successful_analysis_publishes_sast_and_passes(sonar_server, tmp_path):
    state, server_url = sonar_server
    state["task_responses"] = [{
        "status": "PENDING"
    }, {
        "status": "SUCCESS",
        "analysisId": "analysis-1",
    }]

    result, output_file = run_sonar_results(tmp_path, server_url)

    assert result.returncode == 0
    assert json.loads(output_file.read_text(encoding="utf-8")) == G_SAST_REPORT
    requested_paths = [
        urllib.parse.urlsplit(request["path"]).path
        for request in state["requests"]
    ]
    assert "/api/issues/gitlab_sast_export" in requested_paths
    assert "/api/qualitygates/project_status" in requested_paths
    assert all(request["authorization"] for request in state["requests"])


def test_failed_gate_still_publishes_sast(sonar_server, tmp_path):
    state, server_url = sonar_server
    state["task_responses"] = [{
        "status": "SUCCESS",
        "analysisId": "analysis-1",
    }]
    state["gate_status"] = "ERROR"

    result, output_file = run_sonar_results(tmp_path, server_url)

    assert result.returncode != 0
    assert json.loads(output_file.read_text(encoding="utf-8")) == G_SAST_REPORT


def test_failed_processing_does_not_publish_stale_sast(sonar_server, tmp_path):
    state, server_url = sonar_server
    state["task_responses"] = [{
        "status": "FAILED",
        "errorMessage": "processing failed",
    }]

    result, output_file = run_sonar_results(tmp_path, server_url)

    assert result.returncode != 0
    assert not output_file.exists()
    requested_paths = [
        urllib.parse.urlsplit(request["path"]).path
        for request in state["requests"]
    ]
    assert "/api/issues/gitlab_sast_export" not in requested_paths


def test_sonar_pipeline_preserves_results_handoff_contract():
    sonar_jobs = yaml.safe_load((G_REPO_ROOT / ".gitlab" / "ci" /
                                 "sonar-jobs.yml").read_text(encoding="utf-8"))
    properties = dict(
        line.split("=", 1)
        for line in (G_REPO_ROOT / "sonar-project.properties").read_text(
            encoding="utf-8").splitlines() if line)

    assert properties.get("sonar.qualitygate.wait", "false") == "false"
    metadata_path = properties.get("sonar.scanner.metadataFilePath")
    assert metadata_path is None or pathlib.PurePosixPath(
        metadata_path).is_absolute()

    build_artifacts = sonar_jobs["build-sonar"]["artifacts"]["paths"]
    assert any(path.endswith("report-task.txt") for path in build_artifacts)

    results_job = sonar_jobs["sonar-results"]
    build_need = next(need for need in results_job["needs"]
                      if need["job"] == "build-sonar")
    assert build_need["artifacts"] is True
    assert results_job["artifacts"]["reports"].get("sast")
