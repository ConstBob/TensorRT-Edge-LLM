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

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import typing
import urllib.error
import urllib.parse
import urllib.request

IN_PROGRESS_STATUSES = {"PENDING", "IN_PROGRESS"}
FAILED_STATUSES = {"FAILED", "CANCELED"}


class SonarResultsError(RuntimeError):
    """Raised when the submitted SonarQube analysis cannot be published."""


def _parse_arguments(
        argv: typing.Optional[typing.Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish one submitted SonarQube analysis to GitLab.")
    parser.add_argument(
        "--metadata-file",
        default=".scannerwork/report-task.txt",
        help="Metadata written by sonar-scanner after report submission.",
    )
    parser.add_argument(
        "--output-file",
        default="gl-sast-sonar-report.json",
        help="Destination for the GitLab SAST report.",
    )
    parser.add_argument(
        "--timeout-seconds",
        default=300.0,
        type=float,
        help="Maximum time to wait for SonarQube processing.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        default=5.0,
        type=float,
        help="Delay between Compute Engine status requests.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        default=30.0,
        type=float,
        help="Timeout for each SonarQube API request.",
    )
    arguments = parser.parse_args(argv)
    if arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    if arguments.poll_interval_seconds < 0:
        parser.error("--poll-interval-seconds must not be negative")
    if arguments.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be greater than zero")
    return arguments


def _read_metadata(path: pathlib.Path) -> typing.Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SonarResultsError(
            f"cannot read SonarQube task metadata {path}: {error}") from error

    metadata = {}
    for line in lines:
        if not line:
            continue
        if "=" not in line:
            raise SonarResultsError(
                f"invalid SonarQube task metadata line: {line!r}")
        key, value = line.split("=", 1)
        metadata[key] = value

    for key in ("ceTaskUrl", "projectKey", "serverUrl"):
        if not metadata.get(key):
            raise SonarResultsError(
                f"SonarQube task metadata is missing {key}")

    server_url = urllib.parse.urlsplit(metadata["serverUrl"])
    task_url = urllib.parse.urlsplit(metadata["ceTaskUrl"])
    server_origin = (server_url.scheme, server_url.netloc)
    task_origin = (task_url.scheme, task_url.netloc)
    if not all(server_origin) or server_origin != task_origin:
        raise SonarResultsError(
            "SonarQube task URL does not belong to the reported server")
    return metadata


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SonarResultsError(f"{name} must be set")
    return value


def _request_json(url: str, token: str,
                  timeout_seconds: float) -> typing.Dict[str, typing.Any]:
    request_path = urllib.parse.urlsplit(url).path
    credentials = base64.b64encode(f"{token}:".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Basic {credentials}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request,
                                    timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise SonarResultsError(
            f"SonarQube request to {request_path} returned HTTP {error.code}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise SonarResultsError(
            f"SonarQube request to {request_path} failed: {error}") from error

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise SonarResultsError(
            f"SonarQube request to {request_path} returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise SonarResultsError(f"SonarQube request to {request_path} "
                                "did not return a JSON object")
    return payload


def _wait_for_analysis(metadata: typing.Mapping[str, str], token: str,
                       timeout_seconds: float, poll_interval_seconds: float,
                       request_timeout_seconds: float) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while True:
        response = _request_json(metadata["ceTaskUrl"], token,
                                 request_timeout_seconds)
        task = response.get("task")
        if not isinstance(task, dict) or not isinstance(
                task.get("status"), str):
            raise SonarResultsError(
                "SonarQube Compute Engine response has no task status")

        status = task["status"]
        if status != last_status:
            print(f"SonarQube Compute Engine task: {status}")
            last_status = status

        if status == "SUCCESS":
            analysis_id = task.get("analysisId")
            if not isinstance(analysis_id, str) or not analysis_id:
                raise SonarResultsError(
                    "completed SonarQube task has no analysisId")
            return analysis_id
        if status in FAILED_STATUSES:
            detail = task.get("errorMessage", "no error details")
            raise SonarResultsError(
                f"SonarQube Compute Engine task {status}: {detail}")
        if status not in IN_PROGRESS_STATUSES:
            raise SonarResultsError(
                f"unknown SonarQube Compute Engine status: {status}")

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise SonarResultsError(
                f"SonarQube processing exceeded {timeout_seconds:g} seconds")
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def _export_sast_report(metadata: typing.Mapping[str, str], token: str,
                        output_path: pathlib.Path,
                        request_timeout_seconds: float) -> None:
    branch = os.environ.get("CI_COMMIT_BRANCH", "")
    pull_request = os.environ.get("CI_MERGE_REQUEST_IID", "")
    if not branch and not pull_request:
        raise SonarResultsError(
            "CI_COMMIT_BRANCH or CI_MERGE_REQUEST_IID must be set")

    query = urllib.parse.urlencode({
        "projectKey": metadata["projectKey"],
        "branch": branch,
        "pullRequest": pull_request,
    })
    url = (f"{metadata['serverUrl'].rstrip('/')}"
           f"/api/issues/gitlab_sast_export?{query}")
    report = _request_json(url, token, request_timeout_seconds)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8")
    except OSError as error:
        raise SonarResultsError(
            f"cannot write GitLab SAST report {output_path}: {error}"
        ) from error
    print(f"Published GitLab SAST report: {output_path}")


def _quality_gate_status(metadata: typing.Mapping[str, str], analysis_id: str,
                         token: str, request_timeout_seconds: float) -> str:
    query = urllib.parse.urlencode({"analysisId": analysis_id})
    url = (f"{metadata['serverUrl'].rstrip('/')}"
           f"/api/qualitygates/project_status?{query}")
    response = _request_json(url, token, request_timeout_seconds)
    project_status = response.get("projectStatus")
    if not isinstance(project_status, dict) or not isinstance(
            project_status.get("status"), str):
        raise SonarResultsError(
            "SonarQube Quality Gate response has no project status")
    return project_status["status"]


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        token = _required_environment("SONAR_TOKEN")
        metadata = _read_metadata(pathlib.Path(arguments.metadata_file))
        analysis_id = _wait_for_analysis(
            metadata,
            token,
            arguments.timeout_seconds,
            arguments.poll_interval_seconds,
            arguments.request_timeout_seconds,
        )
        _export_sast_report(
            metadata,
            token,
            pathlib.Path(arguments.output_file),
            arguments.request_timeout_seconds,
        )
        gate_status = _quality_gate_status(
            metadata,
            analysis_id,
            token,
            arguments.request_timeout_seconds,
        )
    except SonarResultsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"SonarQube Quality Gate: {gate_status}")
    if gate_status != "OK":
        print("ERROR: SonarQube Quality Gate did not pass", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
