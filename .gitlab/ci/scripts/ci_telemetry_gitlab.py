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

import json
import os
import pathlib
import time
import typing
import urllib.error
import urllib.parse
import urllib.request

API_TIMEOUT_SECONDS = 30
API_MAX_ATTEMPTS = 3
MAX_PORTABLE_RECORD_BYTES = 2 * 1024 * 1024


class ReportError(RuntimeError):
    """Raised when telemetry cannot be retrieved or interpreted."""


class GitLabClient:
    """Minimal GitLab API client for telemetry collection."""

    def __init__(self,
                 api_url: str,
                 project_id: str,
                 token: typing.Optional[str] = None,
                 token_header: str = "PRIVATE-TOKEN") -> None:
        encoded = urllib.parse.quote(project_id, safe="")
        self._project_url = f"{api_url.rstrip('/')}/projects/{encoded}"
        self._headers = {"Accept": "application/json"}
        if token:
            self._headers[token_header] = token

    def _request(
            self,
            path: str) -> typing.Tuple[typing.Any, typing.Mapping[str, str]]:
        request = urllib.request.Request(f"{self._project_url}{path}",
                                         headers=self._headers)
        for attempt in range(API_MAX_ATTEMPTS):
            try:
                with urllib.request.urlopen(
                        request, timeout=API_TIMEOUT_SECONDS) as response:
                    return json.load(response), response.headers
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt + 1 == API_MAX_ATTEMPTS:
                    raise ReportError(
                        f"GitLab API request failed for {path}: HTTP {error.code}"
                    ) from error
                retry_after = error.headers.get("Retry-After", "")
                time.sleep(
                    min(int(retry_after), 10) if retry_after.isdigit(
                    ) else 2**attempt)
            except (urllib.error.URLError, json.JSONDecodeError) as error:
                raise ReportError(
                    f"GitLab API request failed for {path}: {error}"
                ) from error
        raise ReportError(f"GitLab API request failed for {path}")

    def pipeline(self, pipeline_id: str) -> typing.Mapping[str, typing.Any]:
        payload, _ = self._request(f"/pipelines/{pipeline_id}")
        if not isinstance(payload, dict):
            raise ReportError(
                f"pipeline {pipeline_id} response is not an object")
        return payload

    def jobs(self,
             pipeline_id: str) -> typing.List[typing.Mapping[str, typing.Any]]:
        jobs: typing.List[typing.Mapping[str, typing.Any]] = []
        page = "1"
        while page:
            payload, headers = self._request(
                f"/pipelines/{pipeline_id}/jobs?include_retried=true"
                f"&per_page=100&page={page}")
            if not isinstance(payload, list):
                raise ReportError(
                    f"jobs response for pipeline {pipeline_id} is not a list")
            jobs.extend(item for item in payload if isinstance(item, dict))
            page = headers.get("X-Next-Page", "")
        return jobs

    def test_report(self, pipeline_id: str) -> typing.Mapping[str, typing.Any]:
        payload, _ = self._request(f"/pipelines/{pipeline_id}/test_report")
        if not isinstance(payload, dict):
            raise ReportError(
                f"test report for pipeline {pipeline_id} is not an object")
        return payload

    def portable_runner_record(
            self,
            job_id: str) -> typing.Optional[typing.Mapping[str, typing.Any]]:
        artifact = urllib.parse.quote(f".ci-telemetry-artifact/{job_id}.json",
                                      safe="/")
        request = urllib.request.Request(
            f"{self._project_url}/jobs/{job_id}/artifacts/{artifact}",
            headers=self._headers)
        try:
            with urllib.request.urlopen(
                    request, timeout=API_TIMEOUT_SECONDS) as response:
                payload = response.read(MAX_PORTABLE_RECORD_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise ReportError(
                f"GitLab artifact request failed for job {job_id}: HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise ReportError(
                f"GitLab artifact request failed for job {job_id}: {error}"
            ) from error
        if len(payload) > MAX_PORTABLE_RECORD_BYTES:
            raise ReportError(
                f"portable runner record for job {job_id} is oversized")
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ReportError(
                f"portable runner record for job {job_id} is malformed"
            ) from error
        if not isinstance(record, dict):
            raise ReportError(
                f"portable runner record for job {job_id} is not an object")
        return record


def read_token(
    token_file: typing.Optional[pathlib.Path]
) -> typing.Tuple[typing.Optional[str], str]:
    if token_file:
        if token_file.stat().st_size > 16384:
            raise ReportError("token file is unexpectedly large")
        return token_file.read_text(encoding="utf-8").strip(), "PRIVATE-TOKEN"
    if os.environ.get("GITLAB_TOKEN"):
        return os.environ["GITLAB_TOKEN"], "PRIVATE-TOKEN"
    if os.environ.get("CI_JOB_TOKEN"):
        return os.environ["CI_JOB_TOKEN"], "JOB-TOKEN"
    return None, "PRIVATE-TOKEN"
