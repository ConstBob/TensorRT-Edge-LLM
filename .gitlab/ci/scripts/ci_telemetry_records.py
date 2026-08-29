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
import pathlib
import typing

import ci_telemetry_aggregate
import ci_telemetry_gitlab

MAX_RUNNER_RECORDS = 10000
MAX_RUNNER_RECORD_BYTES = 1024 * 1024
MAX_PHASE_RECORDS_PER_JOB = 1000
MAX_PHASE_RECORD_BYTES = 64 * 1024
PORTABLE_RECORD_SCHEMA = "edgellm-ci-runner-usage/v1"
TEST_STATUS_ALIASES = {
    "error": "error",
    "failed": "failed",
    "failure": "failed",
    "passed": "passed",
    "skipped": "skipped",
    "success": "passed",
}

_number = ci_telemetry_aggregate.number
_identifier = ci_telemetry_aggregate.identifier


def function_prefix(test_name: str) -> str:
    """Return a pytest test name without its parameter suffix."""
    return test_name.split("[", 1)[0]


def _test_status(value: typing.Any) -> str:
    return TEST_STATUS_ALIASES.get(str(value).lower(), "other")


def load_runner_usage(
    root: pathlib.Path,
    pipeline_ids: typing.Sequence[str],
) -> typing.Tuple[typing.Dict[typing.Tuple[str, str], typing.Mapping[
        str, typing.Any]], typing.List[str]]:
    """Read only runner records for the requested pipelines."""
    records: typing.Dict[typing.Tuple[str, str],
                         typing.Mapping[str, typing.Any]] = {}
    warnings: typing.List[str] = []
    if not root.is_dir():
        warnings.append(f"runner usage root does not exist: {root}")
        return records, warnings

    pipeline_dirs = set()
    for pipeline_id in pipeline_ids:
        direct = root / pipeline_id
        if direct.is_dir():
            pipeline_dirs.add(direct)
        pipeline_dirs.update(path for path in root.glob(f"*/{pipeline_id}")
                             if path.is_dir())
    paths = sorted(path for directory in pipeline_dirs
                   for path in directory.glob("*.json"))
    if len(paths) > MAX_RUNNER_RECORDS:
        warnings.append(
            f"runner record limit exceeded; reading first {MAX_RUNNER_RECORDS}"
        )
        paths = paths[:MAX_RUNNER_RECORDS]

    for path in paths:
        if path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_RUNNER_RECORD_BYTES:
                warnings.append(f"ignored oversized runner record {path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"ignored runner record {path}: {error}")
            continue
        if not isinstance(payload, dict):
            warnings.append(f"ignored non-object runner record {path}")
            continue
        pipeline_id = _identifier(payload.get("pipeline_id"))
        job_id = _identifier(payload.get("job_id"))
        if pipeline_id not in pipeline_ids or not job_id:
            continue
        record = dict(payload)
        record["_record_path"] = str(path)
        records[(pipeline_id, job_id)] = record
    return records, warnings


def load_fallback_runner_usage(
    client: ci_telemetry_gitlab.GitLabClient,
    pipeline_id: str,
    jobs: typing.Iterable[typing.Mapping[str, typing.Any]],
    records: typing.MutableMapping[typing.Tuple[str, str],
                                   typing.Mapping[str, typing.Any]],
    warnings: typing.List[str],
) -> None:
    """Fill scratch gaps from bounded, sanitized per-job artifacts."""
    for job in jobs:
        job_id = _identifier(job.get("id"))
        key = (pipeline_id, job_id)
        if not job_id or key in records:
            continue
        try:
            payload = client.portable_runner_record(job_id)
        except ci_telemetry_gitlab.ReportError as error:
            warnings.append(str(error))
            continue
        if payload is None:
            continue
        if (payload.get("schema") != PORTABLE_RECORD_SCHEMA
                or _identifier(payload.get("pipeline_id")) != pipeline_id
                or _identifier(payload.get("job_id")) != job_id):
            warnings.append(
                f"ignored mismatched portable runner record for job {job_id}")
            continue
        phases = payload.get("phases", [])
        if not isinstance(phases, list):
            warnings.append(
                f"ignored portable runner record with invalid phases for job {job_id}"
            )
            continue
        record = dict(payload)
        record.pop("phases", None)
        record["_portable"] = True
        record["_portable_phases"] = phases
        records[key] = record


def _valid_phase(payload: typing.Any) -> bool:
    return (isinstance(payload, dict) and payload.get("schema_version") == 1
            and isinstance(payload.get("label"), str)
            and _number(payload.get("elapsed_seconds")) is not None)


def load_phase_records(
    record: typing.Optional[typing.Mapping[str, typing.Any]],
) -> typing.Tuple[typing.List[typing.Mapping[str, typing.Any]], int]:
    """Load bounded phase sidecars from scratch or a portable record."""
    if not record:
        return [], 0
    portable_phases = record.get("_portable_phases")
    if isinstance(portable_phases, list):
        phases = [
            item for item in portable_phases[:MAX_PHASE_RECORDS_PER_JOB]
            if _valid_phase(item)
        ]
        malformed = len(portable_phases) - len(phases)
        return phases, malformed
    if not record.get("_record_path"):
        return [], 0
    record_path = pathlib.Path(str(record["_record_path"]))
    phase_dir = record_path.parent / f"{record_path.stem}.phases"
    if not phase_dir.is_dir() or phase_dir.is_symlink():
        return [], 0

    phases: typing.List[typing.Mapping[str, typing.Any]] = []
    malformed = 0
    paths = sorted(phase_dir.glob("*.json"))
    if len(paths) > MAX_PHASE_RECORDS_PER_JOB:
        malformed += len(paths) - MAX_PHASE_RECORDS_PER_JOB
        paths = paths[:MAX_PHASE_RECORDS_PER_JOB]
    for path in paths:
        if path.is_symlink():
            malformed += 1
            continue
        try:
            if path.stat().st_size > MAX_PHASE_RECORD_BYTES:
                malformed += 1
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        if not _valid_phase(payload):
            malformed += 1
            continue
        phases.append(payload)
    return phases, malformed


def build_job_rows(
    pipeline_id: str,
    jobs: typing.Iterable[typing.Mapping[str, typing.Any]],
    runner_records: typing.Mapping[typing.Tuple[str, str],
                                   typing.Mapping[str, typing.Any]],
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Join API jobs with runner, attempt, and phase telemetry."""
    rows: typing.List[typing.Dict[str, typing.Any]] = []
    for job in jobs:
        job_id = _identifier(job.get("id"))
        runner_record = runner_records.get((pipeline_id, job_id))
        phases, malformed_phases = load_phase_records(runner_record)
        queue_seconds = _number(job.get("queued_duration"))
        execution_seconds = _number(job.get("duration"))
        if runner_record:
            if queue_seconds is None:
                queue_seconds = _number(runner_record.get("queue_seconds"))
            if execution_seconds is None:
                execution_seconds = _number(runner_record.get("run_seconds"))
        phase_metrics = ci_telemetry_aggregate.phase_interval_metrics(phases)
        phase_coverage = 0.0
        if execution_seconds:
            phase_coverage = (100.0 * phase_metrics["phase_unique_seconds"] /
                              execution_seconds)
        runner = job.get("runner") if isinstance(job.get("runner"),
                                                 dict) else {}
        usage_source = ""
        if runner_record:
            usage_source = ("artifact"
                            if runner_record.get("_portable") else "scratch")
        rows.append({
            "pipeline_id":
            pipeline_id,
            "job_id":
            job_id,
            "job_name":
            job.get("name", ""),
            "stage":
            job.get("stage", ""),
            "status":
            job.get("status", ""),
            "created_at":
            job.get("created_at", ""),
            "started_at":
            job.get("started_at", ""),
            "finished_at":
            job.get("finished_at", ""),
            "queue_seconds":
            queue_seconds,
            "execution_seconds":
            execution_seconds,
            "phase_count":
            len(phases),
            **phase_metrics,
            "phase_coverage_percent":
            phase_coverage,
            "malformed_phase_count":
            malformed_phases,
            "publisher_elapsed_seconds":
            _number((runner_record or {}).get("publisher_elapsed_seconds")),
            "runner_id": (runner_record or {}).get("runner_id",
                                                   runner.get("id", "")),
            "runner_description": (runner_record
                                   or {}).get("runner_description",
                                              runner.get("description", "")),
            "runner_node": (runner_record or {}).get("runner_node", ""),
            "runner_usage_found":
            bool(runner_record),
            "runner_usage_source":
            usage_source,
            "web_url":
            job.get("web_url", ""),
            "phases":
            phases,
        })
    ci_telemetry_aggregate.annotate_attempts(rows)
    return rows


def flatten_test_cases(
    pipeline_id: str, report: typing.Mapping[str, typing.Any]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Normalize GitLab test suites into observations."""
    observations: typing.List[typing.Dict[str, typing.Any]] = []
    suites = report.get("test_suites", [])
    if not isinstance(suites, list):
        return observations
    for suite in suites:
        if not isinstance(suite, dict) or not isinstance(
                suite.get("test_cases", []), list):
            continue
        for case in suite.get("test_cases", []):
            if not isinstance(case, dict):
                continue
            name = str(case.get("name", ""))
            scope = str(
                case.get("classname") or suite.get("name") or "unknown")
            observations.append({
                "pipeline_id":
                pipeline_id,
                "function":
                f"{scope}::{function_prefix(name)}",
                "status":
                _test_status(case.get("status")),
                "execution_seconds":
                _number(case.get("execution_time")) or 0.0,
            })
    return observations
