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

import collections
import datetime
import typing

REPORT_SCHEMA_VERSION = 2
TOP_RESULT_COUNT = 20
SUCCESS_STATUSES = frozenset(("success", "passed"))


def number(value: typing.Any) -> typing.Optional[float]:
    """Return a finite numeric telemetry value or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if result == result and result not in (float("inf"), float("-inf")):
            return result
    return None


def identifier(value: typing.Any) -> str:
    """Normalize a GitLab identifier without interpreting its contents."""
    return "" if value is None else str(value)


def _timestamp(value: typing.Any) -> typing.Optional[float]:
    numeric = number(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def phase_interval_metrics(
    phases: typing.Iterable[typing.Mapping[str, typing.Any]],
) -> typing.Dict[str, typing.Any]:
    """Calculate raw, union, and overlap time for phase intervals."""
    phase_list = list(phases)
    raw_seconds = sum(value for value in (number(item.get("elapsed_seconds"))
                                          for item in phase_list)
                      if value is not None)
    intervals: typing.List[typing.Tuple[float, float]] = []
    for phase in phase_list:
        start = _timestamp(phase.get("started_epoch"))
        finish = _timestamp(phase.get("finished_epoch"))
        if start is None or finish is None:
            start = _timestamp(phase.get("started_at"))
            finish = _timestamp(phase.get("finished_at"))
        if start is None or finish is None or finish < start:
            continue
        intervals.append((start, finish))

    interval_seconds = sum(finish - start for start, finish in intervals)
    unique_seconds = 0.0
    if intervals:
        current_start, current_finish = sorted(intervals)[0]
        for start, finish in sorted(intervals)[1:]:
            if start <= current_finish:
                current_finish = max(current_finish, finish)
            else:
                unique_seconds += current_finish - current_start
                current_start, current_finish = start, finish
        unique_seconds += current_finish - current_start

    return {
        "phase_seconds": raw_seconds,
        "phase_interval_count": len(intervals),
        "phase_unique_seconds": unique_seconds,
        "phase_overlap_seconds": max(0.0, interval_seconds - unique_seconds),
        "phase_unbounded_count": len(phase_list) - len(intervals),
    }


def annotate_attempts(
    job_rows: typing.Sequence[typing.Dict[str, typing.Any]], ) -> None:
    """Add deterministic retry and selected-attempt fields in place."""
    groups: typing.Dict[typing.Tuple[str, str], typing.List[typing.Dict[
        str, typing.Any]]] = collections.defaultdict(list)
    for row in job_rows:
        groups[(identifier(row.get("pipeline_id")),
                str(row.get("job_name", "")))].append(row)

    for attempts in groups.values():
        attempts.sort(key=_attempt_sort_key)
        latest_success = next(
            (row for row in reversed(attempts)
             if str(row.get("status", "")).lower() in SUCCESS_STATUSES), None)
        latest_attempt = attempts[-1]
        for attempt_number, row in enumerate(attempts, start=1):
            row["attempt_number"] = attempt_number
            row["attempt_count"] = len(attempts)
            row["is_retry"] = attempt_number > 1
            row["is_latest_attempt"] = row is latest_attempt
            row["is_latest_success"] = row is latest_success
            row["is_superseded_attempt"] = row is not latest_attempt


def _attempt_sort_key(
    row: typing.Mapping[str, typing.Any]
) -> typing.Tuple[float, typing.Tuple[int, typing.Union[int, str]]]:
    created = _timestamp(row.get("created_at"))
    job_id = identifier(row.get("job_id"))
    job_key: typing.Tuple[int, typing.Union[int, str]]
    if job_id.isdigit():
        job_key = (0, int(job_id))
    else:
        job_key = (1, job_id)
    return (created if created is not None else -1.0, job_key)


def aggregate(
    observations: typing.Iterable[typing.Mapping[str, typing.Any]],
    name_field: str,
    duration_field: str,
    statuses: typing.Sequence[str],
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Aggregate named observations by status and elapsed time."""
    aggregates: typing.Dict[str, typing.Dict[
        str, typing.Any]] = collections.defaultdict(dict)
    for observation in observations:
        name = str(observation.get(name_field, ""))
        result = aggregates[name]
        if not result:
            result.update({
                name_field: name,
                "sample_count": 0,
                duration_field: 0.0,
                "other": 0,
            })
            result.update({status: 0 for status in statuses})
        result["sample_count"] += 1
        status = str(observation.get("status", "")).lower()
        result[status if status in statuses else "other"] += 1
        result[duration_field] += number(
            observation.get(duration_field)) or 0.0
    return sorted(aggregates.values(),
                  key=lambda item: (-item[duration_field], item[name_field]))


def aggregate_test_functions(
    observations: typing.Iterable[typing.Mapping[str, typing.Any]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Aggregate parameterized cases by class or suite and function."""
    return aggregate(observations, "function", "execution_seconds",
                     ("passed", "failed", "skipped", "error"))


def build_phase_rows(
    job_rows: typing.Iterable[typing.Mapping[str, typing.Any]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Flatten job phase records into deterministic report rows."""
    rows: typing.List[typing.Dict[str, typing.Any]] = []
    for job in job_rows:
        for phase in job.get("phases", []):
            rows.append({
                "pipeline_id": job.get("pipeline_id", ""),
                "job_id": job.get("job_id", ""),
                "job_name": job.get("job_name", ""),
                "attempt_number": job.get("attempt_number", ""),
                "label": phase.get("label", ""),
                "status": phase.get("status", ""),
                "exit_code": phase.get("exit_code", ""),
                "started_at": phase.get("started_at", ""),
                "finished_at": phase.get("finished_at", ""),
                "elapsed_seconds": phase.get("elapsed_seconds", ""),
            })
    return sorted(rows,
                  key=lambda row: (str(row["pipeline_id"]), str(row["job_id"]),
                                   str(row["started_at"]), str(row["label"])))


def aggregate_phase_labels(
    rows: typing.Iterable[typing.Mapping[str, typing.Any]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Aggregate repeated phase labels across jobs and pipelines."""
    return aggregate(rows, "label", "elapsed_seconds",
                     ("success", "failure", "timeout"))


def _sum_field(rows: typing.Iterable[typing.Mapping[str, typing.Any]],
               name: str) -> float:
    return sum(value for value in (number(row.get(name)) for row in rows)
               if value is not None)


def _pipeline_summary(
    pipeline: typing.Mapping[str, typing.Any],
    rows: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> typing.Dict[str, typing.Any]:
    latest_success = [row for row in rows if row.get("is_latest_success")]
    latest_attempt = [row for row in rows if row.get("is_latest_attempt")]
    superseded = [row for row in rows if row.get("is_superseded_attempt")]
    return {
        "pipeline_id":
        identifier(pipeline.get("id")),
        "status":
        pipeline.get("status", ""),
        "ref":
        pipeline.get("ref", ""),
        "sha":
        pipeline.get("sha", ""),
        "web_url":
        pipeline.get("web_url", ""),
        "attempt_count":
        len(rows),
        "logical_job_count":
        len(latest_attempt),
        "latest_success_job_count":
        len(latest_success),
        "total_consumed_execution_seconds":
        _sum_field(rows, "execution_seconds"),
        "latest_attempt_execution_seconds":
        _sum_field(latest_attempt, "execution_seconds"),
        "latest_success_execution_seconds":
        _sum_field(latest_success, "execution_seconds"),
        "retry_execution_seconds":
        _sum_field(superseded, "execution_seconds"),
        "queue_seconds_total":
        _sum_field(rows, "queue_seconds"),
        "phase_unique_seconds_total":
        _sum_field(rows, "phase_unique_seconds"),
    }


def build_summary(
    pipelines: typing.Sequence[typing.Mapping[str, typing.Any]],
    job_rows: typing.Sequence[typing.Mapping[str, typing.Any]],
    test_functions: typing.Sequence[typing.Mapping[str, typing.Any]],
    phase_labels: typing.Sequence[typing.Mapping[str, typing.Any]],
    warnings: typing.Sequence[str],
) -> typing.Dict[str, typing.Any]:
    """Create the stable machine-readable report summary."""
    queue_samples = [
        value for value in (number(row.get("queue_seconds"))
                            for row in job_rows) if value is not None
    ]
    execution_samples = [
        value for value in (number(row.get("execution_seconds"))
                            for row in job_rows) if value is not None
    ]
    latest_success_rows = [
        row for row in job_rows if row.get("is_latest_success")
    ]
    latest_attempt_rows = [
        row for row in job_rows if row.get("is_latest_attempt")
    ]
    superseded_rows = [
        row for row in job_rows if row.get("is_superseded_attempt")
    ]
    top_jobs = sorted(job_rows,
                      key=lambda row: -(number(row.get("execution_seconds")) or
                                        0.0))[:TOP_RESULT_COUNT]
    top_latest_success = sorted(
        latest_success_rows,
        key=lambda row: -(number(row.get("execution_seconds")) or 0.0
                          ))[:TOP_RESULT_COUNT]
    phase_unique_seconds = _sum_field(job_rows, "phase_unique_seconds")
    execution_seconds = sum(execution_samples)
    pipeline_rows = {
        identifier(pipeline.get("id")): [
            row for row in job_rows
            if identifier(row.get("pipeline_id")) == identifier(
                pipeline.get("id"))
        ]
        for pipeline in pipelines
    }
    pipeline_comparisons = [
        _pipeline_summary(pipeline,
                          pipeline_rows[identifier(pipeline.get("id"))])
        for pipeline in pipelines
    ]
    return {
        "schema_version":
        REPORT_SCHEMA_VERSION,
        "execution_time_definition":
        ("GitLab job duration, or collector run_seconds when unavailable; "
         "runner queue and pipeline-to-start time are excluded."),
        "queue_time_definition":
        ("GitLab queued_duration when available; reported separately and "
         "never added to execution time."),
        "attempt_view_definition":
        ("Total consumed includes every executed attempt. Latest-success "
         "includes only the final successful attempt for each logical job. "
         "Retry cost includes superseded attempts."),
        "phase_instrumentation":
        ("Phase coverage is the union of valid phase intervals within each "
         "job. Raw phase sums and overlap are retained as diagnostics."),
        "pipeline_count":
        len(pipelines),
        "job_count":
        len(job_rows),
        "logical_job_count":
        len(latest_attempt_rows),
        "latest_success_job_count":
        len(latest_success_rows),
        "retry_attempt_count":
        len(superseded_rows),
        "runner_usage_sample_count":
        sum(1 for row in job_rows if row.get("runner_usage_found")),
        "fallback_runner_usage_sample_count":
        sum(1 for row in job_rows
            if row.get("runner_usage_source") == "artifact"),
        "phase_sample_count":
        sum(int(row.get("phase_count", 0)) for row in job_rows),
        "phase_interval_sample_count":
        sum(int(row.get("phase_interval_count", 0)) for row in job_rows),
        "malformed_phase_count":
        sum(int(row.get("malformed_phase_count", 0)) for row in job_rows),
        "phase_raw_seconds_total":
        _sum_field(job_rows, "phase_seconds"),
        "phase_unique_seconds_total":
        phase_unique_seconds,
        "phase_overlap_seconds_total":
        _sum_field(job_rows, "phase_overlap_seconds"),
        "phase_unique_coverage_percent":
        (100.0 * phase_unique_seconds /
         execution_seconds if execution_seconds else 0.0),
        "queue_sample_count":
        len(queue_samples),
        "queue_seconds_total":
        sum(queue_samples),
        "execution_sample_count":
        len(execution_samples),
        "execution_seconds_total":
        execution_seconds,
        "total_consumed_execution_seconds":
        execution_seconds,
        "latest_attempt_execution_seconds":
        _sum_field(latest_attempt_rows, "execution_seconds"),
        "latest_success_execution_seconds":
        _sum_field(latest_success_rows, "execution_seconds"),
        "retry_execution_seconds":
        _sum_field(superseded_rows, "execution_seconds"),
        "test_function_count":
        len(test_functions),
        "test_case_sample_count":
        sum(int(row.get("sample_count", 0)) for row in test_functions),
        "pipelines": [{
            key: item.get(key)
            for key in ("id", "status", "ref", "sha", "web_url")
        } for item in pipelines],
        "pipeline_comparisons":
        pipeline_comparisons,
        "top_jobs_by_execution_seconds": [_top_job(row) for row in top_jobs],
        "top_latest_success_jobs_by_execution_seconds":
        [_top_job(row) for row in top_latest_success],
        "top_test_functions_by_execution_seconds":
        list(test_functions[:TOP_RESULT_COUNT]),
        "top_phase_labels_by_elapsed_seconds":
        list(phase_labels[:TOP_RESULT_COUNT]),
        "warnings":
        list(warnings),
    }


def _top_job(
        row: typing.Mapping[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    return {
        "pipeline_id": row.get("pipeline_id"),
        "job_id": row.get("job_id"),
        "job_name": row.get("job_name"),
        "attempt_number": row.get("attempt_number"),
        "attempt_count": row.get("attempt_count"),
        "status": row.get("status"),
        "execution_seconds": row.get("execution_seconds"),
        "queue_seconds": row.get("queue_seconds"),
        "phase_count": row.get("phase_count"),
        "phase_unique_seconds": row.get("phase_unique_seconds"),
    }
