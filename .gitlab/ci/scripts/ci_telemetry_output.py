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

import csv
import json
import os
import pathlib
import typing

import ci_cache_event
import ci_telemetry_aggregate
import ci_telemetry_graph

_number = ci_telemetry_aggregate.number


def _atomic_csv(path: pathlib.Path,
                rows: typing.Sequence[typing.Mapping[str, typing.Any]],
                fieldnames: typing.Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output,
                                    fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: pathlib.Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _format_seconds(value: typing.Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.1f}"


def _markdown_cell(value: typing.Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace(
        "\r", " ").replace("\n", " ")


def render_markdown(summary: typing.Mapping[str, typing.Any]) -> str:
    """Render the concise human-readable report."""
    lines = [
        "# CI telemetry summary",
        "",
        f"- Pipelines: {summary['pipeline_count']}",
        f"- Job attempts: {summary['job_count']}",
        f"- Logical jobs: {summary['logical_job_count']}",
        f"- Runner usage samples: {summary['runner_usage_sample_count']} "
        f"({summary['fallback_runner_usage_sample_count']} from artifacts)",
        f"- Total consumed execution: "
        f"{_format_seconds(summary['total_consumed_execution_seconds'])} seconds",
        f"- Latest-success execution: "
        f"{_format_seconds(summary['latest_success_execution_seconds'])} seconds",
        f"- Superseded retry execution: "
        f"{_format_seconds(summary['retry_execution_seconds'])} seconds",
        f"- Runner queue: {_format_seconds(summary['queue_seconds_total'])} seconds",
        f"- Unique phase coverage: "
        f"{_format_seconds(summary['phase_unique_seconds_total'])} seconds "
        f"({summary['phase_unique_coverage_percent']:.1f}%)",
        f"- Raw phase sum / overlap: "
        f"{_format_seconds(summary['phase_raw_seconds_total'])} / "
        f"{_format_seconds(summary['phase_overlap_seconds_total'])} seconds",
        f"- Test case samples: {summary['test_case_sample_count']}",
        f"- Cache events: {summary['cache_event_count']} "
        f"({summary['cache_hit_count']} hits, "
        f"{summary['cache_miss_count']} misses, "
        f"{summary['invalid_cache_event_count']} invalid)",
        "",
        f"Execution time: {summary['execution_time_definition']}",
        f"Queue time: {summary['queue_time_definition']}",
        f"Attempt views: {summary['attempt_view_definition']}",
        f"Phase coverage: {summary['phase_instrumentation']}",
        "",
        "Summed execution is consumed runner compute, not parallel pipeline wall time.",
        "",
        "## Pipeline comparison",
        "",
        "| Pipeline | Status | Attempts | Logical jobs | Latest successes | Total consumed (s) | Latest success (s) | Retry (s) | Queue (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["pipeline_comparisons"]:
        lines.append(
            f"| {_markdown_cell(row['pipeline_id'])} | "
            f"{_markdown_cell(row['status'])} | {row['attempt_count']} | "
            f"{row['logical_job_count']} | {row['latest_success_job_count']} | "
            f"{_format_seconds(row['total_consumed_execution_seconds'])} | "
            f"{_format_seconds(row['latest_success_execution_seconds'])} | "
            f"{_format_seconds(row['retry_execution_seconds'])} | "
            f"{_format_seconds(row['queue_seconds_total'])} |")
    lines.extend([
        "",
        "## Longest latest-success jobs",
        "",
        "| Pipeline | Job | Attempt | Execution (s) | Queue (s) | Phase union (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["top_latest_success_jobs_by_execution_seconds"]:
        lines.append(f"| {_markdown_cell(row['pipeline_id'])} | "
                     f"{_markdown_cell(row['job_name'])} | "
                     f"{row['attempt_number']}/{row['attempt_count']} | "
                     f"{_format_seconds(row['execution_seconds'])} | "
                     f"{_format_seconds(row['queue_seconds'])} | "
                     f"{_format_seconds(row['phase_unique_seconds'])} |")
    lines.extend([
        "",
        "## Top consumed attempts",
        "",
        "| Pipeline | Job | Status | Attempt | Execution (s) | Queue (s) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for row in summary["top_jobs_by_execution_seconds"]:
        lines.append(f"| {_markdown_cell(row['pipeline_id'])} | "
                     f"{_markdown_cell(row['job_name'])} | "
                     f"{_markdown_cell(row['status'])} | "
                     f"{row['attempt_number']}/{row['attempt_count']} | "
                     f"{_format_seconds(row['execution_seconds'])} | "
                     f"{_format_seconds(row['queue_seconds'])} |")
    lines.extend([
        "",
        "## Top test functions by execution time",
        "",
        "| Function | Samples | Execution (s) | Passed | Failed | Skipped | Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["top_test_functions_by_execution_seconds"]:
        lines.append(
            f"| {_markdown_cell(row['function'])} | {row['sample_count']} | "
            f"{_format_seconds(row['execution_seconds'])} | {row['passed']} | "
            f"{row['failed']} | {row['skipped']} | {row['error']} |")
    lines.extend([
        "",
        "## Top phase labels by elapsed time",
        "",
        "| Phase | Samples | Raw elapsed (s) | Success | Failure | Timeout |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["top_phase_labels_by_elapsed_seconds"]:
        lines.append(
            f"| {_markdown_cell(row['label'])} | {row['sample_count']} | "
            f"{_format_seconds(row['elapsed_seconds'])} | {row['success']} | "
            f"{row['failure']} | {row['timeout']} |")
    if summary.get("cache_decisions"):
        lines.extend([
            "",
            "## Cache decisions",
            "",
            "| Artifact | Operation | Namespace | Decision | Reason | Field | Samples | Time (s) | Read bytes | Written bytes |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in summary["cache_decisions"]:
            lines.append(
                f"| {_markdown_cell(row['artifact'])} | "
                f"{_markdown_cell(row['operation'])} | "
                f"{_markdown_cell(row['namespace'])} | "
                f"{_markdown_cell(row['decision'])} | "
                f"{_markdown_cell(row['reason'])} | "
                f"{_markdown_cell(row['field'])} | {row['sample_count']} | "
                f"{_format_seconds(row['elapsed_seconds'])} | "
                f"{int(row['bytes_read'])} | {int(row['bytes_written'])} |")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {_markdown_cell(item)}"
                     for item in summary["warnings"])
    return "\n".join(lines) + "\n"


def write_reports(output_dir: pathlib.Path,
                  pipelines: typing.Sequence[typing.Mapping[str, typing.Any]],
                  job_rows: typing.Sequence[typing.Mapping[str, typing.Any]],
                  test_functions: typing.Sequence[typing.Mapping[str,
                                                                 typing.Any]],
                  phase_rows: typing.Sequence[typing.Mapping[str, typing.Any]],
                  phase_labels: typing.Sequence[typing.Mapping[str,
                                                               typing.Any]],
                  warnings: typing.Sequence[str]) -> None:
    """Atomically write machine-readable and graphical report artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_rows, invalid_cache_events = ci_cache_event.build_rows(job_rows)
    cache_decisions = ci_cache_event.aggregate(cache_rows)
    summary = ci_telemetry_aggregate.build_summary(pipelines, job_rows,
                                                   test_functions,
                                                   phase_labels, warnings)
    summary.update({
        "cache_event_count":
        len(cache_rows),
        "invalid_cache_event_count":
        invalid_cache_events,
        "cache_hit_count":
        sum(row["decision"] == "hit" for row in cache_rows),
        "cache_miss_count":
        sum(row["decision"] in ci_cache_event.MISS_DECISIONS
            for row in cache_rows),
        "cache_decisions":
        cache_decisions,
    })
    svg = ci_telemetry_graph.render_svg(summary)
    _atomic_text(output_dir / "summary.json",
                 json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_csv(output_dir / "jobs.csv", job_rows, [
        "pipeline_id", "job_id", "job_name", "stage", "status",
        "attempt_number", "attempt_count", "is_retry", "is_latest_attempt",
        "is_latest_success", "is_superseded_attempt", "created_at",
        "started_at", "finished_at", "queue_seconds", "execution_seconds",
        "phase_count", "phase_interval_count", "phase_seconds",
        "phase_unique_seconds", "phase_overlap_seconds",
        "phase_unbounded_count", "phase_coverage_percent",
        "malformed_phase_count", "publisher_elapsed_seconds", "runner_id",
        "runner_description", "runner_node", "runner_usage_found",
        "runner_usage_source", "web_url"
    ])
    _atomic_csv(output_dir / "test_functions.csv", test_functions, [
        "function", "sample_count", "passed", "failed", "skipped", "error",
        "other", "execution_seconds"
    ])
    _atomic_csv(output_dir / "phases.csv", phase_rows, [
        "pipeline_id", "job_id", "job_name", "attempt_number", "label",
        "status", "exit_code", "started_at", "finished_at", "elapsed_seconds"
    ])
    _atomic_csv(output_dir / "cache_events.csv", cache_rows, [
        "pipeline_id", "job_id", "job_name", "attempt_number", "artifact",
        "operation", "namespace", "key_prefix", "decision", "reason", "field",
        "lane", "elapsed_seconds", "files", "bytes_read", "bytes_written",
        "materialization", "build_skipped", "inference_executed"
    ])
    _atomic_text(output_dir / "summary.md", render_markdown(summary))
    _atomic_text(output_dir / "summary.svg", svg)
    _atomic_text(output_dir / "index.html",
                 ci_telemetry_graph.render_html(summary, svg))
    _atomic_text(output_dir / "metrics.txt",
                 ci_telemetry_graph.render_metrics(summary))
