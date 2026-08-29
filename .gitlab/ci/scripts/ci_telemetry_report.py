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
import pathlib
import sys
import typing

SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import ci_telemetry_aggregate
import ci_telemetry_gitlab
import ci_telemetry_output
import ci_telemetry_records

aggregate_test_functions = ci_telemetry_aggregate.aggregate_test_functions
aggregate_phase_labels = ci_telemetry_aggregate.aggregate_phase_labels
build_phase_rows = ci_telemetry_aggregate.build_phase_rows
build_summary = ci_telemetry_aggregate.build_summary
build_job_rows = ci_telemetry_records.build_job_rows
flatten_test_cases = ci_telemetry_records.flatten_test_cases
function_prefix = ci_telemetry_records.function_prefix
load_phase_records = ci_telemetry_records.load_phase_records
load_runner_usage = ci_telemetry_records.load_runner_usage
render_markdown = ci_telemetry_output.render_markdown
write_reports = ci_telemetry_output.write_reports


def _pipeline_id(value: str) -> str:
    if not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError(
            "pipeline ID must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dependency-free GitLab CI timing reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Authentication:
  --token-file takes precedence over GITLAB_TOKEN and CI_JOB_TOKEN.

Outputs:
  index.html, summary.svg, summary.md, summary.json, metrics.txt, jobs.csv,
  phases.csv, cache_events.csv, and test_functions.csv

Timing:
  Execution excludes runner queue and pipeline scheduling wait. Local, wheel,
  and relayed board phases contribute to interval coverage.
""")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--pipeline-id",
                        action="append",
                        required=True,
                        dest="pipeline_ids",
                        type=_pipeline_id)
    parser.add_argument("--token-file", type=pathlib.Path)
    parser.add_argument("--runner-usage-root",
                        required=True,
                        type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--exclude-job-id", action="append", default=[])
    return parser


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        token, token_header = ci_telemetry_gitlab.read_token(args.token_file)
        client = ci_telemetry_gitlab.GitLabClient(args.api_url,
                                                  args.project_id, token,
                                                  token_header)
        runner_records, warnings = load_runner_usage(args.runner_usage_root,
                                                     args.pipeline_ids)
        pipelines: typing.List[typing.Mapping[str, typing.Any]] = []
        job_rows: typing.List[typing.Dict[str, typing.Any]] = []
        observations: typing.List[typing.Dict[str, typing.Any]] = []
        excluded = set(args.exclude_job_id)
        for pipeline_id in args.pipeline_ids:
            pipelines.append(client.pipeline(pipeline_id))
            jobs = [
                job for job in client.jobs(pipeline_id)
                if ci_telemetry_aggregate.identifier(job.get("id")) not in
                excluded
            ]
            ci_telemetry_records.load_fallback_runner_usage(
                client, pipeline_id, jobs, runner_records, warnings)
            job_rows.extend(build_job_rows(pipeline_id, jobs, runner_records))
            observations.extend(
                flatten_test_cases(pipeline_id,
                                   client.test_report(pipeline_id)))
        job_rows.sort(key=lambda row: (str(row[
            "pipeline_id"]), str(row["job_name"]), str(row["job_id"])))
        test_functions = aggregate_test_functions(observations)
        phase_rows = build_phase_rows(job_rows)
        phase_labels = aggregate_phase_labels(phase_rows)
        write_reports(args.output_dir, pipelines, job_rows, test_functions,
                      phase_rows, phase_labels, warnings)
    except (OSError, ci_telemetry_gitlab.ReportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"CI telemetry reports written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
