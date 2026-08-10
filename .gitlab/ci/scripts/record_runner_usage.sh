#!/bin/sh
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

set +e

usage_root="${RUNNER_USAGE_DIR:-${EDGE_LLM_CACHE_DIR:-/scratch.edge_llm_cache}/runner_usage}"
project_slug="${CI_PROJECT_PATH_SLUG:-tensorrt-edge-llm}"
pipeline_id="${CI_PIPELINE_ID:-manual}"
job_id="${CI_JOB_ID:-unknown}"
out_dir="${usage_root}/${project_slug}/${pipeline_id}"
out_file="${out_dir}/${job_id}.json"

mkdir -p "$out_dir" 2>/dev/null || {
    echo "WARN: cannot create runner usage dir $out_dir"
    exit 0
}

now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"

to_epoch() {
    if [ -z "$1" ]; then
        printf ''
        return
    fi
    date -u -d "$1" +%s 2>/dev/null || printf ''
}

duration() {
    start="$1"
    end="$2"
    if [ -n "$start" ] && [ -n "$end" ]; then
        printf '%s' "$((end - start))"
    else
        printf 'null'
    fi
}

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_string() {
    if [ -n "$1" ]; then
        printf '"%s"' "$(json_escape "$1")"
    else
        printf 'null'
    fi
}

json_number() {
    if printf '%s' "$1" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
        printf '%s' "$1"
    else
        printf 'null'
    fi
}

pipeline_epoch="$(to_epoch "${CI_PIPELINE_CREATED_AT:-}")"
job_started_epoch="$(to_epoch "${CI_JOB_STARTED_AT:-}")"
job_finished_epoch="$(to_epoch "$now_utc")"
pipeline_to_job_start_seconds="$(duration "$pipeline_epoch" "$job_started_epoch")"
run_seconds="$(duration "$job_started_epoch" "$job_finished_epoch")"
queue_seconds="${CI_JOB_QUEUED_DURATION:-}"
if [ -z "$queue_seconds" ] && [ "$pipeline_to_job_start_seconds" != "null" ]; then
    queue_seconds="$pipeline_to_job_start_seconds"
fi

runner_hostname="$(hostname 2>/dev/null || true)"
runner_node="${NODE_NAME:-$runner_hostname}"
source_commit_sha="${CI_MERGE_REQUEST_SOURCE_BRANCH_SHA:-}"
# GitLab leaves the source branch SHA empty for detached MR pipelines.
if [ -z "$source_commit_sha" ] \
    && [ "${CI_MERGE_REQUEST_EVENT_TYPE:-}" = "detached" ]; then
    source_commit_sha="${CI_COMMIT_SHA:-}"
fi

tmp_file="${out_file}.tmp.$$"
cat > "$tmp_file" <<JSON
{
  "schema_version": 2,
  "project": $(json_string "${CI_PROJECT_PATH:-}"),
  "ref": $(json_string "${CI_COMMIT_REF_NAME:-}"),
  "commit_sha": $(json_string "${CI_COMMIT_SHA:-}"),
  "merge_request_iid": $(json_string "${CI_MERGE_REQUEST_IID:-}"),
  "merge_request_event_type": $(json_string "${CI_MERGE_REQUEST_EVENT_TYPE:-}"),
  "source_commit_sha": $(json_string "$source_commit_sha"),
  "target_commit_sha": $(json_string "${CI_MERGE_REQUEST_TARGET_BRANCH_SHA:-}"),
  "pipeline_id": $(json_string "${CI_PIPELINE_ID:-}"),
  "pipeline_iid": $(json_string "${CI_PIPELINE_IID:-}"),
  "pipeline_source": $(json_string "${CI_PIPELINE_SOURCE:-}"),
  "pipeline_url": $(json_string "${CI_PIPELINE_URL:-}"),
  "pipeline_created_at": $(json_string "${CI_PIPELINE_CREATED_AT:-}"),
  "job_id": $(json_string "${CI_JOB_ID:-}"),
  "job_name": $(json_string "${CI_JOB_NAME:-}"),
  "job_stage": $(json_string "${CI_JOB_STAGE:-}"),
  "job_url": $(json_string "${CI_JOB_URL:-}"),
  "job_status": $(json_string "${CI_JOB_STATUS:-}"),
  "job_started_at": $(json_string "${CI_JOB_STARTED_AT:-}"),
  "job_finished_at": $(json_string "$now_utc"),
  "queue_seconds": $(json_number "$queue_seconds"),
  "pipeline_to_job_start_seconds": $pipeline_to_job_start_seconds,
  "run_seconds": $run_seconds,
  "runner_id": $(json_string "${CI_RUNNER_ID:-}"),
  "runner_description": $(json_string "${CI_RUNNER_DESCRIPTION:-}"),
  "runner_tags": $(json_string "${CI_RUNNER_TAGS:-}"),
  "runner_executable_arch": $(json_string "${CI_RUNNER_EXECUTABLE_ARCH:-}"),
  "runner_node": $(json_string "$runner_node"),
  "runner_hostname": $(json_string "$runner_hostname"),
  "runner_concurrent_id": $(json_string "${CI_CONCURRENT_ID:-}"),
  "runner_concurrent_project_id": $(json_string "${CI_CONCURRENT_PROJECT_ID:-}"),
  "board_ip": $(json_string "${BOARD_IP:-}"),
  "board_user": $(json_string "${BOARD_USER:-}"),
  "remote_workspace": $(json_string "${REMOTE_WORKSPACE:-}"),
  "embedded_target": $(json_string "${EMBEDDED_TARGET:-}"),
  "image": $(json_string "${IMAGE:-${CI_JOB_IMAGE:-}}"),
  "experimental_docker_image": $(json_string "${EXPERIMENTAL_DOCKER_IMAGE:-}"),
  "trt_package_dir": $(json_string "${TRT_PACKAGE_DIR:-}"),
  "cuda_version": $(json_string "${CUDA_VERSION:-}")
}
JSON

mv -f "$tmp_file" "$out_file" 2>/dev/null || {
    echo "WARN: cannot write runner usage file $out_file"
    rm -f "$tmp_file" 2>/dev/null
    exit 0
}

echo "Runner usage written to $out_file"
exit 0
