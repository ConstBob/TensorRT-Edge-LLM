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

if [ "${CI_TELEMETRY_ENABLED:-1}" = "0" ]; then
    exit 0
fi

publisher_started_epoch="$(date +%s 2>/dev/null || printf '0')"
usage_root="${RUNNER_USAGE_DIR:-${EDGE_LLM_CACHE_DIR:-/scratch.edge_llm_cache}/runner_usage}"
project_slug="${CI_PROJECT_PATH_SLUG:-tensorrt-edge-llm}"
pipeline_id="${CI_PIPELINE_ID:-manual}"
job_id="${CI_JOB_ID:-unknown}"
out_dir="${usage_root}/${project_slug}/${pipeline_id}"
out_file="${out_dir}/${job_id}.json"
portable_dir="${CI_TELEMETRY_ARTIFACT_DIR:-${CI_PROJECT_DIR:-$PWD}/.ci-telemetry-artifact}"
portable_file="${portable_dir}/${job_id}.json"
phase_source_dir="${CI_TELEMETRY_DIR:-${CI_PROJECT_DIR:-$PWD}/.ci-telemetry}/phases/${job_id}"
phase_out_dir="${out_dir}/${job_id}.phases"
max_phase_records=1000
max_phase_bytes=65536
max_portable_phase_bytes=1572864

raw_enabled=1
portable_enabled=1
if ! mkdir -p "$out_dir" 2>/dev/null; then
    echo "WARN: cannot create runner usage dir $out_dir"
    raw_enabled=0
fi
if ! mkdir -p "$portable_dir" 2>/dev/null; then
    echo "WARN: cannot create portable CI telemetry dir $portable_dir"
    portable_enabled=0
fi
if [ "$raw_enabled" -eq 0 ] && [ "$portable_enabled" -eq 0 ]; then
    exit 0
fi

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

runner_hostname="$(hostname 2>/dev/null || true)"
runner_node="${NODE_NAME:-$runner_hostname}"
source_commit_sha="${CI_MERGE_REQUEST_SOURCE_BRANCH_SHA:-}"
if [ -z "$source_commit_sha" ] \
    && [ "${CI_MERGE_REQUEST_EVENT_TYPE:-}" = "detached" ]; then
    source_commit_sha="${CI_COMMIT_SHA:-}"
fi

valid_phase_record() {
    phase_path="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c '
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as phase_file:
        phase = json.load(phase_file)
except (OSError, ValueError):
    sys.exit(1)
required = ("label", "status", "exit_code", "started_at", "started_epoch",
            "finished_at", "finished_epoch", "elapsed_seconds")
valid = (isinstance(phase, dict) and phase.get("schema_version") == 1
         and all(name in phase for name in required))
sys.exit(0 if valid else 1)
' "$phase_path" >/dev/null 2>&1
        return $?
    fi
    grep -Eq '"schema_version"[[:space:]]*:[[:space:]]*1' "$phase_path" \
        && grep -Eq '"elapsed_seconds"[[:space:]]*:[[:space:]]*[0-9]+' "$phase_path" \
        && tail -c 2 "$phase_path" | grep -Eq '}[[:space:]]*$'
}

phase_fragment=""
if [ "$portable_enabled" -eq 1 ]; then
    phase_fragment="${portable_dir}/.phases-${job_id}.tmp.$$"
    : > "$phase_fragment" 2>/dev/null || portable_enabled=0
fi
if [ "$raw_enabled" -eq 1 ] && [ -d "$phase_source_dir" ]; then
    mkdir -p "$phase_out_dir" 2>/dev/null || {
        echo "WARN: cannot create CI telemetry phase directory $phase_out_dir"
        raw_enabled=0
    }
fi

phase_records_seen=0
portable_phase_bytes=0
portable_phase_count=0
if [ -d "$phase_source_dir" ]; then
    for phase_file in "$phase_source_dir"/*.json; do
        [ -f "$phase_file" ] && [ ! -L "$phase_file" ] || continue
        if [ "$phase_records_seen" -ge "$max_phase_records" ]; then
            echo "WARN: reached CI telemetry phase record limit $max_phase_records"
            break
        fi
        phase_records_seen=$((phase_records_seen + 1))
        phase_size="$(wc -c < "$phase_file" 2>/dev/null | tr -d '[:space:]')"
        case "$phase_size" in
            ''|*[!0-9]*)
                echo "WARN: cannot determine CI telemetry phase record size $phase_file"
                continue
                ;;
        esac
        if [ "$phase_size" -gt "$max_phase_bytes" ]; then
            echo "WARN: ignoring oversized CI telemetry phase record $phase_file"
            continue
        fi
        if ! valid_phase_record "$phase_file"; then
            echo "WARN: ignoring malformed CI telemetry phase record $phase_file"
            continue
        fi

        if [ "$raw_enabled" -eq 1 ]; then
            phase_name="$(basename "$phase_file")"
            phase_tmp="${phase_out_dir}/.${phase_name}.tmp.$$"
            if ! cp "$phase_file" "$phase_tmp" 2>/dev/null \
                || ! mv -f "$phase_tmp" "${phase_out_dir}/${phase_name}" 2>/dev/null; then
                echo "WARN: cannot copy CI telemetry phase record $phase_file"
                rm -f "$phase_tmp" 2>/dev/null
            fi
        fi

        if [ "$portable_enabled" -eq 1 ]; then
            next_portable_bytes=$((portable_phase_bytes + phase_size))
            if [ "$next_portable_bytes" -gt "$max_portable_phase_bytes" ]; then
                echo "WARN: reached portable CI telemetry phase size limit $max_portable_phase_bytes"
                portable_enabled=0
                rm -f "$phase_fragment" 2>/dev/null
                continue
            fi
            if [ "$portable_phase_count" -gt 0 ]; then
                printf ',\n' >> "$phase_fragment"
            fi
            cat "$phase_file" >> "$phase_fragment"
            portable_phase_bytes="$next_portable_bytes"
            portable_phase_count=$((portable_phase_count + 1))
        fi
    done
fi

publisher_finished_epoch="$(date +%s 2>/dev/null || printf '%s' "$publisher_started_epoch")"
publisher_elapsed_seconds="$(duration "$publisher_started_epoch" "$publisher_finished_epoch")"

if [ "$portable_enabled" -eq 1 ]; then
    portable_tmp="${portable_file}.tmp.$$"
    {
        cat <<JSON
{
  "schema": "edgellm-ci-runner-usage/v1",
  "sanitized": true,
  "pipeline_id": $(json_string "${CI_PIPELINE_ID:-}"),
  "pipeline_iid": $(json_string "${CI_PIPELINE_IID:-}"),
  "pipeline_source": $(json_string "${CI_PIPELINE_SOURCE:-}"),
  "commit_sha": $(json_string "${CI_COMMIT_SHA:-}"),
  "merge_request_iid": $(json_string "${CI_MERGE_REQUEST_IID:-}"),
  "job_id": $(json_string "${CI_JOB_ID:-}"),
  "job_name": $(json_string "${CI_JOB_NAME:-}"),
  "job_stage": $(json_string "${CI_JOB_STAGE:-}"),
  "job_status": $(json_string "${CI_JOB_STATUS:-}"),
  "job_started_at": $(json_string "${CI_JOB_STARTED_AT:-}"),
  "job_finished_at": $(json_string "$now_utc"),
  "queue_seconds": $(json_number "$queue_seconds"),
  "pipeline_to_job_start_seconds": $pipeline_to_job_start_seconds,
  "run_seconds": $run_seconds,
  "publisher_elapsed_seconds": $publisher_elapsed_seconds,
  "phases": [
JSON
        cat "$phase_fragment"
        printf '\n  ]\n}\n'
    } > "$portable_tmp" 2>/dev/null
    if ! mv -f "$portable_tmp" "$portable_file" 2>/dev/null; then
        echo "WARN: cannot write portable CI telemetry file $portable_file"
        rm -f "$portable_tmp" 2>/dev/null
    else
        echo "Portable runner usage written to $portable_file"
    fi
    rm -f "$phase_fragment" 2>/dev/null
fi

if [ "$raw_enabled" -eq 1 ]; then
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
  "publisher_elapsed_seconds": $publisher_elapsed_seconds,
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

    if ! mv -f "$tmp_file" "$out_file" 2>/dev/null; then
        echo "WARN: cannot write runner usage file $out_file"
        rm -f "$tmp_file" 2>/dev/null
    else
        echo "Runner usage written to $out_file"
    fi
fi

exit 0
