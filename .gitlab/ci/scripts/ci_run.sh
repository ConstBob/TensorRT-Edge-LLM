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

set -u

usage() {
    echo "Usage: $0 <timeout-seconds; 0 disables> <label> [--] <command> [args...]" >&2
    exit 2
}

if [ "$#" -lt 3 ]; then
    usage
fi

timeout_seconds="$1"
label="$2"
shift 2

if [ "${1:-}" = "--" ]; then
    shift
fi

case "$timeout_seconds" in
    0|[1-9]|[1-9][0-9]*) ;;
    *) usage ;;
esac
if [ "$#" -eq 0 ]; then
    usage
fi

if [ "$timeout_seconds" -ne 0 ]; then
    kill_after_seconds="${CI_COMMAND_KILL_AFTER_SECONDS:-30}"
    case "$kill_after_seconds" in
        [1-9]|[1-9][0-9]*) ;;
        *)
            echo "ERROR: CI_COMMAND_KILL_AFTER_SECONDS must be a positive integer" >&2
            exit 2
            ;;
    esac

    if ! command -v timeout >/dev/null 2>&1; then
        echo "ERROR: timeout is required to run '$label'" >&2
        exit 127
    fi
fi

if [ "$timeout_seconds" -eq 0 ]; then
    echo "Starting $label (no timeout)"
else
    echo "Starting $label (timeout: ${timeout_seconds}s)"
fi
start_seconds="$(date +%s)"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$timeout_seconds" -eq 0 ]; then
    "$@"
else
    timeout --signal=TERM --kill-after="${kill_after_seconds}s" "${timeout_seconds}s" "$@"
fi
status=$?
finish_seconds="$(date +%s)"
finish_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds="$((finish_seconds - start_seconds))"

if [ "$status" -eq 0 ]; then
    phase_status="success"
    echo "Completed $label in ${elapsed_seconds}s"
elif [ "$timeout_seconds" -ne 0 ] \
    && { [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; }; then
    phase_status="timeout"
    echo "ERROR: $label timed out after ${timeout_seconds}s" >&2
else
    phase_status="failure"
    echo "ERROR: $label failed after ${elapsed_seconds}s with status $status" >&2
fi

write_phase_record() {
    if [ "${CI_TELEMETRY_ENABLED:-1}" = "0" ] || [ -z "${CI_JOB_ID:-}" ]; then
        return
    fi

    telemetry_dir="${CI_TELEMETRY_DIR:-${CI_PROJECT_DIR:-$PWD}/.ci-telemetry}/phases/${CI_JOB_ID}"
    if ! mkdir -p "$telemetry_dir" 2>/dev/null; then
        echo "WARN: cannot create CI telemetry phase directory $telemetry_dir" >&2
        return
    fi

    escaped_label="$(
        printf '%s' "$label" \
            | LC_ALL=C tr '\t\r\n' '   ' \
            | LC_ALL=C tr -cd ' -~' \
            | sed 's/\\/\\\\/g; s/"/\\"/g'
    )"
    record_name="phase-${start_seconds}-${BASHPID:-$$}-${RANDOM:-0}.json"
    record_path="$telemetry_dir/$record_name"
    tmp_path="${record_path}.tmp.$$"
    if ! cat > "$tmp_path" <<JSON
{
  "schema_version": 1,
  "label": "$escaped_label",
  "status": "$phase_status",
  "exit_code": $status,
  "started_at": "$start_utc",
  "started_epoch": $start_seconds,
  "finished_at": "$finish_utc",
  "finished_epoch": $finish_seconds,
  "elapsed_seconds": $elapsed_seconds
}
JSON
    then
        echo "WARN: cannot write CI telemetry phase record $record_path" >&2
        rm -f "$tmp_path" 2>/dev/null || true
        return
    fi
    if ! mv -f "$tmp_path" "$record_path" 2>/dev/null; then
        echo "WARN: cannot publish CI telemetry phase record $record_path" >&2
        rm -f "$tmp_path" 2>/dev/null || true
    fi
}

write_phase_record

exit "$status"
