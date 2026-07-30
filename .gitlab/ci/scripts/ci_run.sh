#!/usr/bin/env bash
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

set -uo pipefail

usage() {
    echo "Usage: $0 <timeout-seconds> <label> [--] <command> [args...]" >&2
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

if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || [ "$#" -eq 0 ]; then
    usage
fi

kill_after_seconds="${CI_COMMAND_KILL_AFTER_SECONDS:-30}"
if ! [[ "$kill_after_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CI_COMMAND_KILL_AFTER_SECONDS must be a positive integer" >&2
    exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: timeout is required to run '$label'" >&2
    exit 127
fi

echo "Starting $label (timeout: ${timeout_seconds}s)"
start_seconds="$(date +%s)"
timeout --signal=TERM --kill-after="${kill_after_seconds}s" "${timeout_seconds}s" "$@"
status=$?
elapsed_seconds="$(( $(date +%s) - start_seconds ))"

if [ "$status" -eq 0 ]; then
    echo "Completed $label in ${elapsed_seconds}s"
elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "ERROR: $label timed out after ${timeout_seconds}s" >&2
else
    echo "ERROR: $label failed after ${elapsed_seconds}s with status $status" >&2
fi

exit "$status"
