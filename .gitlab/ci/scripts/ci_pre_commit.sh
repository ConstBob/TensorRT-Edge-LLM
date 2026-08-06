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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_command="$script_dir/ci_run.sh"
bootstrap_attempts="${CI_PRE_COMMIT_BOOTSTRAP_ATTEMPTS:-3}"
bootstrap_timeout_seconds="${CI_PRE_COMMIT_BOOTSTRAP_TIMEOUT_SECONDS:-300}"
check_timeout_seconds="${CI_PRE_COMMIT_CHECK_TIMEOUT_SECONDS:-1200}"
retry_delay_seconds="${CI_PRE_COMMIT_RETRY_DELAY_SECONDS:-15}"

if ! [[ "$bootstrap_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CI_PRE_COMMIT_BOOTSTRAP_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$bootstrap_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CI_PRE_COMMIT_BOOTSTRAP_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$check_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CI_PRE_COMMIT_CHECK_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$retry_delay_seconds" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CI_PRE_COMMIT_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
fi

bootstrap_hooks() {
    local attempt
    local delay_seconds
    local status

    for ((attempt = 1; attempt <= bootstrap_attempts; ++attempt)); do
        if "$run_command" "$bootstrap_timeout_seconds" \
            "Prepare pre-commit hooks (attempt $attempt/$bootstrap_attempts)" -- \
            pre-commit install-hooks; then
            return 0
        else
            status=$?
        fi

        if [ "$attempt" -eq "$bootstrap_attempts" ]; then
            return "$status"
        fi

        delay_seconds="$((retry_delay_seconds * attempt))"
        echo "Retrying pre-commit hook preparation in ${delay_seconds}s" >&2
        sleep "$delay_seconds"
    done
}

bootstrap_hooks || exit $?
"$run_command" "$check_timeout_seconds" "Run pre-commit checks" -- \
    pre-commit run --all-files --show-diff-on-failure
