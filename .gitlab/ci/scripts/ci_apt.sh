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

set -euo pipefail

usage() {
    echo "Usage: $0 install <package> [package...]" >&2
    exit 2
}

if [ "${1:-}" != "install" ] || [ "$#" -lt 2 ]; then
    usage
fi
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_command="$script_dir/ci_run.sh"
update_timeout_seconds="${CI_APT_UPDATE_TIMEOUT_SECONDS:-300}"
install_timeout_seconds="${CI_APT_INSTALL_TIMEOUT_SECONDS:-600}"
network_timeout_seconds="${CI_APT_NETWORK_TIMEOUT_SECONDS:-30}"
lock_timeout_seconds="${CI_APT_LOCK_TIMEOUT_SECONDS:-60}"
retries="${CI_APT_RETRIES:-2}"
update_attempts="${CI_APT_UPDATE_ATTEMPTS:-2}"
update_retry_delay_seconds="${CI_APT_UPDATE_RETRY_DELAY_SECONDS:-10}"

if ! [[ "$update_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CI_APT_UPDATE_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$update_retry_delay_seconds" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CI_APT_UPDATE_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
fi

apt_options=(
    -q
    -o "Acquire::Retries=$retries"
    -o "Acquire::http::Timeout=$network_timeout_seconds"
    -o "Acquire::https::Timeout=$network_timeout_seconds"
    -o "DPkg::Lock::Timeout=$lock_timeout_seconds"
    -o Dpkg::Use-Pty=0
)

update_package_index() {
    local attempt
    local status

    for ((attempt = 1; attempt <= update_attempts; ++attempt)); do
        if "$run_command" "$update_timeout_seconds" \
            "APT package index update (attempt $attempt/$update_attempts)" -- \
            env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" update; then
            return 0
        else
            status=$?
        fi

        if [ "$attempt" -eq "$update_attempts" ]; then
            return "$status"
        fi

        echo "Retrying APT package index update in ${update_retry_delay_seconds}s" >&2
        sleep "$update_retry_delay_seconds"
    done
}

echo "APT packages: $*"
update_package_index
"$run_command" "$install_timeout_seconds" "APT package installation" -- \
    env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y "$@"
