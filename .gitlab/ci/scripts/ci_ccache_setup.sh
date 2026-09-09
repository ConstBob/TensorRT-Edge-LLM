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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this script so compiler-cache exports reach the CI job" >&2
    exit 2
fi

_ci_ccache_setup_main() {
    local mode="${1:-native}"
    local script_dir
    local shard

    case "$mode" in
        native)
            shard="${CI_CCACHE_SHARD:-native-x86_64}"
            ;;
        wheel)
            case "${BUILD_GROUP:-}" in
                x86-*) shard="wheel-$BUILD_GROUP" ;;
                *) return 0 ;;
            esac
            ;;
        *)
            echo "ERROR: unsupported ccache setup mode: $mode" >&2
            return 2
            ;;
    esac

    : "${CI_CCACHE_ENV_FILE:?CI_CCACHE_ENV_FILE must be set}"
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    export EDGELLM_COMPILER_CACHE="AUTO"
    export CI_CCACHE_SHARD="$shard"
    sh "$script_dir/ci_ccache.sh" ensure
    sh "$script_dir/ci_ccache.sh" init "$CI_CCACHE_ENV_FILE"
    # The generated file contains only single-quoted export statements.
    . "$CI_CCACHE_ENV_FILE"
}

_ci_ccache_setup_main "$@"
_ci_ccache_setup_status=$?
unset -f _ci_ccache_setup_main
return "$_ci_ccache_setup_status"
