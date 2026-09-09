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
    echo "Usage: $0 ensure | init <environment-file> | report | report-from-env <environment-file> | prune | prune-ci | delete-mr" >&2
    exit 2
}

warn() {
    echo "WARN: $*" >&2
}

cache_event() {
    operation="$1"
    decision="$2"
    reason="$3"
    files="$4"
    namespace="$5"
    field="$6"
    script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)" \
        || return 0
    event_script="$script_dir/ci_cache_event.py"
    if ! command -v python3 >/dev/null 2>&1 || [ ! -f "$event_script" ]; then
        return 0
    fi

    lane="$(printf '%s' "${CI_CCACHE_SHARD:-native}" \
        | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._/-' '-')"
    namespace="$(printf '%s' "$namespace" \
        | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._:/-' '-')"
    python3 "$event_script" \
        --artifact ccache \
        --operation "$operation" \
        --decision "$decision" \
        --reason "$reason" \
        --elapsed-seconds 0 \
        --namespace "$namespace" \
        --lane "$lane" \
        --field "$field" \
        --files "$files" || true
}

stat_sum() {
    stats="$1"
    names="$2"
    printf '%s\n' "$stats" | awk -v names="$names" '
        BEGIN {
            count = split(names, wanted, " ")
            for (item = 1; item <= count; ++item) {
                selected[wanted[item]] = 1
            }
        }
        $1 in selected { total += $2 }
        END { print total + 0 }
    '
}

record_stats() {
    stats="$1"
    namespace="${CCACHE_NAMESPACE:-${EDGELLM_CCACHE_EPOCH:-1}:${CI_CCACHE_SHARD:-native}}"
    hits="$(stat_sum "$stats" "direct_cache_hit preprocessed_cache_hit")"
    misses="$(stat_sum "$stats" "cache_miss")"
    uncacheable="$(stat_sum "$stats" \
        "autoconf_test bad_compiler_arguments called_for_link called_for_preprocessing compile_failed compiler_produced_no_output compiler_produced_empty_output compiler_produced_stdout could_not_use_modules could_not_use_precompiled_header disabled multiple_source_files no_input_file output_to_stdout preprocessor_error recache unsupported_code_directive unsupported_compiler_option unsupported_environment_variable unsupported_source_encoding unsupported_source_language")"

    cache_event lookup hit hit "$hits" "$namespace" compiler-call
    cache_event lookup miss not-found "$misses" "$namespace" compiler-call
    if [ "$uncacheable" -gt 0 ]; then
        cache_event lookup skipped uncacheable "$uncacheable" "$namespace" compiler-call
    fi

    local_hits="$(stat_sum "$stats" "local_storage_hit")"
    local_misses="$(stat_sum "$stats" "local_storage_miss")"
    local_writes="$(stat_sum "$stats" "local_storage_write")"
    remote_hits="$(stat_sum "$stats" "remote_storage_hit")"
    remote_misses="$(stat_sum "$stats" "remote_storage_miss")"
    remote_writes="$(stat_sum "$stats" "remote_storage_write")"
    remote_errors="$(stat_sum "$stats" \
        "remote_storage_error remote_storage_timeout")"

    if [ "$local_hits" -gt 0 ]; then
        cache_event lookup hit hit "$local_hits" job-local cache-result
    fi
    if [ "$local_misses" -gt 0 ]; then
        cache_event lookup miss not-found "$local_misses" job-local cache-result
    fi
    if [ "$local_writes" -gt 0 ]; then
        cache_event publish published published "$local_writes" job-local storage-entry
    fi
    if [ "$remote_hits" -gt 0 ]; then
        cache_event lookup hit hit "$remote_hits" shared cache-result
    fi
    if [ "$remote_misses" -gt 0 ]; then
        cache_event lookup miss not-found "$remote_misses" shared cache-result
    fi
    if [ "$remote_writes" -gt 0 ]; then
        cache_event publish published published "$remote_writes" shared storage-entry
    fi
    if [ "$remote_errors" -gt 0 ]; then
        cache_event lookup error backend-error "$remote_errors" shared backend-operation
    fi
}

write_export() {
    name="$1"
    value="$2"
    escaped_value="$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
    printf "export %s='%s'\n" "$name" "$escaped_value"
}

validate_controls() {
    case "$EDGELLM_COMPILER_CACHE" in
        AUTO|CCACHE|OFF) ;;
        *)
            echo "ERROR: EDGELLM_COMPILER_CACHE must be AUTO, CCACHE, or OFF" >&2
            return 2
            ;;
    esac
    case "$EDGELLM_CCACHE_MODE" in
        normal|bypass|recache) ;;
        *)
            echo "ERROR: EDGELLM_CCACHE_MODE must be normal, bypass, or recache" >&2
            return 2
            ;;
    esac
}

cache_root() {
    configured="${EDGE_LLM_CI_CACHE_ROOT:-${EDGE_LLM_CACHE_DIR:-/scratch.edge_llm_cache}/gitlab-ci-cache}"
    case "$configured" in
        /*) ;;
        *) return 1 ;;
    esac
    [ "$configured" != "/" ] || return 1
    if [ -L "$configured" ]; then
        return 1
    fi
    printf '%s\n' "$configured"
}

safe_component() {
    value="$1"
    label="$2"
    if [ -z "$value" ] || ! printf '%s' "$value" | grep -Eq '^[A-Za-z0-9._-]+$'; then
        echo "ERROR: unsafe $label cache component: $value" >&2
        return 2
    fi
    printf '%s\n' "$value"
}

ensure_available() {
    [ "$#" -eq 0 ] || usage
    validate_controls || return $?
    if [ "$EDGELLM_COMPILER_CACHE" = "OFF" ] \
        || [ "$EDGELLM_CCACHE_MODE" = "bypass" ] \
        || command -v ccache >/dev/null 2>&1; then
        return 0
    fi

    script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)" || return 1
    apt_script="${CI_CCACHE_APT_SCRIPT:-$script_dir/ci_apt.sh}"
    if command -v apt-get >/dev/null 2>&1 && [ -x "$apt_script" ]; then
        if bash "$apt_script" install ccache \
            && command -v ccache >/dev/null 2>&1; then
            echo "Installed ccache for this CI job"
            return 0
        fi
    elif command -v apk >/dev/null 2>&1; then
        if apk add --no-cache ccache \
            && command -v ccache >/dev/null 2>&1; then
            echo "Installed ccache for this CI job"
            return 0
        fi
    fi

    if [ "$EDGELLM_COMPILER_CACHE" = "CCACHE" ]; then
        echo "ERROR: EDGELLM_COMPILER_CACHE=CCACHE but ccache could not be installed" >&2
        return 1
    fi
    warn "ccache could not be installed; AUTO will continue without it"
    return 0
}

resolve_shared_policy() {
    CI_CCACHE_WRITE_DIR=""
    CI_CCACHE_TRUSTED_DIR=""
    root="$(cache_root)" || return 0
    project="$(safe_component "${CI_PROJECT_ID:-local}" project)" || return $?
    shard="$(safe_component "${CI_CCACHE_SHARD:-${CI_RUNNER_EXECUTABLE_ARCH:-native}}" shard)" || return $?
    epoch="$(safe_component "$EDGELLM_CCACHE_EPOCH" epoch)" || return $?

    if [ "${CI_PIPELINE_SOURCE:-}" = "merge_request_event" ] \
        && [ -n "${CI_MERGE_REQUEST_IID:-}" ]; then
        iid="$(safe_component "$CI_MERGE_REQUEST_IID" merge-request)" || return $?
        CI_CCACHE_WRITE_DIR="$root/mr/$project/$iid/ccache/v$epoch"
        case "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-main}" in
            main)
                CI_CCACHE_TRUSTED_DIR="$root/trusted/main/ccache/v$epoch/$shard"
                ;;
            release/*)
                release="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME#release/}"
                release="$(safe_component "$release" release)" || return $?
                CI_CCACHE_TRUSTED_DIR="$root/trusted/release/$release/ccache/v$epoch/$shard"
                ;;
        esac
    elif [ "${CI_COMMIT_REF_PROTECTED:-false}" = "true" ]; then
        case "${CI_COMMIT_BRANCH:-}" in
            main)
                CI_CCACHE_WRITE_DIR="$root/trusted/main/ccache/v$epoch/$shard"
                ;;
            release/*)
                release="${CI_COMMIT_BRANCH#release/}"
                release="$(safe_component "$release" release)" || return $?
                CI_CCACHE_WRITE_DIR="$root/trusted/release/$release/ccache/v$epoch/$shard"
                ;;
        esac
    fi
}

initialize() {
    [ "$#" -eq 1 ] || usage
    environment_file="$1"
    project_dir="${CI_PROJECT_DIR:-$PWD}"
    ccache_dir="${CCACHE_DIR:-$project_dir/.cache/ccache}"
    max_size="${CI_CCACHE_MAX_SIZE:-2G}"
    compression_level="${CI_CCACHE_COMPRESSION_LEVEL:-6}"

    validate_controls || return $?
    if ! printf '%s' "$max_size" \
        | grep -Eq '^[1-9][0-9]*([.][0-9]+)?[kKmMgGtT]?$'; then
        echo "ERROR: CI_CCACHE_MAX_SIZE has an unsupported value: $max_size" >&2
        return 2
    fi
    case "$compression_level" in
        [1-9]) ;;
        *)
            echo "ERROR: CI_CCACHE_COMPRESSION_LEVEL must be between 1 and 9" >&2
            return 2
            ;;
    esac

    project_dir="$(cd "$project_dir" 2>/dev/null && pwd -P)" || {
        echo "ERROR: CI_PROJECT_DIR is not accessible" >&2
        return 2
    }
    case "$ccache_dir" in
        /*) ;;
        *) ccache_dir="$project_dir/$ccache_dir" ;;
    esac
    case "$ccache_dir/" in
        "$project_dir"/*) ;;
        *)
            echo "ERROR: CCACHE_DIR must be job-local under CI_PROJECT_DIR" >&2
            return 2
            ;;
    esac
    mkdir -p "$ccache_dir" || return 1
    ccache_dir="$(cd "$ccache_dir" && pwd -P)" || return 1
    case "$ccache_dir/" in
        "$project_dir"/*) ;;
        *)
            echo "ERROR: resolved CCACHE_DIR escapes CI_PROJECT_DIR" >&2
            return 2
            ;;
    esac

    effective_cache="$EDGELLM_COMPILER_CACHE"
    if [ "$EDGELLM_CCACHE_MODE" = "bypass" ]; then
        effective_cache="OFF"
    elif [ "$effective_cache" != "OFF" ] && ! command -v ccache >/dev/null 2>&1; then
        if [ "$effective_cache" = "CCACHE" ]; then
            echo "ERROR: EDGELLM_COMPILER_CACHE=CCACHE but ccache is unavailable" >&2
            return 1
        fi
        warn "ccache is unavailable; continuing with compiler cache disabled"
        effective_cache="OFF"
    fi

    secondary_storage=""
    CI_CCACHE_WRITE_DIR=""
    CI_CCACHE_TRUSTED_DIR=""
    if [ "$effective_cache" != "OFF" ] \
        && [ "${CI_CCACHE_SHARED_ENABLED:-1}" = "1" ] \
        && ccache --help 2>&1 | grep -q -- '--trim-dir'; then
        resolve_shared_policy || return $?
        if [ -n "$CI_CCACHE_WRITE_DIR" ]; then
            mkdir -p "$CI_CCACHE_WRITE_DIR" || {
                warn "shared ccache overlay is unavailable; using the job-local cache"
                CI_CCACHE_WRITE_DIR=""
            }
        fi
        if [ -n "$CI_CCACHE_WRITE_DIR" ]; then
            secondary_storage="file:$CI_CCACHE_WRITE_DIR|umask=002|update-mtime=true"
        fi
        if [ -n "$CI_CCACHE_TRUSTED_DIR" ] && [ -d "$CI_CCACHE_TRUSTED_DIR" ]; then
            trusted="file:$CI_CCACHE_TRUSTED_DIR|read-only=true|update-mtime=true"
            secondary_storage="${secondary_storage:+$secondary_storage }$trusted"
        fi
    elif [ "$effective_cache" != "OFF" ] \
        && [ "${CI_CCACHE_SHARED_ENABLED:-1}" = "1" ]; then
        warn "ccache lacks shared file-backend support; using the job-local cache"
    fi

    namespace="$EDGELLM_CCACHE_EPOCH:${CI_CCACHE_SHARD:-${CI_RUNNER_EXECUTABLE_ARCH:-native}}"
    environment_dir="$(dirname "$environment_file")"
    mkdir -p "$environment_dir" || return 1
    temporary_file="${environment_file}.tmp.$$"
    {
        write_export EDGELLM_COMPILER_CACHE "$effective_cache"
        write_export EDGELLM_CCACHE_MODE "$EDGELLM_CCACHE_MODE"
        write_export EDGELLM_CCACHE_EPOCH "$EDGELLM_CCACHE_EPOCH"
        write_export CCACHE_DIR "$ccache_dir"
        write_export CCACHE_BASEDIR "$project_dir"
        write_export CCACHE_COMPILERCHECK content
        write_export CCACHE_COMPRESS true
        write_export CCACHE_COMPRESSLEVEL "$compression_level"
        write_export CCACHE_MAXSIZE "$max_size"
        write_export CCACHE_NAMESPACE "$namespace"
        write_export CCACHE_STATS true
        if [ "$effective_cache" != "OFF" ]; then
            write_export CMAKE_C_COMPILER_LAUNCHER ccache
            write_export CMAKE_CXX_COMPILER_LAUNCHER ccache
            write_export CMAKE_CUDA_COMPILER_LAUNCHER ccache
        fi
        if [ "$EDGELLM_CCACHE_MODE" = "recache" ]; then
            write_export CCACHE_RECACHE true
        fi
        if [ -n "$secondary_storage" ]; then
            write_export CI_CCACHE_SHARED_DIR "$CI_CCACHE_WRITE_DIR"
            write_export CI_CCACHE_WRITE_DIR "$CI_CCACHE_WRITE_DIR"
            write_export CI_CCACHE_TRUSTED_DIR "$CI_CCACHE_TRUSTED_DIR"
            write_export CCACHE_SECONDARY_STORAGE "$secondary_storage"
        fi
    } > "$temporary_file" || {
        rm -f "$temporary_file"
        return 1
    }
    mv -f "$temporary_file" "$environment_file" || {
        rm -f "$temporary_file"
        return 1
    }

    if [ "$effective_cache" != "OFF" ]; then
        CCACHE_DIR="$ccache_dir" CCACHE_BASEDIR="$project_dir" \
            CCACHE_COMPILERCHECK=content CCACHE_COMPRESS=true \
            CCACHE_COMPRESSLEVEL="$compression_level" \
            CCACHE_MAXSIZE="$max_size" CCACHE_NAMESPACE="$namespace" \
            CCACHE_STATS=true ccache --zero-stats >/dev/null 2>&1 \
            || warn "ccache statistics could not be reset"
    fi
    echo "ccache environment written to $environment_file"
    if [ -n "$secondary_storage" ]; then
        echo "ccache shared storage: $secondary_storage"
    fi
}

report() {
    if [ "${EDGELLM_COMPILER_CACHE:-AUTO}" = "OFF" ]; then
        echo "ccache disabled"
        cache_event lookup policy-disabled disabled 0 disabled compiler-call
        return 0
    fi
    if ! command -v ccache >/dev/null 2>&1; then
        warn "ccache is unavailable; no statistics to report"
        cache_event lookup policy-disabled unavailable 0 disabled compiler-call
        return 0
    fi
    ccache --show-stats || warn "ccache statistics could not be read"
    stats="$(ccache --print-stats 2>/dev/null)" \
        && record_stats "$stats" \
        || warn "ccache machine-readable statistics could not be read"
    if [ -n "${CI_CCACHE_WRITE_DIR:-}" ]; then
        echo "ccache writable backend: $CI_CCACHE_WRITE_DIR"
    fi
    if [ -n "${CI_CCACHE_TRUSTED_DIR:-}" ]; then
        echo "ccache trusted read backend: $CI_CCACHE_TRUSTED_DIR"
    fi
}

report_from_env() {
    [ "$#" -eq 1 ] || usage
    environment_file="$1"
    if [ -n "$environment_file" ] && [ -f "$environment_file" ]; then
        . "$environment_file"
        report
    fi
}

prune_dir() {
    directory="$1"
    maximum="$2"
    if [ -d "$directory" ] && [ ! -L "$directory" ]; then
        ccache --trim-dir "$directory" --trim-max-size "$maximum" \
            --trim-method mtime || warn "could not trim $directory"
    fi
}

prune() {
    [ "$#" -eq 0 ] || usage
    command -v ccache >/dev/null 2>&1 || {
        warn "ccache is unavailable; shared cache was not pruned"
        return 0
    }
    resolve_shared_policy || return $?
    maximum="${CI_CCACHE_SHARED_MAX_SIZE:-5Gi}"
    if [ -n "$CI_CCACHE_WRITE_DIR" ]; then
        case "$CI_CCACHE_WRITE_DIR" in
            "$root"/trusted/*)
                parent="$(dirname "$CI_CCACHE_WRITE_DIR")"
                for directory in "$parent"/*; do
                    prune_dir "$directory" "$maximum"
                done
                ;;
            *) prune_dir "$CI_CCACHE_WRITE_DIR" "$maximum" ;;
        esac
    fi
}

prune_ci() {
    [ "$#" -eq 0 ] || usage
    EDGELLM_COMPILER_CACHE="AUTO"
    ensure_available || return $?
    prune
}

delete_mr() {
    [ "$#" -eq 0 ] || usage
    root="$(cache_root)" || {
        echo "ERROR: unsafe cache root" >&2
        return 2
    }
    project="$(safe_component "${CI_PROJECT_ID:-}" project)" || return $?
    iid="$(safe_component "${CI_MERGE_REQUEST_IID:-}" merge-request)" || return $?
    directory="$root/mr/$project/$iid/ccache"
    case "$directory" in
        "$root"/mr/"$project"/"$iid"/ccache) ;;
        *) return 2 ;;
    esac
    rm -rf -- "$directory"
    rmdir "$root/mr/$project/$iid" "$root/mr/$project" 2>/dev/null || true
}

EDGELLM_COMPILER_CACHE="${EDGELLM_COMPILER_CACHE:-AUTO}"
EDGELLM_CCACHE_MODE="${EDGELLM_CCACHE_MODE:-normal}"
EDGELLM_CCACHE_EPOCH="${EDGELLM_CCACHE_EPOCH:-1}"

case "${1:-}" in
    ensure)
        shift
        ensure_available "$@"
        ;;
    init)
        shift
        initialize "$@"
        ;;
    report)
        shift
        [ "$#" -eq 0 ] || usage
        report
        ;;
    report-from-env)
        shift
        report_from_env "$@"
        ;;
    prune)
        shift
        prune "$@"
        ;;
    prune-ci)
        shift
        prune_ci "$@"
        ;;
    delete-mr)
        shift
        delete_mr "$@"
        ;;
    *) usage ;;
esac
