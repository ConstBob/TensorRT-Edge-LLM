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

set -eu

: "${CI_REGISTRY:?CI_REGISTRY must be set}"
: "${CI_REGISTRY_USER:?CI_REGISTRY_USER must be set}"
: "${CI_REGISTRY_PASSWORD:?CI_REGISTRY_PASSWORD must be set}"
: "${CI_REGISTRY_IMAGE:?CI_REGISTRY_IMAGE must be set}"
: "${CI_DEPENDENCY_CACHE_FAMILY:?CI_DEPENDENCY_CACHE_FAMILY must be set}"
: "${CI_DEPENDENCY_BASE_IMAGE:?CI_DEPENDENCY_BASE_IMAGE must be set}"
: "${CI_DEPENDENCY_PROFILE:?CI_DEPENDENCY_PROFILE must be set}"
: "${CI_DEPENDENCY_OUTPUT_IMAGE:?CI_DEPENDENCY_OUTPUT_IMAGE must be set}"
: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set}"

if [ "$CI_REGISTRY" != "gitlab-master.nvidia.com:5005" ]; then
    echo "ERROR: unexpected container registry: $CI_REGISTRY" >&2
    exit 2
fi

registry_auth="$(printf '%s:%s' "$CI_REGISTRY_USER" "$CI_REGISTRY_PASSWORD" \
    | base64 | tr -d '\n')"
mkdir -p "$HOME/.docker"
printf '{"auths":{"%s":{"auth":"%s"}}}' \
    "$CI_REGISTRY" "$registry_auth" > "$HOME/.docker/config.json"

if [ "${CI_PIPELINE_SOURCE:-}" = "merge_request_event" ]; then
    : "${CI_MERGE_REQUEST_IID:?CI_MERGE_REQUEST_IID must be set for an MR pipeline}"
    write_scope="mr-$CI_MERGE_REQUEST_IID"
    case "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-main}" in
        release/*)
            release_series="$(printf "%s" "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME#release/}" | tr "." "-")"
            trusted_scope="release-$release_series"
            ;;
        *) trusted_scope="main" ;;
    esac
elif [ "${CI_COMMIT_REF_PROTECTED:-false}" = "true" ]; then
    case "${CI_COMMIT_BRANCH:-}" in
        main)
            write_scope="main"
            ;;
        release/*)
            release_series="$(printf "%s" "${CI_COMMIT_BRANCH#release/}" | tr "." "-")"
            write_scope="release-$release_series"
            ;;
        *)
            echo "ERROR: refusing to publish a shared dependency cache from unsupported protected branch: ${CI_COMMIT_BRANCH:-<unset>}" >&2
            exit 2
            ;;
    esac
    trusted_scope="$write_scope"
else
    echo "ERROR: refusing to publish a shared dependency cache from an unprotected branch" >&2
    exit 2
fi

trusted_cache="$CI_REGISTRY_IMAGE/ci-dependency/buildkit:$CI_DEPENDENCY_CACHE_FAMILY-$trusted_scope"
write_cache="$CI_REGISTRY_IMAGE/ci-dependency/buildkit:$CI_DEPENDENCY_CACHE_FAMILY-$write_scope"
telemetry_script="$CI_PROJECT_DIR/.gitlab/ci/scripts/ci_run.sh"

exec sh "$telemetry_script" "${CI_JOB_PHASE_TIMEOUT_SECONDS:-0}" \
    "Build dependency image $CI_DEPENDENCY_CACHE_FAMILY" -- \
    buildctl-daemonless.sh build \
    --frontend dockerfile.v0 \
    --local "context=$CI_PROJECT_DIR" \
    --local "dockerfile=$CI_PROJECT_DIR/.gitlab/ci/images" \
    --opt "filename=Dockerfile.dependencies" \
    --opt "target=$CI_DEPENDENCY_PROFILE" \
    --opt "build-arg:BASE_IMAGE=$CI_DEPENDENCY_BASE_IMAGE" \
    --import-cache "type=registry,ref=$trusted_cache" \
    --import-cache "type=registry,ref=$write_cache" \
    --export-cache "type=registry,ref=$write_cache,mode=max" \
    --output "type=image,name=$CI_DEPENDENCY_OUTPUT_IMAGE,push=true"
