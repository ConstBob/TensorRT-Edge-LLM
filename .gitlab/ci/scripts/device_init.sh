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

: "${REMOTE_WORKSPACE:?REMOTE_WORKSPACE must be set}"
: "${CI_APT_SCRIPT:?CI_APT_SCRIPT must be set}"

case "$REMOTE_WORKSPACE" in
  "$HOME"/tensorrt-edge-llm*) ;;
  *)
    echo "Unsafe remote workspace: $REMOTE_WORKSPACE" >&2
    exit 1
    ;;
esac

ci_script_dir="$(dirname "$CI_APT_SCRIPT")"
expected_ci_script_dir="$REMOTE_WORKSPACE/.gitlab/ci/scripts"
if [ "$ci_script_dir" != "$expected_ci_script_dir" ]; then
  echo "Unsafe CI script directory: $ci_script_dir" >&2
  exit 1
fi
trap 'rm -rf -- "$ci_script_dir"' EXIT

echo "Setting up device environment on $HOME directory"
board_password={BOARDPASSWORD}

# Clean up stale tensorrt-edge-llm workspaces from previous CI runs.
# Only remove directories older than 120 minutes to preserve workspaces
# from concurrent jobs running on the same board (TRT10 + TRT11 in parallel).
echo "Cleaning stale tensorrt-edge-llm workspaces (older than 120 minutes)"
for dir in $HOME/tensorrt-edge-llm*; do
  if [ -d "$dir" ]; then
    age_check=$(find "$dir" -maxdepth 0 -mmin +120 2>/dev/null || true)
    if [ -n "$age_check" ]; then
      echo "  removing stale: $dir"
      printf '%s\n' "$board_password" | sudo -S chmod -R 777 "$dir" 2>/dev/null || true
      printf '%s\n' "$board_password" | sudo -S rm -rf "$dir"
    else
      echo "  keeping (recent): $dir"
    fi
  fi
done

echo "Installing dependencies"
printf '%s\n' "$board_password" | sudo -S bash "$CI_APT_SCRIPT" install \
  python3 python3-pip git curl nfs-common cmake rsync

# Check if scratch.edge_llm_cache folder exists
if [ -d "/scratch.edge_llm_cache" ] ; then
  ls /scratch.edge_llm_cache
  echo "/scratch.edge_llm_cache folder is mounted"
else
  echo "/scratch.edge_llm_cache folder is not mounted." && exit 1
fi

df -h /home

echo "Environment is ready!"
