#!/usr/bin/env python3
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
"""Delete or bound the current latest-engine cache namespace."""

import argparse
import json
import os
import pathlib
import re
import shutil
import time


def _namespace():
    project = os.environ.get("CI_PROJECT_ID", "local")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", project):
        raise ValueError("CI_PROJECT_ID is not a safe cache component")
    prefix = "project-{}".format(project)
    merge_request = os.environ.get("CI_MERGE_REQUEST_IID", "")
    if merge_request:
        if not merge_request.isdecimal():
            raise ValueError("CI_MERGE_REQUEST_IID must be decimal")
        return "{}-mr-{}".format(prefix, merge_request), False
    branch = os.environ.get("CI_COMMIT_BRANCH", "")
    protected = os.environ.get("CI_COMMIT_REF_PROTECTED", "").lower() == "true"
    if protected and branch == "main":
        return "{}-main".format(prefix), True
    if protected and branch.startswith("release/"):
        release = branch[len("release/"):]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", release):
            raise ValueError("release branch is not a safe cache component")
        return "{}-release-{}".format(prefix, release), True
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-",
                  os.environ.get("CI_COMMIT_REF_SLUG",
                                 "local")).strip(".-")[:64]
    return "{}-ref-{}".format(prefix, slug or "local"), False


def _bytes(root):
    total = 0
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("symlinked cache entry is not allowed")
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("delete", "prune"))
    args = parser.parse_args()
    root_value = os.environ.get("EDGE_LLM_CI_CACHE_ROOT", "")
    root = pathlib.Path(root_value)
    if not root_value or not root.is_absolute() or root == pathlib.Path("/"):
        raise ValueError("EDGE_LLM_CI_CACHE_ROOT is not a safe absolute path")
    namespace, trusted = _namespace()
    namespace_root = root / "engines" / namespace
    if args.operation == "delete":
        if os.environ.get("CI_MERGE_REQUEST_IID") and namespace_root.is_dir():
            shutil.rmtree(str(namespace_root))
            decision = "deleted"
        else:
            decision = "kept"
        print(json.dumps({"decision": decision, "namespace": namespace}))
        return 0
    if not namespace_root.is_dir():
        print(json.dumps({"decision": "empty", "namespace": namespace}))
        return 0
    name = ("EDGE_LLM_ENGINE_TRUSTED_CACHE_BYTES"
            if trusted else "EDGE_LLM_ENGINE_MR_CACHE_BYTES")
    default = 200 * 1024**3 if trusted else 20 * 1024**3
    value = os.environ.get(name, str(default))
    if not value.isdecimal():
        raise ValueError("{} must be a nonnegative integer".format(name))
    removed = []
    entries = []
    now = time.time()
    for entry in namespace_root.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        if entry.name.startswith("."):
            if now - entry.stat().st_mtime > 24 * 60 * 60:
                shutil.rmtree(str(entry))
                removed.append(entry.name)
            continue
        entries.append((entry.stat().st_mtime, _bytes(entry), entry))
    cutoff = now - 30 * 24 * 60 * 60
    for item in list(entries):
        if not trusted and item[0] < cutoff:
            shutil.rmtree(str(item[2]))
            removed.append(item[2].name)
            entries.remove(item)
    remaining = sum(item[1] for item in entries)
    for _, size, entry in sorted(entries):
        if remaining <= int(value):
            break
        shutil.rmtree(str(entry))
        removed.append(entry.name)
        remaining -= size
    print(
        json.dumps(
            {
                "decision": "pruned",
                "namespace": namespace,
                "remaining_bytes": remaining,
                "removed": removed,
            },
            sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
