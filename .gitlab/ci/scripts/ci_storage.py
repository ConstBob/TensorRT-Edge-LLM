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

import argparse
import os
import re
import shlex
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROTECTED_MODE = 0o755
JOB_MODE = 0o777

DECIMAL_PATTERN = re.compile(r"^[0-9]+$")
SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
PIPELINE_DIRECTORY_PATTERN = re.compile(r"^pipeline_([0-9]+)$")


class StorageContractError(RuntimeError):
    """Raised when an operation would violate the L0 storage contract."""


@dataclass(frozen=True)
class StorageLayout:
    """Validated paths for one L0 pipeline context."""

    scratch_root: Path
    storage_kind: str
    storage_id: str

    @property
    def l0_root(self) -> Path:
        return self.scratch_root / "L0"

    @property
    def storage_parent(self) -> Path:
        if self.storage_kind == "mr":
            return self.l0_root
        return self.l0_root / "STABILITY"

    @property
    def storage_root(self) -> Path:
        if self.storage_kind == "mr":
            return self.storage_parent / f"MR_{self.storage_id}"
        return self.storage_parent / self.storage_id

    @property
    def onnx_cache_root(self) -> Path:
        return self.storage_root / "onnx_cache"

    @property
    def workspace_root(self) -> Path:
        return self.storage_root / "workspace"

    def pipeline_root(self, pipeline_id: str) -> Path:
        if self.storage_kind == "stability":
            return self.workspace_root
        return self.workspace_root / f"pipeline_{pipeline_id}"

    def job_root(self, pipeline_id: str, job_id: str, job_slug: str) -> Path:
        return self.pipeline_root(pipeline_id) / f"job_{job_id}_{job_slug}"


def _required_value(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "")
    if not value:
        raise StorageContractError(f"{name} must be set")
    return value


def _required_decimal(environ: Mapping[str, str], name: str) -> str:
    value = _required_value(environ, name)
    if DECIMAL_PATTERN.fullmatch(value) is None:
        raise StorageContractError(
            f"{name} must be a non-empty decimal integer, got {value!r}")
    return value


def _required_slug(environ: Mapping[str, str]) -> str:
    value = _required_value(environ, "CI_JOB_NAME_SLUG")
    if SLUG_PATTERN.fullmatch(value) is None:
        raise StorageContractError(
            "CI_JOB_NAME_SLUG must contain only lowercase letters, digits, "
            f"and internal hyphens, got {value!r}")
    return value


def _stability_storage_id(environ: Mapping[str, str], pipeline_id: str) -> str:
    raw_created_at = _required_value(environ, "CI_PIPELINE_CREATED_AT")
    normalized_created_at = raw_created_at
    if raw_created_at.endswith("Z"):
        normalized_created_at = f"{raw_created_at[:-1]}+00:00"
    try:
        created_at = datetime.fromisoformat(normalized_created_at)
    except ValueError as error:
        raise StorageContractError(
            "CI_PIPELINE_CREATED_AT must be an ISO 8601 timestamp with a "
            f"timezone, got {raw_created_at!r}") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise StorageContractError(
            "CI_PIPELINE_CREATED_AT must include a timezone, "
            f"got {raw_created_at!r}")

    utc_created_at = created_at.astimezone(timezone.utc)
    timestamp = utc_created_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{timestamp}_pipeline_{pipeline_id}"


def _storage_layout(environ: Mapping[str, str]) -> StorageLayout:
    raw_root = _required_value(environ, "EDGE_LLM_CACHE_DIR")
    root = Path(raw_root)
    if not root.is_absolute():
        raise StorageContractError(
            f"EDGE_LLM_CACHE_DIR must be absolute, got {raw_root!r}")
    if ".." in root.parts or root == Path("/"):
        raise StorageContractError(
            f"unsafe EDGE_LLM_CACHE_DIR value: {raw_root!r}")

    storage_kind = _required_value(environ, "L0_STORAGE_KIND")
    if storage_kind == "mr":
        storage_id = _required_decimal(environ, "CI_MERGE_REQUEST_IID")
    elif storage_kind == "stability":
        pipeline_id = _required_decimal(environ, "CI_PIPELINE_ID")
        storage_id = _stability_storage_id(environ, pipeline_id)
    else:
        raise StorageContractError(
            "L0_STORAGE_KIND must be either 'mr' or 'stability', "
            f"got {storage_kind!r}")

    return StorageLayout(scratch_root=root,
                         storage_kind=storage_kind,
                         storage_id=storage_id)


def _optional_relative_path(environ: Mapping[str, str],
                            name: str) -> Optional[Path]:
    value = environ.get(name, "")
    if not value:
        return None

    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in (".", "..")
                                                   for part in path.parts):
        raise StorageContractError(
            f"{name} must be a relative path without traversal, got {value!r}")
    return path


def _protected_paths(layout: StorageLayout,
                     pipeline_id: str) -> List[Tuple[Path, str]]:
    paths = [
        (layout.storage_root, "storage root"),
        (layout.onnx_cache_root, "ONNX cache root"),
        (layout.workspace_root, "workspace root"),
    ]
    pipeline_root = layout.pipeline_root(pipeline_id)
    if pipeline_root != layout.workspace_root:
        paths.append((pipeline_root, "pipeline workspace root"))
    return paths


def _assert_directory(path: Path, purpose: str) -> os.stat_result:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise StorageContractError(
            f"{purpose} does not exist: {path}") from error

    if stat.S_ISLNK(path_stat.st_mode):
        raise StorageContractError(f"{purpose} must not be a symlink: {path}")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise StorageContractError(f"{purpose} is not a directory: {path}")
    return path_stat


def _mode_details(role: str, path: Path, expected_mode: int,
                  path_stat: os.stat_result) -> str:
    return (f"role={role} path={path} expected_mode={expected_mode:04o} "
            f"actual_uid={path_stat.st_uid} actual_gid={path_stat.st_gid} "
            f"actual_mode={stat.S_IMODE(path_stat.st_mode):04o}")


def _ensure_directory(path: Path, mode: int, purpose: str,
                      role: str) -> os.stat_result:
    try:
        path.mkdir()
    except FileExistsError:
        pass

    _assert_directory(path, purpose)
    path.chmod(mode)
    path_stat = _assert_directory(path, purpose)
    actual_mode = stat.S_IMODE(path_stat.st_mode)
    if actual_mode != mode:
        raise StorageContractError(
            f"{purpose} has an invalid mode: "
            f"{_mode_details(role, path, mode, path_stat)}")
    return path_stat


def _prepare_protected_paths(layout: StorageLayout, pipeline_id: str) -> int:
    _assert_directory(layout.scratch_root, "scratch root")
    _assert_directory(layout.l0_root, "L0 root")
    if not os.access(layout.l0_root, os.W_OK | os.X_OK):
        raise StorageContractError(
            f"lifecycle identity cannot write and traverse {layout.l0_root}")

    protected_paths = _protected_paths(layout, pipeline_id)
    if layout.storage_parent != layout.l0_root:
        protected_paths.insert(
            0, (layout.storage_parent, "storage context parent"))
    owner_uid: Optional[int] = None
    for path, purpose in protected_paths:
        path_stat = _ensure_directory(path, PROTECTED_MODE, purpose,
                                      "lifecycle")
        if owner_uid is None:
            owner_uid = path_stat.st_uid
        elif path_stat.st_uid != owner_uid:
            raise StorageContractError(
                f"{purpose} owner UID {path_stat.st_uid} does not match "
                f"lifecycle UID {owner_uid}: {path}")

    if owner_uid is None:
        raise StorageContractError("no protected storage paths were prepared")
    return owner_uid


def _prepare_storage(environ: Mapping[str, str]) -> None:
    layout = _storage_layout(environ)
    pipeline_id = _required_decimal(environ, "CI_PIPELINE_ID")
    owner_uid = _prepare_protected_paths(layout, pipeline_id)
    print(f"Prepared L0 storage: kind={layout.storage_kind} "
          f"path={layout.storage_root} "
          f"lifecycle_uid={owner_uid} mode={PROTECTED_MODE:04o}")


def _job_environment(layout: StorageLayout, pipeline_id: str, job_id: str,
                     job_slug: str, environ: Mapping[str,
                                                     str]) -> Dict[str, str]:
    job_root = layout.job_root(pipeline_id, job_id, job_slug)
    variables = {
        "L0_ONNX_CACHE_ROOT": str(layout.onnx_cache_root),
        "PIPELINE_WORKSPACE_ROOT": str(layout.pipeline_root(pipeline_id)),
        "JOB_WORKSPACE_ROOT": str(job_root),
        "TEST_LOG_DIR": str(job_root / "logs"),
    }

    writes_engines = environ.get("L0_JOB_WRITES_ENGINES", "")
    if writes_engines == "1":
        variables["ENGINE_DIR"] = str(job_root / "engines")
    elif writes_engines:
        raise StorageContractError(
            "L0_JOB_WRITES_ENGINES must be '1' when set")

    cache_relative_path = _optional_relative_path(
        environ, "L0_ONNX_CACHE_RELATIVE_PATH")
    if cache_relative_path is not None:
        variables["ONNX_DIR"] = str(layout.onnx_cache_root /
                                    cache_relative_path)
    return variables


def _write_environment_file(path: Path, variables: Mapping[str, str]) -> None:
    contents = "".join(f"export {name}={shlex.quote(value)}\n"
                       for name, value in variables.items())
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o600)


def _prepare_job(environ: Mapping[str, str],
                 environment_file: Optional[Path]) -> None:
    layout = _storage_layout(environ)
    pipeline_id = _required_decimal(environ, "CI_PIPELINE_ID")
    job_id = _required_decimal(environ, "CI_JOB_ID")
    job_slug = _required_slug(environ)
    job_environment = _job_environment(layout, pipeline_id, job_id, job_slug,
                                       environ)
    _prepare_protected_paths(layout, pipeline_id)

    job_root = layout.job_root(pipeline_id, job_id, job_slug)
    _ensure_directory(job_root, JOB_MODE, "job workspace root", "runner")

    if environment_file is not None:
        _write_environment_file(environment_file, job_environment)
    print(f"Prepared L0 job workspace: path={job_root} mode={JOB_MODE:04o}")


def _validated_cleanup_target(path: Path, parent: Path, purpose: str) -> None:
    if path.parent != parent:
        raise StorageContractError(
            f"{purpose} is not an immediate child of {parent}: {path}")
    _assert_directory(path, purpose)


def _remove_tree(path: Path, parent: Path, purpose: str) -> None:
    _validated_cleanup_target(path, parent, purpose)
    try:
        shutil.rmtree(path)
    except OSError as error:
        try:
            path_stat = path.lstat()
            details = (f"uid={path_stat.st_uid} gid={path_stat.st_gid} "
                       f"mode={stat.S_IMODE(path_stat.st_mode):04o}")
        except OSError:
            details = "ownership unavailable"
        raise StorageContractError(
            f"failed to remove {purpose} {path} ({details}): {error}"
        ) from error
    if path.exists():
        raise StorageContractError(
            f"{purpose} still exists after cleanup: {path}")


def _prune_pipeline_workspaces(environ: Mapping[str, str]) -> None:
    layout = _storage_layout(environ)
    if layout.storage_kind != "mr":
        print(f"Pipeline workspace pruning does not apply to "
              f"storage kind={layout.storage_kind}")
        return

    current_pipeline_id = int(_required_decimal(environ, "CI_PIPELINE_ID"))
    try:
        layout.workspace_root.lstat()
    except FileNotFoundError:
        print(f"No L0 pipeline workspaces to prune beneath "
              f"{layout.workspace_root}")
        return
    _assert_directory(layout.workspace_root, "workspace root")

    candidates: List[Tuple[int, Path]] = []
    for path in layout.workspace_root.iterdir():
        match = PIPELINE_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            print(f"Skipping unrecognized L0 workspace entry: {path}",
                  file=sys.stderr)
            continue
        _assert_directory(path, "pipeline workspace")
        candidates.append((int(match.group(1)), path))

    for pipeline_id, path in candidates:
        if pipeline_id >= current_pipeline_id:
            print(f"Preserving current or newer L0 pipeline workspace: "
                  f"pipeline_id={pipeline_id}")
            continue
        _remove_tree(path, layout.workspace_root, "pipeline workspace")
        print(f"Pruned L0 pipeline workspace: {path}")


def _delete_storage(environ: Mapping[str, str]) -> None:
    layout = _storage_layout(environ)
    _assert_directory(layout.scratch_root, "scratch root")
    _assert_directory(layout.l0_root, "L0 root")
    try:
        layout.storage_root.lstat()
    except FileNotFoundError:
        print(f"L0 storage already absent: {layout.storage_root}")
        return

    _remove_tree(layout.storage_root, layout.storage_parent, "storage root")
    print(f"Deleted L0 storage: {layout.storage_root}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage validated TensorRT Edge-LLM L0 scratch storage")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("prepare-storage")
    prepare_job_parser = subparsers.add_parser("prepare-job")
    prepare_job_parser.add_argument("--environment-file", type=Path)
    subparsers.add_parser("prune-pipeline-workspaces")
    subparsers.add_parser("delete-storage")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one storage operation and return its process exit status."""
    args = _parser().parse_args(argv)
    try:
        if args.operation == "prepare-storage":
            _prepare_storage(os.environ)
        elif args.operation == "prepare-job":
            _prepare_job(os.environ, args.environment_file)
        elif args.operation == "prune-pipeline-workspaces":
            _prune_pipeline_workspaces(os.environ)
        elif args.operation == "delete-storage":
            _delete_storage(os.environ)
        else:
            raise StorageContractError(
                f"unsupported storage operation: {args.operation}")
    except (OSError, StorageContractError) as error:
        print(f"Storage contract failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
