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

import argparse
import json
import pathlib
import shutil
import time
import typing


def _safe_root(value: str) -> pathlib.Path:
    candidate = pathlib.Path(value).expanduser()
    if candidate.is_symlink():
        raise argparse.ArgumentTypeError("cleanup root must not be a symlink")
    root = candidate.resolve()
    if root == pathlib.Path("/") or root.name != "runner_usage":
        raise argparse.ArgumentTypeError(
            "cleanup root must be a directory named runner_usage")
    return root


def _is_finished_record(path: pathlib.Path) -> bool:
    try:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (isinstance(payload, dict) and str(payload.get(
        "job_status", "")).lower() not in ("created", "pending", "preparing",
                                           "running", "waiting_for_resource"))


def _resolves_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def cleanup(root: pathlib.Path,
            max_age_days: float,
            active_grace_hours: float,
            now: typing.Optional[float] = None,
            dry_run: bool = False) -> typing.Dict[str, int]:
    """Remove expired finished records without following symlinks."""
    current_time = time.time() if now is None else now
    expiry_cutoff = current_time - max_age_days * 24 * 60 * 60
    active_cutoff = current_time - active_grace_hours * 60 * 60
    summary = {
        "records_removed": 0,
        "phase_directories_removed": 0,
        "empty_directories_removed": 0,
        "records_preserved": 0,
    }
    if not root.is_dir() or root.is_symlink():
        return summary

    for path in sorted(root.glob("*/*/*.json")):
        if (not _resolves_within(path, root) or not path.parent.name.isdigit()
                or not path.stem.isdigit() or path.is_symlink()):
            summary["records_preserved"] += 1
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            summary["records_preserved"] += 1
            continue
        if (modified >= expiry_cutoff or modified >= active_cutoff
                or not _is_finished_record(path)):
            summary["records_preserved"] += 1
            continue
        phase_directory = path.parent / f"{path.stem}.phases"
        phase_exists = (phase_directory.is_dir()
                        and not phase_directory.is_symlink())
        if not dry_run:
            try:
                if phase_exists:
                    shutil.rmtree(phase_directory)
                path.unlink(missing_ok=True)
            except OSError:
                summary["records_preserved"] += 1
                continue
        summary["records_removed"] += 1
        summary["phase_directories_removed"] += int(phase_exists)

    for pattern in ("*/*", "*"):
        for directory in sorted(root.glob(pattern), reverse=True):
            if (not _resolves_within(directory, root)
                    or not directory.is_dir() or directory.is_symlink()):
                continue
            try:
                if directory.stat().st_mtime >= active_cutoff:
                    continue
                empty = not any(directory.iterdir())
            except OSError:
                continue
            if empty:
                try:
                    if not dry_run:
                        directory.rmdir()
                except OSError:
                    continue
                summary["empty_directories_removed"] += 1
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservatively expire finished CI telemetry records.")
    parser.add_argument("--root", required=True, type=_safe_root)
    parser.add_argument("--max-age-days", type=float, default=30.0)
    parser.add_argument("--active-grace-hours", type=float, default=24.0)
    parser.add_argument("--now-epoch", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_age_days <= 0 or args.active_grace_hours < 0:
        print("ERROR: retention values must be positive")
        return 2
    summary = cleanup(args.root, args.max_age_days, args.active_grace_hours,
                      args.now_epoch, args.dry_run)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
