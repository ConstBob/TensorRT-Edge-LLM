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
import datetime
import json
import math
import os
import pathlib
import re
import sys
import time
import typing

SCHEMA = "edgellm-ci-cache-event/v1"
ARTIFACTS = {"image", "ccache", "onnx", "engine", "timing"}
OPERATIONS = {
    "build", "deserialize", "lookup", "materialize", "prune", "publish",
    "pull", "push", "resolve", "restore", "serialize", "validate"
}
DECISIONS = {
    "corruption", "descriptor-change", "deserialization-failure", "error",
    "forced-cold", "hit", "identical-publisher-race", "invalid-inventory",
    "invalid-manifest", "key-collision", "miss", "not-found", "oversized",
    "policy-disabled", "published", "quota-eviction", "restore-cost-gate",
    "schema-mismatch", "skipped", "unplanned-producer", "unrepresentable-input"
}
MISS_DECISIONS = DECISIONS - {
    "hit", "identical-publisher-race", "policy-disabled", "published",
    "skipped"
}
TOKEN = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,79}$")
KEY_PREFIX = re.compile(r"^[0-9a-f]{8,16}$")


class _FailOpenArgumentParser(argparse.ArgumentParser):

    def error(self, message: str) -> typing.NoReturn:
        raise ValueError(message)


def _token(value: typing.Any,
           name: str,
           allowed: typing.Optional[typing.Set[str]] = None,
           required: bool = True) -> str:
    text = "" if value is None else str(value)
    if not text and not required:
        return ""
    if not TOKEN.fullmatch(
            text) or allowed is not None and text not in allowed:
        raise ValueError(f"invalid cache event {name}")
    return text


def _number(value: typing.Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid cache event {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid cache event {name}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid cache event {name}")
    return number


def normalize(
        payload: typing.Mapping[str,
                                typing.Any]) -> typing.Dict[str, typing.Any]:
    """Return a bounded cache event containing only the public schema."""
    if payload.get("schema") != SCHEMA or payload.get("event_type") != "cache":
        raise ValueError("invalid cache event schema")
    result = {
        "schema_version":
        1,
        "schema":
        SCHEMA,
        "event_type":
        "cache",
        "label":
        _token(payload.get("label"), "label"),
        "status":
        _token(payload.get("status"), "status", {"success"}),
        "exit_code":
        0,
        "started_at":
        str(payload.get("started_at", "")),
        "started_epoch":
        _number(payload.get("started_epoch"), "started_epoch"),
        "finished_at":
        str(payload.get("finished_at", "")),
        "finished_epoch":
        _number(payload.get("finished_epoch"), "finished_epoch"),
        "elapsed_seconds":
        _number(payload.get("elapsed_seconds"), "elapsed_seconds"),
        "artifact":
        _token(payload.get("artifact"), "artifact", ARTIFACTS),
        "operation":
        _token(payload.get("operation"), "operation", OPERATIONS),
        "decision":
        _token(payload.get("decision"), "decision", DECISIONS),
        "reason":
        _token(payload.get("reason"), "reason"),
    }
    for name in ("namespace", "lane", "materialization", "field"):
        value = _token(payload.get(name), name, required=False)
        if value:
            result[name] = value
    key_prefix = str(payload.get("key_prefix") or "")
    if key_prefix:
        if not KEY_PREFIX.fullmatch(key_prefix):
            raise ValueError("invalid cache event key_prefix")
        result["key_prefix"] = key_prefix
    for name in ("files", "bytes_read", "bytes_written"):
        value = payload.get(name)
        if value is not None:
            number = _number(value, name)
            if not number.is_integer():
                raise ValueError(f"invalid cache event {name}")
            result[name] = int(number)
    for name in ("build_skipped", "inference_executed"):
        value = payload.get(name)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"invalid cache event {name}")
            result[name] = value
    return result


def record(
    event: typing.Mapping[str, typing.Any],
    environment: typing.Optional[typing.Mapping[str, str]] = None
) -> typing.Optional[pathlib.Path]:
    """Write one fail-open cache event into the ordinary phase transport."""
    env = os.environ if environment is None else environment
    if env.get("CI_TELEMETRY_ENABLED", "1") == "0" or not env.get("CI_JOB_ID"):
        return None
    try:
        finished = time.time()
        elapsed = _number(event.get("elapsed_seconds"), "elapsed_seconds")
        started = finished - elapsed
        payload = {
            **event,
            "schema":
            SCHEMA,
            "event_type":
            "cache",
            "label":
            f"cache-{event.get('artifact')}-{event.get('operation')}",
            "status":
            "success",
            "started_at":
            datetime.datetime.fromtimestamp(started,
                                            datetime.timezone.utc).isoformat(),
            "started_epoch":
            started,
            "finished_at":
            datetime.datetime.fromtimestamp(finished,
                                            datetime.timezone.utc).isoformat(),
            "finished_epoch":
            finished,
        }
        normalized = normalize(payload)
        root = pathlib.Path(
            env.get(
                "CI_TELEMETRY_DIR",
                str(
                    pathlib.Path(env.get("CI_PROJECT_DIR", os.getcwd())) /
                    ".ci-telemetry")))
        directory = root / "phases" / env["CI_JOB_ID"]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"cache-{time.time_ns()}-{os.getpid()}.json"
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(normalized, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, target)
        return target
    except (OSError, ValueError) as error:
        print(f"WARN: cannot record CI cache event: {error}", file=sys.stderr)
        return None


def build_rows(
    job_rows: typing.Iterable[typing.Mapping[str, typing.Any]]
) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], int]:
    rows: typing.List[typing.Dict[str, typing.Any]] = []
    invalid = 0
    for job in job_rows:
        for phase in job.get("phases", []):
            if phase.get("event_type") != "cache":
                continue
            try:
                event = normalize(phase)
            except (AttributeError, ValueError):
                invalid += 1
                continue
            rows.append({
                "pipeline_id": job.get("pipeline_id", ""),
                "job_id": job.get("job_id", ""),
                "job_name": job.get("job_name", ""),
                "attempt_number": job.get("attempt_number", ""),
                **{
                    name: event.get(name, "")
                    for name in ("artifact", "operation", "namespace", "key_prefix", "decision", "reason", "field", "lane", "elapsed_seconds", "files", "bytes_read", "bytes_written", "materialization", "build_skipped", "inference_executed")
                },
            })
    return rows, invalid


def aggregate(
    rows: typing.Iterable[typing.Mapping[str, typing.Any]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    groups: typing.Dict[typing.Tuple[str, ...], typing.Dict[str,
                                                            typing.Any]] = {}
    names = ("artifact", "operation", "namespace", "decision", "reason",
             "field", "lane")
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in names)
        result = groups.setdefault(
            key, {
                **dict(zip(names, key)),
                "sample_count": 0,
                "elapsed_seconds": 0.0,
                "files": 0,
                "bytes_read": 0,
                "bytes_written": 0,
            })
        result["sample_count"] += 1
        for name in ("elapsed_seconds", "files", "bytes_read",
                     "bytes_written"):
            result[name] += float(row.get(name) or 0)
    return sorted(
        groups.values(),
        key=lambda row:
        (-row["elapsed_seconds"], ) + tuple(str(row[name]) for name in names))


def _parser() -> argparse.ArgumentParser:
    parser = _FailOpenArgumentParser(description="Record one CI cache event.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--elapsed-seconds", required=True, type=float)
    for name in ("namespace", "key-prefix", "lane", "materialization",
                 "field"):
        parser.add_argument(f"--{name}")
    for name in ("files", "bytes-read", "bytes-written"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--build-skipped", action="store_true", default=None)
    parser.add_argument("--inference-executed",
                        action="store_true",
                        default=None)
    return parser


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    try:
        values = vars(_parser().parse_args(argv))
    except ValueError as error:
        print(f"WARN: cannot record CI cache event: {error}", file=sys.stderr)
        return 0
    event = {
        name.replace("-", "_"): value
        for name, value in values.items() if value is not None
    }
    record(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
