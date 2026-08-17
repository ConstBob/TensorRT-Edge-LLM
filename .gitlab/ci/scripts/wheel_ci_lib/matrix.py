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
"""Qualification configuration and generated GitLab wheel matrices."""

from __future__ import annotations

import typing

from wheellib import config

_GENERATED_CI = config.REPO_ROOT / ".gitlab" / "ci" / "wheel-generated.yml"
_QUALIFICATION = config.REPO_ROOT / ".gitlab" / "ci" / "wheel-qualification.toml"
_CI_FIELDS = frozenset({
    "variant_id",
    "ci_board_ip",
    "ci_board_user",
    "ci_build_image",
    "ci_build_mode",
    "ci_build_runner",
    "ci_python_headers",
    "ci_remote",
    "ci_sysroot",
    "ci_target_trt_wheel",
    "ci_target_work_dir",
    "ci_test_image",
    "ci_test_python_abis",
    "ci_test_runner",
    "ci_test_trt_requirement",
    "ci_toolchain",
    "ci_trt_package",
})
_CI_REQUIRED_FIELDS = frozenset({
    "variant_id",
    "ci_build_image",
    "ci_build_mode",
    "ci_build_runner",
    "ci_remote",
    "ci_test_image",
    "ci_test_runner",
    "ci_trt_package",
})
_MAX_NEEDED_JOB_NAME = 128
_BUILD_GROUP_FIELDS = (
    "cpu_arch",
    "ci_build_runner",
    "ci_build_image",
    "ci_build_mode",
    "ci_trt_package",
    "ci_toolchain",
    "ci_sysroot",
    "ci_python_headers",
)


def _validate_qualification_row(value: typing.Any,
                                index: int) -> typing.Dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Qualification row {index} must be a table.")
    row = dict(value)
    unexpected = sorted(set(row) - _CI_FIELDS)
    missing = sorted(_CI_REQUIRED_FIELDS - set(row))
    if unexpected or missing:
        raise RuntimeError(
            f"Qualification row {index} has unexpected={unexpected}, missing={missing}."
        )
    variant_id = str(row["variant_id"])
    build_mode = row["ci_build_mode"]
    if build_mode not in {"native", "cross"}:
        raise RuntimeError(
            f"Variant {variant_id} ci_build_mode must be native or cross.")
    if build_mode == "cross":
        missing = sorted(field for field in ("ci_toolchain", "ci_sysroot",
                                             "ci_python_headers")
                         if not row.get(field))
        if missing:
            raise RuntimeError(
                f"Variant {variant_id} is missing cross-build fields: {missing}."
            )
    remote = row["ci_remote"]
    if not isinstance(remote, bool):
        raise RuntimeError(f"Variant {variant_id} ci_remote must be Boolean.")
    if remote:
        missing = sorted(field for field in (
            "ci_board_ip",
            "ci_board_user",
            "ci_target_trt_wheel",
            "ci_test_python_abis",
        ) if not row.get(field))
        if missing:
            raise RuntimeError(
                f"Variant {variant_id} is missing remote-test fields: {missing}."
            )
    test_python_abis = row.get("ci_test_python_abis")
    if test_python_abis is not None and (
            not isinstance(test_python_abis, list) or not test_python_abis
            or any(not isinstance(item, str) or not item
                   for item in test_python_abis)
            or len(set(test_python_abis)) != len(test_python_abis)):
        raise RuntimeError(
            f"Variant {variant_id} ci_test_python_abis must be a non-empty list of unique strings."
        )
    return row


def load_qualification() -> typing.Tuple[typing.Dict[
    str, typing.Any], typing.List[typing.Dict[str, object]]]:
    matrix, variants = config.load_matrix(config.REPO_ROOT / "packaging" /
                                          "variants.toml")
    private = config.load_toml(_QUALIFICATION)
    if private.get("schema_version") != 1:
        raise RuntimeError(
            "wheel-qualification.toml must use schema_version = 1.")
    values = private.get("qualification")
    if not isinstance(values, list) or not values:
        raise RuntimeError(
            "wheel-qualification.toml contains no qualification rows.")
    qualification = [
        _validate_qualification_row(value, index)
        for index, value in enumerate(values)
    ]
    qualified_python_abis = set(matrix["qualified_python_abis"])
    for row in qualification:
        requested = set(row.get("ci_test_python_abis", qualified_python_abis))
        unsupported = sorted(requested - qualified_python_abis)
        if unsupported:
            raise RuntimeError(
                f"Variant {row['variant_id']} requests unsupported test Python ABIs: {unsupported}."
            )
    by_id = {str(row["variant_id"]): row for row in qualification}
    if len(by_id) != len(qualification):
        raise RuntimeError(
            "wheel-qualification.toml contains duplicate variant IDs.")
    public_ids = {str(row["variant_id"]) for row in variants}
    private_ids = set(by_id)
    if public_ids != private_ids:
        raise RuntimeError(
            "Wheel qualification coverage differs from the public matrix: "
            f"missing={sorted(public_ids - private_ids)}, "
            f"extra={sorted(private_ids - public_ids)}.")
    merged = [{**row, **by_id[str(row["variant_id"])]} for row in variants]
    return matrix, merged


def _yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return '"1"' if value else '"0"'
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _matrix_row(values: typing.Mapping[str, object]) -> typing.List[str]:
    items = iter(values.items())
    name, value = next(items)
    lines = [f"      - {name}: {_yaml_value(value)}"]
    lines.extend(f"        {item_name}: {_yaml_value(item_value)}"
                 for item_name, item_value in items)
    return lines


def _build_group_name(rows: typing.List[typing.Mapping[str, object]]) -> str:
    first = str(rows[0]["variant_id"])
    additional = len(rows) - 1
    return first if additional == 0 else f"{first}-plus-{additional}"


def build_groups(
    rows: typing.List[typing.Mapping[str, object]]
) -> typing.List[typing.Tuple[str, typing.List[typing.Mapping[str, object]]]]:
    grouped: typing.Dict[typing.Tuple[object, ...],
                         typing.List[typing.Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in _BUILD_GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)
    return [(_build_group_name(group), group) for group in grouped.values()]


def _validate_job_name(job: str, values: typing.Tuple[object, ...]) -> None:
    suffix = ", ".join(str(value) for value in values)
    name = f"{job}: [{suffix}]"
    if len(name) > _MAX_NEEDED_JOB_NAME:
        raise RuntimeError(
            f"Generated matrix job name exceeds {_MAX_NEEDED_JOB_NAME} "
            f"characters: {name}")


def _validate_matrix_job_names(
        rows: typing.List[typing.Mapping[str, object]]) -> None:
    for group_name, group in build_groups(rows):
        first = group[0]
        job = ("wheel_payload_x86"
               if first["cpu_arch"] == "x86_64" else "wheel_payload_aarch64")
        _validate_job_name(
            job,
            (group_name, first["ci_build_runner"], first["ci_build_image"]),
        )
    for row in rows:
        job = ("wheel_integration_x86"
               if row["cpu_arch"] == "x86_64" else "wheel_integration_aarch64")
        _validate_job_name(
            job,
            (row["variant_id"], row["ci_test_runner"], row["ci_test_image"]),
        )


def _matrix_lines(values: typing.Iterable[typing.Mapping[str, object]]) -> str:
    return "\n".join(line for value in values for line in _matrix_row(value))


def _build_matrix(rows: typing.List[typing.Mapping[str, object]],
                  cpu_arch: str) -> str:
    values = []
    for group_name, group in build_groups(rows):
        if group[0]["cpu_arch"] != cpu_arch:
            continue
        values.append({
            "BUILD_GROUP": group_name,
            "BUILD_RUNNER_TAG": group[0]["ci_build_runner"],
            "BUILD_IMAGE": group[0]["ci_build_image"],
        })
    return _matrix_lines(values)


def _test_matrix(rows: typing.List[typing.Mapping[str, object]],
                 cpu_arch: str) -> str:
    return _matrix_lines({
        "VARIANT": row["variant_id"],
        "TEST_RUNNER_TAG": row["ci_test_runner"],
        "TEST_IMAGE": row["ci_test_image"],
    } for row in rows if row["cpu_arch"] == cpu_arch)


def generated_ci() -> str:
    """Render concrete payload and integration matrices from variants.toml."""
    _, rows = load_qualification()
    _validate_matrix_job_names(rows)
    x86_builds = _build_matrix(rows, "x86_64")
    aarch64_builds = _build_matrix(rows, "aarch64")
    x86_tests = _test_matrix(rows, "x86_64")
    aarch64_tests = _test_matrix(rows, "aarch64")
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated by: sh .gitlab/ci/scripts/wheel_ci.sh generate-ci
# Do not edit this file directly.

wheel_payload_x86:
  extends: [.wheel_payload_template]
  parallel:
    matrix:
{x86_builds}

wheel_payload_aarch64:
  extends: [.wheel_payload_template]
  parallel:
    matrix:
{aarch64_builds}

wheel_assemble_x86:
  extends: [.wheel_assemble_template]
  variables:
    WHEEL_ARCH: "x86_64"
  needs:
    - job: wheel_python_base
      artifacts: true
    - job: wheel_payload_x86
      artifacts: true

wheel_assemble_aarch64:
  extends: [.wheel_assemble_template]
  variables:
    WHEEL_ARCH: "aarch64"
  needs:
    - job: wheel_python_base
      artifacts: true
    - job: wheel_payload_aarch64
      artifacts: true

wheel_integration_x86:
  extends: [.wheel_integration_template]
  needs:
    - job: wheel_assemble_x86
      artifacts: true
  parallel:
    matrix:
{x86_tests}

wheel_integration_aarch64:
  extends: [.wheel_integration_template]
  needs:
    - job: wheel_assemble_aarch64
      artifacts: true
  parallel:
    matrix:
{aarch64_tests}
"""


def generate_ci(*, check: bool = False) -> None:
    """Write or validate the checked-in generated GitLab matrix."""
    rendered = generated_ci()
    if check:
        actual = (_GENERATED_CI.read_text(
            encoding="utf-8") if _GENERATED_CI.is_file() else "")
        if actual != rendered:
            raise RuntimeError(
                f"{_GENERATED_CI} is stale; run wheel_ci.sh generate-ci.")
        return
    _GENERATED_CI.write_text(rendered, encoding="utf-8")


def variant_row(variant: str) -> typing.Mapping[str, object]:
    _, rows = load_qualification()
    row = next((value for value in rows if value["variant_id"] == variant),
               None)
    if row is None:
        raise RuntimeError(f"Unknown scheduled variant {variant}.")
    return row


def scheduled_build_group(
        group_name: str) -> typing.List[typing.Mapping[str, object]]:
    _, rows = load_qualification()
    groups = dict(build_groups(rows))
    try:
        return groups[group_name]
    except KeyError as error:
        raise RuntimeError(
            f"Unknown scheduled build group {group_name!r}.") from error
