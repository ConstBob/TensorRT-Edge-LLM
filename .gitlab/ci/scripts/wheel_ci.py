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
"""Command-line entrypoint for wheel packaging CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
import typing

from wheel_ci_lib import build_jobs, integration, matrix


def _generate_ci_command(values: typing.Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="wheel_ci.py generate-ci")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(values)
    matrix.generate_ci(check=args.check)


def _no_arguments(function: typing.Callable[[], None],
                  values: typing.Sequence[str]) -> None:
    if values:
        raise RuntimeError(f"This command takes no arguments: {values}.")
    function()


def main(values: typing.Optional[typing.Sequence[str]] = None) -> int:
    """Dispatch one internal wheel CI command."""
    arguments = list(values if values is not None else sys.argv[1:])
    commands = {
        "generate-ci":
        _generate_ci_command,
        "ci-precheck":
        lambda args: _no_arguments(build_jobs.precheck, args),
        "ci-build-base":
        lambda args: _no_arguments(build_jobs.build_base_ci, args),
        "ci-build-payload":
        lambda args: _no_arguments(build_jobs.build_payload_ci, args),
        "ci-assemble":
        lambda args: _no_arguments(build_jobs.assemble_ci, args),
        "ci-integration":
        lambda args: _no_arguments(integration.integration_ci, args),
        "ci-integration-gate":
        lambda args: _no_arguments(integration.integration_gate, args),
    }
    if not arguments or arguments[0] not in commands:
        available = ", ".join(sorted(commands))
        raise RuntimeError(
            f"Usage: wheel_ci.py <command> [args]\nCommands: {available}")
    commands[arguments[0]](arguments[1:])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
