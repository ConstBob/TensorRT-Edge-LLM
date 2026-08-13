#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Build, verify, and qualify TensorRT Edge-LLM wheels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Callable, Dict, Optional, Sequence

from wheellib import assemble, base, ci, cutedsl, payload, source, verify
from wheellib.config import REPO_ROOT, load_matrix


def _validate_matrix(_: Sequence[str]) -> None:
    load_matrix(REPO_ROOT / "packaging" / "variants.toml")


def _generate_ci(values: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="wheel_cli.py generate-ci")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(values)
    ci.generate_ci(check=args.check)


def _no_arguments(function: Callable[[], None], values: Sequence[str]) -> None:
    if values:
        raise RuntimeError(f"This command takes no arguments: {values}.")
    function()


def main(values: Optional[Sequence[str]] = None) -> int:
    """Dispatch one supported packaging or CI command."""
    arguments = list(values if values is not None else sys.argv[1:])
    commands: Dict[str, Callable[[Sequence[str]], None]] = {
        "validate-source":
        source.main,
        "validate-matrix":
        _validate_matrix,
        "generate-ci":
        _generate_ci,
        "build-base":
        base.main,
        "prepare-cutedsl":
        cutedsl.main,
        "build-payload":
        payload.main,
        "verify-payload":
        verify.main,
        "assemble":
        assemble.main,
        "ci-precheck":
        lambda args: _no_arguments(ci.precheck, args),
        "ci-build-base":
        lambda args: _no_arguments(ci.build_base_ci, args),
        "ci-build-payload":
        lambda args: _no_arguments(ci.build_payload_ci, args),
        "ci-assemble":
        lambda args: _no_arguments(ci.assemble_ci, args),
        "ci-integration":
        lambda args: _no_arguments(ci.integration_ci, args),
        "ci-integration-gate":
        lambda args: _no_arguments(ci.integration_gate, args),
    }
    if not arguments or arguments[0] not in commands:
        available = ", ".join(sorted(commands))
        raise RuntimeError(
            f"Usage: python packaging/wheel_cli.py <command> [args]\n"
            f"Commands: {available}")
    commands[arguments[0]](arguments[1:])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
