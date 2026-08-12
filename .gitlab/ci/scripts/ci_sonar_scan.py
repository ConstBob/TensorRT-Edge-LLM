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
"""Run SonarScanner without making SonarQube outages pipeline-blocking."""

import argparse
import os
import subprocess
import sys
import typing
import urllib.error
import urllib.parse
import urllib.request


def _parse_server_url(value: str) -> str:
    parsed_url = urllib.parse.urlsplit(value)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise argparse.ArgumentTypeError(
            "SonarQube server URL must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def _parse_arguments(
        argv: typing.Optional[typing.Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SonarScanner when SonarQube is available.")
    parser.add_argument(
        "--probe-timeout-seconds",
        default=10.0,
        type=float,
        help="Timeout for each SonarQube availability request.",
    )
    parser.add_argument("server_url", type=_parse_server_url)
    parser.add_argument("scanner_command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    if arguments.probe_timeout_seconds <= 0:
        parser.error("--probe-timeout-seconds must be greater than zero")
    if not arguments.scanner_command:
        parser.error("a scanner command is required")
    return arguments


def _sonar_is_available(server_url: str, timeout_seconds: float) -> bool:
    request = urllib.request.Request(
        f"{server_url}/api/server/version",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds):
            return True
    except urllib.error.HTTPError as error:
        if error.code not in (408, 429) and error.code < 500:
            return True
        print(f"SonarQube probe returned HTTP {error.code}", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"SonarQube probe failed: {error}", file=sys.stderr)
    return False


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    arguments = _parse_arguments(argv)

    print("Checking SonarQube availability")
    if not _sonar_is_available(arguments.server_url,
                               arguments.probe_timeout_seconds):
        print("WARNING: SonarQube is unavailable; skipping source analysis",
              file=sys.stderr)
        return os.EX_TEMPFAIL

    try:
        scanner_result = subprocess.run(arguments.scanner_command, check=False)
    except OSError as error:
        print(f"ERROR: cannot run SonarScanner: {error}", file=sys.stderr)
        return 1

    if scanner_result.returncode == 0:
        return 0
    if _sonar_is_available(arguments.server_url,
                           arguments.probe_timeout_seconds):
        return scanner_result.returncode

    print("WARNING: SonarQube is unavailable; skipping source analysis",
          file=sys.stderr)
    return os.EX_TEMPFAIL


if __name__ == "__main__":
    sys.exit(main())
