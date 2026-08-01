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

import os
import stat
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).parents[3] / "tests"
sys.path.insert(0, str(TESTS_ROOT))

import pytest_helpers
from conftest import RemoteConfig


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_remote_commands_create_shared_storage_modes(monkeypatch, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_transport = fake_bin / "sshpass"
    fake_transport.write_text("""#!/bin/sh
for remote_command
do
  :
done
exec sh -c "$remote_command"
""")
    fake_transport.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    remote_config = RemoteConfig(host="board",
                                 user="tensorrt_user",
                                 password="secret",
                                 remote_workspace=str(tmp_path))

    result = pytest_helpers.run_command(
        ["sh", "-c", "mkdir board-output && : > board-output/audio.wav"],
        remote_config)

    assert result["success"]
    assert _mode(tmp_path / "board-output") == 0o777
    assert _mode(tmp_path / "board-output" / "audio.wav") == 0o666


def test_local_commands_preserve_the_caller_umask(tmp_path):
    previous_umask = os.umask(0o027)
    try:
        result = pytest_helpers.run_command(
            ["mkdir", str(tmp_path / "runner-output")], None)
    finally:
        os.umask(previous_umask)

    assert result["success"]
    assert _mode(tmp_path / "runner-output") == 0o750
