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
import pathlib
import shutil
import subprocess

G_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
G_CI_SCRIPT_DIR = G_REPO_ROOT / ".gitlab" / "ci" / "scripts"


def run_script(script_name, *args, env=None, harness_timeout=10):
    return subprocess.run(
        ["bash", str(G_CI_SCRIPT_DIR / script_name), *args],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=harness_timeout,
    )


def install_fake_apt(tmp_path, body):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command_name in ("apt", "apt-get"):
        apt_command = fake_bin / command_name
        apt_command.write_text("#!/usr/bin/env bash\n" + body,
                               encoding="utf-8")
        apt_command.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["APT_CALL_LOG"] = str(tmp_path / "apt.log")
    return env, pathlib.Path(env["APT_CALL_LOG"])


def install_side_effect_spies(tmp_path):
    fake_bin = tmp_path / "side-effect-bin"
    fake_bin.mkdir()
    call_log = tmp_path / "side-effects.log"
    body = 'printf "%s\\n" "$0 $*" >> "$SIDE_EFFECT_LOG"\n'
    for command_name in ("apt", "apt-get", "sudo"):
        command = fake_bin / command_name
        command.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        command.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SIDE_EFFECT_LOG"] = str(call_log)
    return env, call_log


def install_side_effect_script(script_path):
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        'printf "package helper ran\\n" >> "$SIDE_EFFECT_LOG"\n',
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def install_fake_pre_commit(tmp_path, body):
    fake_bin = tmp_path / "pre-commit-bin"
    fake_bin.mkdir()
    command = fake_bin / "pre-commit"
    command.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    command.chmod(0o755)
    sleep_command = fake_bin / "sleep"
    sleep_command.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SLEEP_CALL_LOG"\n',
        encoding="utf-8",
    )
    sleep_command.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PRE_COMMIT_CALL_LOG"] = str(tmp_path / "pre-commit.log")
    env["SLEEP_CALL_LOG"] = str(tmp_path / "sleep.log")
    env["CI_PRE_COMMIT_BOOTSTRAP_ATTEMPTS"] = "2"
    env["CI_PRE_COMMIT_BOOTSTRAP_TIMEOUT_SECONDS"] = "5"
    env["CI_PRE_COMMIT_CHECK_TIMEOUT_SECONDS"] = "5"
    env["CI_PRE_COMMIT_RETRY_DELAY_SECONDS"] = "0"
    return env, pathlib.Path(env["PRE_COMMIT_CALL_LOG"])


def read_command_calls(call_log):
    return [
        line.split()
        for line in call_log.read_text(encoding="utf-8").splitlines()
    ]


def test_ci_run_preserves_output_and_exit_status():
    success = run_script(
        "ci_run.sh",
        "5",
        "successful command",
        "--",
        "sh",
        "-c",
        "printf child-stdout; printf child-stderr >&2",
    )
    assert success.returncode == 0
    assert "child-stdout" in success.stdout
    assert "child-stderr" in success.stderr
    assert "successful command" in success.stdout + success.stderr

    failure = run_script(
        "ci_run.sh",
        "5",
        "failing command",
        "--",
        "sh",
        "-c",
        "printf failure-stderr >&2; exit 17",
    )
    assert failure.returncode == 17
    assert "failure-stderr" in failure.stderr
    assert "failing command" in failure.stdout + failure.stderr


def test_ci_run_stops_a_hung_command():
    env = os.environ.copy()
    env["CI_COMMAND_KILL_AFTER_SECONDS"] = "1"

    result = run_script(
        "ci_run.sh",
        "1",
        "hung command",
        "--",
        "sh",
        "-c",
        "trap '' TERM; sleep 30",
        env=env,
        harness_timeout=5,
    )

    assert result.returncode != 0
    assert "hung command" in result.stdout + result.stderr


def test_ci_run_fails_before_execution_without_timeout(tmp_path):
    bash_path = shutil.which("bash")
    assert bash_path is not None

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "bash").symlink_to(bash_path)
    marker = tmp_path / "command-ran"
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    result = run_script(
        "ci_run.sh",
        "5",
        "missing timeout command",
        "--",
        "bash",
        "-c",
        'printf ran > "$1"',
        "test-command",
        str(marker),
        env=env,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert "missing timeout command" in result.stdout + result.stderr


def test_ci_pre_commit_retries_bootstrap_then_checks_once(tmp_path):
    marker = tmp_path / "first-bootstrap-failed"
    body = """printf "%s\\n" "$*" >> "$PRE_COMMIT_CALL_LOG"
if [ "$1" = "install-hooks" ] && [ ! -e "$PRE_COMMIT_FAILURE_MARKER" ]; then
    touch "$PRE_COMMIT_FAILURE_MARKER"
    exit 42
fi
"""
    env, call_log = install_fake_pre_commit(tmp_path, body)
    env["PRE_COMMIT_FAILURE_MARKER"] = str(marker)

    result = run_script("ci_pre_commit.sh", env=env)

    assert result.returncode == 0
    calls = read_command_calls(call_log)
    bootstrap_calls = [call for call in calls if call[0] == "install-hooks"]
    check_calls = [call for call in calls if call[0] == "run"]
    assert len(bootstrap_calls) == 2
    assert len(check_calls) == 1
    assert "--all-files" in check_calls[0]
    assert calls.index(bootstrap_calls[-1]) < calls.index(check_calls[0])


def test_ci_pre_commit_backs_off_then_stops_after_bootstrap_failure(tmp_path):
    body = """printf "%s\\n" "$*" >> "$PRE_COMMIT_CALL_LOG"
if [ "$1" = "install-hooks" ]; then
    exit 42
fi
"""
    env, call_log = install_fake_pre_commit(tmp_path, body)
    env["CI_PRE_COMMIT_BOOTSTRAP_ATTEMPTS"] = "3"
    env["CI_PRE_COMMIT_RETRY_DELAY_SECONDS"] = "15"

    result = run_script("ci_pre_commit.sh", env=env)

    assert result.returncode == 42
    calls = read_command_calls(call_log)
    assert len(calls) == 3
    assert all(call[0] == "install-hooks" for call in calls)
    sleep_calls = read_command_calls(pathlib.Path(env["SLEEP_CALL_LOG"]))
    assert sleep_calls == [["15"], ["30"]]


def test_ci_pre_commit_does_not_retry_failed_checks(tmp_path):
    body = """printf "%s\\n" "$*" >> "$PRE_COMMIT_CALL_LOG"
if [ "$1" = "run" ]; then
    exit 17
fi
"""
    env, call_log = install_fake_pre_commit(tmp_path, body)

    result = run_script("ci_pre_commit.sh", env=env)

    assert result.returncode == 17
    calls = read_command_calls(call_log)
    assert [call[0] for call in calls] == ["install-hooks", "run"]
    assert "--all-files" in calls[-1]


def test_ci_apt_updates_then_installs_requested_packages(tmp_path):
    env, call_log = install_fake_apt(
        tmp_path, 'printf "%s\\n" "$*" >> "$APT_CALL_LOG"\n')

    result = run_script("ci_apt.sh", "install", "python3", "git", env=env)

    assert result.returncode == 0
    calls = read_command_calls(call_log)
    update_calls = [call for call in calls if "update" in call]
    install_calls = [call for call in calls if "install" in call]
    assert update_calls
    assert install_calls
    assert calls.index(update_calls[0]) < calls.index(install_calls[0])
    assert any({"python3", "git"}.issubset(call) for call in install_calls)


def test_ci_apt_keeps_package_manager_output_visible(tmp_path):
    body = 'printf "apt-stdout-marker\\n"\nprintf "apt-stderr-marker\\n" >&2\n'
    env, _ = install_fake_apt(tmp_path, body)

    result = run_script("ci_apt.sh", "install", "git", env=env)

    assert result.returncode == 0
    assert "apt-stdout-marker" in result.stdout
    assert "apt-stderr-marker" in result.stderr


def test_ci_apt_retries_a_transient_update_before_installing(tmp_path):
    marker = tmp_path / "first-update-failed"
    body = """printf "%s\\n" "$*" >> "$APT_CALL_LOG"
if [[ " $* " == *" update "* ]] && [ ! -e "$APT_FAILURE_MARKER" ]; then
    touch "$APT_FAILURE_MARKER"
    exit 42
fi
"""
    env, call_log = install_fake_apt(tmp_path, body)
    env["APT_FAILURE_MARKER"] = str(marker)
    env["CI_APT_UPDATE_RETRY_DELAY_SECONDS"] = "0"

    result = run_script("ci_apt.sh", "install", "git", env=env)

    assert result.returncode == 0
    calls = read_command_calls(call_log)
    update_calls = [call for call in calls if "update" in call]
    install_calls = [call for call in calls if "install" in call]
    assert len(update_calls) >= 2
    assert install_calls
    assert calls.index(update_calls[-1]) < calls.index(install_calls[0])


def test_ci_apt_does_not_install_after_update_failure(tmp_path):
    body = """printf "%s\\n" "$*" >> "$APT_CALL_LOG"
if [[ " $* " == *" update "* ]]; then
    exit 42
fi
"""
    env, call_log = install_fake_apt(tmp_path, body)
    env["CI_APT_UPDATE_RETRY_DELAY_SECONDS"] = "0"

    result = run_script("ci_apt.sh", "install", "git", env=env)

    assert result.returncode == 42
    calls = read_command_calls(call_log)
    assert any("update" in call for call in calls)
    assert not any("install" in call for call in calls)
    assert result.stderr


def test_ci_apt_stops_a_hung_update_before_installing(tmp_path):
    body = """printf "%s\\n" "$*" >> "$APT_CALL_LOG"
if [[ " $* " == *" update "* ]]; then
    sleep 30
fi
"""
    env, call_log = install_fake_apt(tmp_path, body)
    env["CI_APT_UPDATE_TIMEOUT_SECONDS"] = "1"
    env["CI_APT_UPDATE_ATTEMPTS"] = "1"
    env["CI_COMMAND_KILL_AFTER_SECONDS"] = "1"

    result = run_script("ci_apt.sh",
                        "install",
                        "git",
                        env=env,
                        harness_timeout=5)

    assert result.returncode != 0
    calls = read_command_calls(call_log)
    assert any("update" in call for call in calls)
    assert not any("install" in call for call in calls)
    assert result.stderr


def test_ci_apt_rejects_empty_package_list_before_package_manager(tmp_path):
    env, call_log = install_fake_apt(
        tmp_path, 'printf "%s\\n" "$*" >> "$APT_CALL_LOG"\n')

    result = run_script("ci_apt.sh", "install", env=env)

    assert result.returncode != 0
    assert result.stderr
    assert not call_log.exists()


def test_device_init_accepts_workspace_scoped_helper(tmp_path):
    env, side_effect_log = install_side_effect_spies(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    remote_workspace = home / "tensorrt-edge-llm-123"
    apt_script = remote_workspace / ".gitlab" / "ci" / "scripts" / "ci_apt.sh"
    install_side_effect_script(apt_script)
    env["HOME"] = str(home)
    env["REMOTE_WORKSPACE"] = str(remote_workspace)
    env["CI_APT_SCRIPT"] = str(apt_script)
    env["BOARD_PASSWORD"] = "test-password"

    run_script("device_init.sh", env=env)

    assert side_effect_log.exists()


def test_device_init_rejects_workspace_before_side_effects(tmp_path):
    env, side_effect_log = install_side_effect_spies(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    remote_workspace = tmp_path / "outside" / "tensorrt-edge-llm-123"
    apt_script = remote_workspace / ".gitlab" / "ci" / "scripts" / "ci_apt.sh"
    install_side_effect_script(apt_script)
    sentinel = remote_workspace / "keep-me"
    sentinel.write_text("protected", encoding="utf-8")
    env["HOME"] = str(home)
    env["REMOTE_WORKSPACE"] = str(remote_workspace)
    env["CI_APT_SCRIPT"] = str(apt_script)

    result = run_script("device_init.sh", env=env)

    assert result.returncode != 0
    assert result.stderr
    assert not side_effect_log.exists()
    assert sentinel.read_text(encoding="utf-8") == "protected"


def test_device_init_rejects_external_helper_before_side_effects(tmp_path):
    env, side_effect_log = install_side_effect_spies(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    remote_workspace = home / "tensorrt-edge-llm-123"
    remote_workspace.mkdir()
    apt_script = tmp_path / "external-helper" / "ci_apt.sh"
    install_side_effect_script(apt_script)
    sentinel = apt_script.parent / "keep-me"
    sentinel.write_text("protected", encoding="utf-8")
    env["HOME"] = str(home)
    env["REMOTE_WORKSPACE"] = str(remote_workspace)
    env["CI_APT_SCRIPT"] = str(apt_script)

    result = run_script("device_init.sh", env=env)

    assert result.returncode != 0
    assert result.stderr
    assert not side_effect_log.exists()
    assert sentinel.read_text(encoding="utf-8") == "protected"
