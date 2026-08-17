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
"""Fresh-environment wheel qualification and integration evidence gating."""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import tempfile
import typing

from wheel_ci_lib import common, matrix, python_setup
from wheellib import config


def _model_dir() -> pathlib.Path:
    explicit = os.environ.get("WHEEL_MODEL_DIR")
    if explicit:
        return pathlib.Path(explicit).resolve(strict=True)
    roots = tuple(
        pathlib.Path(value) for value in (
            os.environ.get("LLM_MODELS_DIR"),
            "/scratch.trt_llm_data/llm-models",
            "/home/scratch.trt_llm_data/llm-models",
        ) if value)
    candidates = tuple(candidate for root in roots for candidate in (
        root / "Qwen2.5-0.5B-Instruct",
        root / "Qwen" / "Qwen2.5-0.5B-Instruct",
    ))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "Set WHEEL_MODEL_DIR to the Qwen2.5-0.5B-Instruct checkpoint. "
        f"Searched: {', '.join(str(path) for path in candidates)}")


def _trt_python_wheel(trt_dir: pathlib.Path, python_abi: str) -> pathlib.Path:
    wheels = sorted(
        path
        for path in (trt_dir / "python").glob(f"tensorrt-*-{python_abi}-*.whl")
        if path.is_file())
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one TensorRT {python_abi} wheel under {trt_dir / 'python'}."
        )
    return wheels[0].resolve()


def _gpu_uuid_for_sm(gpu_sm: int) -> str:
    explicit = os.environ.get("WHEEL_GPU_UUID")
    if explicit:
        return explicit
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ],
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Cannot query GPU UUIDs for wheel qualification.") from error
    matches = []
    for line in result.stdout.splitlines():
        try:
            uuid, capability = (value.strip() for value in line.split(",", 1))
            major, minor = (int(value) for value in capability.split(".", 1))
        except ValueError as error:
            raise RuntimeError(
                f"Unexpected nvidia-smi GPU row: {line!r}.") from error
        if major * 10 + minor == gpu_sm:
            matches.append(uuid)
    if not matches:
        raise RuntimeError(f"No visible GPU reports required SM {gpu_sm}.")
    return sorted(matches)[0]


def _integration_environment(
        trt_dir: pathlib.Path,
        gpu_sm: typing.Optional[int]) -> typing.Dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("LLM_SDK_DIR", None)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(trt_dir / "lib"),
                            environment.get("LD_LIBRARY_PATH", "")) if value)
    if gpu_sm is not None:
        environment["CUDA_VISIBLE_DEVICES"] = _gpu_uuid_for_sm(gpu_sm)
    return environment


def _local_integration(row: typing.Mapping[str, object], python_abi: str,
                       wheel: pathlib.Path, result: pathlib.Path) -> None:
    variant = str(row["variant_id"])
    version = common.ABI_INTERPRETERS[python_abi].removeprefix("python")
    python_bin = python_setup.install_host_python(version)
    trt_dir = common.normalized_trt_package(
        pathlib.Path(config.required_environment("TRT_PACKAGE_DIR")).resolve())
    model_dir = _model_dir()
    environment = _integration_environment(trt_dir, int(row["gpu_sm"]))
    with tempfile.TemporaryDirectory(
            prefix="edgellm-wheel-test-") as temporary:
        root = pathlib.Path(temporary)
        venv = root / "venv"
        config.run_checked([python_bin, "-m", "venv", str(venv)])
        interpreter = venv / "bin" / "python"
        config.run_checked(
            [str(interpreter), "-m", "pip", "install", "--upgrade", "pip"],
            env=environment)
        trt_requirement = row.get("ci_test_trt_requirement")
        if trt_requirement:
            trt_install = [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                os.environ.get("WHEEL_EXTRA_INDEX_URL",
                               "https://pypi.nvidia.com"),
                str(trt_requirement),
            ]
        else:
            trt_install = [
                str(interpreter),
                "-m",
                "pip",
                "install",
                str(_trt_python_wheel(trt_dir, python_abi)),
            ]
        config.run_checked(trt_install, env=environment)
        install = [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            os.environ.get("WHEEL_EXTRA_INDEX_URL", "https://pypi.nvidia.com"),
            str(wheel),
        ]
        config.run_checked(install, env=environment)
        raw_result = root / "result.json"
        config.run_checked([
            str(interpreter),
            "-I",
            str(config.REPO_ROOT / "examples" / "python" /
                "installed_wheel_build_and_infer.py"),
            str(model_dir),
            str(root / "engine"),
            "--expected-variant",
            variant,
            "--result",
            str(raw_result),
        ],
                           cwd=root,
                           env=environment)
        evidence = json.loads(raw_result.read_text(encoding="utf-8"))
    evidence.update({
        "variant_id": variant,
        "python_abi": python_abi,
        "wheel": wheel.name,
        "wheel_sha256": config.sha256(wheel),
    })
    config.write_json(result, evidence)


def _ssh_prefix(operation: str) -> typing.List[str]:
    return [
        "sshpass",
        "-e",
        operation,
        "-o",
        "StrictHostKeyChecking=no",
    ]


def _ssh_environment(password: str) -> typing.Dict[str, str]:
    environment = dict(os.environ)
    environment["SSHPASS"] = password
    return environment


def _copy_remote_model(scp: typing.Sequence[str], target: str, remote: str,
                       model_dir: pathlib.Path,
                       environment: typing.Mapping[str, str]) -> str:
    target_model = str(pathlib.PurePosixPath(remote) / "model")
    print(f"Copying model checkpoint to {target}:{target_model}")
    config.run_checked(
        [*scp, "-r", str(model_dir), f"{target}:{target_model}"],
        env=environment)
    return target_model


def _remote_integration(variant: str, python_abi: str, wheel: pathlib.Path,
                        result: pathlib.Path) -> None:
    board_user = config.required_environment("BOARD_USER")
    board_ip = config.required_environment("BOARD_IP")
    password = config.required_environment("BOARD_PASSWORD_NVKS")
    target_trt_template = config.required_environment("WHEEL_TARGET_TRT_WHEEL")
    if "{python_abi}" not in target_trt_template:
        raise RuntimeError(
            "WHEEL_TARGET_TRT_WHEEL must contain the {python_abi} placeholder."
        )
    target_trt = target_trt_template.replace("{python_abi}", python_abi)
    model_dir = _model_dir()
    job_id = config.required_environment("CI_JOB_ID")
    target = f"{board_user}@{board_ip}"
    remote_root = os.environ.get("WHEEL_TARGET_WORK_DIR",
                                 f"/home/{board_user}")
    if not remote_root.startswith("/"):
        raise RuntimeError("WHEEL_TARGET_WORK_DIR must be an absolute path.")
    remote = str(
        pathlib.PurePosixPath(remote_root) / f"edgellm-wheel-{job_id}")
    script = config.REPO_ROOT / "examples" / "python" / "installed_wheel_build_and_infer.py"
    ssh_environment = _ssh_environment(password)
    ssh = _ssh_prefix("ssh")
    scp = _ssh_prefix("scp")
    config.run_checked([
        *ssh,
        target,
        f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}",
    ],
                       env=ssh_environment)
    try:
        config.run_checked(
            [*scp, str(wheel),
             str(script), f"{target}:{remote}/"],
            env=ssh_environment)
        python_bin = "python3"
        probe = subprocess.run([
            *ssh,
            target,
            f"{python_bin} -c " + shlex.quote(
                "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
            ),
        ],
                               check=False,
                               env=ssh_environment,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        if probe.returncode != 0:
            raise RuntimeError("Cannot identify the target Python ABI.")
        target_abi = probe.stdout.strip()
        if target_abi != python_abi:
            raise RuntimeError(
                f"Target {python_bin} provides {target_abi}, not {python_abi}."
            )
        target_model = _copy_remote_model(scp, target, remote, model_dir,
                                          ssh_environment)
        command = " && ".join([
            f"{python_bin} -m venv --without-pip {shlex.quote(remote + '/venv')}",
            (f"{python_bin} -m pip install --ignore-installed "
             f"--prefix {shlex.quote(remote + '/venv')} pip"),
            f"{shlex.quote(remote + '/venv/bin/python')} -m pip install {shlex.quote(target_trt)}",
            (f"{shlex.quote(remote + '/venv/bin/python')} -m pip install "
             f"--extra-index-url {shlex.quote(os.environ.get('WHEEL_EXTRA_INDEX_URL', 'https://pypi.nvidia.com'))} "
             f"{shlex.quote(remote + '/' + wheel.name)}"),
            (f"cd {shlex.quote(remote)} && "
             f"{shlex.quote(remote + '/venv/bin/python')} -I "
             f"{shlex.quote(remote + '/' + script.name)} "
             f"{shlex.quote(target_model)} {shlex.quote(remote + '/engine')} "
             f"--expected-variant {shlex.quote(variant)} "
             f"--result {shlex.quote(remote + '/result.json')}"),
        ])
        config.run_checked([*ssh, target, command], env=ssh_environment)
        result.parent.mkdir(parents=True, exist_ok=True)
        config.run_checked(
            [*scp, f"{target}:{remote}/result.json",
             str(result)],
            env=ssh_environment)
    finally:
        subprocess.run(
            [*ssh, target, f"rm -rf {shlex.quote(remote)}"],
            check=False,
            env=ssh_environment,
        )
    evidence = json.loads(result.read_text(encoding="utf-8"))
    evidence.update({
        "variant_id": variant,
        "python_abi": python_abi,
        "wheel": wheel.name,
        "wheel_sha256": config.sha256(wheel),
    })
    config.write_json(result, evidence)


def _integrate_wheel(row: typing.Mapping[str, object],
                     python_abi: str) -> None:
    variant = str(row["variant_id"])
    environment_fields = (common.REMOTE_INTEGRATION_ENVIRONMENT
                          if row["ci_remote"] else
                          common.LOCAL_INTEGRATION_ENVIRONMENT)
    common.apply_variant_environment(row, environment_fields)
    wheel = common.single_wheel(str(row["cpu_arch"]), python_abi)
    result = (config.REPO_ROOT / "artifacts" / "integration" /
              f"{variant}-{python_abi}.json")
    if os.environ.get("REMOTE_TARGET") == "1":
        _remote_integration(variant, python_abi, wheel, result)
    else:
        _local_integration(row, python_abi, wheel, result)


def integration_ci() -> None:
    """Install, build, and infer with every configured ABI wheel."""
    qualification, _ = matrix.load_qualification()
    row = matrix.variant_row(config.required_environment("VARIANT"))
    python_abis = row.get("ci_test_python_abis",
                          qualification["qualified_python_abis"])
    if not row["ci_remote"]:
        versions = tuple(
            common.ABI_INTERPRETERS[python_abi].removeprefix("python")
            for python_abi in python_abis)
        with common.phase(f"integration {row['variant_id']}: host Pythons"):
            python_setup.prepare_host_pythons(versions)
    for python_abi in python_abis:
        with common.phase(f"integration {row['variant_id']}/{python_abi}"):
            _integrate_wheel(row, python_abi)


def _validate_integration_result(key: typing.Tuple[str, str],
                                 value: typing.Mapping[str, object],
                                 wheel: pathlib.Path) -> None:
    if value.get("wheel") != wheel.name or value.get(
            "wheel_sha256") != config.sha256(wheel):
        raise RuntimeError(
            f"Integration evidence for {key} references another wheel.")
    output_count = value.get("output_token_count")
    if not isinstance(output_count, int) or output_count < 1:
        raise RuntimeError(f"Integration evidence for {key} has no output.")
    output_text = value.get("output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError(
            f"Integration evidence for {key} has no generated text.")


def integration_gate() -> None:
    """Require successful evidence for every selected variant and ABI."""
    qualification, rows = matrix.load_qualification()
    expected = {
        (str(row["variant_id"]), abi): row
        for row in matrix.integration_rows(rows)
        for abi in row.get("ci_test_python_abis",
                           qualification["qualified_python_abis"])
    }
    evidence_dir = config.REPO_ROOT / "artifacts" / "integration"
    found = {}
    for path in evidence_dir.glob("*.json"):
        if path.name == "gate.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (str(value.get("variant_id")), str(value.get("python_abi")))
        if key in found:
            raise RuntimeError(f"Duplicate integration evidence for {key}.")
        found[key] = value
    if set(found) != set(expected):
        raise RuntimeError("Incomplete integration evidence: "
                           f"missing={sorted(set(expected) - set(found))}, "
                           f"extra={sorted(set(found) - set(expected))}.")
    for key, value in found.items():
        row = expected[key]
        wheel = common.single_wheel(str(row["cpu_arch"]), key[1])
        _validate_integration_result(key, value, wheel)
    config.write_json(
        config.REPO_ROOT / "artifacts" / "integration" / "gate.json",
        {
            "schema_version":
            1,
            "qualified_rows":
            len(found),
            "wheel_sha256":
            sorted({value["wheel_sha256"]
                    for value in found.values()}),
        },
    )
