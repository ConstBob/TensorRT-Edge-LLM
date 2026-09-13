#!/usr/bin/env bash
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

set -euo pipefail

: "${REMOTE_WORKSPACE:?REMOTE_WORKSPACE must be set}"
: "${TRT_PACKAGE_DIR:?TRT_PACKAGE_DIR must be set}"
: "${PRIORITY:?PRIORITY must be set}"
: "${JUNIT_PREFIX:?JUNIT_PREFIX must be set}"

cd "${REMOTE_WORKSPACE}"
ci_run="${REMOTE_WORKSPACE}/.gitlab/ci/scripts/ci_run.sh"
pip_timeout_seconds="${CI_PIP_INSTALL_TIMEOUT_SECONDS:-1200}"
job_phase_timeout_seconds="${CI_JOB_PHASE_TIMEOUT_SECONDS:-0}"
report_name="${PYTHON_PLUGIN_REPORT_NAME:-l0_python_plugin_ut}"
report_dir="${PYTHON_PLUGIN_REPORT_DIR:-logs}"

library_dir="${TRT_LIBRARY_DIR:-${TRT_PACKAGE_DIR}/lib}"
export LD_LIBRARY_PATH="${library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Resolving Python plugin-test environment on $(hostname)"
use_system_site_packages="${PYTHON_PLUGIN_USE_SYSTEM_SITE_PACKAGES:-0}"
venv_args=()
requirements=(tests/requirements.txt)
if [[ "${use_system_site_packages}" == "1" ]]; then
    venv_args+=(--system-site-packages)
else
    requirements+=(tests/requirements-ut.txt)
fi

python_tag="cp$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
trt_wheel=""
if [[ -d "${TRT_PACKAGE_DIR}/python" ]]; then
    trt_wheel=$(find "${TRT_PACKAGE_DIR}/python" -maxdepth 1 -type f \
        -name "tensorrt-*${python_tag}*aarch64*.whl" -print -quit)
fi

trt_source=""
if [[ "${use_system_site_packages}" == "1" ]] \
        && system_trt_version=$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null); then
    trt_source="system:${system_trt_version}"
elif [[ -n "${trt_wheel}" ]]; then
    trt_source="wheel:$(sha256sum "${trt_wheel}" | cut -d' ' -f1)"
elif [[ -n "${TRT_PYTHON_VERSION:-}" ]]; then
    trt_source="package:${TRT_PYTHON_VERSION}"
else
    echo "No ${python_tag} TensorRT wheel found in ${TRT_PACKAGE_DIR}/python"
    exit 1
fi

cache_root="${PYTHON_PLUGIN_VENV_CACHE_DIR:-${HOME}/.cache/tensorrt-edge-llm/python-plugin-venvs}"
mkdir -p "${cache_root}"
cache_key=$(
    {
        printf '%s\n' "device-plugin-venv-v1"
        python3 -c 'import platform, sys; print(platform.python_implementation(), sys.version)'
        printf 'system_site_packages=%s\n' "${use_system_site_packages}"
        printf 'trt_source=%s\n' "${trt_source}"
        for requirement in "${requirements[@]}"; do
            sha256sum "${requirement}"
        done
    } | sha256sum | cut -d' ' -f1
)
venv_dir="${cache_root}/${cache_key}"
ready_file="${venv_dir}/.ready"
lock_file="${cache_root}/${cache_key}.lock"

exec {cache_lock_fd}>"${lock_file}"
flock "${cache_lock_fd}"
if [[ -f "${ready_file}" ]] \
        && "${venv_dir}/bin/python3" -c 'import pytest, tensorrt, torch' >/dev/null 2>&1; then
    echo "Reusing cached Python plugin-test environment ${venv_dir}"
else
    echo "Creating cached Python plugin-test environment ${venv_dir}"
    rm -rf "${venv_dir}"
    python3 -m venv "${venv_args[@]}" "${venv_dir}"
    bash "${ci_run}" "${pip_timeout_seconds}" "Upgrade pip" -- \
        "${venv_dir}/bin/python3" -m pip install --upgrade pip
    install_args=()
    for requirement in "${requirements[@]}"; do
        install_args+=(-r "${requirement}")
    done
    bash "${ci_run}" "${pip_timeout_seconds}" "Install plugin-test dependencies" -- \
        "${venv_dir}/bin/python3" -m pip install "${install_args[@]}"

    if ! "${venv_dir}/bin/python3" -c 'import tensorrt' >/dev/null 2>&1; then
        if [[ -n "${trt_wheel}" ]]; then
            bash "${ci_run}" "${pip_timeout_seconds}" "Install TensorRT Python package" -- \
                "${venv_dir}/bin/python3" -m pip install "${trt_wheel}"
        else
            bash "${ci_run}" "${pip_timeout_seconds}" "Install TensorRT Python bindings" -- \
                "${venv_dir}/bin/python3" -m pip install --no-deps \
                "tensorrt_cu13==${TRT_PYTHON_VERSION}" \
                "tensorrt_cu13_bindings==${TRT_PYTHON_VERSION}"
        fi
    fi
    "${venv_dir}/bin/python3" -c 'import pytest, tensorrt, torch'
    printf '%s\n' "${cache_key}" > "${ready_file}"
fi
flock -u "${cache_lock_fd}"
exec {cache_lock_fd}>&-

# shellcheck disable=SC1091
source "${venv_dir}/bin/activate"

export LLM_SDK_DIR="${REMOTE_WORKSPACE}"
cpp_unit_build_dir="${CPP_UNIT_BUILD_DIR:-${REMOTE_WORKSPACE}/build-cpp-unit}"
export EDGELLM_PLUGIN_LIB="${EDGELLM_PLUGIN_LIB:-${cpp_unit_build_dir}/libNvInfer_edgellm_plugin.so}"
mkdir -p "${report_dir}"

test -f "${EDGELLM_PLUGIN_LIB}"
grep -qx 'ENABLE_CUTE_DSL:STRING=ALL' "${cpp_unit_build_dir}/CMakeCache.txt"
if [[ -n "${TRT_PYTHON_VERSION:-}" ]]; then
    python3 -c 'import os, tensorrt; assert tensorrt.__version__ == os.environ["TRT_PYTHON_VERSION"]'
fi
python3 -c "import tensorrt, torch; assert torch.cuda.is_available(), 'CUDA not available'"

echo "Running Python plugin tests with pytest. Time:$(date)"
bash "${ci_run}" "${job_phase_timeout_seconds}" "Run Python plugin tests" -- \
    python3 -m pytest tests/python-unittests/*plugin*.py \
    --priority="${PRIORITY}" -v --maxfail=1 --color=yes \
    --html="${report_dir}/test_report_${report_name}.html" --self-contained-html \
    --junit-prefix="${JUNIT_PREFIX}" \
    --junitxml="${report_dir}/test_report_${report_name}.xml"
