#!/bin/bash
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
#
# Runs the plugin Python unit tests ON an aarch64 board (Thor/Orin). Unlike the
# pipeline tests, these drive the TensorRT Python API + torch CUDA directly, so
# the pytest process must run where the GPU is. Executed over ssh with:
#   REMOTE_WORKSPACE=... TRT_PACKAGE_DIR=... PRIORITY=... bash -s < this-script
set -euo pipefail

: "${REMOTE_WORKSPACE:?REMOTE_WORKSPACE must be set}"
: "${TRT_PACKAGE_DIR:?TRT_PACKAGE_DIR must be set}"
: "${PRIORITY:?PRIORITY must be set}"

cd "$REMOTE_WORKSPACE"

echo "Setting up python environment on $(hostname)"
python3 -m venv ut_venv
# shellcheck disable=SC1091
source ut_venv/bin/activate
pip3 install -q --upgrade pip
pip3 install -q \
    -r tests/requirements.txt \
    -r tests/requirements-ut.txt

echo "Installing TensorRT python package from $TRT_PACKAGE_DIR"
# Pick the wheel matching this board's python ABI (the package ships several
# cp3XX wheels); fall back to whatever is there so the error names the wheel.
PY_TAG="cp$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
TRT_WHL=$(ls "$TRT_PACKAGE_DIR"/python/tensorrt-*"$PY_TAG"*aarch64*.whl 2>/dev/null | head -n 1)
if [ -z "$TRT_WHL" ]; then
    echo "No $PY_TAG TensorRT wheel found, available wheels:"
    ls "$TRT_PACKAGE_DIR"/python/
    TRT_WHL=$(ls "$TRT_PACKAGE_DIR"/python/tensorrt-*aarch64*.whl | head -n 1)
fi
pip3 install -q "$TRT_WHL"

python3 -c "import torch; assert torch.cuda.is_available(), 'torch has no CUDA support on this board'"

export LLM_SDK_DIR="$REMOTE_WORKSPACE"
export ONNX_DIR="$REMOTE_WORKSPACE/onnx"
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ONNX_DIR" logs

# test_build_project_with_pybind (l0_python_ut list) builds _edgellm_runtime.
export PYTHONPATH="$REMOTE_WORKSPACE/build/pybind${PYTHONPATH:+:$PYTHONPATH}"

echo "Running unit tests with pytest. Time:$(date)"
# Test selection is driven by the tests/test_lists/$PRIORITY.yml list.
python3 -m pytest tests/ --priority="$PRIORITY" -v --color=yes \
    --html="logs/test_report_$PRIORITY.html" --self-contained-html \
    --junitxml="logs/test_report_$PRIORITY.xml"
