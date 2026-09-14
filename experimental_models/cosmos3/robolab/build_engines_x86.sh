#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unattended x86 build of the Cosmos3 Policy-DROID runtime and its TensorRT
# engines, sized for the RoboLab serving contract (chunk 32 x raw dim 8,
# fps 15, one clean state row). Runs on a single datacenter GPU inside a
# CUDA 13 container and stores every reusable artifact under $WORK_ROOT so a
# later serving replica only has to load the engines.
#
# The script is idempotent: an existing $WORK_ROOT/engines/READY short-circuits
# the export and engine build. Concurrent replicas serialize on flocks.
#
#   EDGELLM_SRC   checkout to build (default: this repository)
#   WORK_ROOT     artifact root, normally on shared storage
#   CKPT_REPO     Hugging Face checkpoint id
#   CKPT_LOCAL    local checkpoint directory (shared HF cache by default)
set -euxo pipefail

EDGELLM_SRC="${EDGELLM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/mnt/cosmos-eval/edgellm-b}"
CKPT_REPO="${CKPT_REPO:-nvidia/Cosmos3-Edge-Policy-DROID}"
CKPT_LOCAL="${CKPT_LOCAL:-/mnt/cosmos-eval/hf-cache/local/nvidia--Cosmos3-Edge-Policy-DROID}"
BUILD_DIR="${BUILD_DIR:-/tmp/edgellm-build}"
ONNX_DIR="${WORK_ROOT}/onnx"
ENGINE_DIR="${WORK_ROOT}/engines"

# Served contract. These are not defaults in the exporter: the canonical
# request is 16 x 10 at fps 5, which is not Policy-DROID.
ACTION_CHUNK_SIZE=32
NUM_FRAMES=33
FPS=15

mkdir -p "${WORK_ROOT}" "${ONNX_DIR}" "${ENGINE_DIR}" "${BUILD_DIR}"
NATIVE_DIR="${WORK_ROOT}/native"

# The RoboLab policy-server image's python is only in /workspace/.venv and
# has no pip. Keep that interpreter; install deps with uv (does not need pip).
# Do not strip the venv from PATH: there is no /usr/bin/python3 in this image
# (v686 died immediately with PY=).
if [ -x /workspace/.venv/bin/python3 ]; then
    PY=/workspace/.venv/bin/python3
else
    PY="$(command -v python3)"
fi
echo "python: ${PY} ($(${PY} -V 2>&1))"

py_install() {
    uv pip install --python "${PY}" --no-cache-dir "$@"
}

echo "=== stage: toolchain ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build \
    libnvinfer-dev libnvonnxparsers-dev

# TensorRT reports capability as "9.0"; artifact tags and nvcc want "90".
SM="${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
echo "target SM: ${SM}"

echo "=== stage: cute dsl kernels ==="
# The attention plugins need FMHA. Prefer a checked-in prebuilt archive for this
# SM; generate it on the target when absent. A build without CuTe DSL still
# links, so fall back rather than abort and lose the GPU slot.
CUTE_MODE=fmha
CUTE_ARGS=(-DENABLE_CUTE_DSL=fmha "-DCUTE_DSL_ARTIFACT_TAG=sm_${SM}")
if [ ! -e "${EDGELLM_SRC}/kernelSrcs/cuteDSLPrebuilt/cutedsl_x86_64_sm_${SM}_cuda13.tar.gz" ] \
   && [ ! -d "${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact/x86_64/sm_${SM}" ]; then
    if py_install "nvidia-cutlass-dsl[cu13]==4.7.0" cupy-cuda13x cuda-python \
       && "${PY}" "${EDGELLM_SRC}/kernelSrcs/build_cutedsl.py" \
            --kernels fmha --gpu_arch "sm_${SM}" --arch x86_64; then
        echo "generated CuTe DSL fmha artifacts for sm_${SM}"
    else
        echo "WARNING: no CuTe DSL artifacts for sm_${SM}; configuring with ENABLE_CUTE_DSL=OFF"
        CUTE_MODE=off
        CUTE_ARGS=(-DENABLE_CUTE_DSL=OFF)
    fi
fi

POLICY_BUILD="${NATIVE_DIR}/cosmos3_policy_build"
POLICY_INFER="${NATIVE_DIR}/cosmos3_policy_inference"
PLUGIN_SO="${NATIVE_DIR}/libNvInfer_edgellm_plugin.so"
if [ -x "${POLICY_BUILD}" ] && [ -x "${POLICY_INFER}" ] && [ -e "${PLUGIN_SO}" ] \
   && [ "$(cat "${NATIVE_DIR}/SM" 2>/dev/null || true)" = "${SM}" ]; then
    echo "=== reusing native binaries from ${NATIVE_DIR} ==="
else
    echo "=== stage: c++ build (cute_dsl=${CUTE_MODE}) ==="
    cmake -S "${EDGELLM_SRC}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCUDA_CTK_VERSION=13.0 \
        -DCMAKE_CUDA_ARCHITECTURES="${SM}" \
        -DBUILD_EXPERIMENTAL_MODELS=ON \
        "${CUTE_ARGS[@]}"
    cmake --build "${BUILD_DIR}" --parallel "$(nproc)" \
        --target NvInfer_edgellm_plugin cosmos3_policy_build cosmos3_policy_inference
    mkdir -p "${NATIVE_DIR}"
    cp -f "${BUILD_DIR}/experimental_models/cosmos3/examples/cosmos3_policy_build" "${POLICY_BUILD}"
    cp -f "${BUILD_DIR}/experimental_models/cosmos3/examples/cosmos3_policy_inference" "${POLICY_INFER}"
    cp -f "${BUILD_DIR}/libNvInfer_edgellm_plugin.so" "${PLUGIN_SO}"
    printf '%s\n' "${SM}" > "${NATIVE_DIR}/SM"
    chmod +x "${POLICY_BUILD}" "${POLICY_INFER}"
fi
export EDGELLM_PLUGIN_PATH="${PLUGIN_SO}"

if [ -e "${ENGINE_DIR}/READY" ]; then
    echo "=== engines already present, skipping export/build ==="
else
    echo "=== stage: checkpoint ==="
    mkdir -p "${CKPT_LOCAL}"
    # The A-arm endpoint populates the same shared cache; take the same lock.
    exec 9>"${WORK_ROOT}/checkpoint.lock"
    flock 9
    uvx --with click hf@1.16.4 download "${CKPT_REPO}" \
        --repo-type model --revision main --local-dir "${CKPT_LOCAL}"
    flock -u 9

    echo "=== stage: python package ==="
    # Do not `pip install -e`: scikit-build would rebuild C++, and the image
    # venv has no pip. Export only needs the in-tree package plus extras.
    if ! "${PY}" -c "import torch, transformers, onnx, onnxscript, safetensors, numpy, onnx_graphsurgeon, PIL"; then
        py_install \
            "torch==2.13.0" "transformers==5.14.1" "onnx==1.19.0" \
            "onnxscript==0.7.1" "safetensors==0.8.0" "numpy==2.2.6" \
            "onnx-graphsurgeon==0.6.1" pillow
    fi
    export PYTHONPATH="${EDGELLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

    echo "=== stage: onnx export ==="
    exec 8>"${WORK_ROOT}/build.lock"
    flock 8
    PYTHONNOUSERSITE=1 "${PY}" -m tensorrt_edgellm.scripts.export \
        "${CKPT_LOCAL}" "${ONNX_DIR}" \
        --task policy --dtype float16 \
        --action-chunk-size "${ACTION_CHUNK_SIZE}" \
        --num-frames "${NUM_FRAMES}" \
        --fps "${FPS}"

    echo "=== stage: engine build ==="
    "${POLICY_BUILD}" --onnxDir "${ONNX_DIR}" --engineDir "${ENGINE_DIR}"
    touch "${ENGINE_DIR}/READY"
    flock -u 8
fi

echo "=== stage: contract check ==="
CHECK_DIR="$(mktemp -d)"
"${PY}" - "${CHECK_DIR}/observation.png" <<'PY'
import sys

import numpy as np
from PIL import Image

# RoboLab hands the policy a 640x540 concat view; the runtime resizes it.
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, size=(540, 640, 3), dtype=np.uint8)
Image.fromarray(frame).save(sys.argv[1])
PY

"${POLICY_INFER}" \
    --engineDir "${ENGINE_DIR}" \
    --image "${CHECK_DIR}/observation.png" \
    --prompt "Pick up the mustard bottle and place it in the right bin." \
    --viewPoint concat_view \
    --guidance 3.0 \
    --steps 4 \
    --state "0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,1.0" \
    --output "${CHECK_DIR}/action.json"

"${PY}" - "${CHECK_DIR}/action.json" <<'PY'
import json
import sys

import numpy as np

payload = json.load(open(sys.argv[1]))
action = np.asarray(payload["action"], dtype=np.float32).reshape(-1, 8)
assert action.shape == (32, 8), f"expected (32, 8), got {action.shape}"
assert np.isfinite(action).all(), "action chunk contains non-finite values"
print("contract ok:", action.shape, "range", float(action.min()), float(action.max()))
PY

# A missing state must fail loudly rather than silently serving a stateless
# chunk, which is what produced the earlier 16 x 10 mismatch.
if "${POLICY_INFER}" \
    --engineDir "${ENGINE_DIR}" \
    --image "${CHECK_DIR}/observation.png" \
    --prompt "Pick up the mustard bottle and place it in the right bin." \
    --output "${CHECK_DIR}/stateless.json" 2>"${CHECK_DIR}/stateless.err"; then
    echo "ERROR: stateless request unexpectedly succeeded" >&2
    exit 1
fi
echo "stateless request rejected as expected"

cp -f "${CHECK_DIR}/action.json" "${WORK_ROOT}/contract-action.json"
echo "engines: ${ENGINE_DIR}"
echo "B_ENGINE_BUILD_OK"
