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

# Native trt::RotaryEmbedding / Attention need TensorRT >= 10.16. Staging
# 10.16 on neb-cdg (driver 570 / CUDA 12.8) parses then segfaults in the
# compiler backend — both +cuda13.2 and +cuda12.9, even with a local copy.
# Default: decompose those ops at export and build with the image 10.14.
# Do not call the staged version TRT_VERSION: NGC images already export that.
EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN="${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN:-1}"
export EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN
ONNX_MODE="policy-32x8-decomp${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN}"
EDGELLM_TRT_VERSION="${EDGELLM_TRT_VERSION:-10.16.1.11-1+cuda12.9}"
EDGELLM_TRT_REPO="${EDGELLM_TRT_REPO:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64}"
EDGELLM_USE_IMAGE_TRT="${EDGELLM_USE_IMAGE_TRT:-}"
if [ -z "${EDGELLM_USE_IMAGE_TRT}" ]; then
    if [ "${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN}" = "1" ]; then
        EDGELLM_USE_IMAGE_TRT=1
    else
        EDGELLM_USE_IMAGE_TRT=0
    fi
fi

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

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

py_install() {
    # NGC images expose the interpreter as the distro python3, which carries
    # Debian's EXTERNALLY-MANAGED marker; there is no venv to install into.
    uv pip install --python "${PY}" --no-cache-dir "$@" \
        || uv pip install --python "${PY}" --no-cache-dir --break-system-packages "$@"
}

echo "=== stage: toolchain ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build
nvidia-smi -L || true
nvcc --version | head -6 || true

echo "=== stage: tensorrt (image=${EDGELLM_USE_IMAGE_TRT} decompose=${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN}) ==="
# Decomposed export: stay on pytorch:25.12 image TRT 10.14.1 (matches driver).
# Native-op export: unpack 10.16 side by side. Do not apt-upgrade (cudaError 35).
TRT_STAGE="${WORK_ROOT}/tensorrt/${EDGELLM_TRT_VERSION}"
TRT_LOCAL="${TRT_LOCAL:-/opt/edgellm-trt/${EDGELLM_TRT_VERSION}}"
if [ "${EDGELLM_USE_IMAGE_TRT}" = "1" ]; then
    TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"
    echo "using image TensorRT at ${TRT_PACKAGE_DIR}"
else
    exec 7>"${WORK_ROOT}/tensorrt.lock"
    flock 7
    if [ ! -e "${TRT_STAGE}/READY" ]; then
        rm -rf "${TRT_STAGE}"
        mkdir -p "${TRT_STAGE}"
        for pkg in libnvinfer-headers-dev libnvinfer-headers-plugin-dev \
                   libnvinfer-dev libnvinfer10 \
                   libnvinfer-plugin-dev libnvinfer-plugin10 \
                   libnvonnxparsers-dev libnvonnxparsers10; do
            deb="${pkg}_${EDGELLM_TRT_VERSION}_amd64.deb"
            curl -fsSL -o "${BUILD_DIR}/${deb}" "${EDGELLM_TRT_REPO}/${deb}"
            dpkg-deb -x "${BUILD_DIR}/${deb}" "${TRT_STAGE}"
        done
        touch "${TRT_STAGE}/READY"
    fi
    flock -u 7
    if [ ! -e "${TRT_LOCAL}/READY" ]; then
        rm -rf "${TRT_LOCAL}"
        mkdir -p "${TRT_LOCAL}"
        tar -C "${TRT_STAGE}" --exclude='*.a' -cf - . | tar -C "${TRT_LOCAL}" -xf -
    fi
    TRT_PACKAGE_DIR="${TRT_LOCAL}/usr"
    export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib/x86_64-linux-gnu:${TRT_STAGE}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
ls -l "${TRT_PACKAGE_DIR}/lib/x86_64-linux-gnu/libnvinfer.so".* \
      "${TRT_PACKAGE_DIR}/lib/x86_64-linux-gnu/libnvonnxparser.so".* \
      2>/dev/null || ls -l "${TRT_PACKAGE_DIR}/lib/"*nvinfer* 2>/dev/null || true

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
if [ "${EDGELLM_USE_IMAGE_TRT}" = "1" ]; then
    NATIVE_KEY="sm${SM}-trtIMAGE-cute${CUTE_MODE}-decomp${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN}"
else
    NATIVE_KEY="sm${SM}-trt${EDGELLM_TRT_VERSION}-cute${CUTE_MODE}-decomp${EDGELLM_COSMOS3_DECOMPOSE_NATIVE_ATTN}"
fi
if [ -x "${POLICY_BUILD}" ] && [ -x "${POLICY_INFER}" ] && [ -e "${PLUGIN_SO}" ] \
   && [ "$(cat "${NATIVE_DIR}/KEY" 2>/dev/null || true)" = "${NATIVE_KEY}" ]; then
    echo "=== reusing native binaries from ${NATIVE_DIR} ==="
else
    echo "=== stage: c++ build (cute_dsl=${CUTE_MODE}) ==="
    CUDA_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
    if [ -z "${CUDA_DIR}" ] || [ ! -e "${CUDA_DIR}/include/cuda_runtime_api.h" ]; then
        CUDA_DIR="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    fi
    CUDA_CTK_VERSION="$(echo "${CUDA_DIR}" | sed -n 's/.*cuda-\([0-9][0-9.]*\).*/\1/p')"
    CUDA_CTK_VERSION="${CUDA_CTK_VERSION:-13.0}"
    echo "CUDA_DIR=${CUDA_DIR} CUDA_CTK_VERSION=${CUDA_CTK_VERSION} TRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"
    rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
    cmake -S "${EDGELLM_SRC}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
        -DCUDA_DIR="${CUDA_DIR}" \
        -DCUDA_CTK_VERSION="${CUDA_CTK_VERSION}" \
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
    printf '%s\n' "${NATIVE_KEY}" > "${NATIVE_DIR}/KEY"
    chmod +x "${POLICY_BUILD}" "${POLICY_INFER}"
fi
export EDGELLM_PLUGIN_PATH="${PLUGIN_SO}"

if [ "$(cat "${ONNX_DIR}/EXPORT_MODE" 2>/dev/null || true)" != "${ONNX_MODE}" ]; then
    echo "=== onnx mode changed to ${ONNX_MODE}; invalidating old ONNX/engines ==="
    rm -f "${ONNX_DIR}/EXPORT_OK" "${ENGINE_DIR}/READY"
fi

if [ -e "${ENGINE_DIR}/READY" ]; then
    echo "=== engines already present, skipping export/build ==="
else
    echo "=== stage: checkpoint ==="
    mkdir -p "${CKPT_LOCAL}"
    exec 9>"${WORK_ROOT}/checkpoint.lock"
    flock 9
    uvx --with click hf@1.16.4 download "${CKPT_REPO}" \
        --repo-type model --revision main --local-dir "${CKPT_LOCAL}"
    flock -u 9

    exec 8>"${WORK_ROOT}/build.lock"
    flock 8
    if [ -e "${ONNX_DIR}/EXPORT_OK" ] \
       && [ "$(cat "${ONNX_DIR}/EXPORT_MODE" 2>/dev/null || true)" = "${ONNX_MODE}" ]; then
        echo "=== onnx already present, skipping export ==="
    else
        echo "=== stage: python package ==="
        "${PY}" -c "import torch" || py_install "torch==2.13.0"
        py_install \
            "transformers==5.14.1" "onnx==1.19.0" "onnxscript==0.7.1" \
            "safetensors==0.8.0" "numpy==2.2.6" "onnx-graphsurgeon==0.6.1" pillow
        export PYTHONPATH="${EDGELLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
        "${PY}" -c "from transformers import Gemma4AudioConfig; import tensorrt_edgellm.scripts.export"

        echo "=== stage: onnx export (mode=${ONNX_MODE}) ==="
        PYTHONNOUSERSITE=1 "${PY}" -m tensorrt_edgellm.scripts.export \
            "${CKPT_LOCAL}" "${ONNX_DIR}" \
            --task policy --dtype float16 \
            --action-chunk-size "${ACTION_CHUNK_SIZE}" \
            --num-frames "${NUM_FRAMES}" \
            --fps "${FPS}"
        printf '%s\n' "${ONNX_MODE}" > "${ONNX_DIR}/EXPORT_MODE"
        touch "${ONNX_DIR}/EXPORT_OK"
    fi

    echo "=== stage: engine build ==="
    # Torch's bundled CUDA/TRT must not win over the builder's libnvinfer.
    _engine_ld=""
    IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
    for _p in "${_ld_parts[@]}"; do
        case "${_p}" in
            *torch*|*torch_tensorrt*) continue ;;
        esac
        _engine_ld="${_engine_ld:+${_engine_ld}:}${_p}"
    done
    export LD_LIBRARY_PATH="${_engine_ld}"
    ldd "${POLICY_BUILD}" | grep -E "nvinfer|nvonnx|cudart|cuda" || true
    ldd "${PLUGIN_SO}" | grep -E "nvinfer|nvonnx|cudart|cuda" || true
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
