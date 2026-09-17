#!/usr/bin/env bash
# Bring up Cosmos3-Edge Reasoner OpenAI server on a Lepton GPU node.
# Expects a source checkout at EDGELLM_SRC (cloned by the endpoint command).
set -euxo pipefail

EDGELLM_SRC="${EDGELLM_SRC:-/tmp/edgellm}"
WORK_ROOT="${WORK_ROOT:-/mnt/cosmos-eval/edgellm-reasoner-1643}"
HF_HOME="${HF_HOME:-/mnt/cosmos-eval/hf-cache}"
BUILD_DIR="${BUILD_DIR:-/tmp/edgellm-build}"
CACHE_DIR="${CACHE_DIR:-${WORK_ROOT}/serve-cache}"
REASONING_CHECKPOINT="${REASONING_CHECKPOINT:-nvidia/Cosmos3-Edge}"
export HF_HOME
mkdir -p "${WORK_ROOT}" "${CACHE_DIR}" "${BUILD_DIR}"

echo REASONER_SERVE_STARTING
git -C "${EDGELLM_SRC}" log -1 --oneline
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nvcc --version | tail -3

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build python3-dev pkg-config

if [ -x /workspace/.venv/bin/python3 ]; then
  PY=/workspace/.venv/bin/python3
else
  PY="$(command -v python3)"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv pip install --python "${PY}" --no-cache-dir \
    pybind11==3.0.4 fastapi==0.139.2 uvicorn==0.51.0 \
    'huggingface-hub>=1.5.0,<2.0' av==17.1.0 python-multipart==0.0.32 \
    'numpy==2.2.6' || \
uv pip install --python "${PY}" --no-cache-dir --break-system-packages \
    pybind11==3.0.4 fastapi==0.139.2 uvicorn==0.51.0 \
    'huggingface-hub>=1.5.0,<2.0' av==17.1.0 python-multipart==0.0.32 \
    'numpy==2.2.6'

SM="${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
echo "target SM: ${SM}"

# Prefer image TensorRT (pytorch:25.12-py3 = 10.14). Reasoner is not the
# Policy native-op graph; fall back to already-staged 10.16 on NFS if headers
# are missing. Do not write into the Policy WORK_ROOT.
if [ -f /usr/include/NvInfer.h ]; then
  TRT_PACKAGE_DIR=/usr
elif [ -d /usr/include/x86_64-linux-gnu ] && ls /usr/include/x86_64-linux-gnu/NvInfer.h >/dev/null 2>&1; then
  TRT_PACKAGE_DIR=/usr
else
  TRT_STAGE_CANDIDATE="${TRT_STAGE:-/mnt/cosmos-eval/edgellm-b/tensorrt/10.16.1.11-1+cuda12.9}"
  TRT_LOCAL="/opt/edgellm-trt-reasoner"
  if [ -e "${TRT_STAGE_CANDIDATE}/READY" ] && [ ! -e "${TRT_LOCAL}/READY" ]; then
    mkdir -p "${TRT_LOCAL}"
    tar -C "${TRT_STAGE_CANDIDATE}" --exclude='*.a' -cf - . | tar -C "${TRT_LOCAL}" -xf -
    touch "${TRT_LOCAL}/READY"
  fi
  TRT_PACKAGE_DIR="${TRT_LOCAL}/usr"
  export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
test -f "${TRT_PACKAGE_DIR}/include/NvInfer.h" -o -f /usr/include/NvInfer.h

CUTE_ARGS=(-DENABLE_CUTE_DSL=fmha "-DCUTE_DSL_ARTIFACT_TAG=sm_${SM}")
if [ ! -d "${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact/x86_64/sm_${SM}" ] \
   && [ ! -e "${EDGELLM_SRC}/kernelSrcs/cuteDSLPrebuilt/cutedsl_x86_64_sm_${SM}_cuda13.tar.gz" ]; then
  if uv pip install --python "${PY}" --no-cache-dir 'nvidia-cutlass-dsl[cu13]==4.7.0' cupy-cuda13x cuda-python \
     && "${PY}" "${EDGELLM_SRC}/kernelSrcs/build_cutedsl.py" --kernels fmha --gpu_arch "sm_${SM}" --arch x86_64; then
    echo "generated CuTe DSL fmha artifacts for sm_${SM}"
  else
    echo "WARNING: CuTe generate failed; ENABLE_CUTE_DSL=OFF"
    CUTE_ARGS=(-DENABLE_CUTE_DSL=OFF)
  fi
fi

CUDA_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
if [ -z "${CUDA_DIR}" ] || [ ! -e "${CUDA_DIR}/include/cuda_runtime_api.h" ]; then
  CUDA_DIR="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
CUDA_CTK_VERSION="$(echo "${CUDA_DIR}" | sed -n 's/.*cuda-\([0-9][0-9.]*\).*/\1/p')"
CUDA_CTK_VERSION="${CUDA_CTK_VERSION:-13.1}"
PYBIND_DIR="$("${PY}" -m pybind11 --cmakedir)"

echo "CUDA_DIR=${CUDA_DIR} CUDA_CTK_VERSION=${CUDA_CTK_VERSION} TRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"
cmake -S "${EDGELLM_SRC}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
    -DCUDA_DIR="${CUDA_DIR}" \
    -DCUDA_CTK_VERSION="${CUDA_CTK_VERSION}" \
    -DCMAKE_CUDA_ARCHITECTURES="${SM}" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -Dpybind11_DIR="${PYBIND_DIR}" \
    "${CUTE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel 16 --target _edgellm_runtime NvInfer_edgellm_plugin

SO_DIR="$(dirname "$(find "${BUILD_DIR}" -name '*_edgellm_runtime*.so' | head -1)")"
export PYTHONPATH="${SO_DIR}:${EDGELLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export EDGELLM_PLUGIN_PATH="$(find "${BUILD_DIR}" -name 'libNvInfer_edgellm_plugin.so' | head -1)"
"${PY}" -c "import _edgellm_runtime, experimental.server.cli; print('PYBIND_OK', _edgellm_runtime.__file__)"
echo REASONER_PYBIND_OK

echo OPENAI_SERVE_STARTING
exec "${PY}" -m experimental.server.cli \
    "${REASONING_CHECKPOINT}" \
    --host 0.0.0.0 \
    --port 8000 \
    --cache-dir "${CACHE_DIR}" \
    --max-image-tokens 4096 \
    --max-image-tokens-per-image 4096
