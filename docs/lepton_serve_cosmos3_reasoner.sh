#!/usr/bin/env bash
# Bring up Cosmos3-Edge Reasoner OpenAI server on a Lepton GPU node.
# Expects a source checkout at EDGELLM_SRC (cloned by the endpoint command).
# Keep the Lepton GPU if a step fails. Bare `set -e` + exit releases the replica.
set -euo pipefail
hold_gpu() {
  echo "REASONER_HOLDING_GPU status=${1:-unknown}"
  sleep infinity
}
trap 'hold_gpu ERR' ERR

EDGELLM_SRC="${EDGELLM_SRC:-/tmp/edgellm}"
WORK_ROOT="${WORK_ROOT:-/mnt/cosmos-eval/edgellm-reasoner-1643}"
HF_HOME="${HF_HOME:-/mnt/cosmos-eval/hf-cache}"
BUILD_DIR="${BUILD_DIR:-/tmp/edgellm-build}"
CACHE_DIR="${CACHE_DIR:-${WORK_ROOT}/serve-cache}"
REASONING_CHECKPOINT="${REASONING_CHECKPOINT:-nvidia/Cosmos3-Edge}"
# Do not name this TRT_VERSION: NGC images already export that (10.14.1.48+cuda13.0).
EDGELLM_TRT_REPO="${EDGELLM_TRT_REPO:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64}"
EDGELLM_TRT_FALLBACK="${EDGELLM_TRT_FALLBACK:-10.16.1.11-1+cuda12.9}"
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
    'numpy==2.2.6' cuda-python || \
uv pip install --python "${PY}" --no-cache-dir --break-system-packages \
    pybind11==3.0.4 fastapi==0.139.2 uvicorn==0.51.0 \
    'huggingface-hub>=1.5.0,<2.0' av==17.1.0 python-multipart==0.0.32 \
    'numpy==2.2.6' cuda-python

SM="${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
echo "target SM: ${SM}"

# pytorch:25.12-py3 ships TensorRT 10.14 *runtime*, not NvInfer.h. Reasoner
# wants that 10.14 path (AttentionPlugin / rope-as-input), not Policy's native
# RotaryEmbedding graph. Never apt-get libnvinfer-dev (pulls CUDA 13.4).
# Unpack matching header/dev debs, reuse a staged NFS tree, or download 10.16.
locate_nvinfer_h() {
  local p root="${1:-}"
  local candidates=(
    /usr/include/NvInfer.h
    /usr/include/x86_64-linux-gnu/NvInfer.h
    /usr/local/include/NvInfer.h
  )
  if [ -n "${root}" ]; then
    candidates+=(
      "${root}/include/NvInfer.h"
      "${root}/include/x86_64-linux-gnu/NvInfer.h"
      "${root}/usr/include/NvInfer.h"
      "${root}/usr/include/x86_64-linux-gnu/NvInfer.h"
    )
  fi
  for p in "${candidates[@]}"; do
    if [ -f "${p}" ]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  return 1
}

unpack_trt_debs() {
  local ver="$1" dest="$2"
  mkdir -p "${dest}" "${BUILD_DIR}"
  local pkg deb
  for pkg in libnvinfer-headers-dev libnvinfer-headers-plugin-dev \
             libnvinfer-dev libnvinfer10 \
             libnvinfer-plugin-dev libnvinfer-plugin10 \
             libnvonnxparsers-dev libnvonnxparsers10; do
    deb="${pkg}_${ver}_amd64.deb"
    echo "TRT_DEB fetching ${deb}"
    curl -fsSL -o "${BUILD_DIR}/${deb}" "${EDGELLM_TRT_REPO}/${deb}"
    dpkg-deb -x "${BUILD_DIR}/${deb}" "${dest}"
  done
  touch "${dest}/READY"
}

unpack_trt_python_debs() {
  local ver="$1" dest="$2"
  local pkg deb
  # The existing Policy stage already has the 2.4 GB libnvinfer-dev package.
  # Only these small packages are missing for the checkpoint-direct Python
  # builder (roughly 1 MB total).
  for pkg in libnvinfer-vc-plugin10 python3-libnvinfer; do
    deb="${pkg}_${ver}_amd64.deb"
    echo "TRT_PY_DEB fetching ${deb}"
    curl -fsSL -o "${BUILD_DIR}/${deb}" "${EDGELLM_TRT_REPO}/${deb}"
    dpkg-deb -x "${BUILD_DIR}/${deb}" "${dest}"
  done
  touch "${dest}/READY_FULL"
}

sync_trt_local() {
  local src="$1" dest="$2"
  mkdir -p "${dest}"
  tar -C "${src}" --exclude='*.a' -cf - . | tar -C "${dest}" -xf -
  touch "${dest}/READY"
}

TRT_LOCAL="${TRT_LOCAL:-/opt/edgellm-trt-reasoner}"
# The checkpoint-direct builder uses INetworkDefinition.add_rotary_embedding,
# introduced in TensorRT 10.16. The image's 10.14 Python binding cannot expose
# that method even though its C++ headers are present. Stage both native libs
# and python3-libnvinfer 10.16 side-by-side in this Reasoner-owned WORK_ROOT.
STAGED="${WORK_ROOT}/tensorrt/${EDGELLM_TRT_FALLBACK}"
mkdir -p "${WORK_ROOT}/tensorrt"
exec 7>"${WORK_ROOT}/tensorrt.lock"
flock 7
if [ ! -e "${STAGED}/READY_FULL" ]; then
  mkdir -p "${STAGED}"
  POLICY_STAGE="/mnt/cosmos-eval/edgellm-b/tensorrt/${EDGELLM_TRT_FALLBACK}"
  if [ ! -e "${STAGED}/READY" ] && [ -e "${POLICY_STAGE}/READY" ]; then
    echo "TRT_SEED_FROM_POLICY_STAGE ${POLICY_STAGE}"
    tar -C "${POLICY_STAGE}" --exclude='*.a' -cf - . | tar -C "${STAGED}" -xf -
  fi
  if [ ! -e "${STAGED}/READY" ]; then
    unpack_trt_debs "${EDGELLM_TRT_FALLBACK}" "${STAGED}"
  fi
  unpack_trt_python_debs "${EDGELLM_TRT_FALLBACK}" "${STAGED}"
fi
flock -u 7

rm -rf "${TRT_LOCAL}"
echo "TRT_COPY_LOCAL ${STAGED} -> ${TRT_LOCAL}"
sync_trt_local "${STAGED}" "${TRT_LOCAL}"
TRT_PACKAGE_DIR="${TRT_LOCAL}/usr"
export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TRT_PACKAGE_DIR}/lib/python3.12/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
NVIH="$(locate_nvinfer_h "${TRT_PACKAGE_DIR}" || locate_nvinfer_h "${TRT_LOCAL}" || true)"

if [ -z "${NVIH}" ]; then
  echo "FATAL: NvInfer.h still missing after staging"
  hold_gpu NO_NVINFER_H
fi
echo "using NvInfer.h=${NVIH} TRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"
"${PY}" - <<'PY'
import tensorrt as trt
print("TENSORRT_PYTHON", trt.__version__, trt.__file__)
assert hasattr(trt.INetworkDefinition, "add_rotary_embedding"), (
    "TensorRT Python binding lacks add_rotary_embedding"
)
PY

if [ ! -f "${EDGELLM_SRC}/3rdParty/xgrammar/include/xgrammar/xgrammar.h" ]; then
  echo "INIT_SUBMODULES xgrammar (shallow clone has no gitlinks)"
  git -C "${EDGELLM_SRC}" submodule update --init --recursive --depth 1
fi
if [ ! -f "${EDGELLM_SRC}/3rdParty/xgrammar/include/xgrammar/xgrammar.h" ]; then
  echo "FATAL: 3rdParty/xgrammar header still missing"
  hold_gpu NO_XGRAMMAR
fi

CUTE_ROOT="${WORK_ROOT}/cutedsl"
CUTE_ARTIFACT="${CUTE_ROOT}/x86_64/sm_${SM}"
mkdir -p "${CUTE_ROOT}" "$(dirname "${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact")"
rm -rf "${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact"
ln -s "${CUTE_ROOT}" "${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact"
CUTE_ARGS=(-DENABLE_CUTE_DSL=fmha "-DCUTE_DSL_ARTIFACT_TAG=sm_${SM}")
if [ ! -e "${CUTE_ARTIFACT}/metadata.json" ]; then
  echo "CUTE_FMHA_GENERATE sm_${SM}"
  # build_cutedsl.py pins CuPy exactly. An unpinned install selected 14.2.0,
  # failed its dependency check (requires 13.6.0), and silently disabled the
  # only ViTAttentionPlugin backend that supports Cosmos3's head size 72.
  uv pip install --python "${PY}" --no-cache-dir --break-system-packages \
      'nvidia-cutlass-dsl[cu13]==4.7.0' 'cupy-cuda13x==13.6.0' cuda-python
  "${PY}" "${EDGELLM_SRC}/kernelSrcs/build_cutedsl.py" \
      --kernels fmha --gpu_arch "sm_${SM}" --arch x86_64 \
      --output_dir "${CUTE_ROOT}" --clean
fi
if [ ! -s "${CUTE_ARTIFACT}/libcutedsl_x86_64.a" ] \
   || [ ! -s "${CUTE_ARTIFACT}/metadata.json" ]; then
  echo "FATAL: CuTe FMHA artifact missing for sm_${SM}"
  hold_gpu NO_CUTE_FMHA
fi
echo "CUTE_FMHA_READY ${CUTE_ARTIFACT}"

CUDA_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
if [ -z "${CUDA_DIR}" ] || [ ! -e "${CUDA_DIR}/include/cuda_runtime_api.h" ]; then
  CUDA_DIR="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
CUDA_CTK_VERSION="$(echo "${CUDA_DIR}" | sed -n 's/.*cuda-\([0-9][0-9.]*\).*/\1/p')"
CUDA_CTK_VERSION="${CUDA_CTK_VERSION:-13.1}"
PYBIND_DIR="$("${PY}" -m pybind11 --cmakedir)"

echo "CUDA_DIR=${CUDA_DIR} CUDA_CTK_VERSION=${CUDA_CTK_VERSION} TRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"
NATIVE_DIR="${WORK_ROOT}/native"
# Key only native build inputs. Serve-script/doc-only commits should not force
# another 368-object rebuild, while any C++/kernel/CMake change still does.
NATIVE_SOURCE_KEY="$(git -C "${EDGELLM_SRC}" ls-files -s \
    CMakeLists.txt cmake cpp kernelSrcs pybind | sha256sum | cut -d' ' -f1)"
PYBIND_KEY="sm${SM}-trt${EDGELLM_TRT_FALLBACK}-cutefmha-native${NATIVE_SOURCE_KEY}"
mkdir -p "${NATIVE_DIR}"
RUNTIME_SO="$(find "${NATIVE_DIR}" -name '*_edgellm_runtime*.so' -print -quit || true)"
PLUGIN_SO="$(find "${NATIVE_DIR}" -name 'libNvInfer_edgellm_plugin.so' -print -quit || true)"
if [ -n "${RUNTIME_SO}" ] && [ -n "${PLUGIN_SO}" ] \
   && [ "$(cat "${NATIVE_DIR}/PYBIND_KEY" 2>/dev/null || true)" = "${PYBIND_KEY}" ]; then
  echo "REUSING_PYBIND ${PYBIND_KEY}"
else
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
  RUNTIME_SO="$(find "${BUILD_DIR}" -name '*_edgellm_runtime*.so' -print -quit)"
  PLUGIN_SO="$(find "${BUILD_DIR}" -name 'libNvInfer_edgellm_plugin.so' -print -quit)"
  if [ -z "${RUNTIME_SO}" ] || [ -z "${PLUGIN_SO}" ]; then
    echo "FATAL: missing pybind or plugin so runtime=${RUNTIME_SO:-none} plugin=${PLUGIN_SO:-none}"
    hold_gpu NO_PYBIND_SO
  fi
  cp -f "${RUNTIME_SO}" "${PLUGIN_SO}" "${NATIVE_DIR}/"
  printf '%s\n' "${PYBIND_KEY}" > "${NATIVE_DIR}/PYBIND_KEY"
  RUNTIME_SO="$(find "${NATIVE_DIR}" -name '*_edgellm_runtime*.so' -print -quit)"
  PLUGIN_SO="$(find "${NATIVE_DIR}" -name 'libNvInfer_edgellm_plugin.so' -print -quit)"
fi
SO_DIR="$(dirname "${RUNTIME_SO}")"
# experimental.server._import_runtime does not honor top-level
# ``import _edgellm_runtime`` from PYTHONPATH. It looks for
# EDGELLM_PYBIND_DIR, BUILD_DIR/pybind, then repo build/pybind.
export EDGELLM_PYBIND_DIR="${SO_DIR}"
mkdir -p "${EDGELLM_SRC}/build/pybind"
ln -sfn "${RUNTIME_SO}" "${EDGELLM_SRC}/build/pybind/$(basename "${RUNTIME_SO}")"
ln -sfn "${PLUGIN_SO}" "${EDGELLM_SRC}/build/pybind/$(basename "${PLUGIN_SO}")"
export PYTHONPATH="${SO_DIR}:${EDGELLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export EDGELLM_PLUGIN_PATH="${PLUGIN_SO}"
"${PY}" -c "import _edgellm_runtime, experimental.server.cli; print('PYBIND_OK', _edgellm_runtime.__file__)"
cd "${EDGELLM_SRC}"
"${PY}" -c "from experimental.server.runtime.engine import _import_runtime; m=_import_runtime(); print('RUNTIME_IMPORT_OK', getattr(m, '__file__', m))"
echo REASONER_PYBIND_OK

echo OPENAI_SERVE_STARTING
# ``python -m experimental.server.cli`` only imports the module (no
# ``if __name__``) and exits 0 in ~400ms. The package entry is
# ``python -m experimental.server`` / ``experimental.server.cli:main``.
# Disable ERR trap around serve: a non-zero python exit must retry,
# not sleep-infinity the GPU.
while true; do
  trap - ERR
  set +e
  # Visual profile can emit 4096 image tokens. Default max_input_len=4096 then
  # 400s EDGELLM_INPUT_TOO_LONG on any HD eval image plus the text prompt
  # (BlinkDepth 2048px: prefill 4118). KV 16384 leaves room for eval
  # max_tokens=8192 after a full visual prefill.
  "${PY}" -m experimental.server \
    "${REASONING_CHECKPOINT}" \
    --host 0.0.0.0 \
    --port 8000 \
    --cache-dir "${CACHE_DIR}" \
    --max-input-len 8192 \
    --max-kv-cache-capacity 16384 \
    --max-image-tokens 4096 \
    --max-image-tokens-per-image 4096
  serve_rc=$?
  set -e
  trap 'hold_gpu ERR' ERR
  echo "REASONER_SERVE_EXIT_${serve_rc}"
  echo "REASONER_SERVE_DIED_HOLDING_GPU"
  sleep 30
done
