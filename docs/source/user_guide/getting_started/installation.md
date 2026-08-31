# Installation

## Choose a deployment workflow

| Workflow | Use it when | Data flow |
|---|---|---|
| Python wheel | The application builds engines directly from checkpoints and uses the Python API or optional HTTP server on a configured target. | Hugging Face checkpoint → checkpoint-direct builder → TensorRT engine → Python inference |
| ONNX and C++ source deployment | The deployment consumes the C++ API, or separates host conversion from target engine build. | Hugging Face checkpoint → optional quantization → ONNX export → C++ engine build → C++ inference |

The wheel workflow is the recommended Python path on configured x86_64,
Jetson, DRIVE, and DGX Spark systems. After obtaining a matching wheel file,
installation is one command. The source workflow remains available for C++
applications and ONNX-based deployments.

## Python wheel deployment

The packaging design maximizes what is carried in one architecture wheel so the
normal user flow is a single installation command. The tooling produces one
x86_64 and one aarch64 wheel for each supported CPython minor version. Each
wheel contains the Python frontend, checkpoint-direct builder, runtime API,
server implementation, and every configured native payload for that CPU
architecture. Payload selection happens at runtime; users do not choose an SM
or TensorRT major during installation.

The x86_64 wheel contains more native payloads because it covers several GPU
SMs and TensorRT majors. The aarch64 wheel covers the intended edge deployment
paths with fewer payload rows. Developers who need a smaller custom artifact
can build a subset wheel without changing the architecture-wide release
contract.

### Platform prerequisites

Install the CUDA and TensorRT versions supplied by the configured platform before
the EdgeLLM wheel. The NVIDIA driver, CUDA runtime, TensorRT runtime and Python
binding, and model checkpoints are not duplicated in the wheel. TensorRT is not
a declared pip dependency because pip cannot select the platform-qualified
TensorRT major from the detected SDK and GPU; declaring a broad range could
replace a compatible platform package with an incompatible major. See the
[wheel packaging matrix](support-matrix.md#wheel-packaging-matrix) for exact
platform rows.

### Current wheel availability

TensorRT Edge-LLM wheels are not currently published to a Python package index.
Build a matching wheel from source as described below, then install that file
directly.

### Install a wheel file

Install a locally built or downloaded wheel into a clean environment:

```bash
export EDGELLM_WHEEL=/absolute/path/to/tensorrt_edgellm-0.10.0-cp312-cp312-manylinux_2_35_x86_64.whl
python -m pip install "$EDGELLM_WHEEL"
```

The base wheel installs the complete checkpoint-direct engine build and Python
inference path. Pip resolves its declared Python dependencies and validates the
wheel against the host CPU architecture and CPython ABI. EdgeLLM then validates
the platform release, CUDA and TensorRT SONAMEs, and visible GPU SM before
loading native code.

Optional dependencies remain available without forcing HTTP, media, PyTorch,
or ONNX packages into every runtime environment. Use a PEP 508 direct reference
to request an extra from a local wheel:

```bash
# OpenAI-compatible HTTP server and checkpoint download
python -m pip install \
    "tensorrt-edgellm[server] @ file://$EDGELLM_WHEEL"

# PyTorch/ONNX export frontend
python -m pip install \
    "tensorrt-edgellm[export] @ file://$EDGELLM_WHEEL"

# Export, quantization, LoRA, vocabulary, and audio tools
python -m pip install \
    "tensorrt-edgellm[tools] @ file://$EDGELLM_WHEEL"
```

### Build an engine and run a prompt

Save the following standalone script as `run_edgellm.py`. It imports only the
installed package, builds a TensorRT engine through the public
`experimental.server.LLM` API, and runs one prompt through the selected native
payload:

```python
import sys
from pathlib import Path

from experimental.server import LLM, SamplingParams


model_dir = Path(sys.argv[1]).resolve(strict=True)
engine_dir = Path(sys.argv[2]).resolve()
llm = LLM(
    model=str(model_dir),
    cache_dir=str(engine_dir),
    clear_engine_cache=True,
    max_input_len=128,
    max_kv_cache_capacity=256,
    max_batch_size=1,
)
try:
    outputs = llm.generate(
        ["Please introduce NVIDIA."],
        SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=32,
        ),
    )
finally:
    llm.close()

print(outputs[0].text)
```

Run it with a local checkpoint and an engine-cache directory:

```bash
python run_edgellm.py \
    /path/to/Qwen2.5-0.5B-Instruct /tmp/edgellm-engine
```

Keep the model checkpoint at its build-time path while using the engine;
checkpoint-backed weights remain external to reduce engine duplication.

### Hosts with different GPU architectures

If visible GPUs have different SMs, EdgeLLM fails before loading native code
rather than selecting an ambiguous payload. Run one process per selected GPU,
using the stable UUID reported by `nvidia-smi`:

```bash
nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv,noheader

CUDA_VISIBLE_DEVICES="GPU-<SM86-UUID>" \
    python run_edgellm.py /path/to/model /tmp/engine-sm86
CUDA_VISIBLE_DEVICES="GPU-<SM120-UUID>" \
    python run_edgellm.py /path/to/model /tmp/engine-sm120
```

`CUDA_VISIBLE_DEVICES` remaps the selected physical GPU to CUDA device `0`
inside the process. A numeric ordinal can refer to a different physical GPU
than the index printed by `nvidia-smi` on heterogeneous hosts, so UUIDs are the
reliable choice. The same installed x86_64 wheel serves both processes.

### Build a custom wheel from source

External users can build a wheel for the current target, a compatible subset of
SMs, or a complete CPU architecture. See
[`packaging/README.md`](../../../../packaging/README.md) for the public build
container, CuTeDSL prerequisites, and high- and low-level commands. Packaging
tools are build-only dependencies and are not installed into runtime
environments.

---

## Source workflow: export and quantization

The Python frontend exports Hugging Face checkpoints and optionally quantizes
FP16/BF16 checkpoints before export. Export runs on CPU. Quantization requires
an NVIDIA GPU.

### System Requirements

- **Platform**: x86-64 Linux system
- **Recommended OS**: Ubuntu 22.04, 24.04
- **GPU for quantization**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- **CUDA for quantization**: 12.x or 13.x
- **Python**: 3.10+

#### Memory Requirements

- Export: at least 1.5 times the checkpoint size in CPU memory. No GPU is
  required.
- Quantization: GPU memory at least equal to the FP16 checkpoint size.

**Verify Your Prerequisites:**

```bash
# Check CUDA installation when quantizing
nvcc --version
# Should show CUDA 12.x or 13.x

# Check the GPU and available memory when quantizing
nvidia-smi
# Look for GPU memory (e.g., "24576MiB" for 24GB)

# Check Python version
python3 --version
# Should show Python 3.10 or higher
```

**If CUDA is not installed:**

Download and install CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). Choose version 12.x or 13.x for your system.

After installation, verify with `nvcc --version` and `nvidia-smi`.

### Installing

For a containerized environment for clean installation, it is recommended to use the NVIDIA PyTorch Docker image:

```bash
# Pull the recommended Docker image
docker pull nvcr.io/nvidia/pytorch:25.12-py3

# Run the container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash
```

**1. Clone Repository**

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**2. Install Python Dependencies**

If you are not using container, it is recommended to use a virtual environment:
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

Install the dependency set for the host-side ONNX workflow:

```bash
# PyTorch/ONNX checkpoint exporter
pip3 install -e ".[export]"

# Export plus quantization, LoRA, vocabulary, and audio tools
pip3 install -e ".[tools]"
```

The `tools` extra remains a superset of `export`. Checkpoint-direct engine build
and Python inference are covered by the wheel workflow above; keeping them out
of this source-export procedure avoids mixing the two deployment paths.

> **Note:** Accuracy evaluation dependencies live under `examples/accuracy/requirements.txt`.

**3. Verify the Checkpoint Export Workflow**

Use the virtual environment created in Step 2 for this checkout. Do not mix
packages from older release branches into the same environment.

Export an unquantized or supported pre-quantized Hugging Face checkpoint with
`tensorrt-edgellm-export`. Run `tensorrt-edgellm-quantize` first only when you
need to create a quantized checkpoint from an FP16/BF16 source checkpoint.

```bash
# Included in the base package
tensorrt-edgellm-export --help

# Available after installing the tools extra
tensorrt-edgellm-quantize --help
tensorrt-edgellm-merge-lora --help
tensorrt-edgellm-reduce-vocab --help
```

**4. Configure HuggingFace Access (Optional)**

Some models on HuggingFace require you to accept terms before downloading.

**Models that require HuggingFace login:**
- Llama family (Llama 3.x)
- Phi-4-Multimodal
- Alpamayo-R1-10B
- Other models marked as "gated" on HuggingFace

**To configure access:**

```bash
# Install HuggingFace CLI and login
hf auth login
# Enter your HuggingFace access token when prompted
```

> **How to get a token:** Visit [HuggingFace Settings - Tokens](https://huggingface.co/settings/tokens), create a new token (read access is sufficient), and copy it.

**You're done with export pipeline setup!** You can now quantize and export models with the checkpoint-based workflow. The ONNX files will be transferred to the Edge device for runtime deployment.

---

## Source workflow: C++ runtime

The C++ runtime builds TensorRT engines and runs inference on the target. For
the authoritative JetPack, DriveOS, CUDA, TensorRT, and TensorRT Edge-LLM
compatibility table, see the [Official Support Matrix](support-matrix.md).
Then use the matching platform command below for your device or SDK image.

Jetson Orin does not support FP8, MXFP8, FP4, or NVFP4 runtime precision in
this release. Use FP16, INT8, or INT4 checkpoints for Orin.

### System Requirements

- CUDA and TensorRT from the target JetPack, DriveOS SDK, or DGX Spark software release
- Disk space: ~20-50GB for ONNX files and TensorRT engines

### Build Instructions

**1. Install System Dependencies (on Edge device)**

```bash
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git
```

**2. Verify CUDA and TensorRT Installation**

After JetPack is installed, inside the DriveOS SDK Docker image, or on DGX
Spark, TensorRT should be installed in `/usr`.

```bash
# Check CUDA version
nvcc --version  # Should match the CUDA_CTK_VERSION for your platform below

# Check TensorRT version
dpkg -l | grep tensorrt  # Should show TensorRT 10.x+
```

**3. Clone Repository (on Edge device)**

```bash
# Clone to your chosen source directory
cd /path/to/parent-directory
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**4. Configure Build**

Use the CMake command for your platform. All commands enable CuTe DSL kernels
because Qwen3.5 and several other model paths require them.

**JetPack 7.0/7.1 Thor**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL
```

**JetPack 7.2 Thor**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

**DriveOS 7.2 Thor**

Run this inside the DriveOS SDK Docker image, then copy `build/` to the DRIVE
system.

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=auto-thor \
    -DCUDA_CTK_VERSION=13.3 \
    -DENABLE_CUTE_DSL=ALL
```

**IGX Thor with an RTX SM120 GPU**

An IGX Thor system with both the integrated Thor GPU and an attached RTX
Blackwell GPU can use one plugin/runtime build. Generate one AArch64 CuTe DSL
artifact containing every supported SM110 and SM120 kernel, then compile the
CUDA sources for both architectures in one CMake build:

```bash
python kernelSrcs/build_cutedsl.py \
    --kernels ALL \
    --gpu_arch sm_110,sm_120 \
    --arch aarch64 \
    --cuda-version 13 \
    --clean

cmake -S . -B build-igx-thor \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=igx-thor \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL

cmake --build build-igx-thor --parallel
```

`EMBEDDED_TARGET=igx-thor` compiles the CUDA sources for `110a` and `120`,
requires CUDA 13 or newer, and selects the combined `sm_110_sm_120` CuTe DSL
artifact. On a native IGX host, set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, then use
`CUDA_VISIBLE_DEVICES=0` for Thor or `CUDA_VISIBLE_DEVICES=1` for RTX. Confirm
the selection before building or running an engine:

```bash
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'
```

In an NVIDIA container use `NVIDIA_VISIBLE_DEVICES=0` or
`NVIDIA_VISIBLE_DEVICES=1`. The selected physical GPU appears as CUDA device 0
inside the container, so do not copy the physical ordinal into
`CUDA_VISIBLE_DEVICES` there. GPU selection is fixed for the life of an
inference process. To switch GPUs, stop the process, update the selector, use
the engine built for the selected SM, and restart. TensorRT engines remain
architecture specific even though the plugin binary is shared.

**DGX Spark (GB10)**

Run this directly on the DGX Spark system. Use `gb10` as the embedded target
and CUDA Toolkit 13.0.

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=gb10 \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL
```

**JetPack 7.2 Orin**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

**Alternative: Building on x86 GPU Systems (Optional for Developers)**

If you want to build and test on an x86 workstation with NVIDIA GPU (for development purposes before deploying to Edge devices), you can use this configuration instead:

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
    -DCUTE_DSL_ARTIFACT_TAG=<YOUR_SM> \
    -DENABLE_CUTE_DSL=ALL
```

> **Note:** Replace `/usr/local/TensorRT-10.x.x` with your actual TensorRT installation path. Use `dpkg -l | grep tensorrt` to find it, or download from [NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt). Replace `<YOUR_CUDA_VERSION>` with your actual CUDA version (e.g., `13.0`). Use `nvcc --version` to check your CUDA version.
> Replace `<YOUR_SM>` with the generated CuTe DSL artifact tag, for example
> `sm_80`, `sm_100`, or `sm_120`.

**CMake Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `TRT_PACKAGE_DIR` | Path to TensorRT installation. Auto-detected; manual hint to disambiguate multiple versions. | N/A |
| `CMAKE_TOOLCHAIN_FILE` | **Required for Edge devices**: Use `cmake/aarch64_linux_toolchain.cmake` for Edge device builds. **Not needed for GPU builds** | N/A |
| `EMBEDDED_TARGET` | **Required for Edge devices**: `jetson-thor` (Jetson Thor), `igx-thor` (IGX Thor plus RTX SM120), `auto-thor` (DRIVE Thor / DriveOS), `gb10` (DGX Spark), or `jetson-orin` (Jetson Orin). **Not needed for GPU builds** | N/A |
| `CUDA_CTK_VERSION` | CUDA Toolkit version. Use the platform command above to select `13.3`, `13.2`, or `13.0`. Do not pass `-DCUDA_VERSION`; CMake reserves that name for CUDA headers and rejects it. | target default |
| `BUILD_UNIT_TESTS` | Build unit tests | OFF |
| `ENABLE_COVERAGE` | Enable gcov code coverage instrumentation (see [Code Coverage](../../developer_guide/testing/code-coverage.md)) | OFF |
| `ENABLE_CUTE_DSL` | Select generated CuTe DSL kernels: `fmha`, `ALL`, or a group list such as `gdn`, `gemm`, or `ssd`. Any selection also links `fmha`, which the attention plugins require. Use `ALL` for customer builds. | fmha |
| `CUTE_DSL_ARTIFACT_TAG` | Artifact tag under `cpp/kernels/cuteDSLArtifact/<arch>/`, for example `sm_87`, `sm_110`, `sm_110_sm_120`, or `sm_121`. Edge targets infer it from `EMBEDDED_TARGET`; pass it explicitly for x86 prebuilt artifacts or when multiple local tags exist for one CPU architecture. | auto |

**CuTe DSL Kernel Artifacts**

CuTe DSL binaries are generated with `kernelSrcs/build_cutedsl.py` before
configuring CMake. A normal build defaults to the canonical `fmha` family and
therefore requires a matching artifact. This family provides Context/ViT
attention on supported GPUs and adds the optimized Blackwell implementation
on SM100/SM101/SM110 when available.

The platform commands above pass `-DENABLE_CUTE_DSL=ALL` because Qwen3.5 and
several other model paths require optional groups. Selecting a narrower group
still includes the `fmha` baseline; for example, `-DENABLE_CUTE_DSL=gdn`
enables both GDN and FMHA.

If you have multiple local artifact tags for the same CPU architecture, also
pass `-DCUTE_DSL_ARTIFACT_TAG=<tag>`.

For B200 or other SM100 build hosts without a matching prebuilt artifact, install
the CuTe DSL package expected by `kernelSrcs/build_cutedsl.py`, then generate the
artifact before running CMake:

```bash
pip install 'nvidia-cutlass-dsl==4.7.0'
python kernelSrcs/build_cutedsl.py --gpu_arch sm_100
```

For cross-compilation, pass `--arch aarch64` when the artifact must be consumed
by an AArch64 target build.

> **For supported model families, precisions, and hardware notes**, see [Supported Models](supported-models.md).

**5. Build Project**

```bash
make -j$(nproc)
```

Build time: ~1-2 minutes depending on hardware.

**6. Verify Build**

```bash
# Test C++ examples
./examples/llm/llm_build --help
./examples/llm/llm_inference --help
```

**You're done with C++ runtime setup!** You can now build engines and run inference on the Edge device.

---

## Next Steps

After installation, proceed to the [Quick Start Guide](quick-start-guide.md) for a complete end-to-end workflow, or see the [Examples](../examples/index.md) for detailed pipeline stages and advanced use cases.

---

## Troubleshooting

### Common Installation Issues

**Issue: Python module import errors**

Solution: Activate the virtual environment and reinstall the package from the
current checkout:
```bash
source venv/bin/activate
python -m pip install -e .
tensorrt-edgellm-export --help
```

**Issue: `nvcc: command not found`**

Solution: Ensure the target JetPack release, DriveOS SDK Docker image, or DGX
Spark software stack is installed with CUDA support:
```bash
# Verify CUDA installation
nvcc --version
# Should match the CUDA_CTK_VERSION used for CMake
```

**Issue: `TensorRT not found` during CMake**

Solution: Specify TensorRT package directory. This directory should contain `lib` and `include` directories, and we are looking for the `nvinfer` library and header:
```bash
cmake .. \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=<jetson-thor|igx-thor|auto-thor|gb10|jetson-orin> \
    -DCUDA_CTK_VERSION=<target CUDA version> \
    -DENABLE_CUTE_DSL=ALL
```

**Issue: Thread issue during C++ build**

Solution: Reduce parallel jobs or even use sequential build:
```bash
make -j  # Instead of make -j$(nproc)
```

### Getting Help

- **Documentation**: Check the `docs/source/developer_guide` directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Community**: Join the NVIDIA Developer Forums

## Uninstalling

**Quantization and `tensorrt_edgellm` (x86 Host):**
- Deactivate and remove virtual environment: `deactivate && rm -rf venv`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`

**C++ Runtime (Edge Device):**
- Remove build directory: `rm -rf build`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`
