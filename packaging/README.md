# TensorRT Edge-LLM wheel tooling

All supported operations use:

    python packaging/wheel_cli.py <command>

Install development-only tools with:

    python -m pip install -r packaging/wheel-toolchain-requirements.txt

These PEP 517/build/qualification dependencies are not installed for normal use.
`packaging/docker/Dockerfile` builds a reference environment with all three
qualified interpreters. Its default base is the concrete x86_64 Ubuntu
24.04/CUDA 13 build environment. Supply `--build-arg BASE_IMAGE=...` for a
different native or cross-build matrix environment.

## Local sequence

    python packaging/wheel_cli.py validate-matrix
    python packaging/wheel_cli.py validate-source --output artifacts/provenance/source.json
    python packaging/wheel_cli.py build-base --output-dir artifacts/base/cp312
    python packaging/wheel_cli.py prepare-cutedsl --variant VARIANT --artifact-dir kernelSrcs/cuteDSLPrebuilt
    python packaging/wheel_cli.py build-payload --variant VARIANT --python-abi cp312 --trt-package-dir TRT --output-dir artifacts/payloads/VARIANT-cp312
    python packaging/wheel_cli.py verify-payload --stage artifacts/payloads/VARIANT-cp312
    python packaging/wheel_cli.py assemble --base-wheel BASE.whl --payload-root artifacts/payloads --cpu-arch x86_64 --python-abi cp312 --output-dir dist/x86_64/cp312

variants.toml is authoritative for runtime identity and CI scheduling.
generate-ci writes .gitlab/ci/wheel-generated.yml; CI checks for drift.

Build runners only need a reproducible compiler, CUDA toolkit, TensorRT SDK,
CuTe DSL archive, and the requested CPython development files. Target GPU and
board runners are used later to install and qualify the assembled wheel. A
native build runner may also be the target when a maintained cross environment
is unavailable, as for DGX Spark.

scikit-build-core produces the Python/metadata base. Qualified runners build
native payloads, which are folded into six final wheels. Packaging and runtime
share tensorrt_edgellm/_native/contract.py.

Wheel CI installs each final wheel in a clean environment, builds a
Qwen2.5-0.5B engine through the installed CLI, and runs a prompt through the
installed runtime for every variant and ABI. Build images and remote targets
must provide all three qualified CPython minors. Container jobs bootstrap
missing minors from the configured PPA; unprivileged native shell runners use a
pinned `uv` installation to provision managed CPython without `sudo`. System
TensorRT installations may use the standard multiarch include and library
layout. Remote TensorRT wheel paths use a `{python_abi}` placeholder. Remote
jobs default to the board user home; a
variant may set `ci_target_work_dir` when qualification needs another filesystem
(D7L uses `/mnt/bigspace`). Qualification requires all evidence.

## Packaging precedent

The Python base uses scikit-build-core as the PEP 517 frontend for CMake. This
matches the established CMake-driven approach used by projects such as
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/setup.py),
[vLLM](https://github.com/vllm-project/vllm/blob/main/setup.py), and
[TensorRT](https://github.com/NVIDIA/TensorRT/blob/main/python/packaging/frontend_sdist/CMakeLists.txt).
TensorRT-LLM and vLLM also support consuming precompiled native content, while
[SGLang](https://github.com/sgl-project/sglang/blob/main/python/pyproject.toml)
publishes its kernel package separately.
[ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/pyproject.toml)
uses setuptools, includes C++/CUDA sources as package data, and
[JIT-compiles optional extensions](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/utils/cpp_extension.py)
through PyTorch. EdgeLLM cannot assume a compiler toolchain on edge targets, so
it keeps scikit-build for the standard Python/CMake boundary and adds only the
repository-specific aggregation needed to place precompiled platform/SM
payloads in one architecture-wide wheel. Each native payload compiles only the
SM declared by its matrix row; assembly combines those payloads into the final
architecture-wide wheel.
