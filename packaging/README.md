# TensorRT Edge-LLM wheel tooling

All supported operations use:

    python packaging/wheel_cli.py <command>

Install development-only tools with:

    python -m pip install -r packaging/wheel-toolchain-requirements.txt

These PEP 517/build/qualification dependencies are not installed for normal use.
`packaging/docker/Dockerfile` builds an x86_64 or aarch64 reference image with
all three qualified interpreters.

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

scikit-build-core produces the Python/metadata base. Qualified runners build
native payloads, which are folded into six final wheels. Packaging and runtime
share tensorrt_edgellm/_native/contract.py.

Wheel CI installs each final wheel in a clean environment, builds a
Qwen2.5-0.5B engine through the installed CLI, and runs a prompt through the
installed runtime for every variant and ABI. Build images and remote targets
must provide all three qualified CPython minors; remote TensorRT wheel paths
use a `{python_abi}` placeholder. Remote jobs default to the board user home; a
variant may set `ci_target_work_dir` when qualification needs another filesystem
(D7L uses `/mnt/bigspace`). Qualification requires all evidence.
