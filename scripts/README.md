# Release Helper Scripts

## OSS Release Sanitizer

Release policy lives at the repo root in `oss_release_manifest.json`, following
the same centralized-policy shape as TensorRT's `oss_components.yml`.

Use `DO_NOT_RELEASE` for whole-file, whole-directory, and glob exclusions from
the release staging tree. Use `{$edge-llm-internal-release begin}` and
`{$edge-llm-internal-release end}` for internal regions inside files that
otherwise remain public.

Expected release/mirror usage:

```bash
OSS_DIR="$(mktemp -d)"
git archive HEAD | tar -x -C "$OSS_DIR"

python3 scripts/strip_internal_release.py --root "$OSS_DIR"
```

The tool consumes `DO_NOT_RELEASE` when present, deletes manifest-listed
internal-only files, strips guarded regions, and fails if any manifest-listed
forbidden pattern remains. Run it on a clean archive or clone, not on an
arbitrary developer build directory.

For CI or local validation of the current tracked tree, run:

```bash
python3 scripts/check_oss_release_sanitizer.py
```

## TensorRT CI D7L compatibility check

`run_trt_ci_d7l.py` is the downstream compatibility entry point for TensorRT
CI. It synchronizes this checkout to a Linux build host, creates a package view
from candidate TensorRT source and build trees, cross-builds the full default
Edge-LLM target set plus `unitTest`, and runs the tests on D7L. The GTest exit
code is returned unchanged.

```bash
python3 scripts/run_trt_ci_d7l.py \
  --trt-root /remote/trt/source \
  --trt-build-dir /remote/trt/build \
  --build-host build.example.nvidia.com --build-user trt-ci \
  --test-host d7l.example.nvidia.com --test-user root \
  --identity-file "$HOME/.ssh/trt-ci" \
  --artifacts-dir "$PWD/artifacts/trt-ci-d7l"
```

The TRT paths are paths on the **build host**, not on the controller. By
default the script reads the CUDA toolkit version from
`/usr/local/cuda/bin/nvcc --version` there; use `--cuda-version` only when CI
must override it. Separate `--build-port`/`--test-port` and
`--build-jump-host`/`--test-jump-host` options support independent endpoints.
Host keys are verified strictly by default; `--host-key-policy accept-new` is a
reasonable bootstrap policy for ephemeral CI hosts.

Prerequisites:

- Python 3.8 or newer plus controller-side `ssh`, `scp`, and `rsync`;
- initialized Edge-LLM submodules (`git submodule update --init`);
- key- or agent-based non-interactive SSH access (passwords are unsupported);
- Bash, CMake, AArch64 Linux compilers, CUDA, `find`, `grep`, `readlink`,
  `sha256sum`, `sort`, `tar`, and `rsync` on the build host;
- an AArch64 D7L rootfs with `/etc/nvidia/version-ubuntu-rootfs.txt`, CUDA,
  Bash, `awk`, `ldd`, `readlink`, `sha256sum`, `tar`, and `tee`;
- candidate headers under `<trt-root>/include` and `<trt-root>/parsers/onnx`,
  generated `NvInferVersion.h` under `<trt-build-dir>/include`, and a complete
  candidate `Release/lib` directory.

Each run uses only `<workspace>/run-<run-id>` on both hosts. Successful runs
remove that child unless `--keep-workspace` is set; failed runs preserve it for
debugging. Phase logs, D7L `ldd` output, GTest output, and GTest XML are retained
under `<artifacts-dir>/run-<run-id>`.

Result collection is best effort and runs even after an earlier failure. A
collection error is reported as a warning and never replaces either a passing
status or the primary build/test exit code. Use `--keep-workspace` when remote
results must survive a controller-side transfer failure. Run
`python3 scripts/run_trt_ci_d7l.py --help` for filters, concurrency, timeout,
workspace, and SSH options.
