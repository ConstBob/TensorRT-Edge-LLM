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

## TensorRT CI Edge-LLM validation

`run_trt_ci.py` is the x86 first draft of the internal TRT downstream check.
It takes exactly four positional inputs:

```bash
python3 scripts/run_trt_ci.py \
  x86 \
  /absolute/trt/prebuilt/on-build-host \
  trt-ci@build.example.nvidia.com \
  trt-ci@run.example.nvidia.com
```

The TRT location on the build host may be either:

- a package/install root with `include/` and `lib/`; or
- a built TRT source root with `include/`, `parsers/onnx/`,
  `build/include/`, and `build/Release/lib/`.

Both are CodeManager `PRE_BUILT` inputs; the script does not rebuild or
repackage TRT. SSH endpoints use `[user@]host[:port]` or an OpenSSH host alias.
Aliases may provide identity and `ProxyJump` settings. Run-host SSH
configuration must be available on the build host.

The script delegates infrastructure work to TRT Dev Toolkit:

1. The controller uses `CommandManager` and `RemoteConnectionManager` to
   stage this Edge-LLM checkout on the x86 build host.
2. A build-host worker gives `CodeManager` the TRT `PRE_BUILT` target and an
   Edge-LLM source `BUILD` target. `ContainerManager` owns the build
   container.
3. The exact `RunResult` from `CodeManager.plan_and_execute()` is passed to
   `CodeManager.deploy_runtime()`, which deploys TRT and Edge-LLM directly to
   the run host. Runtime artifacts and resources are not relayed through the
   controller.
4. `CommandManager` checks candidate TRT linkage, runs `unitTest`, builds a
   Qwen2.5-0.5B-Instruct FP16 engine, and runs `llm_inference` with
   `tests/test_cases/llm_basic.json`.

The run host must expose that model's exported ONNX tree at
`/home/edge_llm_cache/trt-ci/onnx/Qwen2.5-0.5B-Instruct/llm-fp16-fp16`.
Set `TRT_CI_ONNX_DIR` on the controller to override the ONNX root while
keeping the public command unchanged. D7L cross-compilation is intentionally
deferred until this native x86 flow has run successfully in TRT CI.

Prerequisites:

- Python 3 and TRT Dev Toolkit on the controller and build host;
- initialized Edge-LLM submodules on the controller;
- key/agent/OpenSSH-config authentication from the controller to the build host
  and from the build host to the run host;
- TRT container tooling and rsync on the controller/build host; and
- an x86 NVIDIA GPU environment plus Bash, CUDA, `ldd`, `awk`, and
  `readlink` on the run host.

Each run owns `/tmp/edgellm-trt-ci/run-<id>` on both remote hosts. Failed runs
retain their workspaces. Successful runs clean them, while controller logs and
streamed worker output remain under `artifacts/trt-ci/run-<id>`. Set
`TRT_CI_JOBS`, `TRT_CI_ARTIFACTS_DIR`, or `TRT_CI_ONNX_DIR` only when a
CI default needs an override. The controller forwards its discovered toolkit
source path to the build worker; use `TRT_CI_TOOLKIT_PYTHONPATH` only when the
build host exposes that checkout at a different shared path.
