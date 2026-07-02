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

`run_trt_ci_d7l.py` is the internal TensorRT CI downstream check for Edge-LLM.
It uses TRT Dev Toolkit for every command, container build, remote transfer,
and deployment:

1. RemoteConnectionManager stages this checkout on the Linux build host.
2. A build-host worker gives CodeManager a TRT `PRE_BUILT` package and an
   Edge-LLM source `BUILD` target for D7L.
3. RemoteConnectionManager relays both runtime trees to the controller.
4. `CodeManager.deploy_runtime()` deploys them to D7L, where CommandManager
   checks candidate TRT linkage and runs `unitTest`.

```bash
python3.12 scripts/run_trt_ci_d7l.py \
  --trt-root /remote/trt/source \
  --trt-build-dir /remote/trt/build \
  --trt-branch main \
  --build-host build.example.nvidia.com --build-user trt-ci \
  --test-host d7l.example.nvidia.com --test-user root \
  --artifacts-dir "$PWD/artifacts/trt-ci-d7l"
```

The TRT paths are absolute paths on the **build host**. The worker assembles the
CodeManager `PRE_BUILT` package from `<trt-root>/include`, parser headers,
`<trt-build-dir>/include/NvInferVersion.h`, and
`<trt-build-dir>/Release/lib`. CodeManager then cross-builds the full default
Edge-LLM target set with C++ unit tests enabled, `auto-thor`, and CuTe DSL
disabled. Use `--cuda-version` only when the toolkit default does not match the
candidate environment.

Prerequisites:

- Python 3.12 and TRT Dev Toolkit on the controller and build host;
- initialized Edge-LLM submodules;
- key/agent/OpenSSH-config authentication from the controller to both hosts;
- TRT container tooling, GNU `timeout`, rsync, and the normal D7L cross-build
  toolchain on the build host; and
- Bash, CUDA, `awk`, `ldd`, `readlink`, and `tee` on D7L.

The toolkit does not expose a target identity-file override, so keys must be
available through the CI user's normal SSH agent/config. Separate ports and
`--build-jump-host`/`--test-jump-host` values support independent endpoints.
Host keys are strict by default; use `--known-hosts-file` for a CI-owned file or
`--host-key-policy no` only in an isolated ephemeral environment.

Each run owns only `<workspace>/run-<run-id>` on each remote host. A successful
run removes those directories unless `--keep-workspace` is set; failures retain
them. Controller logs, build-host logs, D7L GTest output, and XML are collected
under `<artifacts-dir>/run-<run-id>`. Collection and cleanup warnings never
replace the primary build or test status. Run
`python3.12 scripts/run_trt_ci_d7l.py --help` for all CI options.
