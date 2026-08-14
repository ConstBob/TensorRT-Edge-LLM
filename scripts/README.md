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

`ci_dependency_tests/run_trt_dependency_ci.py` is the internal x86/D7L TRT downstream check. It takes
architecture, TRT location, build-host JSON, and run-host JSON:

```bash
python3 scripts/ci_dependency_tests/run_trt_dependency_ci.py \
  x86 \
  /absolute/trt/prebuilt/on-build-host \
  /path/to/build-host.json \
  /path/to/run-host.json
```

Append `--no-trt-containers` to build Edge-LLM and run its Python E2E tests natively on their selected hosts.

Native x86 builds must receive the Conan CUDA toolkit selected by TensorRT CI:

```bash
python3 scripts/ci_dependency_tests/run_trt_dependency_ci.py \
  x86 /absolute/trt/prebuilt/on-build-host \
  /path/to/build-host.json /path/to/run-host.json \
  --no-trt-containers \
  --cuda-root /absolute/conan/cuda/package/root \
  --cuda-version 13.3
```

`TRT_CI_CUDA_ROOT` and `TRT_CI_CUDA_VERSION` are equivalent environment
inputs. The runner validates `bin/nvcc`, pins CMake to that toolkit, and
forwards its activated library and include paths to the native build host.
`--run-python /absolute/python` selects the interpreter used for native E2E
tests when CI provides one with Edge-LLM's Python dependencies.

D7L cross-builds use separate generic CUDA and Thor target directories. They
default to `/usr/local/cuda/targets/aarch64-linux` and
`/usr/local/cuda/thor/targets/aarch64-linux`, respectively. Override them with
`--cuda-dir` and `--cuda-target-dir`, or with the equivalent
`TRT_CI_CUDA_DIR` and `TRT_CI_CUDA_TARGET_DIR` environment variables.

Each host argument may be an inline JSON object or a JSON file.

Local host:

```json
{
  "host": "localhost"
}
```

Direct SSH target; use an empty password for SSH-key authentication:

```json
{
  "host": "compute-host.example.com",
  "port": 22,
  "user": "ci-user",
  "password": ""
}
```

SSH target reached through a jump host:

```json
{
  "host": "192.168.1.3",
  "port": 22,
  "user": "board-user",
  "password": "board-password",
  "jump_host": {
    "host": "jump-host.example.com",
    "port": 22,
    "user": "ci-user",
    "password": ""
  }
}
```

`jump_host` is optional. The script constructs devtoolkit `RemoteConfig`
objects directly and never depends on the target registry or OpenSSH aliases.
Passwords should be supplied through protected JSON files in CI rather than
inline command arguments.

The accepted architectures are `x86`, `x86_64`, and `d7l`. The TRT directory
is passed unchanged to CodeManager as a `PRE_BUILT` artifact usable as
`TRT_PACKAGE_DIR`. Either host can use `{"host":"localhost"}`. For a remote
build, the checkout and TRT package must be visible at the same absolute paths
on the controller and build host.

Pass `--no-trt-containers` to run Edge-LLM's CMake/make build on the
build host and its Python E2E tests on the run host through devtoolkit
`CommandManager`, without launching TRT containers. CodeManager still records
the TRT `PRE_BUILT` artifact, deploys the resulting native Edge-LLM artifact,
and writes the sourced runtime environment script, including `LD_LIBRARY_PATH`
for Edge-LLM and TRT. Both hosts must already provide their required native
toolchain; the run host must also provide Python and pytest.

One controller process owns the flow; there is no hidden worker or script
self-invocation:

1. `CodeManager.plan_and_execute()` binds TRT. By default it builds the
   current Edge-LLM checkout through `ContainerManager`; `--no-trt-containers`
   runs the CMake/make build through `CommandManager` on the build target. The
   checkout and TRT package must be visible at the same absolute paths on a
   remote build host.
2. The default build container explicitly disables the NVIDIA runtime. With
   `--no-trt-containers`, the native build host supplies its toolchain instead.
   CodeManager's generated setup script configures the deployed Edge-LLM/TRT
   library paths before the direct E2E command runs.
3. `RemoteConnectionManager` supplies direct JSON-configured SSH targets and
   CodeManager's deployment transport. A remote run uses
   `CodeManager.deploy_runtime()`; x86 requests direct rsync, while D7L lets
   the toolkit try NFS before its direct-copy fallbacks. D7L first derives a
   runtime `RunResult` that omits build-only TRT static archives. If NFS is
   selected, D7L ONNX staging writes into the mounted export rather than
   copying ONNX directly to the board. A local run uses
   `CodeManager.write_environment_setup_script()` without a copy.
4. By default, x86 uses a CodeManager-owned runtime container; D7L runs the
   Python E2E cases directly on the deployed board runtime. `--no-trt-containers`
   makes x86 use that same direct `CommandManager` execution path.
5. Remote JUnit, logs, and inference outputs are copied back to the local
   artifact directory. Engines remain in the remote run workspace and all
   script-owned remote workspaces are then removed best-effort.

The script always runs the built-in smoke set and accepts additional model
cases with `--additional-model-cases`. The argument is a JSON array, or a path
to a JSON file containing that array. Each object maps directly to `ModelCase`:

```json
[
  {
    "name": "MyModel",
    "repository": "org/my-model",
    "checkpoint_dir": "MyModel",
    "pipeline_param": "MyModel-fp16-mxsl2048-mxbs1-mxil1024"
  }
]
```

Each entry supplies the test name, HuggingFace repository, Edge-LLM source
checkpoint layout, and pipeline test parameter. It schedules optional
checkpoint download, optional ONNX export, engine build, and `llm_basic`
inference. `TRT_CI_ONNX_DIR` is always
required: without `--export_onnx` it must point at existing ONNX packages, and
with `--export_onnx` it is the build-host output directory populated by
`tests/defs/test_checkpoint_export.py::test_checkpoint_export`. Pass
`--download_hf_checkpoint` to download missing HuggingFace checkpoints on the
build host before export. `TRT_CI_HF_CHECKPOINT_DIR` is required when either
`--download_hf_checkpoint` or `--export_onnx` is used. The build-host account
must already have HuggingFace access. Some checkpoints may require prior
repository access approval and `hf auth login`. Without
`--download_hf_checkpoint`, existing HF checkpoints are reused.
C++ unit tests are not built or run by this flow. The controller's active Python environment supplies pytest and the E2E
dependencies; its prefix must be visible at the same absolute path on the run
host and is mounted into the x86 test container automatically. D7L runs use
the board's `python3`, which must provide the Edge-LLM E2E dependencies,
including `examples/accuracy/requirements.txt` for ROUGE validation.

**Remote-build path requirement:** the controller and build host must see the
Edge-LLM checkout and supplied TRT directory at identical absolute paths. The
current CodeManager Edge-LLM remote-build flow does not stage source or sync
its output back, while deployment reads the controller-visible `RunResult`
paths.

The caller must set `TRT_CI_ONNX_DIR` to the ONNX root used by
`test_engine_build` and `test_inference`. For D7L, the script stages only the
configured model ONNX subdirectories to the board runtime workspace. The caller must also set
`TRT_CI_HF_CHECKPOINT_DIR` when downloading checkpoints or exporting ONNX.
`TRT_CI_ARTIFACTS_DIR` is needed only when a remote build requires a
caller-supplied shared path. `TRT_CI_JOBS` and
`TRT_CI_BRANCH` are optional tuning overrides; the script makes git-trt
noninteractive itself. Python 3 and TRT Dev Toolkit are required on the controller.
Remote targets also require controller-side SSH authentication; the toolkit
falls back to tar-over-SSH when a board does not provide rsync. Only the run target needs NVIDIA runtime
support; the build target may be CPU-only. A D7L board is executed directly and
does not require Docker or git-trt.

The last console line reports `TRT CI validation PASSED` or
`TRT CI validation FAILED`, together with the run ID and local artifact path.
Build artifacts and logs remain under `artifacts/trt-ci/run-<id>` or
`TRT_CI_ARTIFACTS_DIR`. An x86 remote run uses `/tmp/edgellm-trt-ci/run-<id>`.
A D7L remote run uses `/home/<run-user>/edgellm-trt-ci/run-<id>` by default to
avoid placing large ONNX and engine artifacts in tmpfs. Set
`TRT_CI_RUN_WORKSPACE_ROOT` to override the run-host root, for example
`/dev/shm/edgellm-trt-ci` on lab boards whose root filesystem is full. Remote
run workspaces are removed best-effort after either outcome.

## Release Documentation

Build published documentation from the sanitized tree, never from the raw
source: `DO_NOT_RELEASE` entries such as
`docs/source/user_guide/examples/omni_internal.md` otherwise render into public
HTML, and internal C++ headers reach the API pages and every nav sidebar.

`docs/source/conf.py` links each page back to the commit it was generated from.
That commit must be the public GitHub one — a `git archive` export has no
repository to read, and a GitLab SHA does not resolve on github.com. Pass it
explicitly when building release docs:

```bash
(
  cd "$OSS_DIR/docs"
  doxygen
  EDGELLM_DOCS_COMMIT_SHA=<public github commit> make html
)
```

`conf.py` fails if `docs/cpp_docs/xml/index.xml` is absent, and the Makefile
does not run Doxygen itself.

The build fails rather than emitting links to a commit that does not exist, so
an unset variable outside a repository is reported instead of silently
published.
