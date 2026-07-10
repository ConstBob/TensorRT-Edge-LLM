# TRT CI Edge-LLM Validation (Issue 451)

## Status

Implemented and validated on x86 and a D7L board through a password-authenticated
jump host.

## Goal

Build Edge-LLM against a supplied PRE_BUILT TRT, hand both artifacts to the run
target, and run Python E2E engine-build and inference cases for an explicit,
easy-to-extend model-family list.

## Public interface

```bash
python3 scripts/run_trt_dependency_ci.py \
  x86 \
  /absolute/trt/prebuilt/on-build-host \
  /path/to/build-host.json \
  /path/to/run-host.json
```

The four positional inputs are architecture, TRT location, build-host JSON, and
run-host JSON. `x86`, `x86_64`, and `d7l` are accepted. A host argument may be
an inline JSON object or a JSON file. `{"host":"localhost"}` selects a
`LocalTarget`. Remote objects require `host`, `port`, `user`, and `password`;
an optional nested `jump_host` object uses the same fields. The script builds
devtoolkit `RemoteConfig` objects directly and never queries the target registry
or resolves OpenSSH aliases.

The default Edge-LLM source-build target sets CodeManager's
`no_nvidia_runtime` flag. Consequently, the controller may be a CPU-only
local build host while the run host supplies the GPU. Passwords should be
passed in protected JSON files in CI; an empty password selects SSH key
authentication.

`--build-locally` uses CommandManager to execute the same Edge-LLM CMake/make
configuration on the selected build host without a build container. The script
adds the native result to the CodeManager run result so deployment remains
toolkit-owned. It requires the host compiler, CMake, CUDA, and any target
cross-toolchain to be preinstalled; x86 test execution is unchanged.

TRT is passed unchanged as a CodeManager `PRE_BUILT` build dependency usable
as Edge-LLM `TRT_PACKAGE_DIR`. D7L deployment omits build-only static archives
from the runtime copy; CodeManager still owns the remote deployment.

## Process and toolkit ownership

There is no hidden worker or self-invocation. One controller process owns:

- non-local SSH target handling and CodeManager deployment transport with
  `RemoteConnectionManager`;
- TRT binding and Edge-LLM build with `CodeManager.plan_and_execute()` and
  `ContainerManager`;
- remote handoff with
  `CodeManager.deploy_runtime()`: direct rsync on x86 and toolkit-selected
  NFS/direct-copy fallback on D7L; local handoff uses
  `CodeManager.write_environment_setup_script()`; and
- x86 test-container execution on the run target with `ContainerManager`; and
- direct deployed-runtime execution on D7L, where the board is the run target.

The exact successful `RunResult` is used for either handoff.

## Remote-build path contract

A remote build requires the controller and build host to see the current
Edge-LLM checkout and TRT directory at the same absolute paths. CodeManager's
public `deploy_runtime()` API deploys completed artifacts to a run target; it
does not stage a source checkout to a build host. The current Edge-LLM remote
build generator also does not sync output back, so these paths must be shared.

## Flow

```text
parse four positional arguments
  -> parse build/run JSON into local or direct SSH targets
  -> CodeManager plan_and_execute(TRT PRE_BUILT, Edge-LLM BUILD)
  -> remote run: CodeManager deploy_runtime(actual RunResult,
     x86 RSYNC or D7L AUTO)
     local run: CodeManager write_environment_setup_script(actual RunResult)
  -> x86: launch the normalized Edge-LLM CUDA profile on the run target
     D7L: execute directly in the deployed board runtime
  -> source setup_environment.sh and configure Python E2E paths
  -> pytest test_engine_build for each configured LLM family
  -> pytest test_inference with llm_basic for each configured LLM family
  -> collect remote JUnit, logs, and inference outputs locally
  -> remove the x86 test container and the script-owned remote workspace
```

Edge-LLM is built in the current checkout through CodeManager. Remote x86
runs deploy below `/tmp/edgellm-trt-ci/run-<id>/runtime`; D7L uses
`/dev/shm/edgellm-trt-ci/run-<id>/runtime` to leave room for generated engines.
Local runs use the build result in place. The run target must expose the ONNX tree at
`/home/edge_llm_cache/trt-ci/onnx` or the caller-supplied
`TRT_CI_ONNX_DIR` override. `_ONNX_MODELS` maps each test family to its
HuggingFace repository and Edge-LLM checkpoint-cache path; adding one entry
schedules engine build and `llm_basic` inference. By default the flow reuses
existing ONNX. On x86, `--download_onnx` downloads each checkpoint on the run
host and invokes Edge-LLM checkpoint export in the test container before those
E2E cases. The active controller Python prefix supplies the
E2E dependencies and must be visible at the same absolute path on the run host;
the toolkit-managed container mounts it automatically. Branch, job-count, and local artifact-root
overrides are optional; the script configures git-trt for noninteractive CI
without requiring caller environment variables.

One CodeManager instance intentionally receives both `exec_target=build_host`
and `remote_connection_manager=run_host`: the first selects where source is
built, while the second is the deployment transport for that exact successful
`RunResult`. For x86 remote tests, `ContainerManager.launch()` and
`exec_progress()` both receive the run host as `exec_target`, so the container
is launched and executed remotely rather than launched on the controller.

## Failure and cleanup

Every completed flow reports a final `TRT CI validation PASSED` or
`TRT CI validation FAILED` line with its run ID and artifact path. Failures
return nonzero. The test container is always removed, and script-owned remote
workspaces are removed best-effort after either outcome. Controller logs and builds remain under
`artifacts/trt-ci/run-<id>` or `TRT_CI_ARTIFACTS_DIR`.

## Focused tests

Public-toolkit fakes cover the four arguments, x86/D7L validation, inline/file
JSON, localhost/direct SSH/jump-host targets, CodeManager targets, one-process
ordering, remote deployment versus local handoff, test execution, failure
status, cleanup, registry absence, and hidden worker/self-invocation absence.
