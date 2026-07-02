# TRT CI Edge-LLM Validation (Issue 451)

## Status

Implementation-ready, x86 first draft.

## Goal

Provide a small internal TRT CI entry point that builds Edge-LLM against a
supplied PRE_BUILT TRT, deploys both artifacts to a separate run host, and runs
one fixed engine-build-plus-inference E2E case. D7L is deferred until the native
x86 flow is proven.

## Public interface

```bash
python3 scripts/run_trt_ci.py \
  x86 \
  /absolute/trt/prebuilt/on-build-host \
  ci-user@build-host[:port] \
  ci-user@run-host[:port]
```

The four positional inputs are architecture, TRT location, build-host SSH
information, and run-host SSH information. The first draft accepts `x86` or
`x86_64` and rejects D7L clearly.

The TRT location is a CodeManager-compatible artifact directory usable as
Edge-LLM `TRT_PACKAGE_DIR` and passed as an opaque `PRE_BUILT` input. This entry
point does not inspect or normalize the directory. SSH endpoints use `[user@]host[:port]`; OpenSSH aliases can supply
identity and ProxyJump settings. The run-host alias is resolved on the build
host.

## Toolkit ownership

- `CommandManager`: every local, build-host, and run-host command.
- `RemoteConnectionManager`: source upload and both SSH connections.
- `ContainerManager`: Edge-LLM build and run-host test container lifecycles.
- `CodeManager`: TRT PRE_BUILT binding, Edge-LLM source build, artifact
  planning, and direct runtime deployment.

There is no controller-side runtime relay or synthetic deployment result. The
exact `RunResult` returned by `plan_and_execute()` is passed directly to
`deploy_runtime(..., preferred_mode=RSYNC)`.

## Flow

```text
controller
  -> stage current Edge-LLM checkout
  -> RemoteConnectionManager upload to x86 build host
  -> hidden build-host worker with forwarded TRT Dev Toolkit PYTHONPATH
       -> CodeManager plan_and_execute(TRT PRE_BUILT, Edge-LLM BUILD)
       -> CodeManager deploy_runtime(actual RunResult, x86 run host)
       -> resolve the normalized Edge-LLM CUDA profile
       -> launch the profile on the run host
            -> source CodeManager setup_environment.sh
            -> focused TRT-facing unitTest subset
            -> Qwen2.5-0.5B FP16 llm_build
            -> llm_inference with llm_basic.json
       -> remove the test container
```

Edge-LLM is built in-source at
`<workspace>/run-<id>/runtime/edgellm` and deployed to the identical absolute
runtime root. This preserves source-relative test resources without separate
resource copies.

The fixed E2E expects the exported ONNX tree under
`/home/edge_llm_cache/trt-ci/onnx` at the same absolute path on both hosts;
`TRT_CI_ONNX_DIR` is an optional controller-side override. The public CLI
remains four positional arguments.

## Failure and cleanup

CodeManager and containerized-test failures return nonzero. The test container
is always removed, while failed workspaces remain for debugging. Successful
remote workspaces are removed best-effort. Controller logs and streamed worker
output remain under `artifacts/trt-ci/run-<id>`.

## Focused tests

Fake public toolkit services cover:

- x86 parsing and D7L rejection;
- SSH aliases, users, ports, and jump-host-preserving resolution;
- opaque TRT PRE_BUILT and Edge-LLM source target construction;
- controller source upload and hidden-worker invocation;
- identity-preserving `plan_and_execute` to `deploy_runtime`;
- normalized Edge-LLM profile resolution and run-host container lifecycle;
- setup-environment, unit, engine-build, and inference command construction;
- run-host status and cleanup behavior; and
- structural absence of runtime/resource/result copy logic.
