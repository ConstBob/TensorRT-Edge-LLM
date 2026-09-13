<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT engine cache

This first draft caches the complete TensorRT engine bundle produced for one
local or remote `TestConfig`. It changes only GitLab CI and the test harness;
it does not change the Edge-LLM C++ library, builder API, Python package, or
CMake target structure.

This hooks the ONNX-to-TensorRT `execute_build_test` path. The experimental
checkpoint-direct builder also produces TensorRT engines, but uses a separate
bundle contract and is not changed by this first draft.

## Hit and miss contract

A lookup is eligible for local and remote builds with engine caching enabled.
Remote jobs fingerprint their target-side builders, libraries, GPU, driver, and
architecture. The cache key contains exact SHA-256 identities for:

- every payload file in every ONNX input tree (excluding `.export_meta` cache metadata);
- every normalized builder command, option, and timeout;
- each builder executable, dynamically linked library, and discovered Edge-LLM
  plugin library;
- GPU name and compute capability, driver, CUDA, TensorRT, TRT-RTX, CI image,
  host architecture, and the explicit cache epoch.

Input and output directory paths are normalized, so moving otherwise identical
workspaces does not cause a miss. Any change to the inputs above causes a miss.
Each logical build lane keeps one latest bundle with `.engine_meta`. A hit
requires an exact fingerprint and output-root match, followed by successful
TensorRT deserialization of every cached `*.engine`. Local jobs probe through
the TensorRT Python runtime; remote jobs use target-side `trtexec --loadEngine`
with inference disabled and the discovered Edge-LLM plugins loaded. Restore is
staged and installed atomically. Missing, malformed, incompatible, or
unavailable state fails open to a normal build. Successful builds are probed
before atomically replacing the previous latest bundle. An MR can set the
`ci:force-engine-build` label to bypass reads and refresh its writable cache.

The cache is deliberately performance-gated off for NVIDIA GeForce RTX 3090.
The experiment measured about 393 seconds of materialization and validation on
that lane, making the warm job 18.2% slower than a rebuild.

## Namespace and storage policy

Entries live under:

```text
/scratch.edge_llm_cache/gitlab-ci-cache/engines/<namespace>/<lane>/
```

MR pipelines read their writable `project-<id>-mr-<iid>` overlay first and then
the target branch's read-only `project-<id>-main` or
`project-<id>-release-<series>` namespace. Only protected main/release
stability pipelines may write trusted namespaces. The backend filesystem ACL
must enforce that trust boundary in addition to the helper's write policy.

Each logical build configuration receives a stable `trt-engine-<digest>` lane.
The full fingerprint remains stricter and includes exact ONNX, builder, library,
and platform content. The lane is only a directory bucket; its `.engine_meta`
fingerprint decides the hit and prevents a stale bundle from matching.

The measured selected-job smoke subset occupied about 11 GiB. A complete latest
trusted generation is estimated at 120-200 GiB; that is an estimate until
production telemetry inventories all engine objects. MR engine namespaces are
soft-capped at 20 GiB. Trusted namespaces are soft-capped at 200 GiB.

`prune_trt_engine_cache` runs in every L0 setup stage. MR entries older than 30
days are removed and the namespace is trimmed by modification time to 20 GiB.
Publication directly replaces the previous entry, so protected caches retain
only the latest bundle per logical lane, with a 200 GiB soft cap. Interrupted
staging directories expire after 24 hours.

## Runtime evidence

Complete engine hits reduced selected warm jobs by:

- 58.6% on Thor 2 TRT 11;
- 52.7% on A30 TRT 11;
- 51.2% on RTX 5080;
- 47.8% on Jedha;
- 42.9% on A30;
- 37.6% on Thor 2.

The weighted reduction across those completed hits was about 49%. For ordinary
MRs, 15-35% whole-job savings are expected because source, ONNX, platform, or
profile changes legitimately invalidate some engines. On an eligible exact hit,
40-50% whole-job savings are the expected range. These measurements include
cache restore and validation time but exclude runner queue time.

## Timing-cache follow-up

This MR does not serialize TensorRT timing caches. The current builders have no
shared timing-cache CLI contract without the previously explored C++ builder
changes. Timing cache support should follow the builder/CMake isolation MR so it
can use a separate key that includes builder, TensorRT, GPU, tactic-affecting
options, and plugin identities but intentionally excludes ONNX weights. That
will accelerate legitimate engine misses without weakening complete-engine hit
validation.
