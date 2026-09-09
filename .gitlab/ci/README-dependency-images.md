<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI dependency images

The `build_ci_dependency_images` matrix builds the repeated export, x86 test,
and x86 wheel environments once in the setup stage. BuildKit imports the
applicable protected main or release cache plus the merge request's overlay and
publishes the current scope. Consumer jobs use a pipeline-specific image tag,
so a job cannot accidentally run an image produced for a different commit.

The cache identity is the base image, Dockerfile instruction, selected target,
and that target's bind-mounted dependency files. The pipeline image reads the
integration and accuracy requirements, the export image reads `pyproject.toml`
and the integration requirements, and the wheel image reads no repository
dependency file. Unit-test requirements remain job-local so the pipeline image
does not upgrade packages that the base image intentionally provides. Source
files are not copied into an image, so unrelated dependency-file changes and
ordinary C++, CUDA, test, and exporter implementation changes retain the
unaffected dependency layers.

Compatible L0 and L1 x86 jobs consume the matching pipeline or export image.
The L1 CUDA 12.9 compatibility jobs and remote device jobs keep their existing
images because the dependency matrix currently produces only CUDA 13 images.

The cache-smoke experiment measured about 8.4-8.5 GB per wheel image,
14.6-14.7 GB per export or pipeline image, and 29.6 GB for all shared BuildKit
cache records. The registry, rather than `/scratch.edge_llm_cache`, owns these
objects. Pipeline output tags use the project's version-shaped retention
pattern and expire after seven days. Protected main/release and MR BuildKit
references are overwritten in place; stale closed-MR references require a
scheduled registry sweep until GitLab exposes merge-time cleanup to these jobs.
