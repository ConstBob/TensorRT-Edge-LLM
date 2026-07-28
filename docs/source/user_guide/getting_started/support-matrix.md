<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Official Support Matrix

This page lists the TensorRT Edge-LLM platform and software stack combinations
that are officially supported or compatibility-tested for the current release.
Use this page together with [Supported Models](supported-models.md) for model and
precision coverage, and [Installation](installation.md) for platform-specific
CMake commands.

## Support Levels

| Level | Meaning |
|-------|---------|
| Official | Documented and tested by TensorRT Edge-LLM for the listed release. |
| Compatible | Expected to work with the listed constraints, but not the primary release target. |
| Developer-only | Useful for development or CI coverage; not an edge deployment support target. |

## TensorRT Edge-LLM 0.9.1

| Platform | Support Level | OS / SDK Release | CUDA Toolkit | TensorRT | Build Location | Precision Notes |
|----------|---------------|------------------|--------------|----------|----------------|-----------------|
| Jetson Thor | Official | JetPack 7.0 / 7.1 | 13.0 | JetPack-bundled TensorRT | Jetson device | See [Supported Models](supported-models.md). |
| Jetson Thor | Official | JetPack 7.2 | 13.2 | JetPack-bundled TensorRT | Jetson device | See [Supported Models](supported-models.md). |
| NVIDIA DRIVE Thor | Official | DriveOS 7.2 | 13.3 | DriveOS SDK-bundled TensorRT | DriveOS SDK Docker image, then copy `build/` to the DRIVE system | See [Supported Models](supported-models.md). |
| NVIDIA DGX Spark (GB10) | Official | DGX Spark software stack | 13.0 | DGX Spark software stack TensorRT | DGX Spark system | See [Supported Models](supported-models.md). |
| Jetson Orin | Official | JetPack 7.2 | 13.2 | JetPack-bundled TensorRT | Jetson device | FP16, INT8, and INT4 only. |
| Jetson Orin | Compatible | JetPack 6.2+ | 12.6 | JetPack-bundled TensorRT | Jetson device | FP16, INT8, and INT4 only. |
| x86-64 Linux GPU system | Developer-only | Ubuntu 22.04 / 24.04 | 12.x or 13.x | User-provided TensorRT 10.x or newer | x86 workstation | Development, export, and CI validation only. |

Jetson Orin does not support FP8, MXFP8, FP4, or NVFP4 runtime precision in this
release. Use FP16, INT8, or INT4 checkpoints for Orin.

## TensorRT Version Notes

- Edge device deployments normally use the TensorRT package bundled with the
  target JetPack, DriveOS SDK, or DGX Spark software stack.
- Internal CI also validates selected platform combinations with cached TensorRT
  packages, including TensorRT 10.16.0.72 for current JetPack 7 / DriveOS 7
  paths and TensorRT 10.13.3.9 for selected compatibility jobs.
- TensorRT 11 jobs are present in CI as forward-looking validation. TensorRT 11
  is not the default public deployment target for TensorRT Edge-LLM 0.9.1.

## Update Policy

Update this matrix whenever a release changes any of the following:

- TensorRT Edge-LLM release version.
- JetPack, DriveOS, DGX Spark software stack, CUDA Toolkit, or TensorRT version.
- Platform support level.
- Precision constraints for a platform.

DriveOS rows must stay aligned with the DriveOS documentation repository:
`sw-docs/drive/driveos-docs/driveos-guides`. If TensorRT Edge-LLM compatibility
also needs to be advertised in DriveOS documentation, create a separate DriveOS
documentation update after the DriveOS documentation owners approve the public
wording and exact version claims.
