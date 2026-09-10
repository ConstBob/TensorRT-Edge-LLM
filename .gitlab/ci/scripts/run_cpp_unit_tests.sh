#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

: "${TRT_PACKAGE_DIR:?TRT_PACKAGE_DIR must be set}"
: "${CUDA_CTK_VERSION:?CUDA_CTK_VERSION must be set}"

build_dir="${CPP_UNIT_BUILD_DIR:-${repo_root}/build-cpp-unit}"
report_dir="${CPP_UNIT_REPORT_DIR:-${repo_root}/.ci-reports}"
build_jobs="${CPP_UNIT_BUILD_JOBS:-16}"
test_jobs="${CPP_UNIT_TEST_JOBS:-4}"

mkdir -p "${build_dir}" "${report_dir}"

cmake_args=(
    -S "${repo_root}"
    -B "${build_dir}"
    "-DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}"
    "-DCUDA_CTK_VERSION=${CUDA_CTK_VERSION}"
    -DBUILD_UNIT_TESTS=ON
    -DBUILD_PYTHON_BINDINGS=OFF
    -DENABLE_CUTEDSL_MODULE_TEST_HOOK=ON
    -DENABLE_CUTE_DSL=ALL
)

if [[ -n "${CUTE_DSL_ARTIFACT_TAG:-}" ]]; then
    cmake_args+=("-DCUTE_DSL_ARTIFACT_TAG=${CUTE_DSL_ARTIFACT_TAG}")
fi

if [[ -n "${EMBEDDED_TARGET:-}" ]]; then
    cmake_args+=(
        "-DEMBEDDED_TARGET=${EMBEDDED_TARGET}"
        "-DCMAKE_TOOLCHAIN_FILE=${repo_root}/cmake/aarch64_linux_toolchain.cmake"
    )
fi

if [[ "${ENABLE_MULTI_DEVICE:-OFF}" == "ON" ]]; then
    cmake_args+=(-DENABLE_MULTI_DEVICE=ON)
fi
if [[ "${ENABLE_MULTI_DEVICE_MPI:-OFF}" == "ON" ]]; then
    cmake_args+=(-DENABLE_MULTI_DEVICE_MPI=ON)
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target unitTests --parallel "${build_jobs}"

library_path="${TRT_LIBRARY_DIR:-${TRT_PACKAGE_DIR}/lib}"
export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GTEST_OUTPUT="xml:${report_dir}/"

ctest --test-dir "${build_dir}" \
    --parallel "${test_jobs}" \
    --stop-on-failure \
    --output-on-failure
