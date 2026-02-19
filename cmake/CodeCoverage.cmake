# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# CodeCoverage.cmake - Adds gcov instrumentation flags for coverage analysis.
#
# When ENABLE_COVERAGE is ON, this module appends --coverage to the C++ and
# linker flags so that gcno/gcda files are produced during build and test
# execution.
#
# CUDA .cu files are NOT instrumented. nvcc generates temporary stub files in
# /tmp/ during compilation that are deleted after the build completes, leaving
# behind .gcno files with unresolvable source references. gcov and gcovr cannot
# process these files and will fail. Coverage is therefore limited to C++ (.cpp)
# source files, which is where the meaningful host-side logic resides.
#
# The resulting gcov data can be consumed directly by SonarQube's C/C++ analyzer
# (sonar.cfamily.gcov.reportsPath) or post-processed with gcovr.

if(ENABLE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(
      WARNING
        "Code coverage results with a non-Debug build type may be unreliable. "
        "Consider setting -DCMAKE_BUILD_TYPE=Debug.")
  endif()

  # Verify gcov is available
  find_program(GCOV_EXECUTABLE gcov)
  if(NOT GCOV_EXECUTABLE)
    message(FATAL_ERROR "gcov not found — required for coverage builds.")
  endif()
  message(STATUS "gcov found: ${GCOV_EXECUTABLE}")

  # C++ coverage flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-abs-path")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

  # NOTE: CMAKE_CUDA_FLAGS is intentionally left unchanged. Passing -Xcompiler
  # --coverage to nvcc produces .gcno files that reference ephemeral
  # /tmp/tmpxft_* stub sources, causing gcov/gcovr failures.

  message(STATUS "Code coverage ENABLED (gcov, C++ sources only)")
endif()
