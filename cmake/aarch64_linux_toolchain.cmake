# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# aarch64_toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER
    ${CMAKE_CXX_COMPILER}
    CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_FLAGS
    " -Xcompiler=\"-fPIC \""
    CACHE STRING "" FORCE)

# Specify the architecture for CUDA
if("${EMBEDDED_TARGET}" STREQUAL "auto-thor")
  set(CMAKE_CUDA_ARCHITECTURES 101)
  set(CUDA_VERSION 12.8)
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
elseif("${EMBEDDED_TARGET}" STREQUAL "jetson-thor")
  set(CMAKE_CUDA_ARCHITECTURES 110)
  set(CUDA_VERSION 13.0)
  set(CUDA_DIR
      /usr/local/cuda/targets/sbsa-linux
      CACHE STRING "CUDA toolkit dir")
elseif("${EMBEDDED_TARGET}" STREQUAL "orin")
  set(CMAKE_CUDA_ARCHITECTURES 87)
  set(CUDA_VERSION 12.6)
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
elseif("${EMBEDDED_TARGET}" STREQUAL "n1")
  set(CMAKE_CUDA_ARCHITECTURES 121)
  set(CUDA_VERSION 13.0)
  set(CUDA_DIR
      /usr/local/cuda/targets/sbsa-linux
      CACHE STRING "CUDA toolkit dir")
endif()

# Tell CMake how to search for the libraries and programs
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Set variable to indicate CMake is running aarch64 build
set(AARCH64_BUILD TRUE)
