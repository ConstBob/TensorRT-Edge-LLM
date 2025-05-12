# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(
  driveos_llm_sdk
  VERSION 0.0.3.3
  LANGUAGES CXX CUDA)

# Specify the C++ standard and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# CUDA flags and dependencies.
if(NOT DEFINED AARCH64_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
endif()

set_ifndef(CUDA_VERSION 12.8)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_VERSION})

find_path(
  CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES include)

find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64)

find_library(
  CUDA_LIB cuda
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64)

find_path(
  CURAND_KERNEL_INCLUDE_DIR
  NAMES curand_kernel.h
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES include)

set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                     ${CURAND_KERNEL_INCLUDE_DIR})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Set TensorRT header includes and library dependencies.
if(NOT DEFINED TRT_PACKAGE_DIR)
  message(
    FATAL_ERROR "Please specify the -DTRT_PACKAGE_DIR when invoking CMake.")
endif()

find_path(
  TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_path(ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h HINTS ${TRT_INCLUDE_DIR})
if(NOT ONNX_PARSER_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "NvOnnxParser.h not found in TensorRT headers, please specify the -DONNX_PARSER_INCLUDE_DIR when invoking CMake"
  )
endif()

find_library(
  NVINFER_LIB nvinfer
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_library(
  NV_ONNX_PARSER_LIB nvonnxparser
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 x86_64-linux-gnu)

set(COMMON_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIR} ${TRT_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/examples/multimodal)

add_library(commonLibraryExt INTERFACE)
target_link_libraries(commonLibraryExt INTERFACE ${NVINFER_LIB} ${CUDART_LIB})
# Workaround for aarch64 cross compilation where libvninfer.so links to
# libnvdla_compilers. The libnvinfer dependencies shall be resolved at execution
# time on an embedded board.
target_link_options(commonLibraryExt INTERFACE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")

# Build Core libraries of the project. attentionPlugin shared library and
# llmSdkCore/tokenizer static library will be generated.
add_subdirectory(cpp)

# Build C++ examples to showcase the usage
add_subdirectory(examples)

option(BUILD_UNIT_TESTS "Enable building unit tests" OFF)

if(BUILD_UNIT_TESTS)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Disable Google Mock")
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/googletest)
  file(GLOB_RECURSE UNIT_TESTS_SRCS ${CMAKE_SOURCE_DIR}/unittests/*.cpp
       ${CMAKE_SOURCE_DIR}/unittests/*.cu)
  add_executable(unitTest ${UNIT_TESTS_SRCS})
  target_include_directories(unitTest PRIVATE ${COMMON_INCLUDE_DIRS})
  target_link_libraries(unitTest PRIVATE gtest gtest_main llmSdkPlugin
                                         llmSdkCore ${CUDART_LIB} ${CUDA_LIB})
endif()
