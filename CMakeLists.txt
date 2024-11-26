#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(
  driveos_llm_sdk
  VERSION 7.0.1.0
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
if(NOT DEFINED AARCH64_CROSS_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
endif()
set_ifndef(CUDA_VERSION 12.6)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_VERSION})

find_path(
  CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES include)
find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64)

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

set(COMMON_INCLUDE_DIRS ${CUDA_INCLUDE_DIR} ${TRT_INCLUDE_DIR}
                        ${CMAKE_SOURCE_DIR}/cpp)

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
