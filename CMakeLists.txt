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

project(drive_llm VERSION 1.0 LANGUAGES CXX CUDA)

# Specify the C++ standard and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")

macro(set_ifndef var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# CUDA flags and dependencies.
message(STATUS "CUDA_DIR is configured to ${CUDA_DIR}")
message(STATUS "CUDA_LIBRARIES is configured to ${CUDA_LIBRARIES}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set_ifndef(CMAKE_CUDA_ARCHITECTURES 80;86;89)

set_ifndef(CUDA_VERSION 12.6)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_VERSION})

find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS ${CUDA_DIR} PATH_SUFFIXES include)
find_library(CUDART_LIB cudart HINTS ${CUDA_DIR} PATH_SUFFIXES lib lib64)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Set TensorRT header includes and library dependencies.
if(NOT DEFINED TRT_PACKAGE_DIR)
    message(FATAL_ERROR "Please specify the -DTRT_PACKAGE_DIR when invoking CMake.")
endif()

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_PACKAGE_DIR} PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_path(ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h HINTS ${TRT_INCLUDE_DIR})
if(NOT ONNX_PARSER_INCLUDE_DIR)
    message(FATAL_ERROR "NvOnnxParser.h not found in TensorRT headers, please specify the -DONNX_PARSER_INCLUDE_DIR when invoking CMake")
endif()

find_library(NVINFER_LIB nvinfer HINTS ${TRT_PACKAGE_DIR} PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_library(NV_ONNX_PARSER_LIB nvonnxparser HINTS ${TRT_PACKAGE_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)

# Include directories
include_directories(${TRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)

# Add executable with sources
add_subdirectory(plugins)
add_subdirectory(tokenizer)
add_subdirectory(sampler)
add_subdirectory(decoder)

target_include_directories(sampler PUBLIC ${CMAKE_SOURCE_DIR}/sampler/include)
target_link_libraries(decoder PUBLIC sampler)

add_executable(builder builder.cpp)
target_link_libraries(builder PRIVATE ${NV_ONNX_PARSER_LIB} ${NVINFER_LIB} ${CUDART_LIB} ${CMAKE_DL_LIBS})
# Workaround for aarch64 cross compilation where libvninfer.so links to libnvdla_compilers.
# The libnvinfer dependencies shall be resolved at execution time on an embedded board.
target_link_options(builder PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs")       
target_include_directories(builder PUBLIC ${ONNX_PARSER_INCLUDE_DIR})

add_executable(runtime runtime.cpp)
target_link_libraries(runtime PRIVATE decoder tokenizer ${NVINFER_LIB} ${CUDART_LIB} ${CMAKE_DL_LIBS})
# There are two workarounds for aarch64 cross compilation. It needs to override the libcudla symbols
# and also need to points to the aarch64 version of libcudart_static lib.
# TODO: Adjust CMake system to avoid introducing -lcudadevrt -lcudart_static -lrt -lpthread -ldl to the linker.
#       From local experiment, the additional cuda libraries are brought by decoder static library.
target_link_options(runtime PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs" "-L${CUDA_DIR}/lib")   
target_include_directories(runtime PUBLIC ${CMAKE_SOURCE_DIR}/tokenizer ${CMAKE_SOURCE_DIR}/decoder $(CMAKE_SOURCE_DIR)/sampler)
