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

project(drive_llm VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(3RDPARTY_DIR ${ROOT_DIR}/3rdparty)

macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# TRT dependencies
set_ifndef(CUDA_VERSION 12.5)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_VERSION})
set_ifndef(TRT_LIB_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set_ifndef(TRT_INCLUDE_DIR /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

find_path(_CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS ${CUDA_DIR} PATH_SUFFIXES include)
set_ifndef(CUDA_INCLUDE_DIR ${_CUDA_INCLUDE_DIR})

find_library(_CUDART_LIB cudart HINTS ${CUDA_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(CUDART_LIB ${_CUDART_LIB})

find_path(_TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_INCLUDE_DIR} PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set_ifndef(TRT_INCLUDE_DIR ${_TRT_INCLUDE_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# Include directories
include_directories(${TRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(
  ${3RDPARTY_DIR}/json/include
  ${3RDPARTY_DIR}/cxxopts/include)

# Add executable with sources
add_executable(runtime runtime.cpp)

# Link libraries
link_directories(${TRT_LIB_DIR})
link_directories(${CUDA_DIR}/lib64)

target_link_libraries(runtime
    ${NVINFER_LIB}
    ${CUDART_LIB}
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
