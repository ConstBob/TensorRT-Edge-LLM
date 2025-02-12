# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

add_executable(vlm_build vlm_build.cpp)
target_link_libraries(vlm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt)
target_include_directories(vlm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})

file(GLOB_RECURSE QWEN2VL_CPP_SRCS "qwen2vl/*.cpp")

add_executable(vlm_chat vlm_chat.cpp ${QWEN2VL_CPP_SRCS})

target_link_libraries(vlm_chat PRIVATE llmSdkCore tokenizer commonLibraryExt)
target_include_directories(vlm_chat PRIVATE ${COMMON_INCLUDE_DIRS}
                                            ${CMAKE_SOURCE_DIR}/3rdParty/stb)

add_executable(vlm_benchmark vlm_benchmark.cpp ${QWEN2VL_CPP_SRCS})

target_link_libraries(vlm_benchmark PRIVATE llmSdkCore tokenizer
                                            commonLibraryExt)
target_include_directories(
  vlm_benchmark PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdParty/stb)
add_executable(vlm_accuracy vlm_accuracy.cpp ${QWEN2VL_CPP_SRCS})

target_link_libraries(vlm_accuracy PRIVATE llmSdkCore tokenizer
                                           commonLibraryExt)
target_include_directories(
  vlm_accuracy PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdParty/stb)

if(DEFINED AARCH64_BUILD)
  target_link_options(vlm_chat PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(vlm_benchmark PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(vlm_accuracy PRIVATE "-L${CUDA_DIR}/lib")
endif()
