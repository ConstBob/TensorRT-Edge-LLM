# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

add_executable(llm_eagle_build llm_eagle_build.cpp)
target_link_libraries(llm_eagle_build PRIVATE ${NV_ONNX_PARSER_LIB}
                                              commonLibraryExt llmSdkCore)
target_include_directories(llm_eagle_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                   ${COMMON_INCLUDE_DIRS})

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_eagle_chat llm_eagle_chat.cpp)
target_link_libraries(llm_eagle_chat PRIVATE llmSdkCore tokenizer
                                             commonLibraryExt)
target_include_directories(llm_eagle_chat PRIVATE ${COMMON_INCLUDE_DIRS})

add_executable(llm_eagle_benchmark llm_eagle_benchmark.cpp)
target_link_libraries(llm_eagle_benchmark PRIVATE llmSdkCore tokenizer
                                                  commonLibraryExt)
target_include_directories(llm_eagle_benchmark PRIVATE ${COMMON_INCLUDE_DIRS})

add_executable(llm_eagle_accuracy llm_eagle_accuracy.cpp)
target_link_libraries(llm_eagle_accuracy PRIVATE llmSdkCore tokenizer
                                                 commonLibraryExt)
target_include_directories(llm_eagle_accuracy PRIVATE ${COMMON_INCLUDE_DIRS})

# Resolve linker error for AArch64 cross build. cudadevrt and cudart_static are
# introduced into the linking of executables
if(DEFINED AARCH64_BUILD)
  target_link_options(llm_eagle_chat PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(llm_eagle_build PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(llm_eagle_benchmark PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(llm_eagle_accuracy PRIVATE "-L${CUDA_DIR}/lib")
endif()
