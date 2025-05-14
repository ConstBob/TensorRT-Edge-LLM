# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

add_executable(llm_build llm_build.cpp)
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt)
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_build)

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_chat llm_chat.cpp)
target_link_libraries(llm_chat PRIVATE llmSdkCore tokenizer commonLibraryExt)
target_include_directories(llm_chat PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_chat)

add_executable(llm_benchmark llm_benchmark.cpp)
target_link_libraries(llm_benchmark PRIVATE llmSdkCore commonLibraryExt)
target_include_directories(llm_benchmark PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_benchmark)

add_executable(llm_accuracy llm_accuracy.cpp)
target_link_libraries(llm_accuracy PRIVATE llmSdkCore tokenizer
                                           commonLibraryExt)
target_include_directories(llm_accuracy PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_accuracy)
