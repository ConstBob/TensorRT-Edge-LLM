# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated"
)

# Define additional CUDA flags for kernel compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0")
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)
set(CMAKE_CUDA_FLAGS_DEBUG "")

# Generate attention plugin shared library
file(GLOB_RECURSE PLUGIN_CPP_SRCS "attentionPlugin/*.cpp")
file(GLOB_RECURSE PLUGIN_CU_SRCS "attentionPlugin/*.cu")

add_library(AttentionPlugin SHARED ${PLUGIN_CPP_SRCS} ${PLUGIN_CU_SRCS})
set_target_properties(AttentionPlugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(AttentionPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR})
target_include_directories(AttentionPlugin PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(AttentionPlugin commonLibraryExt cuda)
# avoid linker error from aarch64 cross build
if(DEFINED AARCH64_BUILD)
  target_link_options(AttentionPlugin PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(AttentionPlugin PRIVATE "-L${CUDA_DIR}/lib/stubs")
endif()

# Generate optional int4GroupwiseGemm plugin shared library
file(GLOB_RECURSE INT4_GEMM_PLUGIN_CPP_SRCS "int4GroupwiseGemmPlugin/*.cpp")
file(GLOB_RECURSE INT4_GEMM_PLUGIN_CU_SRCS "int4GroupwiseGemmPlugin/*.cu")

add_library(Int4GemmPlugin SHARED ${INT4_GEMM_PLUGIN_CPP_SRCS}
                                  ${INT4_GEMM_PLUGIN_CU_SRCS})
set_target_properties(Int4GemmPlugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(Int4GemmPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                ${CMAKE_BINARY_DIR})
target_include_directories(Int4GemmPlugin PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(Int4GemmPlugin commonLibraryExt cuda)
# avoid linker error from aarch64 cross build
if(DEFINED AARCH64_BUILD)
  target_link_options(Int4GemmPlugin PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(Int4GemmPlugin PRIVATE "-L${CUDA_DIR}/lib/stubs")
endif()

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE SAMPLER_CPP_SRCS "sampler/*.cpp")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE DECODER_CPP_SRCS "decoder/*.cpp")

add_library(llmSdkCore STATIC ${SAMPLER_CPP_SRCS} ${SAMPLER_CU_SRCS}
                              ${DECODER_CPP_SRCS})
target_include_directories(llmSdkCore PRIVATE ${COMMON_INCLUDE_DIRS})

# Generate tokenizer C++ static library
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(tokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(tokenizer PRIVATE ${COMMON_INCLUDE_DIRS})
