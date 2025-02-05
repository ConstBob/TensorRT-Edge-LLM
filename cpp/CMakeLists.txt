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
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated"
)

# Define additional CUDA flags for kernel compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0"
)
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills"
)
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")

# Generate attention plugin shared library
file(GLOB_RECURSE PLUGIN_CPP_SRCS "attentionPlugin/*.cpp")
file(GLOB_RECURSE PLUGIN_CU_SRCS "attentionPlugin/*.cu")

add_library(AttentionPlugin SHARED ${PLUGIN_CPP_SRCS} ${PLUGIN_CU_SRCS})
set_target_properties(AttentionPlugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(AttentionPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(AttentionPlugin PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(AttentionPlugin commonLibraryExt cuda)
# avoid linker error from aarch64 cross build
if (DEFINED AARCH64_BUILD)
  target_link_options(AttentionPlugin PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(AttentionPlugin PRIVATE "-L${CUDA_DIR}/lib/stubs")
endif()

# Generate optional int4GroupwiseGemm plugin shared library
file(GLOB_RECURSE INT4_GEMM_PLUGIN_CPP_SRCS "int4GroupwiseGemmPlugin/*.cpp")
file(GLOB_RECURSE INT4_GEMM_PLUGIN_CU_SRCS "int4GroupwiseGemmPlugin/*.cu")

add_library(Int4GemmPlugin SHARED ${INT4_GEMM_PLUGIN_CPP_SRCS} ${INT4_GEMM_PLUGIN_CU_SRCS})
set_target_properties(Int4GemmPlugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(Int4GemmPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(Int4GemmPlugin PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(Int4GemmPlugin commonLibraryExt cuda)
# avoid linker error from aarch64 cross build
if (DEFINED AARCH64_BUILD)
  target_link_options(Int4GemmPlugin PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(Int4GemmPlugin PRIVATE "-L${CUDA_DIR}/lib/stubs")
endif()

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE SAMPLER_CPP_SRCS "sampler/*.cpp")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE DECODER_CPP_SRCS "decoder/*.cpp")

add_library(llmSdkCore STATIC ${SAMPLER_CPP_SRCS} ${SAMPLER_CU_SRCS} ${DECODER_CPP_SRCS})
target_include_directories(llmSdkCore PRIVATE ${COMMON_INCLUDE_DIRS})

# Generate tokenizer C++ static library
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(tokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(tokenizer PRIVATE ${COMMON_INCLUDE_DIRS})