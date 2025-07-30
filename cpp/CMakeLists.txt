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
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -fPIC"
)

# Define additional CUDA flags for kernel compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0 -fPIC")
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)
set(CMAKE_CUDA_FLAGS_DEBUG "")

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
file(GLOB_RECURSE COMMON_CU_SRCS "common/*.cu")
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE DECODER_CPP_SRCS "decoder/*.cpp")
file(GLOB_RECURSE EAGLE_CU_SRCS "eagle/*.cu")
file(GLOB_RECURSE EAGLE_CPP_SRCS "eagle/*.cpp")
file(GLOB_RECURSE ENGINE_CPP_SRCS "engine/*.cpp")

add_library(llmSdkKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS})
target_include_directories(llmSdkKernels PRIVATE ${COMMON_INCLUDE_DIRS})

add_library(
  llmSdkCore STATIC
  ${COMMON_CPP_SRCS}
  ${COMMON_CU_SRCS}
  ${DECODER_CPP_SRCS}
  ${EAGLE_CPP_SRCS}
  ${EAGLE_CU_SRCS}
  ${ENGINE_CPP_SRCS}
  ${KERNELS_CPP_SRCS}
  ${KERNELS_CU_SRCS}
  ${SAMPLER_CU_SRCS})
target_include_directories(llmSdkCore PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(llmSdkCore PUBLIC llmSdkKernels)

# Generate tokenizer C++ static library
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(tokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(tokenizer PRIVATE ${COMMON_INCLUDE_DIRS})

# Generate the TensorRT plugin shared libraries
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(NvInfer_edgellm_plugin commonLibraryExt llmSdkKernels
                      ${CUDA_DRIVER_LIB})
add_cross_build_link_options(NvInfer_edgellm_plugin)
