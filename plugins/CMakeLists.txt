cmake_minimum_required(VERSION 3.18)
project(llama_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TEST "Build tests which require GTest and Eigen" OFF)

macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

set_ifndef(CMAKE_CUDA_ARCHITECTURES 80;86;89)


# Set TensorRT header includes and library dependencies.
if(NOT DEFINED TRT_PACKAGE_DIR)
    message(FATAL_ERROR "Please specify the -DTRT_PACKAGE_DIR when invoking CMake.")
endif()

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_PACKAGE_DIR} PATH_SUFFIXES include)
find_library(NVINFER_LIB nvinfer HINTS ${TRT_PACKAGE_DIR} PATH_SUFFIXES lib)

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

link_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib/stubs
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0") # -res-usage
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills") # -Werror -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

file(GLOB XQA_CUBIN_SRCS "xqa/cubin/*.cpp")
file(GLOB FMHA_CUBIN_SRCS "fmha-v2/cubin/*.cpp")

message(STATUS "FMHA_CUBIN_SRCS = ${FMHA_CUBIN_SRCS}")

if(BUILD_TEST)
    message(STATUS "Build tests. Looking for GTest and Eigen3")
    find_package(GTest REQUIRED)
    find_package (Eigen3 REQUIRED NO_MODULE)

    add_executable(unitTests tests/xqaTest.cpp tests/fmhaTest.cpp tests/ropeTest.cpp tests/hfAttentionTests.cpp utilKernels.cu contextFMHARunner.cpp
        decoderXQARunner.cpp tests/refAttention.cpp ${XQA_CUBIN_SRCS} ${FMHA_CUBIN_SRCS})
    target_include_directories (unitTests PUBLIC ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${TRT_INCLUDE_DIR})
    target_link_libraries(unitTests PUBLIC GTest::gtest GTest::gtest_main cuda cudart Eigen3::Eigen)
endif()

set(PLUGIN_SRCS attentionPlugin.cpp decoderXQARunner.cpp contextFMHARunner.cpp ${XQA_CUBIN_SRCS} ${FMHA_CUBIN_SRCS} utilKernels.cu)
add_library(LLamaPlugin SHARED ${PLUGIN_SRCS})
set_target_properties(LLamaPlugin PROPERTIES
    VERSION 1.0
    SOVERSION 1
)
target_include_directories(LLamaPlugin PRIVATE ${TRT_INCLUDE_DIR})
target_link_libraries(LLamaPlugin ${NVINFER_LIB} cuda cudart)
