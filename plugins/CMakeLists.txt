cmake_minimum_required(VERSION 3.18)
project(mha LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a-real)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

link_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --expt-relaxed-constexpr -t 0") # -res-usage
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills") # -Werror -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")

file(GLOB CUBIN_SRCS "xqa/cubin/*.cpp")

message(STATUS "STATUS: CUBIN_SRCS= ${CUBIN_SRCS}")

find_package(GTest REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

add_executable(unitTests tests/xqaTest.cpp tests/refAttention.cpp ${CUBIN_SRCS})
target_include_directories (unitTests PUBLIC ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unitTests PUBLIC GTest::gtest GTest::gtest_main cuda cudart Eigen3::Eigen)