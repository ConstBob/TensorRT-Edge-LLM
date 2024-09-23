# aarch64_toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER /usr/local/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/local/bin/aarch64-linux-gnu-g++)

set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)


# Point CUDA to aarch cross targets.
set(CUDA_VERSION 12.4)
set(CUDA_DIR /usr/local/cuda/targets/aarch64-linux CACHE STRING "CUDA ROOT dir")

# Use host nvcc
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_FLAGS " -Xcompiler=\"-fPIC \"" CACHE STRING "" FORCE)

# Specify the architecture for CUDA
set(CMAKE_CUDA_ARCHITECTURES 87)  # 87 is for Jetson Orin (Ampere GPU with compute capability 8.7)

# Tell CMake how to search for the libraries and programs
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
