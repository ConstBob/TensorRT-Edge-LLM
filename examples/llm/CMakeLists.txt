# Build executable for engine generation.
add_executable(llm_build llm_build.cpp)
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt)
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_chat llm_chat.cpp)
target_link_libraries(llm_chat PRIVATE llmSdkCore tokenizer commonLibraryExt)
target_include_directories(llm_chat PRIVATE ${COMMON_INCLUDE_DIRS})

add_executable(llm_benchmark llm_benchmark.cpp)
target_link_libraries(llm_benchmark PRIVATE llmSdkCore commonLibraryExt)
target_include_directories(llm_benchmark PRIVATE ${COMMON_INCLUDE_DIRS})

add_executable(llm_accuracy llm_accuracy.cpp)
target_link_libraries(llm_accuracy PRIVATE llmSdkCore tokenizer
                                           commonLibraryExt)
target_include_directories(llm_accuracy PRIVATE ${COMMON_INCLUDE_DIRS})

# Resolve linker error for AArch64 cross build. cudadevrt and cudart_static are
# introduced into the linking of executables
if(DEFINED AARCH64_CROSS_BUILD)
  target_link_options(llm_chat PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(llm_benchmark PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(llm_accuracy PRIVATE "-L${CUDA_DIR}/lib")
endif()
