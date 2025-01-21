# Build executable for engine generation.
add_executable(vlm_build vlm_build.cpp)
target_link_libraries(vlm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt)
target_include_directories(vlm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})

file(GLOB_RECURSE QWEN2VL_CPP_SRCS "qwen2vl/*.cpp")

add_executable(vlm_chat vlm_chat.cpp ${QWEN2VL_CPP_SRCS})

target_link_libraries(vlm_chat PRIVATE llmSdkCore tokenizer commonLibraryExt)
target_include_directories(vlm_chat PRIVATE ${COMMON_INCLUDE_DIRS}
                                            ../../dependencies/stb)
if(DEFINED AARCH64_CROSS_BUILD)
  target_link_options(vlm_chat PRIVATE "-L${CUDA_DIR}/lib")
endif()
