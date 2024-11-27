file(GLOB_RECURSE QWEN2VL_CPP_SRCS "qwen2vl/*.cpp")

add_executable(multimodal_runner multimodal_model_runner.cpp
                                 ${QWEN2VL_CPP_SRCS})

target_link_libraries(multimodal_runner PRIVATE llmSdkCore tokenizer
                                                commonLibraryExt)
target_include_directories(multimodal_runner PRIVATE ${COMMON_INCLUDE_DIRS}
                                                     ../../dependencies/stb)
