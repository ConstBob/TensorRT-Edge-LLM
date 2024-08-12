file(GLOB CU_SOURCES "*.cu")
file(GLOB CPP_SOURCES "*.cpp")
add_library(SamplingTopkLayer STATIC ${CU_SOURCES} ${CPP_SOURCES})
