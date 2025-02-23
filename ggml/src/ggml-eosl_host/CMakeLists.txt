message(STATUS "Using EOSL_HOST backend")

# add_compile_options() command must be before the add_library()/add_executable
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wno-gnu-statement-expression-from-macro-expansion>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-gnu-statement-expression-from-macro-expansion>)
endif()

ggml_add_backend_library(ggml-eosl_host
                         ggml-eosl_host.cpp
                         ../ggml-eosl_common/device_memory.cpp
                         ../ggml-eosl_common/device_memory.hpp
                         ../../include/ggml-eosl_host.h)

file(GLOB   GGML_HEADERS_EOSL_HOST "*.hpp" "*.h")
file(GLOB   GGML_SOURCES_EOSL_HOST "*.cpp" "*.c")
target_sources(ggml-eosl_host PRIVATE ${GGML_HEADERS_EOSL_HOST} ${GGML_SOURCES_EOSL_HOST})

#target_link_libraries(ggml-eosl_host PRIVATE ggml-base)
target_compile_features(ggml-eosl_host PRIVATE c_std_11 cxx_std_17)
#target_compile_features(ggml-eosl_host PRIVATE c_std_11 cxx_std_11)
target_include_directories(ggml-eosl_host PRIVATE . .. $ENV{XDMA_ROOT})
target_include_directories(ggml-eosl_host PRIVATE $ENV{FREERTOS_DIR}/eosl/vcu_118)
target_compile_definitions(ggml-eosl_host PUBLIC GGML_USE_EOSL_HOST)
target_link_libraries(ggml-eosl_host PRIVATE uuid)

