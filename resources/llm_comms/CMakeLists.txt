cmake_minimum_required(VERSION 3.14...3.28)
project(llm_comms C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Version del motor llm_comms
set(LLMCOMMS_VERSION_MAJOR 1)
set(LLMCOMMS_VERSION_MINOR 0)
set(LLMCOMMS_VERSION_PATCH 0)
set(LLMCOMMS_VERSION_BASE "${LLMCOMMS_VERSION_MAJOR}.${LLMCOMMS_VERSION_MINOR}.${LLMCOMMS_VERSION_PATCH}")
set(LLMCOMMS_VERSION "${LLMCOMMS_VERSION_BASE}-llm_comms")
set(LLMCOMMS_BUILD_COMMIT "llm_comms-v1")
set(BUILD_NUMBER 1)
set(BUILD_COMMIT "llm_comms-v1")

# Compilar dependencias internas (ggml) de forma estatica para generar un unico llm_comms.dll autocontenido
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

# Desactivar suites externas y utilidades innecesarias
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLMCOMMS_BUILD_UI OFF CACHE BOOL "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# 1. Compilar motor de tensores interno (estatico)
add_subdirectory(ggml)

# 2. Compilar libreria unificada llm_comms (dll + lib) y exportarla a server/libraries/llm_comms
add_subdirectory(src)
