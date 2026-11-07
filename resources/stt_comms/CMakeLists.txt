cmake_minimum_required(VERSION 3.14...3.28)
project(stt_comms C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Version del motor stt_comms
set(STTCOMMS_VERSION_MAJOR 1)
set(STTCOMMS_VERSION_MINOR 0)
set(STTCOMMS_VERSION_PATCH 0)
set(STTCOMMS_VERSION_BASE "${STTCOMMS_VERSION_MAJOR}.${STTCOMMS_VERSION_MINOR}.${STTCOMMS_VERSION_PATCH}")
set(STTCOMMS_VERSION "${STTCOMMS_VERSION_BASE}-stt_comms")
set(STTCOMMS_BUILD_COMMIT "stt_comms-v1")
set(BUILD_NUMBER 1)
set(BUILD_COMMIT "stt_comms-v1")

# Compilar dependencias internas (ggml) de forma estatica para generar un unico stt_comms.dll autocontenido
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

# Desactivar suites externas y utilidades innecesarias
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(STTCOMMS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STTCOMMS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

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

# 2. Compilar libreria unificada stt_comms (dll + lib) y exportarla a server/libraries/stt_comms
add_subdirectory(src)
