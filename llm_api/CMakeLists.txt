cmake_minimum_required(VERSION 3.14...3.28)
project(llm_api CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/utf-8 /W3)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Rutas a las librerias precompiladas en server/libraries/
set(LLM_COMMS_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/llm_comms")
set(STT_COMMS_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/stt_comms")
set(TTS_COMMS_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/tts_comms")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLM_COMMS_LIB_DIR}
    ${STT_COMMS_LIB_DIR}
    ${TTS_COMMS_LIB_DIR}
)

if (WIN32)
    set(LLM_COMMS_IMPORT_LIB "${LLM_COMMS_LIB_DIR}/llm_comms.lib")
    set(LLM_COMMS_SHARED_LIB "${LLM_COMMS_LIB_DIR}/llm_comms.dll")
    set(STT_COMMS_IMPORT_LIB "${STT_COMMS_LIB_DIR}/stt_comms.lib")
    set(STT_COMMS_SHARED_LIB "${STT_COMMS_LIB_DIR}/stt_comms.dll")
    set(TTS_COMMS_IMPORT_LIB "${TTS_COMMS_LIB_DIR}/tts_comms.lib")
    set(TTS_COMMS_SHARED_LIB "${TTS_COMMS_LIB_DIR}/tts_comms.dll")
    set(PLATFORM_LIBS ws2_32)
else()
    set(LLM_COMMS_IMPORT_LIB "${LLM_COMMS_LIB_DIR}/libllm_comms.so")
    set(LLM_COMMS_SHARED_LIB "${LLM_COMMS_LIB_DIR}/libllm_comms.so")
    set(STT_COMMS_IMPORT_LIB "${STT_COMMS_LIB_DIR}/libstt_comms.so")
    set(STT_COMMS_SHARED_LIB "${STT_COMMS_LIB_DIR}/libstt_comms.so")
    set(TTS_COMMS_IMPORT_LIB "${TTS_COMMS_LIB_DIR}/libtts_comms.so")
    set(TTS_COMMS_SHARED_LIB "${TTS_COMMS_LIB_DIR}/libtts_comms.so")
    find_package(Threads REQUIRED)
    set(PLATFORM_LIBS Threads::Threads)
endif()

add_compile_definitions(
    LLM_COMMS_USE_SHARED
    STT_COMMS_USE_SHARED
    TTS_COMMS_USE_SHARED
)

# 1. Nucleo de la API
add_library(llm_api_core STATIC
    src/api_core.cpp
    src/http_server.cpp
)
target_link_libraries(llm_api_core PUBLIC
    "${LLM_COMMS_IMPORT_LIB}"
    "${STT_COMMS_IMPORT_LIB}"
    "${TTS_COMMS_IMPORT_LIB}"
    ${PLATFORM_LIBS}
)

# 2. Ejecutable principal del Servidor API
add_executable(llm_api_server
    src/main.cpp
)
target_link_libraries(llm_api_server PRIVATE llm_api_core)

# En Windows, copiar las .dll junto al ejecutable para ejecucion directa
if (WIN32)
    add_custom_command(TARGET llm_api_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LLM_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:llm_api_server>/llm_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${STT_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:llm_api_server>/stt_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TTS_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:llm_api_server>/tts_comms.dll"
    )
endif()
