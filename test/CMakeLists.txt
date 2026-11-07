cmake_minimum_required(VERSION 3.14...3.28)
project(amibot_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/utf-8 /W3)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

set(SERVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(LLM_API_DIR "${SERVER_DIR}/llm_api")
set(LLM_COMMS_LIB_DIR "${SERVER_DIR}/libraries/llm_comms")
set(STT_COMMS_LIB_DIR "${SERVER_DIR}/libraries/stt_comms")
set(TTS_COMMS_LIB_DIR "${SERVER_DIR}/libraries/tts_comms")

include_directories(
    ${LLM_API_DIR}/include
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

enable_testing()

# 1. Suite completa de Unit Tests y Tests de Integracion (63 aserciones)
add_executable(amibot_unit_tests
    unit_tests.cpp
    ${LLM_API_DIR}/src/api_core.cpp
    ${LLM_API_DIR}/src/http_server.cpp
)
target_link_libraries(amibot_unit_tests PRIVATE
    "${LLM_COMMS_IMPORT_LIB}"
    "${STT_COMMS_IMPORT_LIB}"
    "${TTS_COMMS_IMPORT_LIB}"
    ${PLATFORM_LIBS}
)
add_test(NAME AllUnitTests COMMAND amibot_unit_tests)

# 2. Pruebas individuales de cada libreria
add_executable(test_llm_comms test_llm_comms.cpp)
target_link_libraries(test_llm_comms PRIVATE "${LLM_COMMS_IMPORT_LIB}")

add_executable(test_stt_comms test_stt_comms.cpp)
target_link_libraries(test_stt_comms PRIVATE "${STT_COMMS_IMPORT_LIB}")

add_executable(test_tts_comms test_tts_comms.cpp)
target_link_libraries(test_tts_comms PRIVATE "${TTS_COMMS_IMPORT_LIB}")

if (WIN32)
    add_custom_command(TARGET amibot_unit_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LLM_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:amibot_unit_tests>/llm_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${STT_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:amibot_unit_tests>/stt_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TTS_COMMS_SHARED_LIB}" "$<TARGET_FILE_DIR:amibot_unit_tests>/tts_comms.dll"
    )
endif()
