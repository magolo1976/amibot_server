cmake_minimum_required(VERSION 3.14...3.28)
project(tts_comms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TTSCOMMS_VERSION_MAJOR 1)
set(TTSCOMMS_VERSION_MINOR 0)
set(TTSCOMMS_VERSION_PATCH 0)
set(TTSCOMMS_VERSION_BASE "${TTSCOMMS_VERSION_MAJOR}.${TTSCOMMS_VERSION_MINOR}.${TTSCOMMS_VERSION_PATCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options(/utf-8 /W3)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

add_library(tts_comms SHARED
    include/tts_comms.h
    src/tts_comms.cpp
)

set_target_properties(tts_comms PROPERTIES
    VERSION ${TTSCOMMS_VERSION_BASE}
    SOVERSION ${TTSCOMMS_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(tts_comms PUBLIC include)
target_compile_features(tts_comms PRIVATE cxx_std_17)
target_compile_definitions(tts_comms PRIVATE TTS_COMMS_BUILD_SHARED)
target_compile_definitions(tts_comms INTERFACE TTS_COMMS_USE_SHARED)

# Exportar automaticamente la libreria resultante y las voces a server/libraries/tts_comms
set(LIBRARIES_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../../libraries/tts_comms")
if (WIN32)
    add_custom_command(TARGET tts_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}/voices"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:tts_comms>" "${LIBRARIES_OUTPUT_DIR}/tts_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:tts_comms>" "${LIBRARIES_OUTPUT_DIR}/tts_comms.lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/tts_comms.h" "${LIBRARIES_OUTPUT_DIR}/tts_comms.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/tts_comms.js" "${LIBRARIES_OUTPUT_DIR}/tts_comms.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/tts_cli.js" "${LIBRARIES_OUTPUT_DIR}/tts_cli.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/phonemize.js" "${LIBRARIES_OUTPUT_DIR}/phonemize.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/splitter.js" "${LIBRARIES_OUTPUT_DIR}/splitter.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/voices.js" "${LIBRARIES_OUTPUT_DIR}/voices.js"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/voices" "${LIBRARIES_OUTPUT_DIR}/voices"
        COMMENT "Copiando libreria compilada tts_comms y voces (Windows) a server/libraries/tts_comms..."
    )
else()
    add_custom_command(TARGET tts_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}/voices"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:tts_comms>" "${LIBRARIES_OUTPUT_DIR}/libtts_comms.so"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/tts_comms.h" "${LIBRARIES_OUTPUT_DIR}/tts_comms.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/tts_comms.js" "${LIBRARIES_OUTPUT_DIR}/tts_comms.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/tts_cli.js" "${LIBRARIES_OUTPUT_DIR}/tts_cli.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/phonemize.js" "${LIBRARIES_OUTPUT_DIR}/phonemize.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/splitter.js" "${LIBRARIES_OUTPUT_DIR}/splitter.js"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/js/voices.js" "${LIBRARIES_OUTPUT_DIR}/voices.js"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/voices" "${LIBRARIES_OUTPUT_DIR}/voices"
        COMMENT "Copiando libreria compilada tts_comms y voces (Linux) a server/libraries/tts_comms..."
    )
endif()
