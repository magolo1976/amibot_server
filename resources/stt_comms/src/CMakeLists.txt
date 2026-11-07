sttcomms_add_compile_flags()

set(STT_COMMS_CORE_SOURCES
    stt_comms_engine.cpp
    stt_comms.cpp
)

add_library(stt_comms SHARED
    ../include/stt_comms.h
    ../include/sttcomms-core.h
    sttcomms-arch.h
    ${STT_COMMS_CORE_SOURCES}
)

set_target_properties(stt_comms PROPERTIES
    VERSION ${STTCOMMS_VERSION_BASE}
    SOVERSION ${STTCOMMS_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(stt_comms PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(stt_comms PUBLIC ../include ../ggml/include)
target_compile_features(stt_comms PRIVATE cxx_std_17)

target_compile_definitions(stt_comms PRIVATE
    STT_COMMS_BUILD_SHARED
    STTCOMMS_VERSION="${STTCOMMS_VERSION}"
)
target_compile_definitions(stt_comms INTERFACE STT_COMMS_USE_SHARED)

find_package(Threads REQUIRED)
target_link_libraries(stt_comms PRIVATE ggml Threads::Threads)

# Exportar automaticamente la libreria resultante a server/libraries/stt_comms (Windows y Linux/Docker)
set(LIBRARIES_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../../libraries/stt_comms")
if (WIN32)
    add_custom_command(TARGET stt_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:stt_comms>" "${LIBRARIES_OUTPUT_DIR}/stt_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:stt_comms>" "${LIBRARIES_OUTPUT_DIR}/stt_comms.lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/stt_comms.h" "${LIBRARIES_OUTPUT_DIR}/stt_comms.h"
        COMMENT "Copiando libreria compilada stt_comms (Windows) a server/libraries/stt_comms..."
    )
else()
    add_custom_command(TARGET stt_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:stt_comms>" "${LIBRARIES_OUTPUT_DIR}/libstt_comms.so"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/stt_comms.h" "${LIBRARIES_OUTPUT_DIR}/stt_comms.h"
        COMMENT "Copiando libreria compilada stt_comms (Linux) a server/libraries/stt_comms..."
    )
endif()
