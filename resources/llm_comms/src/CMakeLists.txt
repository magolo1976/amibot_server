llmcomms_add_compile_flags()

file(GLOB LLM_MODELS_SOURCES "models/*.cpp")

set(LLM_COMMS_CORE_SOURCES
    llm_comms_engine.cpp
    llm_comms.cpp
    llmcomms-adapter.cpp
    llmcomms-arch.cpp
    llmcomms-batch.cpp
    llmcomms-chat.cpp
    llmcomms-context.cpp
    llmcomms-cparams.cpp
    llmcomms-grammar.cpp
    llmcomms-graph.cpp
    llmcomms-hparams.cpp
    llmcomms-impl.cpp
    llmcomms-io.cpp
    llmcomms-kv-cache.cpp
    llmcomms-kv-cache-iswa.cpp
    llmcomms-kv-cache-dsa.cpp
    llmcomms-kv-cache-dsa-iswa.cpp
    llmcomms-kv-cache-msa.cpp
    llmcomms-kv-cache-dsv4.cpp
    llmcomms-memory.cpp
    llmcomms-moe-cache.cpp
    llmcomms-memory-hybrid.cpp
    llmcomms-memory-hybrid-iswa.cpp
    llmcomms-memory-hybrid-idx.cpp
    llmcomms-memory-recurrent.cpp
    llmcomms-mmap.cpp
    llmcomms-model-loader.cpp
    llmcomms-model-saver.cpp
    llmcomms-model.cpp
    llmcomms-quant.cpp
    llmcomms-sampler.cpp
    llmcomms-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llm_comms SHARED
    ../include/llm_comms.h
    ../include/llmcomms-core.h
    ${LLM_COMMS_CORE_SOURCES}
    unicode.h
    ${LLM_MODELS_SOURCES}
)

set_target_properties(llm_comms PROPERTIES
    VERSION ${LLMCOMMS_VERSION_BASE}
    SOVERSION ${LLMCOMMS_VERSION_MAJOR}
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
    POSITION_INDEPENDENT_CODE ON
)

# Excluir archivos principales de unity build
set_source_files_properties(${LLM_COMMS_CORE_SOURCES} ../include/llm_comms.h ../include/llmcomms-core.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llmcomms-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llmcomms-version.h @ONLY)

target_include_directories(llm_comms PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llm_comms PUBLIC ../include ../ggml/include)
target_compile_features(llm_comms PRIVATE cxx_std_17)

target_compile_definitions(llm_comms PRIVATE LLM_COMMS_BUILD_SHARED)
target_compile_definitions(llm_comms INTERFACE LLM_COMMS_USE_SHARED)

find_package(Threads REQUIRED)
target_link_libraries(llm_comms PRIVATE ggml Threads::Threads)

# Exportar automaticamente la libreria resultante a server/libraries/llm_comms (Windows y Linux/Docker)
set(LIBRARIES_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../../libraries/llm_comms")
if (WIN32)
    add_custom_command(TARGET llm_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:llm_comms>" "${LIBRARIES_OUTPUT_DIR}/llm_comms.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:llm_comms>" "${LIBRARIES_OUTPUT_DIR}/llm_comms.lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/llm_comms.h" "${LIBRARIES_OUTPUT_DIR}/llm_comms.h"
        COMMENT "Copiando libreria compilada llm_comms (Windows) a server/libraries/llm_comms..."
    )
else()
    add_custom_command(TARGET llm_comms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBRARIES_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:llm_comms>" "${LIBRARIES_OUTPUT_DIR}/libllm_comms.so"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/include/llm_comms.h" "${LIBRARIES_OUTPUT_DIR}/llm_comms.h"
        COMMENT "Copiando libreria compilada llm_comms (Linux) a server/libraries/llm_comms..."
    )
endif()
