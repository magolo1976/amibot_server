# ==============================================================================
# Etapa 1: Compilacion nativa en Linux (llm_comms, stt_comms, tts_comms + API)
# ==============================================================================
FROM node:20-bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Copiar librerias base y compilar llm_comms, stt_comms y tts_comms hacia /workspace/libraries/
COPY libraries /workspace/libraries
COPY resources/llm_comms /workspace/resources/llm_comms
COPY resources/stt_comms /workspace/resources/stt_comms
COPY resources/tts_comms /workspace/resources/tts_comms

RUN cmake -S /workspace/resources/llm_comms -B /workspace/resources/llm_comms/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build /workspace/resources/llm_comms/build --config Release -j $(nproc) && \
    cmake -S /workspace/resources/stt_comms -B /workspace/resources/stt_comms/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build /workspace/resources/stt_comms/build --config Release -j $(nproc) && \
    cmake -S /workspace/resources/tts_comms -B /workspace/resources/tts_comms/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build /workspace/resources/tts_comms/build --config Release -j $(nproc)

# Instalar dependencias Node.js en Linux (ONNX Runtime nativo linux-x64 + phonemizer)
RUN cd /workspace/resources/tts_comms && npm install --omit=dev

# 2. Compilar la API C++17 enlazada contra las 3 librerias
COPY llm_api /workspace/llm_api
RUN cmake -S /workspace/llm_api -B /workspace/llm_api/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /workspace/llm_api/build --config Release -j $(nproc)

# ==============================================================================
# Etapa 2: Imagen ligera de produccion para VPS Linux
# ==============================================================================
FROM node:20-bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar librerias compartidas compiladas y registrar en ldconfig
COPY --from=builder /workspace/libraries/llm_comms/libllm_comms.so /usr/local/lib/libllm_comms.so
COPY --from=builder /workspace/libraries/stt_comms/libstt_comms.so /usr/local/lib/libstt_comms.so
COPY --from=builder /workspace/libraries/tts_comms/libtts_comms.so /usr/local/lib/libtts_comms.so
COPY --from=builder /workspace/libraries/tts_comms/voices /app/voices
RUN ldconfig

# Copiar worker neuronal ONNX (Kokoro) y sus modulos nativos de Linux
COPY --from=builder /workspace/resources/tts_comms/package.json /app/package.json
COPY --from=builder /workspace/resources/tts_comms/node_modules /app/node_modules
COPY --from=builder /workspace/libraries/tts_comms/*.js /app/

# Copiar binario de la API
COPY --from=builder /workspace/llm_api/build/bin/llm_api_server /app/llm_api_server

ENV PORT=8080
ENV MODELS_DIR=/app/models

EXPOSE 8080

CMD ["/app/llm_api_server"]
